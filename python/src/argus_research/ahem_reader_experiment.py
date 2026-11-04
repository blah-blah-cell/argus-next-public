"""Source-separated development pilot; frozen projections, trainable reader only.

Targets are cached separately and never passed into the online reader. This is
exploratory evidence, not a rerun or replacement of the original final study.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_contextual_reader import AhemContextualReader
from .ahem_forecast_audit import ProtocolMedian, causal_ewma, forecast_quality
from .ahem_population_audit import sha256, strict_records, select_queries
from .ahem_retrieval_audit import retrieval_quality
from .flow_v3_protocol import condition_record
from .flow_v6_final_evaluate import _load_frozen_checkpoint
from .flow_v6_objective import dynamic_loss, frozen_utility_distribution, retrieval_cross_entropy

INPUTS = ('local_state', 'query', 'keys', 'values', 'successor_dynamic',
          'source_ordinals', 'valid', 'anchor_ordinals', 'context_spans')
BLOCKED = ('thursday-22', 'friday-23', 'cic-ids2017')


def atomic_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@torch.no_grad()
def project_queries(model, record, anchors, *, horizon_packets=10_000):
    """Batch causal projections, with actual prefix lengths at cold start."""
    device = next(model.parameters()).device
    x = record.features
    previous = record.previous_same_flow
    if type(horizon_packets) is not int or horizon_packets < model.local_packets + 2:
        raise ValueError('projection horizon is too short')
    horizon, slots = horizon_packets, model.max_transitions
    batch = len(anchors)
    predecessor = x.new_zeros(batch, model.packet_dim)
    current = x[anchors].clone()
    triple = x.new_zeros(batch, slots, 3, model.packet_dim)
    ordinals = torch.full((batch, slots), -1, dtype=torch.long)
    valid = torch.zeros(batch, slots, dtype=torch.bool)
    key_gap = x.new_zeros(batch, slots)
    value_gap = x.new_zeros(batch, slots)
    query_gap = x.new_zeros(batch)
    for row, anchor in enumerate(anchors):
        oldest = max(0, anchor - horizon + 1)
        chain = [anchor]
        cursor = anchor
        while len(chain) < slots + 2 and int(previous[cursor]) >= oldest:
            cursor = int(previous[cursor])
            chain.append(cursor)
        if len(chain) > 1:
            predecessor[row] = x[chain[1]]
            query_gap[row] = model._normalized_gap(torch.tensor([chain[1]]),
                                                  torch.tensor([anchor]), torch.tensor([float(horizon)]))[0]
        for slot in range(max(0, len(chain) - 2)):
            successor, source, before = chain[slot:slot + 3]
            triple[row, slot] = x[[before, source, successor]]
            ordinals[row, slot] = source
            valid[row, slot] = True
            key_gap[row, slot] = model._normalized_gap(torch.tensor([before]), torch.tensor([source]),
                                                     torch.tensor([float(horizon)]))[0]
            value_gap[row, slot] = model._normalized_gap(torch.tensor([source]), torch.tensor([successor]),
                                                       torch.tensor([float(horizon)]))[0]
    # Do not pad inputs through a biased input projection: evaluate real prefixes.
    local_state = x.new_zeros(batch, model.width, device=device)
    lengths = [min(anchor + 1, model.local_packets) for anchor in anchors]
    for length in sorted(set(lengths)):
        rows = [row for row, size in enumerate(lengths) if size == length]
        packets = torch.stack([x[anchors[row] - length + 1:anchors[row] + 1] for row in rows]).to(device)
        local_state[rows] = model.local(packets)
    triple = triple.to(device)
    keys = model.project_state(triple[:, :, 0].reshape(-1, model.packet_dim),
                               triple[:, :, 1].reshape(-1, model.packet_dim), key_gap.flatten().to(device))
    values = model.project_value(triple[:, :, 1].reshape(-1, model.packet_dim),
                                 triple[:, :, 2].reshape(-1, model.packet_dim), value_gap.flatten().to(device))
    keys = keys.reshape(batch, slots, -1) * valid.to(device)[..., None]
    values = values.reshape(batch, slots, -1) * valid.to(device)[..., None]
    return dict(zip(INPUTS, (local_state, model.project_state(predecessor.to(device), current.to(device),
                                                             query_gap.to(device)), keys, values,
                            triple[:, :, 2, :2], ordinals.to(device), valid.to(device),
                            torch.tensor(anchors, device=device),
                            torch.full((batch,), float(horizon), device=device)), strict=True))


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(token in args.journal.name.lower() for token in BLOCKED):
        raise ValueError('original final-test sources are forbidden in this pilot')
    if args.samples <= 0 or args.batch <= 0:
        raise ValueError('samples and batch must be positive')
    source_hash = sha256(args.journal)
    provenance = args.journal.with_suffix(args.journal.suffix + '.manifest.json')
    if args.require_official_manifest:
        manifest = json.loads(provenance.read_text(encoding='utf-8'))
        if (manifest.get('synthetic_data') is not False or manifest.get('dataset') != 'CSE-CIC-IDS2018'
                or manifest.get('journal', {}).get('sha256') != source_hash
                or not manifest.get('archive', {}).get('source_uri', '').startswith('s3://cse-cic-ids2018/')):
            raise ValueError('official conversion provenance is invalid')
    model, _, _ = _load_frozen_checkpoint(args.checkpoints, args.seed, AHEM_NAME,
                                          EXPECTED_CHECKPOINT_SHA256[args.seed][AHEM_NAME])
    model = model.to(args.device).eval()
    records = [condition_record(record) for record in strict_records(args.journal)]
    coordinates = select_queries(records, args.samples, 20261002)
    if not coordinates:
        raise ValueError('journal has no observed same-flow successor queries')
    chunks = {name: [] for name in (*INPUTS, 'target', 'persistence', 'ewma', 'coordinates', 'anchor_features')}
    for record_index in sorted({index for index, _ in coordinates}):
        record = records[record_index]
        anchors = [anchor for index, anchor in coordinates if index == record_index]
        ewma = causal_ewma(record.features.numpy(), record.previous_same_flow.numpy())
        for start in range(0, len(anchors), args.batch):
            selected = anchors[start:start + args.batch]
            projected = project_queries(model, record, selected)
            for name in INPUTS:
                chunks[name].append(projected[name].cpu())
            chunks['target'].append(record.features[record.next_same_flow[selected], :2])
            chunks['persistence'].append(record.features[selected, :2])
            chunks['anchor_features'].append(record.features[selected])
            chunks['ewma'].append(torch.as_tensor(ewma[selected]))
            chunks['coordinates'].append(torch.tensor([[record_index, anchor] for anchor in selected]))
        print(json.dumps({'stage': 'projection', 'record': record_index, 'queries': len(anchors)}), flush=True)
    payload = {'schema': 'argus.reader-projection-cache/v1', 'role': args.role,
               'source': args.journal.name, 'source_sha256': source_hash, 'seed': args.seed,
               'conversion_manifest_sha256': sha256(provenance) if provenance.exists() else None,
               'official_manifest_verified': args.require_official_manifest,
               'backbone_sha256': EXPECTED_CHECKPOINT_SHA256[args.seed][AHEM_NAME],
               'synthetic_data': False, 'locked_test_opened': False,
               'sampling': 'uniform over all observed-successor queries, including no memory',
               'record_boundaries_preserved': True, 'tensors': {name: torch.cat(parts) for name, parts in chunks.items()}}
    atomic_save(payload, args.output)
    atomic_json({key: value for key, value in payload.items() if key != 'tensors'} |
                {'cache_sha256': sha256(args.output), 'queries': len(coordinates)},
                args.output.with_suffix('.manifest.json'))


def read_cache(path, role, seed):
    cache = torch.load(path, map_location='cpu', weights_only=True)
    manifest = json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    if manifest['cache_sha256'] != sha256(path):
        raise ValueError('cache hash mismatch')
    if cache['schema'] != 'argus.reader-projection-cache/v1' or cache['role'] != role or cache['seed'] != seed:
        raise ValueError('cache role, schema, or seed mismatch')
    if cache['locked_test_opened'] or cache['synthetic_data']:
        raise ValueError('illegal cache provenance')
    if cache['backbone_sha256'] != EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME]:
        raise ValueError('cache backbone mismatch')
    tensors = cache['tensors']
    n = len(tensors['target'])
    if not n or any(len(value) != n or (value.is_floating_point() and not torch.isfinite(value).all())
                    for value in tensors.values()):
        raise ValueError('empty, nonfinite, or unaligned cache')
    return cache


def infer(model, tensors, indices, device):
    return model._read(*(tensors[name][indices].to(device) for name in INPUTS))


def resident_inputs(cache, device):
    """Reuse frozen projections in VRAM without changing batches or RNG."""
    requested = torch.device(device)
    if requested.type == 'cuda' and requested.index is None:
        requested = torch.device('cuda', torch.cuda.current_device())
    resident = cache.get('_resident_inputs')
    if resident is None or any(value.device.type != requested.type or
                               (requested.index is not None and requested.type != 'cpu' and value.device.index != requested.index)
                               for value in resident.values()):
        cache['_resident_inputs'] = {name: cache['tensors'][name].to(device)
                                     for name in (*INPUTS, 'target')}
    return cache['_resident_inputs']


@torch.no_grad()
def evaluate(model, cache, batch, device, median=None, *, infer_fn=infer):
    predictions, weights = [], []
    tensors = cache['tensors']
    inputs = resident_inputs(cache, device)
    for start in range(0, len(tensors['target']), batch):
        out = infer_fn(model, inputs, slice(start, start + batch), device)
        predictions.append(out.prediction.cpu())
        weights.append(out.retrieval_weights.cpu())
    result = {'forecast': forecast_quality(torch.cat(predictions).numpy(), tensors['target'].numpy()),
              'retrieval': retrieval_quality(tensors['successor_dynamic'], tensors['target'],
                                             tensors['valid'], torch.cat(weights))}
    for name in ('persistence', 'ewma'):
        result[name] = forecast_quality(tensors[name].numpy(), tensors['target'].numpy())
    if median is not None:
        result['protocol_median_train_only'] = forecast_quality(median.predict(tensors['anchor_features'].numpy()),
                                                                tensors['target'].numpy())
    return result


def train(args):
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if plan['schema'] != 'argus.contextual-reader-development/v1' or plan['locked_test_access'] is not False:
        raise ValueError('wrong experiment plan')
    if args.output.exists():
        raise FileExistsError('immutable run directory already exists; do not restart a healthy job')
    caches = {role: read_cache(path, role, args.seed) for role, path in
              [('train', args.train), ('validation', args.validation), ('screen', args.screen)]}
    if len({cache['source_sha256'] for cache in caches.values()}) != 3 or len({cache['source'] for cache in caches.values()}) != 3:
        raise ValueError('source-disjoint train/validation/screen required')
    for role, cache in caches.items():
        if cache['source'] != plan['sources'][role] or len(cache['tensors']['target']) != plan['samples'][role]:
            raise ValueError('cache differs from planned source or sample count')
        if plan['require_official_manifest'] and not cache['official_manifest_verified']:
            raise ValueError('official source provenance required by plan')
    torch.manual_seed(args.seed)
    backbone, _, _ = _load_frozen_checkpoint(args.checkpoints, args.seed, AHEM_NAME,
                                             EXPECTED_CHECKPOINT_SHA256[args.seed][AHEM_NAME])
    backbone = backbone.to(args.device).eval()
    candidate = AhemContextualReader(backbone)
    optimizer = torch.optim.AdamW(candidate.reader_head.parameters(), lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
    median = ProtocolMedian.fit(caches['train']['tensors']['anchor_features'].numpy(),
                                caches['train']['tensors']['target'].numpy())
    args.output.mkdir(parents=True)
    binding = {'plan_sha256': sha256(args.plan), 'seed': args.seed,
               'caches': {role: sha256(path) for role, path in [('train', args.train), ('validation', args.validation), ('screen', args.screen)]},
               'backbone_sha256': EXPECTED_CHECKPOINT_SHA256[args.seed][AHEM_NAME],
               'study_role': 'exploratory_development', 'locked_test_opened': False,
               'torch_version': str(torch.__version__), 'candidate': candidate.research_contract(),
               'source_code_sha256': {name: sha256(Path(__file__).parent / name) for name in
                                      ('ahem_reader_experiment.py', 'ahem_contextual_reader.py', 'ahem_v6.py',
                                       'flow_v6_objective.py', 'ahem_retrieval_audit.py', 'ahem_forecast_audit.py')}}
    atomic_json(binding, args.output / 'RUN.json')
    training_inputs = resident_inputs(caches['train'], args.device)
    baseline = evaluate(backbone, caches['validation'], plan['batch'], args.device, median)
    best_loss = float('inf')
    stale = 0
    for epoch in range(1, plan['epochs'] + 1):
        candidate.train()
        order = torch.randperm(len(caches['train']['tensors']['target']))
        total = 0.0
        for start in range(0, len(order), plan['batch']):
            indices = order[start:start + plan['batch']].to(args.device)
            out = infer(candidate, training_inputs, indices, args.device)
            target = training_inputs['target'][indices]
            utility, _ = frozen_utility_distribution(out.successor_dynamic, target, out.valid,
                                                     temperature=plan['utility_temperature'])
            loss = dynamic_loss(out.prediction, target) + plan['retrieval_weight'] * retrieval_cross_entropy(out.retrieval_logits, utility, out.valid)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.reader_head.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        candidate.eval()
        validation = evaluate(candidate, caches['validation'], plan['batch'], args.device, median)
        # Forecast-protection constrained selection; screening is not accessed here.
        forecast = validation['forecast']['weighted_smooth_l1']
        rank_regret = validation['retrieval']['selected_oracle_regret']
        selection = rank_regret if forecast <= baseline['forecast']['weighted_smooth_l1'] * plan['forecast_ceiling_ratio'] and rank_regret is not None else float('inf')
        state = {'schema': 'argus.contextual-reader-checkpoint/v1', 'binding': binding,
                 'epoch': epoch, 'model_state': candidate.state_dict(), 'optimizer_state': optimizer.state_dict(),
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 'validation': validation, 'training_objective': total / len(order)}
        atomic_save(state, args.output / f'epoch-{epoch:03d}.pt')
        atomic_save(state, args.output / 'latest.pt')
        if selection < best_loss:
            best_loss, stale = selection, 0
            atomic_save(state, args.output / 'best.pt')
        else:
            stale += 1
        row = {'epoch': epoch, 'training_objective': state['training_objective'], 'validation': validation,
               'selected': selection < float('inf'), 'epoch_sha256': sha256(args.output / f'epoch-{epoch:03d}.pt')}
        with (args.output / 'progress.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({'epoch': epoch, 'forecast_loss': forecast, 'retrieval_regret': rank_regret}), flush=True)
        if stale >= plan['patience']:
            break
    if not (args.output / 'best.pt').exists():
        atomic_json({'status': 'failed_forecast_protection', 'baseline_validation': baseline}, args.output / 'RESULT.json')
        return
    checkpoint = torch.load(args.output / 'best.pt', map_location=args.device, weights_only=True)
    candidate.load_state_dict(checkpoint['model_state'], strict=True)
    candidate.eval()
    # Freeze selected weights before any development-screen metrics are read.
    atomic_json({'best_sha256': sha256(args.output / 'best.pt'), 'epoch': checkpoint['epoch'],
                 'role': 'development_screen_not_final_test'}, args.output / 'FROZEN.json')
    screen = evaluate(candidate, caches['screen'], plan['batch'], args.device, median)
    original = evaluate(backbone, caches['screen'], plan['batch'], args.device, median)
    criteria = None
    if 'screen_targets' in plan:
        targets = plan['screen_targets']
        accuracy = screen['retrieval']['multi_candidate_top1_tie_aware']
        recency = screen['retrieval']['recency_top1_tie_aware']
        regret = screen['retrieval']['selected_oracle_regret']
        reference_regret = original['retrieval']['selected_oracle_regret']
        criteria = {
            'retrieval_gain': accuracy is not None and recency is not None and
                              accuracy - recency >= targets['tie_aware_multi_top1_gain_over_recency_percentage_points'] / 100,
            'regret_reduction': regret is not None and reference_regret is not None and reference_regret > 0 and
                                regret <= reference_regret * (1 - targets['selected_oracle_regret_relative_reduction_vs_v61']),
            'forecast_protection': screen['forecast']['weighted_smooth_l1'] <= original['forecast']['weighted_smooth_l1'] *
                                   targets['forecast_loss_ratio_ceiling_vs_v61']
        }
    atomic_json({'status': 'complete', 'candidate': screen, 'original_v61': original,
                 'per_seed_screen_criteria': criteria, 'automatic_promotion': False,
                 'study_role': 'exploratory_development', 'publication_claim': False,
                 'locked_test_opened': False}, args.output / 'RESULT.json')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--journal', type=Path, required=True)
    prep.add_argument('--role', choices=('train', 'validation', 'screen'), required=True)
    prep.add_argument('--samples', type=int, required=True)
    prep.add_argument('--batch', type=int, default=256)
    prep.add_argument('--require-official-manifest', action='store_true')
    fit = sub.add_parser('train')
    for name in ('plan', 'train', 'validation', 'screen'):
        fit.add_argument('--' + name, type=Path, required=True)
    for command in (prep, fit):
        command.add_argument('--checkpoints', type=Path, required=True)
        command.add_argument('--seed', type=int, choices=(20260906, 20260907, 20260908), required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
        command.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError('threads must be positive')
    torch.set_num_threads(args.threads)
    (prepare if args.command == 'prepare' else train)(args)


if __name__ == '__main__':
    main()

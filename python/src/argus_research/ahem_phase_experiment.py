"""Bounded phase-reader development; new causal caches, no final-test interface."""
import argparse
import json
import math
import os
from pathlib import Path

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_forecast_audit import causal_ewma, forecast_quality
from .ahem_phase_reader import AhemPhaseReader, PHASE_INPUTS, phase_infer, project_phase_queries
from .ahem_population_audit import sha256, strict_records, select_queries
from .ahem_reader_experiment import INPUTS, atomic_save, atomic_json
from .ahem_reader_ranking import tie_aware_regret_ranking
from .ahem_retrieval_audit import retrieval_quality
from .ahem_top1_reader import best_oracle_margin
from .flow_v3_protocol import condition_record
from .flow_v6_final_evaluate import _load_frozen_checkpoint
from .flow_v6_objective import dynamic_loss


def validate_plan(plan):
    if (plan['schema'] != 'argus.phase-reader-development/v1' or plan['locked_test_access'] is not False
            or plan['runtime_admission'] is not False or plan['synthetic_training'] is not False
            or plan['sources'] != {'train': 'Wednesday-14-02-2018.arguslr', 'validation': 'Thursday-01-03-2018.arguslr'}
            or plan['variants'] != {'phase-soft': None, 'phase-top2': 2}
            or plan['phase_packets'] != 4 or plan['horizon_packets'] != 10000
            or plan['global_context_statistics'] != ['causal_mean', 'causal_population_std']
            or plan['frozen_local_receptive_field'] != 31
            or plan['seeds'] != [20260906, 20260907, 20260908]):
        raise ValueError('unplanned phase protocol or forbidden source')


def prepare_cache(base, journal, role, seed, plan, plan_hash, output, *, official=True, samples=None):
    if output.exists():
        raise FileExistsError('immutable phase cache exists')
    source_hash = sha256(journal)
    provenance_path = journal.with_suffix(journal.suffix + '.manifest.json')
    if official:
        provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        if (journal.name != plan['sources'][role] or source_hash != plan['source_sha256'][role]
                or provenance.get('synthetic_data') is not False
                or provenance.get('dataset') != 'CSE-CIC-IDS2018'
                or provenance.get('journal', {}).get('sha256') != source_hash
                or not provenance.get('archive', {}).get('source_uri', '').startswith('s3://cse-cic-ids2018/')):
            raise ValueError('official source/hash/provenance mismatch')
    records = [condition_record(record) for record in strict_records(journal)]
    count = plan['samples'][role] if samples is None else samples
    coordinates = select_queries(records, count, plan['sampling_seed'])
    if len(coordinates) != count:
        raise ValueError('insufficient observed-successor coordinates')
    candidate = AhemPhaseReader(base, phase_width=plan['phase_width'])
    chunks = {name: [] for name in (*INPUTS, *PHASE_INPUTS, 'target', 'persistence', 'ewma', 'coordinates')}
    for record_index in sorted({i for i, _ in coordinates}):
        record = records[record_index]
        phase_cache = {}
        anchors = [a for i, a in coordinates if i == record_index]
        ewma = torch.as_tensor(causal_ewma(record.features.numpy(), record.previous_same_flow.numpy()))
        for start in range(0, len(anchors), plan['projection_batch']):
            selected = anchors[start:start + plan['projection_batch']]
            inputs = project_phase_queries(candidate, record, selected, horizon_packets=plan['horizon_packets'], phase_cache=phase_cache)
            for name, value in inputs.items():
                chunks[name].append(value.cpu())
            chunks['target'].append(record.features[record.next_same_flow[selected], :2])
            chunks['persistence'].append(record.features[selected, :2])
            chunks['ewma'].append(ewma[selected])
            chunks['coordinates'].append(torch.tensor([[record_index, a] for a in selected]))
        print(json.dumps({'stage': 'phase_projection', 'role': role, 'record': record_index, 'queries': len(anchors)}), flush=True)
    cache = {'schema': 'argus.phase-cache/v1', 'role': role, 'seed': seed, 'plan_sha256': plan_hash,
             'source': journal.name, 'source_sha256': source_hash, 'official': official,
             'conversion_manifest_sha256': sha256(provenance_path) if official else None,
             'backbone_sha256': EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME],
             'synthetic_training': False, 'locked_test_opened': False,
             'tensors': {name: torch.cat(parts) for name, parts in chunks.items()}}
    atomic_save(cache, output)
    atomic_json({k: v for k, v in cache.items() if k != 'tensors'} |
                {'cache_sha256': sha256(output), 'queries': count}, output.with_suffix('.manifest.json'))
    return cache


def load_cache(path, role, seed, plan_hash, plan):
    manifest = json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    if sha256(path) != manifest['cache_sha256']:
        raise ValueError('phase cache digest mismatch')
    cache = torch.load(path, weights_only=True, map_location='cpu')
    if (cache['schema'] != 'argus.phase-cache/v1' or cache['role'] != role or cache['seed'] != seed
            or cache['plan_sha256'] != plan_hash or cache['official'] is not True
            or cache['synthetic_training'] is not False or cache['locked_test_opened'] is not False
            or cache['source'] != plan['sources'][role] or cache['source_sha256'] != plan['source_sha256'][role]
            or cache['backbone_sha256'] != EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME]):
        raise ValueError('phase cache provenance mismatch')
    n = len(cache['tensors']['target'])
    if n != plan['samples'][role] or any(len(t) != n or (t.is_floating_point() and not torch.isfinite(t).all())
                                        for t in cache['tensors'].values()):
        raise ValueError('invalid phase cache contents')
    return cache


@torch.no_grad()
def evaluate_phase(model, tensors, batch, device, *, control=None):
    forecasts, weights, gates = [], [], []
    for start in range(0, len(tensors['target']), batch):
        indices = slice(start, start + batch)
        if control in ('recency', 'uniform', 'original_full_mask'):
            ordinary = [tensors[name][indices].to(device) for name in INPUTS]
            logits = None
            if control != 'original_full_mask':
                ordinary[6] = tensors['phase_valid'][indices].to(device)
                logits = torch.zeros_like(ordinary[6], dtype=ordinary[1].dtype)
                if control == 'recency':
                    chosen = ordinary[6].long().argmax(-1)
                    logits.scatter_(1, chosen[:, None], 10000)
            out = model._read(*ordinary, logits_override=logits)
        elif control is not None:
            raise ValueError('unknown phase control')
        else:
            out = phase_infer(model, tensors, indices, device)
        forecasts.append(out.prediction.cpu())
        weights.append(out.retrieval_weights.cpu())
        gates.append(out.memory_gate.cpu())
    prediction, weight = torch.cat(forecasts), torch.cat(weights)
    mask = tensors['valid'] if control == 'original_full_mask' else tensors['phase_valid']
    mixture = (weight[..., None] * tensors['successor_dynamic']).sum(1)
    available = mask.any(-1)
    return {'forecast': forecast_quality(prediction.numpy(), tensors['target'].numpy()),
            'retrieval': retrieval_quality(tensors['successor_dynamic'], tensors['target'], mask, weight),
            'raw_read_loss_available': float(dynamic_loss(mixture[available], tensors['target'][available])) if available.any() else None,
            'mean_gate': float(torch.cat(gates).mean()),
            'phase_expired_slots': int((tensors['valid'] & ~tensors['phase_valid']).sum())}


def selection(metrics, reference):
    r, b = metrics['retrieval'], reference['retrieval']
    numbers = [metrics['forecast']['weighted_smooth_l1'], reference['forecast']['weighted_smooth_l1'],
               r['multi_candidate_top1_tie_aware'], b['multi_candidate_top1_tie_aware'],
               r['selected_oracle_regret'], b['selected_oracle_regret']]
    if any(value is None for value in numbers):
        return None
    if any(not math.isfinite(value) or value < 0 for value in numbers):
        raise ValueError('nonfinite or negative phase selection metric')
    if (r['multi_candidate_top1_tie_aware'] is None or b['multi_candidate_top1_tie_aware'] is None
            or r['multi_candidate_top1_tie_aware'] < b['multi_candidate_top1_tie_aware']
            or r['selected_oracle_regret'] > b['selected_oracle_regret']):
        return None
    return (-metrics['forecast']['weighted_smooth_l1'], -r['selected_oracle_regret'], r['multi_candidate_top1_tie_aware'])


def fit_phase(base, candidate, caches, plan, output, binding, device):
    if output.exists():
        raise FileExistsError('immutable phase trial exists; never restart healthy jobs')
    output.mkdir(parents=True)
    original = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    reference = evaluate_phase(base, caches['validation']['tensors'], plan['batch'], device)
    score = selection(reference, reference)
    if score is None:
        raise ValueError('validation lacks multi-candidate queries')
    atomic_json(binding | {'candidate': candidate.research_contract()}, output / 'RUN.json')
    controls = {'matched_original': reference}
    for control in ('recency', 'uniform', 'original_full_mask'):
        controls[control] = evaluate_phase(base, caches['validation']['tensors'], plan['batch'], device, control=control)
    for control in ('persistence', 'ewma'):
        tensors = caches['validation']['tensors']
        controls[control] = forecast_quality(tensors[control].numpy(), tensors['target'].numpy())
    atomic_json(controls, output / 'CONTROLS.json')
    atomic_json(evaluate_phase(candidate, caches['validation']['tensors'], plan['batch'], device), output / 'INITIAL.json')
    train = {name: tensor.to(device) for name, tensor in caches['train']['tensors'].items()}
    parameters = [p for p in candidate.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
    generator = torch.Generator().manual_seed(binding['seed'])
    stale, best_epoch = 0, None
    for epoch in range(1, plan['epochs'] + 1):
        candidate.train()
        order = torch.randperm(len(train['target']), generator=generator)
        total = 0.
        for start in range(0, len(order), plan['batch']):
            idx = order[start:start + plan['batch']].to(device)
            out = phase_infer(candidate, train, idx, device)
            target = train['target'][idx]
            margin, _ = best_oracle_margin(out.retrieval_logits, out.successor_dynamic, target, out.valid,
                                          tie_tolerance=plan['tie_tolerance'], margin=plan['margin'])
            regret, _ = tie_aware_regret_ranking(out.retrieval_logits, out.successor_dynamic, target, out.valid,
                                                minimum_regret=0., tie_tolerance=plan['tie_tolerance'], margin=plan['margin'])
            mix = (out.retrieval_weights[..., None] * out.successor_dynamic).sum(1)
            available = out.valid.any(-1)
            read_loss = dynamic_loss(mix[available], target[available]) if available.any() else out.prediction.sum() * 0
            loss = margin + plan['regret_weight'] * regret + plan['raw_read_weight'] * read_loss + plan['forecast_weight'] * dynamic_loss(out.prediction, target)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite phase objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        candidate.eval()
        metrics = evaluate_phase(candidate, caches['validation']['tensors'], plan['batch'], device)
        new_score = selection(metrics, reference)
        improved = new_score is not None and new_score > score
        if improved:
            score, stale, best_epoch = new_score, 0, epoch
        else:
            stale += 1
        state = {'schema': 'argus.phase-checkpoint/v1', 'binding': binding, 'epoch': epoch,
                 'contract': candidate.research_contract(), 'model_state': candidate.state_dict(),
                 'optimizer_state': optimizer.state_dict(), 'sampler_rng': generator.get_state(),
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 'validation': metrics, 'training_loss': total / len(order)}
        path = output / f'epoch-{epoch:03d}.pt'
        atomic_save(state, path)
        atomic_save(state, output / 'latest.pt')
        if improved:
            atomic_save(state, output / 'best.pt')
        with (output / 'progress.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'epoch': epoch, 'selected': improved, 'loss': state['training_loss'],
                                     'validation': metrics, 'checkpoint_sha256': sha256(path)}, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({'epoch': epoch, 'selected': improved, 'forecast': metrics['forecast']['weighted_smooth_l1'],
                          'top1': metrics['retrieval']['multi_candidate_top1_tie_aware']}), flush=True)
        if stale >= plan['patience']:
            break
    if any(not torch.equal(candidate.state_dict()[name].cpu(), value) for name, value in original.items()):
        raise ValueError('frozen backbone changed')
    result = {'selected_epoch': best_epoch, 'backbone_unchanged': True, 'runtime_admission': False,
              'publication_claim': False, 'reference': reference,
              'status': 'development_candidate' if best_epoch else 'baseline_retained'}
    if best_epoch:
        saved = torch.load(output / 'best.pt', weights_only=True, map_location='cpu')
        result |= {'validation': saved['validation'], 'checkpoint_sha256': sha256(output / 'best.pt')}
    atomic_json(result, output / 'RESULT.json')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('prepare', 'train'))
    for name in ('plan', 'checkpoints', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--journal', type=Path)
    parser.add_argument('--role', choices=('train', 'validation'))
    parser.add_argument('--train', type=Path)
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--variant', choices=('phase-soft', 'phase-top2'))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    validate_plan(plan)
    if args.seed not in plan['seeds']:
        raise ValueError('unplanned seed')
    plan_hash = sha256(args.plan)
    torch.manual_seed(args.seed)
    base, _, _ = _load_frozen_checkpoint(args.checkpoints, args.seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[args.seed][AHEM_NAME])
    base = base.to(args.device).eval()
    if args.operation == 'prepare':
        if args.journal is None or args.role is None:
            parser.error('prepare requires --journal and --role')
        prepare_cache(base, args.journal, args.role, args.seed, plan, plan_hash, args.output)
    else:
        if args.variant is None or args.train is None or args.validation is None:
            parser.error('train requires --variant, --train and --validation')
        caches = {role: load_cache(path, role, args.seed, plan_hash, plan) for role, path in [('train', args.train), ('validation', args.validation)]}
        candidate = AhemPhaseReader(base, phase_width=plan['phase_width'], top_k=plan['variants'][args.variant])
        binding = {'seed': args.seed, 'variant': args.variant, 'plan_sha256': plan_hash,
                   'cache_sha256': {role: sha256(path) for role, path in [('train', args.train), ('validation', args.validation)]},
                   'backbone_sha256': EXPECTED_CHECKPOINT_SHA256[args.seed][AHEM_NAME],
                   'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
                   'torch_version': str(torch.__version__), 'device': args.device}
        fit_phase(base, candidate, caches, plan, args.output, binding, args.device)


if __name__ == '__main__':
    main()

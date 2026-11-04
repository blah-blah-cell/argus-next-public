"""Bounded real-laptop engineering study, with later-capture access barrier."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_contextual_reader import AhemContextualReader
from .ahem_forecast_audit import causal_ewma, forecast_quality
from .ahem_population_audit import sha256, strict_records, select_queries
from .ahem_reader_experiment import INPUTS, atomic_json, atomic_save, project_queries
from .ahem_retrieval_audit import retrieval_quality, query_coverage
from .ahem_top1_reader import AhemAnchorReader, AhemAnchorBlendReader
from .ahem_top1_train import fit, observed_inputs, observed_infer
from .flow_v3_protocol import condition_record
from .flow_v6_final_evaluate import _load_frozen_checkpoint
from .flow_v6_objective import dynamic_loss


def validate_sources(plan, directory):
    if (plan['schema'] != 'argus.local-reader-readiness/v1' or plan['screen_access'] is not False
            or plan['runtime_admission'] is not False
            or set(plan['variants']) != {'scalar-top1', 'anchor-top1', 'anchor-blend'}):
        raise ValueError('wrong local protocol')
    paths, dates, hashes = {}, {}, {}
    for role in ('train', 'validation', 'assessment'):
        paths[role], dates[role] = [], set()
        for name in plan['sources'][role]:
            if Path(name).name != name or not name.endswith('.v2.arguslr') or not name.startswith('2026-'):
                raise ValueError('only explicit 2026 real laptop journals accepted')
            path = directory / name
            digest = sha256(path)
            if digest in hashes.values():
                raise ValueError('duplicate source bytes across roles')
            hashes[name] = digest
            paths[role].append(path)
            dates[role].add(name[:10])
    if any(dates[a] & dates[b] for a, b in [('train', 'validation'), ('train', 'assessment'), ('validation', 'assessment')]):
        raise ValueError('local source dates must be disjoint')
    return paths, hashes


def load_role(paths, expected):
    records = {}
    for path in paths:
        if sha256(path) != expected[path.name]:
            raise ValueError('bound journal changed')
        records[path.name] = [condition_record(record) for record in strict_records(path)]
        if not records[path.name]:
            raise ValueError('empty journal')
    return records


@torch.no_grad()
def prepare_role(base, records, role, plan, seed):
    names = (*INPUTS, 'target', 'persistence', 'ewma', 'anchor_features', 'source_index', 'cluster_id')
    pieces = {name: [] for name in names}
    coverage = {}
    for source_index, (name, segments) in enumerate(records.items()):
        coords = select_queries(segments, plan['samples_per_source'][role], plan['sampling_seed'])
        coverage[name] = [query_coverage(r.previous_same_flow.numpy(), r.next_same_flow.numpy()) for r in segments]
        for index in sorted({index for index, _ in coords}):
            record = segments[index]
            anchors = [anchor for i, anchor in coords if i == index]
            ewma = causal_ewma(record.features.numpy(), record.previous_same_flow.numpy())
            clusters = {}
            for start in range(0, len(anchors), plan['batch']):
                selected = anchors[start:start + plan['batch']]
                projected = project_queries(base, record, selected)
                for field in INPUTS:
                    pieces[field].append(projected[field].cpu())
                pieces['target'].append(record.features[record.next_same_flow[selected], :2])
                pieces['persistence'].append(record.features[selected, :2])
                pieces['anchor_features'].append(record.features[selected])
                pieces['ewma'].append(torch.as_tensor(ewma[selected]))
                pieces['source_index'].append(torch.full((len(selected),), source_index, dtype=torch.int64))
                keys = []
                for anchor in selected:
                    flow = int(record.flow_ids[anchor])
                    if flow not in clusters:
                        identity = f'{name}/{record.segment_id}/{flow}'.encode()
                        clusters[flow] = int.from_bytes(hashlib.sha256(identity).digest()[:8], 'little') & ((1 << 63) - 1)
                    keys.append(clusters[flow])
                pieces['cluster_id'].append(torch.tensor(keys))
    return {'role': role, 'seed': seed, 'sources': list(records), 'coverage': coverage,
            'tensors': {name: torch.cat(parts) for name, parts in pieces.items()}}


def candidate_for(variant, base):
    if variant == 'scalar-top1':
        return AhemContextualReader(base)
    if variant == 'anchor-top1':
        return AhemAnchorReader(base)
    if variant == 'anchor-blend':
        return AhemAnchorBlendReader(base)
    raise ValueError('unplanned local reader')


@torch.no_grad()
def predictions(model, cache, device, batch, control=None):
    inputs = observed_inputs(cache, device)
    outputs, weights = [], []
    for start in range(0, len(inputs['target']), batch):
        indices = slice(start, start + batch)
        if control:
            valid = inputs['valid'][indices]
            logits = torch.zeros_like(valid, dtype=inputs['query'].dtype)
            if control == 'recency':
                logits[:, 0] = 10000
            elif control == 'nearest_observed_anchor':
                known = inputs['causal_anchor'][indices, None]
                losses = dynamic_loss(inputs['successor_dynamic'][indices], known.expand_as(inputs['successor_dynamic'][indices]), reduction='none')
                chosen = losses.masked_fill(~valid, torch.inf).argmin(-1)
                logits.scatter_(1, chosen[:, None], 10000)
            elif control != 'uniform':
                raise ValueError('unknown fixed control')
            out = model._read(*(inputs[name][indices] for name in INPUTS), logits_override=logits)
        else:
            out = observed_infer(model, inputs, indices, device)
        outputs.append(out.prediction.cpu())
        weights.append(out.retrieval_weights.cpu())
    return torch.cat(outputs), torch.cat(weights)


def summarize(prediction, weights, cache, mask=None):
    tensors = cache['tensors']
    if mask is None:
        mask = torch.ones(len(prediction), dtype=torch.bool)
    return {'forecast': forecast_quality(prediction[mask].numpy(), tensors['target'][mask].numpy()),
            'retrieval': retrieval_quality(tensors['successor_dynamic'][mask], tensors['target'][mask],
                                            tensors['valid'][mask], weights[mask]),
            'sampled_flow_clusters': len(torch.unique(tensors['cluster_id'][mask]))}


def run(args):
    if args.output.exists():
        raise FileExistsError('immutable local study already exists')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    paths, hashes = validate_sources(plan, args.journals)
    args.output.mkdir(parents=True)
    binding = {'schema': plan['schema'], 'plan_sha256': sha256(args.plan), 'plan': plan,
               'journal_sha256': hashes, 'source_code_sha256': {p.name: sha256(p) for p in
                sorted(Path(__file__).parent.glob('*.py'))}, 'device': args.device,
               'torch_version': str(torch.__version__), 'assessment_parsed': False}
    atomic_json(binding, args.output / 'STUDY.json')
    records = {role: load_role(paths[role], hashes) for role in ('train', 'validation')}
    frozen = {}
    for seed in plan['seeds']:
        base, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base = base.to(args.device).eval()
        caches = {role: prepare_role(base, records[role], role, plan, seed) for role in records}
        for role, cache in caches.items():
            atomic_save(cache, args.output / 'caches' / str(seed) / (role + '.pt'))
        for variant, changes in plan['variants'].items():
            torch.manual_seed(seed)
            candidate = candidate_for(variant, base)
            directory = args.output / 'runs' / f'{seed}-{variant}'
            run_binding = {'seed': seed, 'variant': variant, 'role': plan['role'],
                           'study_sha256': sha256(args.output / 'STUDY.json'),
                           'candidate': candidate.research_contract(), 'assessment_accessed': False,
                           'cache_sha256': {role: sha256(args.output / 'caches' / str(seed) / (role + '.pt')) for role in caches}}
            result = fit(base, candidate, caches, plan | changes, directory, run_binding, args.device)
            checkpoint = directory / 'best.pt' if result['selected_epoch'] else None
            frozen[f'{seed}/{variant}'] = {'result_sha256': sha256(directory / 'RESULT.json'),
                 'selected': result['status'], 'checkpoint_sha256': sha256(checkpoint) if checkpoint else EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME],
                 'uses_original_backbone': checkpoint is None}
        print(json.dumps({'stage': 'seed_frozen', 'seed': seed}), flush=True)
    atomic_json({'selections': frozen, 'assessment_parsed': False}, args.output / 'FROZEN.json')
    # No later-capture targets are parsed before all nine choices are frozen.
    atomic_json({'role': plan['role'], 'frozen_sha256': sha256(args.output / 'FROZEN.json'),
                 'assessment_sources': plan['sources']['assessment'], 'opened_once': True}, args.output / 'ACCESS.json')
    assessment_records = load_role(paths['assessment'], hashes)
    assessment = {}
    for seed in plan['seeds']:
        base, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base = base.to(args.device).eval()
        cache = prepare_role(base, assessment_records, 'assessment', plan, seed)
        rows = {}
        models = {'frozen_ahem': base}
        for variant in plan['variants']:
            item = frozen[f'{seed}/{variant}']
            if item['uses_original_backbone']:
                models[variant] = base
            else:
                path = args.output / 'runs' / f'{seed}-{variant}' / 'best.pt'
                if sha256(path) != item['checkpoint_sha256']:
                    raise ValueError('frozen checkpoint changed')
                model = candidate_for(variant, base).eval()
                state = torch.load(path, weights_only=True, map_location=args.device)
                model.load_state_dict(state['model_state'], strict=True)
                models[variant] = model
        for name, model in models.items():
            prediction, weights = predictions(model, cache, args.device, plan['batch'])
            rows[name] = summarize(prediction, weights, cache)
            rows[name]['per_source'] = {source: summarize(prediction, weights, cache, cache['tensors']['source_index'] == index)
                                        for index, source in enumerate(cache['sources'])}
            atomic_save({'prediction': prediction, 'weights': weights}, args.output / 'assessment' / str(seed) / (name + '.pt'))
        for control in ('recency', 'uniform', 'nearest_observed_anchor'):
            prediction, weights = predictions(base, cache, args.device, plan['batch'], control)
            rows[control] = summarize(prediction, weights, cache)
        for control in ('persistence', 'ewma'):
            rows[control] = {'forecast': forecast_quality(cache['tensors'][control].numpy(), cache['tensors']['target'].numpy())}
        assessment[str(seed)] = rows
        atomic_save(cache, args.output / 'assessment' / str(seed) / 'CACHE.pt')
        atomic_json(rows, args.output / 'assessment' / str(seed) / 'RESULT.json')
        print(json.dumps({'stage': 'assessment_saved', 'seed': seed}), flush=True)
    atomic_json({'role': plan['role'], 'publication_claim': False, 'raw_data_downloaded': False,
                 'original_locked_test_opened': False, 'frozen': frozen, 'assessment': assessment}, args.output / 'RESULT.json')


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'journals', 'checkpoints', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args)


if __name__ == '__main__':
    main()

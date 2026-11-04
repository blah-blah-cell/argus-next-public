"""Additional local capture check of already-frozen convex calibration."""
import argparse
import json
from pathlib import Path

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_forecast_audit import forecast_quality
from .ahem_local_readiness import load_role, prepare_role, predictions, summarize
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json, atomic_save
from .ahem_top1_reader import AhemAnchorReader, AhemAnchorBlendReader
from .flow_v6_final_evaluate import _load_frozen_checkpoint


def run(args):
    if args.output.exists():
        raise FileExistsError('immutable calibrated check exists')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if (plan['schema'] != 'argus.calibrated-local-check/v1' or plan['training'] is not False
            or plan['selection'] is not False):
        raise ValueError('wrong additional-check protocol')
    paths = []
    for name in plan['sources']:
        if Path(name).name != name or not name.startswith('2026-') or not name.endswith('.v2.arguslr'):
            raise ValueError('only local capture files accepted')
        paths.append(args.journals / name)
    calibration = json.loads((args.calibration / 'RESULT.json').read_text(encoding='utf-8'))
    hashes = {str(seed): sha256(args.calibration / str(seed) / 'calibrated.pt') for seed in plan['seeds']}
    for seed, digest in hashes.items():
        if digest != calibration['records'][seed]['checkpoint_sha256']:
            raise ValueError('frozen calibrated checkpoint hash changed')
    args.output.mkdir(parents=True)
    source_hashes = {p.name: sha256(p) for p in paths}
    atomic_json({'plan_sha256': sha256(args.plan), 'plan': plan, 'checkpoint_sha256': hashes,
                 'source_sha256': source_hashes, 'targets_parsed': False,
                 'source_code_sha256': sha256(Path(__file__))}, args.output / 'FROZEN.json')
    atomic_json({'opened_once': True, 'frozen_sha256': sha256(args.output / 'FROZEN.json'),
                 'role': plan['role']}, args.output / 'ACCESS.json')
    records = load_role(paths, source_hashes)
    report = {}
    for seed in plan['seeds']:
        base, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base = base.to(args.device).eval()
        state = torch.load(args.calibration / str(seed) / 'calibrated.pt', weights_only=True, map_location=args.device)
        calibrated = AhemAnchorBlendReader(base).eval()
        calibrated.load_state_dict(state['model_state'], strict=True)
        parent = AhemAnchorReader(base).eval()
        parent.load_state_dict({k: v for k, v in state['model_state'].items() if k != 'persistence_blend'}, strict=True)
        cache = prepare_role(base, records, 'assessment',
                              plan | {'samples_per_source': {'assessment': plan['samples_per_source']}}, seed)
        rows = {}
        for name, model in [('frozen_ahem', base), ('parent_anchor', parent), ('calibrated', calibrated)]:
            forecast, weights = predictions(model, cache, args.device, plan['batch'])
            rows[name] = summarize(forecast, weights, cache)
            rows[name]['per_source'] = {source: summarize(forecast, weights, cache, cache['tensors']['source_index'] == i)
                                        for i, source in enumerate(cache['sources'])}
            atomic_save({'prediction': forecast, 'weights': weights}, args.output / str(seed) / (name + '.pt'))
        for control in ('recency', 'nearest_observed_anchor'):
            forecast, weights = predictions(base, cache, args.device, plan['batch'], control)
            rows[control] = summarize(forecast, weights, cache)
        for control in ('persistence', 'ewma'):
            rows[control] = {'forecast': forecast_quality(cache['tensors'][control].numpy(), cache['tensors']['target'].numpy())}
        atomic_save(cache, args.output / str(seed) / 'CACHE.pt')
        atomic_json(rows, args.output / str(seed) / 'RESULT.json')
        report[str(seed)] = rows
    atomic_json({'role': plan['role'], 'report': report, 'publication_claim': False,
                 'runtime_admission': False, 'parameters_changed': False,
                 'original_locked_test_opened': False}, args.output / 'RESULT.json')
    print(json.dumps({seed: {'forecast': row['calibrated']['forecast']['weighted_smooth_l1'],
                            'persistence': row['persistence']['forecast']['weighted_smooth_l1'],
                            'retrieval': row['calibrated']['retrieval']['multi_candidate_top1_tie_aware'],
                            'original_retrieval': row['frozen_ahem']['retrieval']['multi_candidate_top1_tie_aware'],
                            'recency': row['recency']['retrieval']['multi_candidate_top1_tie_aware']}
                      for seed, row in report.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'journals', 'checkpoints', 'calibration', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args)


if __name__ == '__main__':
    main()

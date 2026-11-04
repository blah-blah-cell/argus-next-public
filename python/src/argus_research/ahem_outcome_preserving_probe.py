"""One frozen outcome-preserving read probe on later local capture dates."""
import argparse
import json
from pathlib import Path

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_phase_branch_audit import audit_model
from .ahem_phase_experiment import prepare_cache
from .ahem_phase_local_study import combine, cleanup_owned_cache
from .ahem_phase_reader import AhemPhaseReader
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json
from .flow_v6_final_evaluate import _load_frozen_checkpoint


def verify_sources(probe, study, journals):
    if (probe['schema'] != 'argus.outcome-preserving-probe/v1'
            or any(probe[key] is not False for key in ('training', 'downloads', 'synthetic_training', 'locked_test_access', 'runtime_admission'))
            or probe['primary_reader'] != 'phase-top2' or probe['primary_forecast'] != 'raw_read_persistence_fallback'
            or probe['seeds'] != [20260906, 20260907, 20260908]):
        raise ValueError('unplanned outcome probe')
    used_dates = {name[:10] for name in study['source_sha256']}
    paths, hashes = [], {}
    for name in probe['sources']:
        path = (journals / name).resolve()
        if (Path(name).name != name or path.parent != journals.resolve()
                or not name.startswith('2026-') or not name.endswith('.v2.arguslr')
                or name[:10] in used_dates):
            raise ValueError('source escape or reused study date')
        digest = sha256(path)
        if digest in study['source_sha256'].values() or digest in hashes.values():
            raise ValueError('duplicate source bytes')
        paths.append(path)
        hashes[name] = digest
    if not paths:
        raise ValueError('no probe sources')
    return paths, hashes


def run(args):
    if args.output.exists():
        raise FileExistsError('immutable probe already exists')
    probe = json.loads(args.plan.read_text(encoding='utf-8'))
    study = json.loads((args.study / 'STUDY.json').read_text(encoding='utf-8'))
    frozen = json.loads((args.study / 'FROZEN.json').read_text(encoding='utf-8'))
    plan = study['plan']
    paths, hashes = verify_sources(probe, study, args.journals)
    args.output.mkdir(parents=True)
    temporary = args.output / 'temporary-projections'
    temporary.mkdir()
    # This immutable binding is emitted before any new source is parsed.
    atomic_json({'plan': probe, 'plan_sha256': sha256(args.plan), 'source_sha256': hashes,
                 'prior_frozen_sha256': sha256(args.study / 'FROZEN.json'),
                 'source_parsed': False, 'device': args.device,
                 'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}, args.output / 'FROZEN.json')
    results, per_source, evidence = {}, {}, {}
    for seed in probe['seeds']:
        base, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base = base.to(args.device).eval()
        caches = []
        for index, path in enumerate(paths):
            if sha256(path) != hashes[path.name]:
                raise ValueError('source changed after binding')
            output = temporary / f'{seed}-probe-{index}.pt'
            caches.append(prepare_cache(base, path, 'assessment', seed, plan, sha256(args.plan), output,
                                        official=False, samples=probe['samples_per_source']))
            evidence[output.name] = sha256(output)
        tensors = combine(caches, 'local_probe')['tensors']
        rows = {'original': audit_model(base, tensors, plan['batch'], args.device)}
        source_rows = {path.name: {'original': audit_model(base, cache['tensors'], plan['batch'], args.device)}
                       for path, cache in zip(paths, caches, strict=True)}
        for variant, top_k in plan['variants'].items():
            selection = frozen['selections'][f'{seed}/{variant}']
            if selection['fallback']:
                raise ValueError('selected model unavailable; do not substitute')
            path = args.study / 'trials' / f'{seed}-{variant}' / 'best.pt'
            if sha256(path) != selection['checkpoint_sha256']:
                raise ValueError('selected model changed')
            model = AhemPhaseReader(base, phase_width=plan['phase_width'], top_k=top_k)
            model.load_state_dict(torch.load(path, weights_only=True, map_location=args.device)['model_state'], strict=True)
            model.eval()
            rows[variant] = audit_model(model, tensors, plan['batch'], args.device)
            for path, cache in zip(paths, caches, strict=True):
                source_rows[path.name][variant] = audit_model(model, cache['tensors'], plan['batch'], args.device)
        results[str(seed)] = rows
        per_source[str(seed)] = source_rows
        atomic_json(rows, args.output / f'{seed}.json')
        atomic_json(source_rows, args.output / f'{seed}-sources.json')
        print(json.dumps({'stage': 'probe_complete', 'seed': seed}), flush=True)
        del caches, tensors, model, base
    atomic_json({'role': probe['role'], 'report': results, 'per_source': per_source, 'training': False,
                 'runtime_admission': False, 'publication_claim': False}, args.output / 'RESULT.json')
    removed = cleanup_owned_cache(temporary, evidence)
    atomic_json({'removed_temporary_projections': removed, 'original_journals_and_checkpoints_preserved': True}, args.output / 'CLEANUP.json')
    print(json.dumps({'stage': 'complete', 'removed_temporary_projections': len(removed)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'journals', 'checkpoints', 'study', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args)


if __name__ == '__main__':
    main()

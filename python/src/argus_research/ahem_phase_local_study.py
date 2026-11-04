"""No-download real local traffic study, followed by narrow temporary-data cleanup."""
import argparse
import json
from pathlib import Path
import time

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_forecast_audit import forecast_quality
from .ahem_phase_experiment import prepare_cache, fit_phase, evaluate_phase
from .ahem_phase_reader import AhemPhaseReader
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json
from .flow_v6_final_evaluate import _load_frozen_checkpoint


def validate_local_sources(plan, journals):
    if (plan['schema'] != 'argus.phase-local-study/v1' or plan['downloads'] is not False
            or plan['synthetic_training'] is not False or plan['locked_test_access'] is not False
            or plan['runtime_admission'] is not False):
        raise ValueError('illegal local experiment scope')
    paths, hashes, dates = {}, {}, {}
    for role in ('train', 'validation', 'assessment'):
        paths[role], dates[role] = [], set()
        for name in plan['sources'][role]:
            if Path(name).name != name or not name.startswith('2026-') or not name.endswith('.v2.arguslr'):
                raise ValueError('only existing local journal filenames accepted')
            path = (journals / name).resolve()
            if path.parent != journals.resolve():
                raise ValueError('journal path escape')
            digest = sha256(path)
            if digest in hashes.values():
                raise ValueError('duplicate source bytes across roles')
            paths[role].append(path)
            hashes[name] = digest
            dates[role].add(name[:10])
    if any(dates[a] & dates[b] for a, b in (('train', 'validation'), ('train', 'assessment'), ('validation', 'assessment'))):
        raise ValueError('capture dates overlap across roles')
    return paths, hashes


def cleanup_owned_cache(directory, evidence):
    root = directory.resolve()
    removed = []
    for name, expected in evidence.items():
        path = (root / name).resolve()
        if (path.parent != root or not path.is_file() or path.suffix != '.pt'
                or sha256(path) != expected):
            raise ValueError('refusing unsafe or changed temporary cache deletion')
    # All exact targets were validated first; no recursive directory deletion.
    for name in evidence:
        (root / name).unlink()
        removed.append(name)
    return removed


def combine(caches, role):
    keys = caches[0]['tensors'].keys()
    if any(c['tensors'].keys() != keys for c in caches):
        raise ValueError('inconsistent phase cache fields')
    return {'role': role, 'tensors': {name: torch.cat([c['tensors'][name] for c in caches]) for name in keys}}


def run(args):
    if args.output.exists():
        raise FileExistsError('immutable local phase study exists')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    paths, hashes = validate_local_sources(plan, args.journals)
    args.output.mkdir(parents=True)
    temporary = args.output / 'temporary-projections'
    temporary.mkdir()
    code = {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}
    binding = {'plan': plan, 'plan_sha256': sha256(args.plan), 'source_sha256': hashes,
               'code_sha256': code, 'device': args.device, 'torch_version': str(torch.__version__),
               'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None,
               'assessment_parsed': False, 'downloads': False, 'synthetic_training': False}
    atomic_json(binding, args.output / 'STUDY.json')
    cache_hashes, selections = {}, {}
    started = time.perf_counter()
    def prepare_role(base, role, seed):
        caches = []
        for index, path in enumerate(paths[role]):
            if sha256(path) != hashes[path.name]:
                raise ValueError('bound source changed during run')
            output = temporary / f'{seed}-{role}-{index}.pt'
            cache = prepare_cache(base, path, role, seed, plan, binding['plan_sha256'], output,
                                  official=False, samples=plan['samples_per_source'][role])
            cache_hashes[output.name] = sha256(output)
            caches.append(cache)
        return combine(caches, role)
    for seed in plan['seeds']:
        base, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base = base.to(args.device).eval()
        caches = {role: prepare_role(base, role, seed) for role in ('train', 'validation')}
        for variant, top_k in plan['variants'].items():
            torch.manual_seed(seed)
            candidate = AhemPhaseReader(base, phase_width=plan['phase_width'], top_k=top_k)
            directory = args.output / 'trials' / f'{seed}-{variant}'
            trial = fit_phase(base, candidate, caches, plan, directory,
                              {'seed': seed, 'variant': variant, 'study_sha256': sha256(args.output / 'STUDY.json'),
                               'role': plan['role'], 'cache_sha256': dict(cache_hashes)}, args.device)
            selected = directory / 'best.pt' if trial['selected_epoch'] else None
            # Strictly validate actual epoch weights, even when baseline is retained.
            for checkpoint in directory.glob('epoch-*.pt'):
                state = torch.load(checkpoint, weights_only=True, map_location='cpu')
                restored = AhemPhaseReader(base.cpu(), phase_width=plan['phase_width'], top_k=top_k)
                restored.load_state_dict(state['model_state'], strict=True)
                if not all(torch.isfinite(t).all() for t in state['model_state'].values()):
                    raise ValueError('nonfinite local epoch checkpoint')
            base = base.to(args.device).eval()
            selections[f'{seed}/{variant}'] = {'selected_epoch': trial['selected_epoch'],
                'checkpoint_sha256': sha256(selected) if selected else EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME],
                'result_sha256': sha256(directory / 'RESULT.json'), 'fallback': selected is None}
        del caches, candidate
        print(json.dumps({'stage': 'seed_complete', 'seed': seed}), flush=True)
    atomic_json({'selections': selections, 'assessment_parsed': False}, args.output / 'FROZEN.json')
    atomic_json({'frozen_sha256': sha256(args.output / 'FROZEN.json'), 'role': 'post_selection_local_engineering_assessment',
                 'previously_available_traffic': True}, args.output / 'ACCESS.json')
    report = {}
    for seed in plan['seeds']:
        base, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base = base.to(args.device).eval()
        cache = prepare_role(base, 'assessment', seed)
        tensors = cache['tensors']
        rows = {'matched_original': evaluate_phase(base, tensors, plan['batch'], args.device)}
        for control in ('recency', 'uniform', 'original_full_mask'):
            rows[control] = evaluate_phase(base, tensors, plan['batch'], args.device, control=control)
        for control in ('persistence', 'ewma'):
            rows[control] = forecast_quality(tensors[control].numpy(), tensors['target'].numpy())
        for variant, top_k in plan['variants'].items():
            selected = selections[f'{seed}/{variant}']
            if selected['fallback']:
                rows[variant] = rows['matched_original'] | {'baseline_fallback': True}
            else:
                path = args.output / 'trials' / f'{seed}-{variant}' / 'best.pt'
                if sha256(path) != selected['checkpoint_sha256']:
                    raise ValueError('frozen checkpoint changed')
                model = AhemPhaseReader(base, phase_width=plan['phase_width'], top_k=top_k)
                model.load_state_dict(torch.load(path, weights_only=True, map_location=args.device)['model_state'], strict=True)
                rows[variant] = evaluate_phase(model, tensors, plan['batch'], args.device)
        report[str(seed)] = rows
        atomic_json(rows, args.output / f'assessment-{seed}.json')
        del cache, tensors
    result = {'role': plan['role'], 'report': report, 'selections': selections,
              'temporary_cache_sha256': cache_hashes, 'elapsed_seconds': time.perf_counter() - started,
              'runtime_admission': False, 'publication_claim': False, 'original_journals_preserved': True}
    atomic_json(result, args.output / 'RESULT.json')
    removed = cleanup_owned_cache(temporary, cache_hashes)
    atomic_json({'removed_study_owned_projection_files': removed, 'original_journals_preserved': True,
                 'checkpoints_and_metrics_preserved': True}, args.output / 'CLEANUP.json')
    print(json.dumps({'stage': 'complete', 'results': str(args.output / 'RESULT.json'),
                      'removed_temporary_files': len(removed)}), flush=True)


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

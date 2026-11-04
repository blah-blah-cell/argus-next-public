"""Cheap causal controls; select blends on validation, never on assessment."""
import argparse
import json
from pathlib import Path

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_forecast_audit import forecast_quality
from .ahem_outcome_gate import ObservedPriorGate
from .ahem_outcome_gate_study import evaluate, project_gate
from .ahem_phase_experiment import prepare_cache
from .ahem_phase_local_study import combine, cleanup_owned_cache, validate_local_sources
from .ahem_phase_reader import AhemPhaseReader
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json
from .flow_v6_final_evaluate import _load_frozen_checkpoint


def blend(current, raw, coefficients):
    if current.shape != raw.shape or current.ndim != 2 or current.shape[1] != 2:
        raise ValueError('aligned two-field observations required')
    coefficient = torch.as_tensor(coefficients, dtype=current.dtype, device=current.device)
    if coefficient.shape != (2,) or not torch.isfinite(coefficient).all() or (coefficient < 0).any() or (coefficient > 1).any():
        raise ValueError('two finite convex coefficients required')
    return current + coefficient * (raw - current)


def select_blend(data, grid=(0., .1, .25, .5, .75, 1.)):
    reference = forecast_quality(data['current'].numpy(), data['target'].numpy())
    best, chosen = reference, (0., 0.)
    for size in grid:
        for delay in grid:
            metrics = forecast_quality(blend(data['current'], data['raw'], (size, delay)).numpy(), data['target'].numpy())
            if (metrics['weighted_smooth_l1'] < best['weighted_smooth_l1']
                    and metrics['size_smooth_l1'] <= reference['size_smooth_l1']
                    and metrics['delay_smooth_l1'] <= reference['delay_smooth_l1']):
                best, chosen = metrics, (size, delay)
    return {'coefficients': list(chosen), 'validation': best, 'grid': list(grid)}


def run(args):
    if args.output.exists():
        raise FileExistsError('immutable control output exists')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    paths, hashes = validate_local_sources(plan, args.journals)
    previous = json.loads((args.study / 'FROZEN.json').read_text(encoding='utf-8'))
    heads = json.loads((args.gates / 'FROZEN.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True)
    temporary = args.output / 'temporary-projections'
    temporary.mkdir()
    atomic_json({'role': 'already_seen_assessment_development_controls', 'source_sha256': hashes,
                 'plan_sha256': sha256(args.plan), 'gate_freeze_sha256': sha256(args.gates / 'FROZEN.json'),
                 'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
                 'assessment_previously_seen': True, 'ewma_alpha': .2,
                 'blend_selection': 'validation only; grid 0,.1,.25,.5,.75,1 separately per field',
                 'runtime_admission': False}, args.output / 'RUN.json')
    owned, models, selections = {}, {}, {}
    def prepare(base, reader, seed, role):
        caches = []
        for index, source in enumerate(paths[role]):
            if sha256(source) != hashes[source.name]:
                raise ValueError('bound source changed')
            path = temporary / f'{seed}-{role}-{index}.pt'
            caches.append(prepare_cache(base, source, role, seed, plan, sha256(args.plan), path,
                                        official=False, samples=plan['samples_per_source'][role]))
            owned[path.name] = sha256(path)
        tensors = combine(caches, role)['tensors']
        return project_gate(reader, tensors, plan['batch'], args.device) | {'ewma': tensors['ewma']}
    for seed in plan['seeds']:
        base, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base = base.to(args.device).eval()
        selected = previous['selections'][f'{seed}/phase-soft']
        reader_path = args.study / 'trials' / f'{seed}-phase-soft' / 'best.pt'
        gate_path = args.gates / str(seed) / 'best.pt'
        if (selected['fallback'] or sha256(reader_path) != selected['checkpoint_sha256']
                or heads['selections'][str(seed)]['fallback']
                or sha256(gate_path) != heads['selections'][str(seed)]['best_sha256']):
            raise ValueError('frozen reader/head digest mismatch')
        reader = AhemPhaseReader(base, phase_width=plan['phase_width']).to(args.device)
        reader.load_state_dict(torch.load(reader_path, weights_only=True, map_location=args.device)['model_state'], strict=True)
        reader.eval().requires_grad_(False)
        gate = ObservedPriorGate().to(args.device).eval()
        gate.load_state_dict(torch.load(gate_path, weights_only=True, map_location=args.device)['model_state'], strict=True)
        selections[str(seed)] = select_blend(prepare(base, reader, seed, 'validation'))
        models[seed] = (base, reader, gate)
    atomic_json({'selections': selections, 'assessment_previously_seen': True}, args.output / 'FROZEN.json')
    report = {}
    for seed, (base, reader, gate) in models.items():
        data = prepare(base, reader, seed, 'assessment')
        rows = {name: forecast_quality(data[field].numpy(), data['target'].numpy())
                for name, field in [('persistence', 'current'), ('ewma_alpha_0_2', 'ewma'), ('raw_read', 'raw')]}
        rows['fixed_blend'] = forecast_quality(blend(data['current'], data['raw'], selections[str(seed)]['coefficients']).numpy(), data['target'].numpy())
        rows['trained_gate'] = evaluate(gate, data, args.device)
        report[str(seed)] = rows
        atomic_json(rows, args.output / f'assessment-{seed}.json')
    if any(sha256(args.journals / name) != digest for name, digest in hashes.items()):
        raise ValueError('source journal changed')
    atomic_json({'report': report, 'selections': selections, 'assessment_previously_seen': True,
                 'publication_claim': False, 'runtime_admission': False}, args.output / 'RESULT.json')
    atomic_json({'removed': cleanup_owned_cache(temporary, owned), 'source_journals_preserved': True}, args.output / 'CLEANUP.json')
    print(json.dumps({'stage': 'complete', 'selections': selections}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'journals', 'checkpoints', 'study', 'gates', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    torch.set_num_threads(4)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

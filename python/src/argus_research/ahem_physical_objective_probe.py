"""Read-only log-loss versus byte-error geometry on existing training data."""
import json
from pathlib import Path
import torch
from .ahem_objective_conflict_probe import geometry, gradients
from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_outcome_gate import ObservedPriorGate
from .ahem_outcome_joint import forward, configure
from .ahem_phase_experiment import prepare_cache
from .ahem_phase_local_study import cleanup_owned_cache
from .ahem_phase_reader import AhemPhaseReader
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json
from .flow_v6_final_evaluate import _load_frozen_checkpoint
from .flow_v6_objective import dynamic_loss


def byte_objective(prediction, target):
    # Same physical transformation as forecast_quality; 1500 is a fixed unit
    # normalization, not a fitted statistic or a clipping threshold.
    error = (torch.expm1(prediction[:, 0].clamp_min(0)) -
             torch.expm1(target[:, 0])).abs()
    if not torch.isfinite(error).all():
        raise ValueError('non-finite physical byte error')
    return error.mean() / 1500.


def run():
    dev = Path('../argus-dev')
    output = dev / 'research-runs/ahem-physical-objective-probe-20261003/round-01'
    if output.exists():
        raise FileExistsError('immutable probe already exists')
    source = dev / 'learning/2026-09-12T16.v2.arguslr'
    study = dev / 'research-runs/ahem-phase-local-study-20261002/round-01'
    recorded = json.loads((study / 'STUDY.json').read_text())
    if sha256(source) != recorded['source_sha256'][source.name]:
        raise ValueError('training journal hash mismatch')
    plan_path = Path('research/ahem-phase-local-study-20261002.json')
    plan = json.loads(plan_path.read_text())
    seeds = [20260906, 20260907, 20260908]
    paths = {str(s): {'reader': study / f'trials/{s}-phase-soft/best.pt',
        'gate': dev / f'research-runs/ahem-observed-prior-gate-20261002/round-01/{s}/best.pt'} for s in seeds}
    bindings = {s: {n: sha256(p) for n, p in ps.items()} for s, ps in paths.items()}
    output.mkdir(parents=True)
    temporary = output / 'temporary-projections'
    temporary.mkdir()
    atomic_json({'source_sha256': sha256(source), 'plan_sha256': sha256(plan_path),
        'checkpoint_sha256': bindings, 'seeds': seeds, 'queries_per_seed': 256,
        'batch': 64, 'optimizer_steps': 0, 'assessment_access': False,
        'physical_objective': 'mean absolute byte error / 1500; no clipping',
        'scope': 'phase_encoder_and_gate', 'role': 'exploratory_training_only',
        'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}, output / 'FROZEN.json')
    owned, reports = {}, {}
    for seed in seeds:
        base, _, _ = _load_frozen_checkpoint(dev / 'checkpoints/ahem-v6-1-development-20260906/extracted/ahem-v6-1-development-20260906', seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base = base.to('cuda').eval()
        cache_path = temporary / f'{seed}.pt'
        with torch.no_grad():
            cache = prepare_cache(base, source, 'train', seed, plan,
                sha256(plan_path), cache_path, official=False, samples=256)
        owned[cache_path.name] = sha256(cache_path)
        reader = AhemPhaseReader(base, phase_width=32).to('cuda').eval()
        gate = ObservedPriorGate().to('cuda').eval()
        reader.load_state_dict(torch.load(paths[str(seed)]['reader'], weights_only=True, map_location='cuda')['model_state'], strict=True)
        gate.load_state_dict(torch.load(paths[str(seed)]['gate'], weights_only=True, map_location='cuda')['model_state'], strict=True)
        parameters = configure(reader, gate, 'joint')
        before = [p.detach().clone() for p in parameters]
        rows = []
        data = cache['tensors']
        for start in range(0, len(data['target']), 64):
            index = slice(start, start + 64)
            prediction, _ = forward(reader, gate, data, index, 'cuda', 1e-12)
            target = data['target'][index].to('cuda')
            log_loss = dynamic_loss(prediction, target)
            physical = byte_objective(prediction, target)
            a, b = gradients(log_loss, parameters), gradients(physical, parameters)
            # Preserve actual vectors for independent cosine verification.
            rows.append({'start': start, 'queries': len(target),
                'log_loss': float(log_loss.detach()), 'mae_bytes': float(physical.detach()) * 1500,
                'geometry': geometry(a, b, weight=1.),
                'log_gradient': a.double().tolist(), 'byte_gradient': b.double().tolist()})
        if any(not torch.equal(p, old) for p, old in zip(parameters, before, strict=True)):
            raise ValueError('probe changed weights')
        reports[str(seed)] = rows
        atomic_json(rows, output / f'{seed}.json')
        print(json.dumps({'seed': seed, 'cosines': [r['geometry']['cosine'] for r in rows]}), flush=True)
        del reader, gate, base, cache, data, parameters, before, prediction, target
        torch.cuda.empty_cache()
    if sha256(source) != recorded['source_sha256'][source.name] or any(sha256(p) != bindings[s][n] for s, ps in paths.items() for n, p in ps.items()):
        raise ValueError('input changed during probe')
    atomic_json({'seeds': seeds, 'optimizer_steps': 0, 'assessment_access': False,
        'negative_cosine_batches': sum(r['geometry']['cosine'] is not None and r['geometry']['cosine'] < 0 for rows in reports.values() for r in rows),
        'batch_count': sum(len(rows) for rows in reports.values()),
        'unique_coordinates': 256, 'evaluations_across_seeds': 768,
        'report_sha256': {str(s): sha256(output / f'{s}.json') for s in seeds}}, output / 'RESULT.json')
    atomic_json({'removed': cleanup_owned_cache(temporary, owned),
        'manifest_sidecars_preserved': True}, output / 'CLEANUP.json')


if __name__ == '__main__':
    torch.set_num_threads(4)
    run()

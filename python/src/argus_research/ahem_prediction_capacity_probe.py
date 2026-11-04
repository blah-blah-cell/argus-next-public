"""Separate read aggregation from gate reachability on existing training traffic.

Target-dependent projections are explicitly unattainable oracle diagnostics,
never model outputs. Median controls use observed historical successors only.
"""
import json
from pathlib import Path
import torch
from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_forecast_audit import forecast_quality
from .ahem_outcome_gate import ObservedPriorGate, gate_inputs
from .ahem_phase_experiment import prepare_cache
from .ahem_phase_local_study import cleanup_owned_cache
from .ahem_phase_reader import AhemPhaseReader, phase_infer
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json
from .flow_v6_final_evaluate import _load_frozen_checkpoint


def interval_oracle(current, prior, target):
    """Unattainable nearest point inside the gate's per-field log interval."""
    return torch.maximum(torch.minimum(target, torch.maximum(current, prior)), torch.minimum(current, prior))


def historical_weighted_median(values, weights, valid, current):
    """Per-field causal weighted median; no future targets or fitted threshold."""
    if values.ndim != 3 or values.shape[-1] != 2 or weights.shape != values.shape[:2] or valid.shape != weights.shape or valid.dtype != torch.bool:
        raise ValueError('unaligned historical observations')
    if current.shape != (len(values), 2) or not torch.isfinite(values).all() or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('finite nonnegative observations and weights required')
    effective = weights * valid.to(weights.dtype)
    result = []
    for field in range(2):
        ordered, indices = values[:, :, field].sort(dim=1, stable=True)
        mass = effective.gather(1, indices)
        total = mass.sum(1, keepdim=True)
        # First order statistic whose cumulative mass reaches half the total.
        selected = (mass.cumsum(1) >= total * .5).to(torch.int64).argmax(1, keepdim=True)
        median = ordered.gather(1, selected).squeeze(1)
        result.append(torch.where(total.squeeze(1) > 0, median, current[:, field]))
    return torch.stack(result, dim=1)


@torch.no_grad()
def diagnose(reader, gate, data):
    parts = {n: [] for n in ('target', 'current', 'raw', 'gate', 'median', 'gate_interval_oracle', 'bank_hull_oracle', 'memory', 'inside_gate', 'inside_bank')}
    for start in range(0, len(data['target']), 64):
        index = slice(start, start+64)
        out = phase_infer(reader, data, index, 'cuda')
        current, target = data['persistence'][index].to('cuda'), data['target'][index].to('cuda')
        features, raw, available = gate_inputs(out, current, max_transitions=reader.max_transitions)
        prediction = gate(features, current, raw, available)
        median = historical_weighted_median(out.successor_dynamic, out.retrieval_weights, out.valid, current)
        minimum = out.successor_dynamic.masked_fill(~out.valid[..., None], torch.inf).amin(1)
        maximum = out.successor_dynamic.masked_fill(~out.valid[..., None], -torch.inf).amax(1)
        minimum, maximum = torch.minimum(current, minimum), torch.maximum(current, maximum)
        bank_oracle = torch.maximum(torch.minimum(target, maximum), minimum)
        gate_oracle = interval_oracle(current, raw, target)
        inside_gate = (target >= torch.minimum(current, raw)) & (target <= torch.maximum(current, raw))
        inside_bank = (target >= minimum) & (target <= maximum)
        values = (target,current,raw,prediction,median,gate_oracle,bank_oracle,available.squeeze(1),inside_gate,inside_bank)
        for name, value in zip(parts, values, strict=True):
            parts[name].append(value.cpu())
    tensors = {n: torch.cat(ps) for n,ps in parts.items()}
    target = tensors['target']
    quality = {n: forecast_quality(tensors[n].numpy(), target.numpy()) for n in ('current','raw','gate','median','gate_interval_oracle','bank_hull_oracle')}
    if quality['gate_interval_oracle']['size_mae_bytes'] > quality['gate']['size_mae_bytes'] + 1e-5 or quality['bank_hull_oracle']['size_mae_bytes'] > quality['gate_interval_oracle']['size_mae_bytes'] + 1e-5:
        raise ValueError('oracle ordering violated')
    memory = tensors['memory']
    return {'quality': quality, 'memory_queries': int(memory.sum()), 'queries': len(target),
        'memory_only_inside_gate_per_field': tensors['inside_gate'][memory].float().mean(0).tolist() if memory.any() else None,
        'memory_only_inside_bank_per_field': tensors['inside_bank'][memory].float().mean(0).tolist() if memory.any() else None,
        'records': {n: t.tolist() for n,t in tensors.items()},
        'oracle_warning': 'Uses actual future target; diagnostic lower bounds only, cannot deploy or count as accuracy.'}


def run():
    dev = Path('../argus-dev')
    output = dev / 'research-runs/ahem-prediction-capacity-20261003/round-01'
    if output.exists():
        raise FileExistsError('immutable probe already exists')
    source = dev / 'learning/2026-09-12T16.v2.arguslr'
    prior = dev / 'research-runs/ahem-phase-local-study-20261002/round-01'
    known = json.loads((prior / 'STUDY.json').read_text())['source_sha256'][source.name]
    if sha256(source) != known:
        raise ValueError('training source changed')
    plan_path = Path('research/ahem-phase-local-study-20261002.json')
    plan = json.loads(plan_path.read_text())
    seeds = [20260906,20260907,20260908]
    paths = {str(s): {'reader': prior / f'trials/{s}-phase-soft/best.pt',
        'gate': dev / f'research-runs/ahem-observed-prior-gate-20261002/round-01/{s}/best.pt'} for s in seeds}
    hashes = {s: {n: sha256(p) for n,p in ps.items()} for s,ps in paths.items()}
    output.mkdir(parents=True)
    temporary = output / 'temporary-projections'
    temporary.mkdir()
    atomic_json({'source_sha256': known, 'plan_sha256': sha256(plan_path), 'parent_sha256': hashes,
        'seeds': seeds, 'queries_per_seed': 256, 'optimizer_steps': 0, 'assessment_access': False,
        'role': 'training-only exploratory capacity diagnosis; target-dependent oracles not performance claims',
        'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}, output / 'FROZEN.json')
    owned, results = {}, {}
    for seed in seeds:
        base,_,_ = _load_frozen_checkpoint(dev / 'checkpoints/ahem-v6-1-development-20260906/extracted/ahem-v6-1-development-20260906', seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base = base.to('cuda').eval()
        path = temporary / f'{seed}.pt'
        with torch.no_grad():
            cache = prepare_cache(base, source, 'train', seed, plan, sha256(plan_path), path, official=False, samples=256)
        owned[path.name] = sha256(path)
        reader, gate = AhemPhaseReader(base, phase_width=32).to('cuda').eval(), ObservedPriorGate().to('cuda').eval()
        reader.load_state_dict(torch.load(paths[str(seed)]['reader'], weights_only=True, map_location='cuda')['model_state'], strict=True)
        gate.load_state_dict(torch.load(paths[str(seed)]['gate'], weights_only=True, map_location='cuda')['model_state'], strict=True)
        reader.requires_grad_(False)
        gate.requires_grad_(False)
        report = diagnose(reader, gate, cache['tensors'])
        atomic_json(report, output / f'{seed}.json')
        results[str(seed)] = {n:v for n,v in report.items() if n != 'records'}
        print(json.dumps({'seed': seed, **results[str(seed)]}), flush=True)
        del base, cache, reader, gate
        torch.cuda.empty_cache()
    if sha256(source) != known or any(sha256(p) != hashes[s][n] for s,ps in paths.items() for n,p in ps.items()):
        raise ValueError('bound input changed')
    atomic_json({'results': results, 'optimizer_steps': 0, 'assessment_access': False}, output / 'RESULT.json')
    atomic_json({'removed': cleanup_owned_cache(temporary, owned), 'manifest_sidecars_preserved': True}, output / 'CLEANUP.json')


if __name__ == '__main__':
    torch.set_num_threads(4)
    run()

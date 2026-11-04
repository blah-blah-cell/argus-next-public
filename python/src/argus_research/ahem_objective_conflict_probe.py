"""Read-only gradient geometry on existing real training traffic; no updates."""
import json
from pathlib import Path
import torch
from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_outcome_gate import ObservedPriorGate
from .ahem_outcome_joint import forward
from .ahem_phase_experiment import prepare_cache
from .ahem_phase_local_study import cleanup_owned_cache
from .ahem_phase_reader import AhemPhaseReader
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json
from .ahem_reader_ranking import tie_aware_regret_ranking
from .flow_v6_final_evaluate import _load_frozen_checkpoint
from .flow_v6_objective import dynamic_loss


def geometry(first, second, weight=.1):
    """Direction of -combined gradient; negative derivative predicts decrease."""
    a, b = first.double(), second.double()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('finite aligned gradients required')
    na, nb = float(a.norm()), float(b.norm())
    dot = float(a @ b)
    return {'first_norm': na, 'second_norm': nb,
            'cosine': dot / (na * nb) if na and nb else None,
            'weighted_second_to_first_norm': weight * nb / na if na else None,
            'first_directional_derivative_combined_descent': -float(a @ (a + weight * b)),
            'second_directional_derivative_combined_descent': -float(b @ (a + weight * b))}


def gradients(loss, parameters):
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten().cpu()
                      for p, v in zip(parameters, values, strict=True)])


def diagnose(reader, gate, data):
    parameters = list(reader.phase_encoder.parameters())
    rows = []
    for start in range(0, len(data['target']), 64):
        index = slice(start, start + 64)
        prediction, out = forward(reader, gate, data, index, 'cuda', 1e-12)
        target = data['target'][index].to('cuda')
        # Unweighted field losses, then the exact existing 1:4 normalized loss.
        error = torch.nn.functional.smooth_l1_loss(prediction, target, reduction='none')
        size, delay = error[:, 0].mean(), error[:, 1].mean()
        forecast = dynamic_loss(prediction, target)
        ranking, counts = tie_aware_regret_ranking(out.retrieval_logits,
            out.successor_dynamic, target, out.valid, minimum_regret=.01, margin=.1)
        gf, gr = gradients(forecast, parameters), gradients(ranking, parameters)
        gs, gd = gradients(size, parameters), gradients(delay, parameters)
        rows.append({'queries': len(target), 'forecast_loss': float(forecast.detach()),
                     'ranking_loss': float(ranking.detach()), 'ranking_counts': counts,
                     'forecast_vs_ranking': geometry(gf, gr),
                     'size_vs_delay': geometry(gs, gd, weight=4.)})
    return rows


def run():
    dev = Path('../argus-dev')
    output = dev / 'research-runs/ahem-objective-conflict-20261003/round-01'
    if output.exists():
        raise FileExistsError('immutable mechanism probe already exists')
    if not torch.cuda.is_available():
        raise RuntimeError('local CUDA required for this bounded probe')
    source = dev / 'learning/2026-09-12T16.v2.arguslr'
    study = dev / 'research-runs/ahem-phase-local-study-20261002/round-01'
    recorded = json.loads((study / 'STUDY.json').read_text())
    if sha256(source) != recorded['source_sha256'][source.name]:
        raise ValueError('existing training source changed')
    plan = json.loads(Path('research/ahem-phase-local-study-20261002.json').read_text())
    seeds = [20260906, 20260907, 20260908]
    paths = {}
    for seed in seeds:
        paths[str(seed)] = {'reader': study / f'trials/{seed}-phase-soft/best.pt',
            'gate': dev / f'research-runs/ahem-observed-prior-gate-20261002/round-01/{seed}/best.pt',
            'joint_last': dev / f'checkpoints/ahem-outcome-vm-20261003/results/{seed}-joint/latest.pt'}
    bindings = {s: {n: sha256(p) for n, p in ps.items()} for s, ps in paths.items()}
    output.mkdir(parents=True)
    temporary = output / 'temporary-projections'
    temporary.mkdir()
    protocol = {'source': source.name, 'source_sha256': sha256(source), 'seeds': seeds,
                'queries_per_seed': 256, 'batch': 64, 'states': ['parent', 'unselected_joint_latest'],
                'ranking_weight': .1, 'minimum_regret': .01, 'ranking_margin': .1,
                'variance_floor': 1e-12, 'checkpoint_sha256': bindings,
                'optimizer_steps': 0, 'assessment_access': False, 'downloads': False,
                'role': 'exploratory_gradient_geometry_on_existing_local_training_traffic',
                'publication_claim': False, 'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    atomic_json(protocol, output / 'FROZEN.json')
    owned, reports = {}, {}
    for seed in seeds:
        base, _, _ = _load_frozen_checkpoint(dev / 'checkpoints/ahem-v6-1-development-20260906/extracted/ahem-v6-1-development-20260906', seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base = base.to('cuda').eval()
        cache_path = temporary / f'{seed}.pt'
        with torch.no_grad():
            cache = prepare_cache(base, source, 'train', seed, plan,
                sha256(Path('research/ahem-phase-local-study-20261002.json')), cache_path,
                official=False, samples=256)
        owned[cache_path.name] = sha256(cache_path)
        reader, gate = AhemPhaseReader(base, phase_width=32).to('cuda').eval(), ObservedPriorGate().to('cuda').eval()
        parent_reader = torch.load(paths[str(seed)]['reader'], weights_only=True, map_location='cuda')
        parent_gate = torch.load(paths[str(seed)]['gate'], weights_only=True, map_location='cuda')
        joint = torch.load(paths[str(seed)]['joint_last'], weights_only=True, map_location='cuda')
        if joint['binding']['parent_sha256'] != {n: bindings[str(seed)][n] for n in ('reader','gate')}:
            raise ValueError('VM parent mismatch')
        seed_reports = {}
        for name, rs, gs in [('parent', parent_reader['model_state'], parent_gate['model_state']),
                              ('unselected_joint_latest', joint['reader_state'], joint['gate_state'])]:
            reader.load_state_dict(rs, strict=True)
            gate.load_state_dict(gs, strict=True)
            reader.phase_encoder.requires_grad_(True)
            gate.requires_grad_(False)
            before = {n: p.detach().clone() for n, p in reader.state_dict().items()}
            seed_reports[name] = diagnose(reader, gate, cache['tensors'])
            if any(not torch.equal(before[n], p) for n, p in reader.state_dict().items()):
                raise ValueError('read-only probe mutated weights')
        reports[str(seed)] = seed_reports
        atomic_json(seed_reports, output / f'{seed}.json')
        print(json.dumps({'stage': 'gradient_probe_complete', 'seed': seed, 'report': seed_reports}), flush=True)
        del cache, base, reader, gate
        torch.cuda.empty_cache()
    if sha256(source) != protocol['source_sha256'] or any(sha256(p) != bindings[s][n] for s, ps in paths.items() for n, p in ps.items()):
        raise ValueError('source/checkpoint changed during diagnostic')
    atomic_json({'report': reports, 'optimizer_steps': 0, 'assessment_access': False}, output / 'RESULT.json')
    atomic_json({'removed': cleanup_owned_cache(temporary, owned), 'original_sources_preserved': True}, output / 'CLEANUP.json')


if __name__ == '__main__':
    torch.set_num_threads(4)
    run()

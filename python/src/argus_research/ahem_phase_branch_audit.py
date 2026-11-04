"""Post-selection branch diagnosis only; no training or deployment changes."""
import argparse
import json
from pathlib import Path

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_forecast_audit import forecast_quality
from .ahem_phase_experiment import prepare_cache
from .ahem_phase_local_study import combine, cleanup_owned_cache, validate_local_sources
from .ahem_phase_reader import AhemPhaseReader, phase_infer
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json
from .flow_v6_final_evaluate import _load_frozen_checkpoint
from .flow_v6_objective import dynamic_loss


def causal_interventions(out, persistence):
    """All forecasts depend solely on observed inputs, never evaluation targets."""
    if persistence.shape != out.prediction.shape or not torch.isfinite(persistence).all():
        raise ValueError('aligned finite persistence required')
    available = out.valid.any(-1, keepdim=True)
    raw = (out.retrieval_weights[..., None] * out.successor_dynamic).sum(1)
    raw = torch.where(available, raw, persistence)
    memory_passthrough = torch.cat((out.memory_candidate[:, :1], raw[:, 1:2]), -1)
    return {
        'original_forecast': out.prediction,
        'local_only': out.local_prediction,
        'memory_only': torch.where(available, out.memory_candidate, out.local_prediction),
        'raw_read_persistence_fallback': raw,
        'memory_time_passthrough': torch.where(available, memory_passthrough, out.local_prediction),
        'gated_time_passthrough': out.local_prediction + out.memory_gate * (memory_passthrough - out.local_prediction),
        'persistence': persistence,
    }


@torch.no_grad()
def audit_model(model, tensors, batch, device):
    predictions, gates, available = {}, [], []
    for start in range(0, len(tensors['target']), batch):
        index = slice(start, start + batch)
        out = phase_infer(model, tensors, index, device)
        for name, value in causal_interventions(out, tensors['persistence'][index].to(device)).items():
            predictions.setdefault(name, []).append(value.cpu())
        gates.append(out.memory_gate.cpu().flatten())
        available.append(out.valid.any(-1).cpu())
    prediction = {name: torch.cat(chunks) for name, chunks in predictions.items()}
    target, gate, mask = tensors['target'], torch.cat(gates), torch.cat(available)
    losses = {name: dynamic_loss(value, target, reduction='none') for name, value in prediction.items()}
    # These comparisons see targets ONLY in this offline evaluator. They are
    # hindsight diagnostics, not a causal oracle router or attainable forecast.
    memory_better = losses['memory_only'] < losses['local_only']
    return {'forecasts': {name: forecast_quality(value.numpy(), target.numpy()) for name, value in prediction.items()},
            'offline_diagnostics': {
                'available_queries': int(mask.sum()), 'mean_gate_available': float(gate[mask].mean()),
                'memory_beats_local_fraction_available': float(memory_better[mask].float().mean()),
                'gate_prefers_worse_branch_fraction_available': float(((gate > .5) != memory_better)[mask].float().mean()),
                'raw_read_beats_neural_memory_fraction_available': float((losses['raw_read_persistence_fallback'] < losses['memory_only'])[mask].float().mean()),
                'hindsight_local_memory_min_loss': float(torch.minimum(losses['local_only'], losses['memory_only']).mean()),
                'hindsight_not_runtime': True}}


def run(args):
    if args.output.exists():
        raise FileExistsError('immutable audit already exists')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    paths, hashes = validate_local_sources(plan, args.journals)
    frozen_path = args.study / 'FROZEN.json'
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    study = json.loads((args.study / 'STUDY.json').read_text(encoding='utf-8'))
    if hashes != study['source_sha256'] or sha256(args.plan) != study['plan_sha256']:
        raise ValueError('source/plan binding changed')
    args.output.mkdir(parents=True)
    temporary = args.output / 'temporary-projections'
    temporary.mkdir()
    atomic_json({'role': 'post_selection_reused_assessment_diagnosis', 'training': False,
                 'publication_claim': False, 'frozen_sha256': sha256(frozen_path),
                 'source_sha256': hashes, 'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}},
                args.output / 'RUN.json')
    results, cache_hashes = {}, {}
    for seed in plan['seeds']:
        base, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base = base.to(args.device).eval()
        caches = []
        for index, path in enumerate(paths['assessment']):
            destination = temporary / f'{seed}-diagnosis-{index}.pt'
            caches.append(prepare_cache(base, path, 'assessment', seed, plan, sha256(args.plan), destination,
                                        official=False, samples=plan['samples_per_source']['assessment']))
            cache_hashes[destination.name] = sha256(destination)
        tensors = combine(caches, 'reused_assessment_diagnosis')['tensors']
        rows = {'original': audit_model(base, tensors, plan['batch'], args.device)}
        for variant, top_k in plan['variants'].items():
            selection = frozen['selections'][f'{seed}/{variant}']
            if selection['fallback']:
                raise ValueError('audit expects actual selected phase weights')
            checkpoint = args.study / 'trials' / f'{seed}-{variant}' / 'best.pt'
            if sha256(checkpoint) != selection['checkpoint_sha256']:
                raise ValueError('selected checkpoint changed')
            model = AhemPhaseReader(base, phase_width=plan['phase_width'], top_k=top_k)
            model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=args.device)['model_state'], strict=True)
            model.eval()
            rows[variant] = audit_model(model, tensors, plan['batch'], args.device)
        results[str(seed)] = rows
        atomic_json(rows, args.output / f'{seed}.json')
        print(json.dumps({'stage': 'diagnosed', 'seed': seed}), flush=True)
        del caches, tensors, model, base
    atomic_json({'role': 'post_selection_reused_assessment_diagnosis', 'report': results,
                 'training': False, 'publication_claim': False, 'runtime_admission': False}, args.output / 'RESULT.json')
    removed = cleanup_owned_cache(temporary, cache_hashes)
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

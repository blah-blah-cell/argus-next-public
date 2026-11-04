"""One fixed-checkpoint development replay; no training or reselection."""
import argparse
import json
from pathlib import Path

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_forecast_audit import forecast_quality
from .ahem_outcome_controls import blend
from .ahem_outcome_gate import ObservedPriorGate
from .ahem_outcome_joint import evaluate
from .ahem_outcome_gate_study import project_gate
from .ahem_phase_experiment import prepare_cache, evaluate_phase
from .ahem_phase_local_study import combine, cleanup_owned_cache, validate_local_sources
from .ahem_phase_reader import AhemPhaseReader
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json
from .flow_v6_final_evaluate import _load_frozen_checkpoint


def run(args):
    if args.output.exists():
        raise FileExistsError('immutable benchmark already exists; do not repeat')
    selection = json.loads((args.trial / 'RESULT.json').read_text(encoding='utf-8'))
    checkpoint = args.trial / 'best.pt'
    if not selection['selected_epoch'] or sha256(checkpoint) != selection['best_sha256']:
        raise ValueError('selected joint checkpoint unavailable or changed')
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if state['arm'] != 'joint':
        raise ValueError('this benchmark requires the selected joint add-on')
    seed = state['seed']
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if sha256(args.plan) != state['binding']['sources_sha256']:
        raise ValueError('source plan differs from frozen training binding')
    paths, hashes = validate_local_sources(plan, args.journals)
    if hashes != state['binding']['source_sha256']:
        raise ValueError('source journals differ from training binding')
    parent_reader_path = args.readers / 'trials' / f'{seed}-phase-soft' / 'best.pt'
    parent_gate_path = args.gates / str(seed) / 'best.pt'
    if (sha256(parent_reader_path) != state['binding']['reader_sha256']
            or sha256(parent_gate_path) != state['binding']['gate_sha256']):
        raise ValueError('parent artifacts changed')
    frozen_controls = json.loads((args.controls / 'FROZEN.json').read_text(encoding='utf-8'))
    coefficients = frozen_controls['selections'][str(seed)]['coefficients']
    args.output.mkdir(parents=True)
    temporary = args.output / 'temporary-projections'
    temporary.mkdir()
    atomic_json({'role': 'previously_seen_assessment_development_replay', 'assessment_previously_seen': True,
                 'seed': seed, 'joint_checkpoint_sha256': sha256(checkpoint),
                 'parent_reader_sha256': sha256(parent_reader_path), 'parent_gate_sha256': sha256(parent_gate_path),
                 'controls_freeze_sha256': sha256(args.controls / 'FROZEN.json'), 'fixed_blend': coefficients,
                 'source_sha256': hashes, 'source_plan_sha256': sha256(args.plan),
                 'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
                 'training': False, 'reselection': False, 'publication_claim': False,
                 'scope': 'forecast and retrieval quality; no inference latency measurement',
                 'other_seeds_failed_selection': [20260907, 20260908]}, args.output / 'FROZEN.json')
    base, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
    base = base.to(args.device).eval()
    parent = AhemPhaseReader(base, phase_width=plan['phase_width']).to(args.device).eval()
    parent.load_state_dict(torch.load(parent_reader_path, weights_only=True, map_location=args.device)['model_state'], strict=True)
    parent_gate = ObservedPriorGate().to(args.device).eval()
    parent_gate.load_state_dict(torch.load(parent_gate_path, weights_only=True, map_location=args.device)['model_state'], strict=True)
    reader = AhemPhaseReader(base, phase_width=plan['phase_width']).to(args.device).eval()
    reader.load_state_dict(state['reader_state'], strict=True)
    gate = ObservedPriorGate().to(args.device).eval()
    gate.load_state_dict(state['gate_state'], strict=True)
    caches, owned = [], {}
    for i, source in enumerate(paths['assessment']):
        destination = temporary / f'{seed}-assessment-{i}.pt'
        caches.append(prepare_cache(base, source, 'assessment', seed, plan, sha256(args.plan), destination,
                                    official=False, samples=plan['samples_per_source']['assessment']))
        owned[destination.name] = sha256(destination)
    inference_plan = state['binding']['plan']
    def quality(tensors):
        parent_projection = project_gate(parent, tensors, plan['batch'], args.device)
        ordinary = evaluate_phase(base, tensors, plan['batch'], args.device)
        phase = evaluate_phase(parent, tensors, plan['batch'], args.device)
        return {
            'original_matched_memory_mask': {'forecast': ordinary['forecast'], 'retrieval': ordinary['retrieval']},
            'phase_reader_original_output_path': {'forecast': phase['forecast'], 'retrieval': phase['retrieval']},
            'parent_observed_prior_addon': evaluate(parent, parent_gate, tensors, inference_plan, args.device),
            'selected_joint_addon': evaluate(reader, gate, tensors, inference_plan, args.device),
            'persistence': {'forecast': forecast_quality(tensors['persistence'].numpy(), tensors['target'].numpy())},
            'ewma_alpha_0_2': {'forecast': forecast_quality(tensors['ewma'].numpy(), tensors['target'].numpy())},
            'fixed_parent_read_blend': {'forecast': forecast_quality(blend(parent_projection['current'], parent_projection['raw'], coefficients).numpy(), tensors['target'].numpy())},
        }
    report = quality(combine(caches, 'assessment')['tensors'])
    per_source = {cache['source']: quality(cache['tensors']) for cache in caches}
    if any(sha256(args.journals / n) != digest for n, digest in hashes.items()):
        raise ValueError('source journal changed during replay')
    if sha256(checkpoint) != selection['best_sha256']:
        raise ValueError('selected checkpoint changed during replay')
    atomic_json({'role': 'development_replay_not_confirmatory', 'seed': seed, 'report': report,
                 'per_source': per_source, 'assessment_previously_seen': True, 'training': False,
                 'runtime_admission': False, 'publication_claim': False,
                 'forecast_delay_mae_is_not_inference_latency': True}, args.output / 'RESULT.json')
    atomic_json({'removed': cleanup_owned_cache(temporary, owned), 'sources_and_checkpoints_preserved': True}, args.output / 'CLEANUP.json')
    print(json.dumps({'stage': 'complete', 'report': report}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'journals', 'checkpoints', 'trial', 'readers', 'gates', 'controls', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    torch.set_num_threads(4)
    with torch.no_grad():
        run(parser.parse_args())


if __name__ == '__main__':
    main()

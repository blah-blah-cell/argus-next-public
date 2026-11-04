"""Fixed VM-selected gate replay on already-seen local development captures."""
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
from .ahem_phase_local_study import combine, cleanup_owned_cache
from .ahem_phase_reader import AhemPhaseReader
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json
from .flow_v6_final_evaluate import _load_frozen_checkpoint


def run():
    dev = Path('../argus-dev')
    study = dev / 'checkpoints/ahem-outcome-vm-20261003'
    trial = study / 'results/20260908-gate_only'
    output = dev / 'research-runs/ahem-vm-selected-benchmark-20261003/round-01'
    if output.exists():
        raise FileExistsError('immutable replay already exists')
    checkpoint = trial / 'best.pt'
    selected = json.loads((trial / 'RESULT.json').read_text())
    if selected['selected_epoch'] != 4 or sha256(checkpoint) != selected['best_sha256']:
        raise ValueError('fixed selected checkpoint mismatch')
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if state['seed'] != 20260908 or state['arm'] != 'gate_only':
        raise ValueError('unexpected selected trial')
    plan = json.loads(Path('research/ahem-phase-local-study-20261002.json').read_text())
    prior = json.loads((dev / 'research-runs/ahem-final-addon-benchmark-20261002/round-01/FROZEN.json').read_text())
    sources = [dev / 'learning' / name for name in ('2026-09-24T13.v2.arguslr', '2026-09-25T10.v2.arguslr')]
    hashes = {p.name: sha256(p) for p in sources}
    if any(prior['source_sha256'][n] != h for n, h in hashes.items()):
        raise ValueError('local replay source changed')
    seed = state['seed']
    reader_path = dev / f'research-runs/ahem-phase-local-study-20261002/round-01/trials/{seed}-phase-soft/best.pt'
    gate_path = dev / f'research-runs/ahem-observed-prior-gate-20261002/round-01/{seed}/best.pt'
    if { 'reader': sha256(reader_path), 'gate': sha256(gate_path)} != state['binding']['parent_sha256']:
        raise ValueError('parent hash mismatch')
    run_record = json.loads((trial / 'RUN.json').read_text())
    coefficients = run_record['controls']['fixed_blend']['coefficients']
    output.mkdir(parents=True)
    temporary = output / 'temporary-projections'
    temporary.mkdir()
    atomic_json({'checkpoint_sha256': sha256(checkpoint), 'seed': seed, 'epoch': 4,
                 'source_sha256': hashes, 'fixed_blend': coefficients,
                 'role': 'previously_seen_local_development_replay_not_confirmatory',
                 'training': False, 'reselection': False, 'latency_measured': False,
                 'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}, output / 'FROZEN.json')
    base, _, _ = _load_frozen_checkpoint(dev / 'checkpoints/ahem-v6-1-development-20260906/extracted/ahem-v6-1-development-20260906', seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = base.to(device).eval()
    reader = AhemPhaseReader(base, phase_width=32).to(device).eval()
    reader.load_state_dict(state['reader_state'], strict=True)
    gate, parent = ObservedPriorGate().to(device).eval(), ObservedPriorGate().to(device).eval()
    gate.load_state_dict(state['gate_state'], strict=True)
    parent.load_state_dict(torch.load(gate_path, weights_only=True, map_location=device)['model_state'], strict=True)
    original_reader = torch.load(reader_path, weights_only=True, map_location='cpu')['model_state']
    if any(not torch.equal(t.cpu(), original_reader[n]) for n, t in reader.state_dict().items()):
        raise ValueError('gate-only reader changed')
    caches, owned = [], {}
    for i, source in enumerate(sources):
        path = temporary / f'{seed}-{i}.pt'
        caches.append(prepare_cache(base, source, 'assessment', seed, plan, sha256(Path('research/ahem-phase-local-study-20261002.json')), path, official=False, samples=1024))
        owned[path.name] = sha256(path)
    def quality(data):
        projected = project_gate(reader, data, 256, device)
        return {'original_matched_memory_mask': evaluate_phase(base, data, 256, device),
                'parent_addon': evaluate(reader, parent, data, state['binding']['plan'], device),
                'vm_selected_gate': evaluate(reader, gate, data, state['binding']['plan'], device),
                'persistence': {'forecast': forecast_quality(data['persistence'].numpy(), data['target'].numpy())},
                'ewma': {'forecast': forecast_quality(data['ewma'].numpy(), data['target'].numpy())},
                'vm_validation_fixed_blend': {'forecast': forecast_quality(blend(projected['current'], projected['raw'], coefficients).numpy(), data['target'].numpy())}}
    result = {'report': quality(combine(caches, 'assessment')['tensors']),
              'per_source': {c['source']: quality(c['tensors']) for c in caches},
              'device': device, 'training': False, 'runtime_admission': False, 'publication_claim': False}
    if sha256(checkpoint) != selected['best_sha256'] or any(sha256(p) != hashes[p.name] for p in sources):
        raise ValueError('bound artifact changed during replay')
    atomic_json(result, output / 'RESULT.json')
    atomic_json({'removed': cleanup_owned_cache(temporary, owned), 'sources_preserved': True}, output / 'CLEANUP.json')
    print(json.dumps({'stage': 'complete', 'report': result['report']}))


if __name__ == '__main__':
    torch.set_num_threads(4)
    with torch.no_grad():
        run()

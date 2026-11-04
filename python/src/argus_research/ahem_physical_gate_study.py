"""Bounded paired gate-only training; physical units enter the objective."""
import copy
import json
from pathlib import Path
import torch
from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_forecast_audit import forecast_quality
from .ahem_outcome_controls import select_blend
from .ahem_outcome_gate import ObservedPriorGate
from .ahem_outcome_gate_study import project_gate, evaluate
from .ahem_outcome_joint import eligible
from .ahem_phase_experiment import prepare_cache
from .ahem_phase_local_study import combine, cleanup_owned_cache
from .ahem_phase_reader import AhemPhaseReader
from .ahem_physical_objective_probe import byte_objective
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json, atomic_save
from .flow_v6_final_evaluate import _load_frozen_checkpoint
from .flow_v6_objective import dynamic_loss


def fit(gate, data, plan, seed, variant, output, binding, ewma):
    output.mkdir(parents=True)
    projected = data['validation']
    blend = select_blend(projected, tuple(plan['blend_grid']))
    controls = {'persistence': forecast_quality(projected['current'].numpy(), projected['target'].numpy()),
        'ewma': forecast_quality(ewma.numpy(), projected['target'].numpy()), 'fixed_blend': blend}
    reference = min((controls['persistence'], controls['ewma'], blend['validation']), key=lambda m: m['weighted_smooth_l1'])
    initial = {'forecast': evaluate(gate, projected, 'cuda')}
    atomic_json(binding | {'seed': seed, 'variant': variant, 'controls': controls,
        'initial_validation': initial, 'trainable_parameters': 210}, output / 'RUN.json')
    optimizer = torch.optim.AdamW(gate.parameters(), lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    train = {n: t.to('cuda') for n,t in data['train'].items()}
    best, selected, stale = float('inf'), None, 0
    for epoch in range(1, plan['epochs']+1):
        order = torch.randperm(len(train['target']), generator=generator)
        gate.train()
        for start in range(0, len(order), plan['batch']):
            index = order[start:start+plan['batch']].to('cuda')
            prediction = gate(*(train[n][index] for n in ('features','current','raw','available')))
            loss = dynamic_loss(prediction, train['target'][index])
            if variant == 'log_plus_bytes':
                loss = loss + plan['physical_weight'] * byte_objective(prediction, train['target'][index])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        gate.eval()
        validation = {'forecast': evaluate(gate, projected, 'cuda')}
        training = {'forecast': evaluate(gate, data['train'], 'cuda')}
        accepted = eligible(validation, reference, initial, best)
        if accepted:
            best, selected, stale = validation['forecast']['weighted_smooth_l1'], epoch, 0
        else:
            stale += 1
        state = {'schema': 'argus.physical-gate-checkpoint/v1', 'seed': seed, 'variant': variant,
            'epoch': epoch, 'model_state': gate.state_dict(), 'optimizer_state': optimizer.state_dict(),
            'torch_rng': torch.get_rng_state(), 'sampler_rng': generator.get_state(),
            'cuda_rng': torch.cuda.get_rng_state_all(), 'binding': binding,
            'validation': validation, 'training': training}
        path = output / f'epoch-{epoch:03d}.pt'
        atomic_save(state, path)
        atomic_save(state, output / 'latest.pt')
        if accepted:
            atomic_save(state, output / 'best.pt')
        with (output / 'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'epoch': epoch, 'selected': accepted, 'validation': validation,
                'training': training, 'sha256': sha256(path)}, allow_nan=False)+'\n')
        print(json.dumps({'seed': seed, 'variant': variant, 'epoch': epoch, 'selected': accepted,
            'loss': validation['forecast']['weighted_smooth_l1'], 'bytes': validation['forecast']['size_mae_bytes']}), flush=True)
        if stale >= plan['patience']:
            break
    for path in output.glob('epoch-*.pt'):
        state = torch.load(path, weights_only=True, map_location='cpu')
        restored = ObservedPriorGate()
        restored.load_state_dict(state['model_state'], strict=True)
        if any(not torch.isfinite(t).all() for t in state['model_state'].values()):
            raise ValueError('invalid checkpoint')
    result = {'selected_epoch': selected, 'baseline_retained': selected is None,
        'best_sha256': sha256(output / 'best.pt') if selected else None, 'runtime_admission': False}
    atomic_json(result, output / 'RESULT.json')
    return result


def run():
    dev = Path('../argus-dev')
    output = dev / 'research-runs/ahem-physical-gate-20261003/round-01'
    if output.exists():
        raise FileExistsError('immutable study already exists')
    plan_path = Path('research/ahem-physical-gate-local-20261003.json')
    plan = json.loads(plan_path.read_text())
    if plan['variants'] != ['log_only','log_plus_bytes'] or plan['physical_normalization_bytes'] != 1500 or any(plan[n] for n in ('assessment_access','downloads','runtime_admission')):
        raise ValueError('unexpected protocol')
    source_plan_path = Path('research/ahem-phase-local-study-20261002.json')
    source_plan = json.loads(source_plan_path.read_text())
    prior = dev / 'research-runs/ahem-phase-local-study-20261002/round-01'
    recorded = json.loads((prior / 'STUDY.json').read_text())['source_sha256']
    sources = {role: [dev / 'learning' / n for n in source_plan['sources'][role]] for role in ('train','validation')}
    hashes = {p.name: sha256(p) for ps in sources.values() for p in ps}
    if any(recorded[n] != h for n,h in hashes.items()):
        raise ValueError('existing source changed')
    parents = {str(s): {'reader': prior / f'trials/{s}-phase-soft/best.pt',
        'gate': dev / f'research-runs/ahem-observed-prior-gate-20261002/round-01/{s}/best.pt'} for s in plan['seeds']}
    parent_hashes = {s: {n: sha256(p) for n,p in ps.items()} for s,ps in parents.items()}
    output.mkdir(parents=True)
    temporary = output / 'temporary-projections'
    temporary.mkdir()
    binding = {'plan': plan, 'plan_sha256': sha256(plan_path), 'source_plan_sha256': sha256(source_plan_path),
        'source_sha256': hashes, 'parent_sha256': parent_hashes, 'assessment_opened': False,
        'torch_version': str(torch.__version__), 'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    atomic_json(binding, output / 'STUDY.json')
    owned, results = {}, {}
    for seed in plan['seeds']:
        base,_,_ = _load_frozen_checkpoint(dev / 'checkpoints/ahem-v6-1-development-20260906/extracted/ahem-v6-1-development-20260906', seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base = base.to('cuda').eval()
        reader = AhemPhaseReader(base, phase_width=32).to('cuda').eval()
        reader.load_state_dict(torch.load(parents[str(seed)]['reader'], weights_only=True, map_location='cuda')['model_state'], strict=True)
        reader.requires_grad_(False)
        data, ewma = {}, None
        for role, ps in sources.items():
            caches = []
            for i, source in enumerate(ps):
                path = temporary / f'{seed}-{role}-{i}.pt'
                with torch.no_grad():
                    caches.append(prepare_cache(base, source, role, seed, source_plan,
                        binding['source_plan_sha256'], path, official=False, samples=plan['samples_per_source'][role]))
                owned[path.name] = sha256(path)
            tensors = combine(caches, role)['tensors']
            data[role] = project_gate(reader, tensors, plan['batch'], 'cuda')
            if role == 'validation':
                ewma = tensors['ewma']
        gate = ObservedPriorGate().to('cuda')
        gate.load_state_dict(torch.load(parents[str(seed)]['gate'], weights_only=True, map_location='cuda')['model_state'], strict=True)
        for variant in plan['variants']:
            results[f'{seed}/{variant}'] = fit(copy.deepcopy(gate), data, plan, seed, variant, output / f'{seed}-{variant}', binding, ewma)
        del data, tensors, caches, reader, base, gate
        torch.cuda.empty_cache()
    if any(sha256(p) != hashes[p.name] for ps in sources.values() for p in ps) or any(sha256(p) != parent_hashes[s][n] for s,ps in parents.items() for n,p in ps.items()):
        raise ValueError('bound input changed')
    atomic_json({'selections': results, 'assessment_opened': False, 'runtime_admission': False}, output / 'FROZEN.json')
    atomic_json({'removed': cleanup_owned_cache(temporary, owned), 'manifest_sidecars_preserved': True}, output / 'CLEANUP.json')
    print(json.dumps({'stage': 'complete', 'results': results}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    run()

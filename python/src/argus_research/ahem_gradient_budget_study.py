"""Three fixed objectives, identical per-seed parents; no final-test access."""
import copy
import json
import math
from pathlib import Path
import torch
from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_forecast_audit import forecast_quality
from .ahem_outcome_controls import select_blend
from .ahem_outcome_gate import ObservedPriorGate
from .ahem_outcome_gate_study import project_gate
from .ahem_outcome_joint import configure, evaluate, eligible, forward
from .ahem_phase_experiment import prepare_cache
from .ahem_phase_local_study import combine, cleanup_owned_cache
from .ahem_phase_reader import AhemPhaseReader
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json, atomic_save
from .ahem_reader_ranking import tie_aware_regret_ranking
from .flow_v6_final_evaluate import _load_frozen_checkpoint
from .flow_v6_objective import dynamic_loss

VARIANTS = ('forecast_only', 'fixed_joint', 'budgeted_joint')


def budget_weight(forecast_norm, ranking_norm, *, maximum=.1, budget=.5, epsilon=1e-12):
    values = (forecast_norm, ranking_norm, maximum, budget, epsilon)
    if any(not math.isfinite(v) or v < 0 for v in values) or epsilon <= 0:
        raise ValueError('finite nonnegative norms/settings and positive epsilon required')
    return min(maximum, budget * forecast_norm / (ranking_norm + epsilon))


def geometry(forecast, ranking, parameters):
    def flattened(loss):
        values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten()
                          for p, v in zip(parameters, values, strict=True)]).double()
    first, second = flattened(forecast), flattened(ranking)
    nf, nr = float(first.norm()), float(second.norm())
    if not math.isfinite(nf) or not math.isfinite(nr):
        raise ValueError('nonfinite objective gradient')
    return nf, nr, float(first @ second) / (nf * nr) if nf and nr else None


def fit(reader, gate, data, plan, variant, seed, output, binding):
    output.mkdir(parents=True)
    parameters = configure(reader, gate, 'joint')
    phase_parameters = list(reader.phase_encoder.parameters())
    frozen = {n: p.cpu().clone() for n, p in reader.state_dict().items() if not n.startswith('phase_encoder.')}
    projected = project_gate(reader, data['validation'], plan['batch'], 'cuda')
    chosen = select_blend(projected, tuple(plan['blend_grid']))
    controls = {'persistence': forecast_quality(projected['current'].numpy(), projected['target'].numpy()),
                'ewma': forecast_quality(data['validation']['ewma'].numpy(), projected['target'].numpy()),
                'fixed_blend': chosen}
    reference = min((controls['persistence'], controls['ewma'], chosen['validation']), key=lambda m: m['weighted_smooth_l1'])
    initial = evaluate(reader, gate, data['validation'], plan, 'cuda')
    atomic_json(binding | {'variant': variant, 'seed': seed, 'initial_validation': initial,
                          'controls': controls, 'trainable_parameters': sum(p.numel() for p in parameters)}, output / 'RUN.json')
    optimizer = torch.optim.AdamW(parameters, lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    train = {n: t.to('cuda') for n, t in data['train'].items()}
    best, selected, stale = float('inf'), None, 0
    for epoch in range(1, plan['epochs'] + 1):
        reader.train()
        gate.train()
        order = torch.randperm(len(train['target']), generator=generator)
        geometry_rows = []
        for start in range(0, len(order), plan['batch']):
            index = order[start:start + plan['batch']].to('cuda')
            prediction, out = forward(reader, gate, train, index, 'cuda', plan['variance_floor'])
            forecast = dynamic_loss(prediction, train['target'][index])
            ranking, counts = tie_aware_regret_ranking(out.retrieval_logits, out.successor_dynamic,
                train['target'][index], out.valid, minimum_regret=plan['minimum_regret'], margin=plan['ranking_margin'])
            nf, nr, cosine = geometry(forecast, ranking, phase_parameters)
            alpha = 0. if variant == 'forecast_only' else plan['ranking_weight']
            if variant == 'budgeted_joint':
                alpha = budget_weight(nf, nr, maximum=plan['ranking_weight'],
                    budget=plan['reader_gradient_budget'], epsilon=plan['gradient_epsilon'])
            ratio = alpha * nr / nf if nf else None
            if variant == 'budgeted_joint' and ratio is not None and ratio > plan['reader_gradient_budget'] + 1e-9:
                raise ValueError('gradient budget violated')
            loss = forecast + alpha * ranking
            if not torch.isfinite(loss):
                raise ValueError('nonfinite objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            geometry_rows.append({'alpha': alpha, 'reader_ratio': ratio, 'cosine': cosine,
                                  'informative_queries': counts['informative_queries']})
        validation = evaluate(reader, gate, data['validation'], plan, 'cuda')
        train_quality = evaluate(reader, gate, data['train'], plan, 'cuda')
        accepted = eligible(validation, reference, initial, best)
        if accepted:
            best, selected, stale = validation['forecast']['weighted_smooth_l1'], epoch, 0
        else:
            stale += 1
        state = {'schema': 'argus.gradient-budget-checkpoint/v1', 'variant': variant, 'seed': seed,
                 'epoch': epoch, 'reader_state': reader.state_dict(), 'gate_state': gate.state_dict(),
                 'optimizer_state': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                 'sampler_rng': generator.get_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                 'binding': binding, 'validation': validation, 'training': train_quality}
        path = output / f'epoch-{epoch:03d}.pt'
        atomic_save(state, path)
        atomic_save(state, output / 'latest.pt')
        if accepted:
            atomic_save(state, output / 'best.pt')
        row = {'epoch': epoch, 'selected': accepted, 'validation': validation,
               'training': train_quality, 'batch_geometry': geometry_rows, 'sha256': sha256(path)}
        with (output / 'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps({'seed': seed, 'variant': variant, 'epoch': epoch, 'selected': accepted,
                         'train_loss': train_quality['forecast']['weighted_smooth_l1'],
                         'validation_loss': validation['forecast']['weighted_smooth_l1']}), flush=True)
        if stale >= plan['patience']:
            break
    for path in output.glob('epoch-*.pt'):
        state = torch.load(path, map_location='cpu', weights_only=True)
        reader.load_state_dict(state['reader_state'], strict=True)
        gate.load_state_dict(state['gate_state'], strict=True)
        if any(not torch.isfinite(v).all() for key in ('reader_state', 'gate_state') for v in state[key].values()):
            raise ValueError('invalid weights')
        if any(not torch.equal(state['reader_state'][n], p) for n, p in frozen.items()):
            raise ValueError('original backbone changed')
    result = {'selected_epoch': selected, 'best_sha256': sha256(output / 'best.pt') if selected else None,
              'baseline_retained': selected is None, 'assessment_opened': False, 'runtime_admission': False}
    atomic_json(result, output / 'RESULT.json')
    return result


def run():
    dev = Path('../argus-dev')
    output = dev / 'research-runs/ahem-gradient-budget-20261003/round-01'
    if output.exists():
        raise FileExistsError('immutable study already exists')
    if not torch.cuda.is_available():
        raise RuntimeError('local CUDA required; no downloads or VM fallback')
    plan_path = Path('research/ahem-gradient-budget-local-20261003.json')
    plan = json.loads(plan_path.read_text())
    if plan['variants'] != list(VARIANTS) or any(plan[k] is not False for k in ('assessment_access','downloads','runtime_admission')):
        raise ValueError('unexpected scope')
    source_plan_path = Path('research/ahem-phase-local-study-20261002.json')
    source_plan = json.loads(source_plan_path.read_text())
    prior = dev / 'research-runs/ahem-phase-local-study-20261002/round-01'
    source_binding = json.loads((prior / 'STUDY.json').read_text())['source_sha256']
    source_paths = {role: [dev / 'learning' / n for n in source_plan['sources'][role]] for role in ('train','validation')}
    hashes = {p.name: sha256(p) for ps in source_paths.values() for p in ps}
    if any(source_binding[n] != h for n, h in hashes.items()):
        raise ValueError('existing source changed')
    parent_paths = {str(seed): {'reader': prior / f'trials/{seed}-phase-soft/best.pt',
        'gate': dev / f'research-runs/ahem-observed-prior-gate-20261002/round-01/{seed}/best.pt'} for seed in plan['seeds']}
    parents = {s: {n: sha256(p) for n,p in ps.items()} for s,ps in parent_paths.items()}
    for seed in plan['seeds']:
        vm = json.loads((dev / f'checkpoints/ahem-outcome-vm-20261003/results/{seed}-joint/RUN.json').read_text())
        if vm['parent_sha256'] != parents[str(seed)]:
            raise ValueError('parent changed')
    output.mkdir(parents=True)
    temporary = output / 'temporary-projections'
    temporary.mkdir()
    binding = {'plan': plan, 'plan_sha256': sha256(plan_path), 'source_plan_sha256': sha256(source_plan_path),
               'source_sha256': hashes, 'parent_sha256': parents, 'assessment_opened': False,
               'torch_version': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
               'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    atomic_json(binding, output / 'STUDY.json')
    owned, selections = {}, {}
    for seed in plan['seeds']:
        base, _, _ = _load_frozen_checkpoint(dev / 'checkpoints/ahem-v6-1-development-20260906/extracted/ahem-v6-1-development-20260906', seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base = base.to('cuda').eval()
        data = {}
        for role, paths in source_paths.items():
            caches = []
            for i, source in enumerate(paths):
                path = temporary / f'{seed}-{role}-{i}.pt'
                with torch.no_grad():
                    caches.append(prepare_cache(base, source, role, seed, source_plan,
                        binding['source_plan_sha256'], path, official=False, samples=plan['samples_per_source'][role]))
                owned[path.name] = sha256(path)
            data[role] = combine(caches, role)['tensors']
        reader, gate = AhemPhaseReader(base, phase_width=32).to('cuda'), ObservedPriorGate().to('cuda')
        reader.load_state_dict(torch.load(parent_paths[str(seed)]['reader'], weights_only=True, map_location='cuda')['model_state'], strict=True)
        gate.load_state_dict(torch.load(parent_paths[str(seed)]['gate'], weights_only=True, map_location='cuda')['model_state'], strict=True)
        for variant in VARIANTS:
            selections[f'{seed}/{variant}'] = fit(copy.deepcopy(reader), copy.deepcopy(gate), data, plan,
                variant, seed, output / f'{seed}-{variant}', binding)
        del data, caches, base, reader, gate
        torch.cuda.empty_cache()
    if any(sha256(p) != hashes[p.name] for ps in source_paths.values() for p in ps):
        raise ValueError('source changed during study')
    if any(sha256(p) != parents[s][n] for s,ps in parent_paths.items() for n,p in ps.items()):
        raise ValueError('parent changed during study')
    atomic_json({'selections': selections, 'assessment_opened': False, 'runtime_admission': False}, output / 'FROZEN.json')
    atomic_json({'removed': cleanup_owned_cache(temporary, owned), 'sources_preserved': True}, output / 'CLEANUP.json')
    print(json.dumps({'stage': 'complete', 'selections': selections}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    run()

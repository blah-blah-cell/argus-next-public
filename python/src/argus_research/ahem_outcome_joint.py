"""Bounded reader/gate ablation on local development traffic, no assessment."""
import argparse
import copy
import json
from pathlib import Path

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_forecast_audit import forecast_quality
from .ahem_outcome_controls import blend, select_blend
from .ahem_outcome_gate import ObservedPriorGate, gate_inputs
from .ahem_outcome_gate_study import project_gate
from .ahem_phase_experiment import prepare_cache
from .ahem_phase_local_study import combine, cleanup_owned_cache, validate_local_sources
from .ahem_phase_reader import AhemPhaseReader, phase_infer
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json, atomic_save
from .ahem_reader_ranking import tie_aware_regret_ranking
from .ahem_retrieval_audit import retrieval_quality
from .flow_v6_final_evaluate import _load_frozen_checkpoint
from .flow_v6_objective import dynamic_loss


ARMS = ('reader_only', 'gate_only', 'joint')


def configure(reader, gate, arm):
    if arm not in ARMS:
        raise ValueError('unplanned ablation')
    reader.requires_grad_(False)
    reader.phase_encoder.requires_grad_(arm != 'gate_only')
    gate.requires_grad_(arm != 'reader_only')
    return [p for module in (reader, gate) for p in module.parameters() if p.requires_grad]


def forward(reader, gate, data, index, device, variance_floor):
    out = phase_infer(reader, data, index, device)
    current = data['persistence'][index].to(device)
    features, raw, available = gate_inputs(out, current, max_transitions=reader.max_transitions,
                                          variance_floor=variance_floor)
    return gate(features, current, raw, available), out


@torch.no_grad()
def evaluate(reader, gate, data, plan, device):
    reader.eval()
    gate.eval()
    predictions, weights = [], []
    for start in range(0, len(data['target']), plan['batch']):
        prediction, out = forward(reader, gate, data, slice(start, start + plan['batch']), device, plan['variance_floor'])
        predictions.append(prediction.cpu())
        weights.append(out.retrieval_weights.cpu())
    return {'forecast': forecast_quality(torch.cat(predictions).numpy(), data['target'].numpy()),
            'retrieval': retrieval_quality(data['successor_dynamic'], data['target'], data['phase_valid'], torch.cat(weights))}


def eligible(metrics, baseline, initial, best):
    m = metrics['forecast']
    return (m['weighted_smooth_l1'] < min(best, initial['forecast']['weighted_smooth_l1'], baseline['weighted_smooth_l1'])
            and m['size_smooth_l1'] <= baseline['size_smooth_l1']
            and m['delay_smooth_l1'] <= baseline['delay_smooth_l1']
            and m['size_mae_bytes'] <= baseline['size_mae_bytes'])


def fit(reader, gate, data, plan, arm, seed, output, binding, device):
    output.mkdir(parents=True)
    parameters = configure(reader, gate, arm)
    original = {n: p.detach().cpu().clone() for n, p in reader.state_dict().items() if not n.startswith('phase_encoder.')}
    fixed_reader = {n: p.detach().cpu().clone() for n, p in reader.phase_encoder.state_dict().items()}
    fixed_gate = {n: p.detach().cpu().clone() for n, p in gate.state_dict().items()}
    projected = project_gate(reader, data['validation'], plan['batch'], device)
    chosen = select_blend(projected, tuple(plan['blend_grid']))
    controls = {'persistence': forecast_quality(projected['current'].numpy(), projected['target'].numpy()),
                'ewma': forecast_quality(data['validation']['ewma'].numpy(), projected['target'].numpy()),
                'fixed_blend': chosen}
    reference = min((controls['persistence'], controls['ewma'], chosen['validation']), key=lambda m: m['weighted_smooth_l1'])
    initial = evaluate(reader, gate, data['validation'], plan, device)
    atomic_json(binding | {'arm': arm, 'trainable_parameters': sum(p.numel() for p in parameters),
                          'initial_validation': initial, 'controls': controls}, output / 'RUN.json')
    optimizer = torch.optim.AdamW(parameters, lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    train = {name: value.to(device) for name, value in data['train'].items()}
    best, selected, stale = float('inf'), None, 0
    for epoch in range(1, plan['epochs'] + 1):
        reader.train()
        gate.train()
        order = torch.randperm(len(train['target']), generator=generator)
        for start in range(0, len(order), plan['batch']):
            index = order[start:start + plan['batch']].to(device)
            prediction, out = forward(reader, gate, train, index, device, plan['variance_floor'])
            target = train['target'][index]
            loss = dynamic_loss(prediction, target)
            if arm != 'gate_only':
                ranking, _ = tie_aware_regret_ranking(out.retrieval_logits, out.successor_dynamic, target, out.valid,
                    minimum_regret=plan['minimum_regret'], margin=plan['ranking_margin'])
                loss = loss + plan['ranking_weight'] * ranking
            if not torch.isfinite(loss):
                raise ValueError('nonfinite joint objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
        metrics = evaluate(reader, gate, data['validation'], plan, device)
        chosen_epoch = eligible(metrics, reference, initial, best)
        if chosen_epoch:
            best, selected, stale = metrics['forecast']['weighted_smooth_l1'], epoch, 0
        else:
            stale += 1
        state = {'schema': 'argus.outcome-joint-development/v1', 'reader_state': reader.state_dict(),
                 'gate_state': gate.state_dict(), 'optimizer_state': optimizer.state_dict(), 'epoch': epoch,
                 'arm': arm, 'seed': seed, 'binding': binding, 'validation': metrics,
                 'torch_rng': torch.get_rng_state(), 'sampler_rng': generator.get_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else []}
        path = output / f'epoch-{epoch:03d}.pt'
        atomic_save(state, path)
        atomic_save(state, output / 'latest.pt')
        if chosen_epoch:
            atomic_save(state, output / 'best.pt')
        with (output / 'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'epoch': epoch, 'selected': chosen_epoch, 'validation': metrics, 'sha256': sha256(path)}, allow_nan=False) + '\n')
        print(json.dumps({'seed': seed, 'arm': arm, 'epoch': epoch, 'selected': chosen_epoch,
                          'validation_loss': metrics['forecast']['weighted_smooth_l1']}), flush=True)
        if stale >= plan['patience']:
            break
    if any(not torch.equal(reader.state_dict()[name].cpu(), value) for name, value in original.items()):
        raise ValueError('original backbone changed')
    if arm == 'gate_only' and any(not torch.equal(reader.phase_encoder.state_dict()[n].cpu(), v) for n, v in fixed_reader.items()):
        raise ValueError('frozen reader changed')
    if arm == 'reader_only' and any(not torch.equal(gate.state_dict()[n].cpu(), v) for n, v in fixed_gate.items()):
        raise ValueError('frozen gate changed')
    for path in output.glob('epoch-*.pt'):
        state = torch.load(path, weights_only=True, map_location='cpu')
        if not all(torch.isfinite(v).all() for key in ('reader_state', 'gate_state') for v in state[key].values()):
            raise ValueError('nonfinite checkpoint weights')
        reader.load_state_dict(state['reader_state'], strict=True)
        gate.load_state_dict(state['gate_state'], strict=True)
    result = {'selected_epoch': selected, 'best_sha256': sha256(output / 'best.pt') if selected else None,
              'baseline_retained': selected is None, 'assessment_opened': False, 'runtime_admission': False}
    atomic_json(result, output / 'RESULT.json')
    return result


def run(args):
    if args.output.exists():
        raise FileExistsError('immutable joint study exists')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if plan['arms'] != list(ARMS) or plan['assessment_access'] is not False or plan['variance_floor'] != 1e-12:
        raise ValueError('unplanned joint protocol')
    source_plan = json.loads(args.sources.read_text(encoding='utf-8'))
    paths, hashes = validate_local_sources(source_plan, args.journals)
    prior_reader = json.loads((args.study / 'FROZEN.json').read_text(encoding='utf-8'))
    prior_gate = json.loads((args.gates / 'FROZEN.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True)
    temporary = args.output / 'temporary-projections'
    temporary.mkdir()
    binding = {'plan': plan, 'plan_sha256': sha256(args.plan), 'sources_sha256': sha256(args.sources),
               'source_sha256': hashes, 'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
               'role': 'local_training_validation_only_not_confirmatory', 'assessment_opened': False,
               'torch_version': str(torch.__version__), 'device': args.device,
               'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None}
    atomic_json(binding, args.output / 'RUN.json')
    owned, results = {}, {}
    for seed in source_plan['seeds']:
        base, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base = base.to(args.device).eval()
        reader_path = args.study / 'trials' / f'{seed}-phase-soft' / 'best.pt'
        gate_path = args.gates / str(seed) / 'best.pt'
        if (sha256(reader_path) != prior_reader['selections'][f'{seed}/phase-soft']['checkpoint_sha256']
                or sha256(gate_path) != prior_gate['selections'][str(seed)]['best_sha256']):
            raise ValueError('parent checkpoint changed')
        reader = AhemPhaseReader(base, phase_width=source_plan['phase_width']).to(args.device)
        reader.load_state_dict(torch.load(reader_path, weights_only=True, map_location=args.device)['model_state'], strict=True)
        gate = ObservedPriorGate().to(args.device)
        gate.load_state_dict(torch.load(gate_path, weights_only=True, map_location=args.device)['model_state'], strict=True)
        data = {}
        for role in ('train', 'validation'):
            caches = []
            for i, source in enumerate(paths[role]):
                if sha256(source) != hashes[source.name]:
                    raise ValueError('source changed')
                path = temporary / f'{seed}-{role}-{i}.pt'
                caches.append(prepare_cache(base, source, role, seed, source_plan, sha256(args.sources), path,
                                            official=False, samples=source_plan['samples_per_source'][role]))
                owned[path.name] = sha256(path)
            data[role] = combine(caches, role)['tensors']
        parents = binding | {'reader_sha256': sha256(reader_path), 'gate_sha256': sha256(gate_path), 'cache_sha256': dict(owned)}
        for arm in ARMS:
            results[f'{seed}/{arm}'] = fit(copy.deepcopy(reader), copy.deepcopy(gate), data, plan, arm, seed,
                                         args.output / f'{seed}-{arm}', parents, args.device)
    if any(sha256(args.journals / n) != v for n, v in hashes.items()):
        raise ValueError('source changed during study')
    atomic_json({'results': results, 'assessment_opened': False, 'runtime_admission': False}, args.output / 'FROZEN.json')
    atomic_json({'removed': cleanup_owned_cache(temporary, owned), 'sources_preserved': True}, args.output / 'CLEANUP.json')
    print(json.dumps({'stage': 'complete', 'results': results}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'sources', 'journals', 'checkpoints', 'study', 'gates', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    torch.set_num_threads(4)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

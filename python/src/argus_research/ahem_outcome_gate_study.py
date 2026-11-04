"""One bounded local gate experiment; frozen readers, no downloads."""
import argparse
import json
from pathlib import Path

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_forecast_audit import forecast_quality
from .ahem_outcome_gate import ObservedPriorGate, gate_inputs
from .ahem_phase_experiment import prepare_cache
from .ahem_phase_local_study import combine, cleanup_owned_cache, validate_local_sources
from .ahem_phase_reader import AhemPhaseReader, phase_infer
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json, atomic_save
from .flow_v6_final_evaluate import _load_frozen_checkpoint
from .flow_v6_objective import dynamic_loss


@torch.no_grad()
def project_gate(reader, tensors, batch, device):
    parts = {name: [] for name in ('features', 'current', 'raw', 'available', 'unchanged')}
    for start in range(0, len(tensors['target']), batch):
        index = slice(start, start + batch)
        out = phase_infer(reader, tensors, index, device)
        current = tensors['persistence'][index].to(device)
        features, raw, available = gate_inputs(out, current, max_transitions=reader.max_transitions)
        for name, value in zip(parts, (features, current, raw, available, out.prediction), strict=True):
            parts[name].append(value.cpu())
    return {name: torch.cat(rows) for name, rows in parts.items()} | {'target': tensors['target']}


@torch.no_grad()
def evaluate(gate, data, device):
    prediction = gate(*(data[name].to(device) for name in ('features', 'current', 'raw', 'available'))).cpu()
    return forecast_quality(prediction.numpy(), data['target'].numpy())


def improves(metrics, reference, best):
    values = [metrics[name] for name in ('weighted_smooth_l1', 'size_smooth_l1', 'delay_smooth_l1')]
    if any(not torch.isfinite(torch.tensor(value)) or value < 0 for value in values):
        raise ValueError('invalid gate selection metric')
    return (metrics['weighted_smooth_l1'] < best
            and metrics['size_smooth_l1'] <= reference['size_smooth_l1']
            and metrics['delay_smooth_l1'] <= reference['delay_smooth_l1'])


def fit(seed, data, plan, directory, binding, device):
    directory.mkdir(parents=True)
    torch.manual_seed(seed)
    gate = ObservedPriorGate().to(device)
    if sum(p.numel() for p in gate.parameters()) != plan['trainable_parameters']:
        raise ValueError('gate parameter contract changed')
    reference = forecast_quality(data['validation']['current'].numpy(), data['validation']['target'].numpy())
    best, selected, stale = reference['weighted_smooth_l1'], None, 0
    optimizer = torch.optim.AdamW(gate.parameters(), lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    train = {name: value.to(device) for name, value in data['train'].items()}
    atomic_json(binding | {'plan': plan, 'persistence_reference': reference}, directory / 'RUN.json')
    for epoch in range(1, plan['epochs'] + 1):
        gate.train()
        order = torch.randperm(len(train['target']), generator=generator)
        for start in range(0, len(order), plan['batch']):
            index = order[start:start + plan['batch']].to(device)
            prediction = gate(*(train[name][index] for name in ('features', 'current', 'raw', 'available')))
            loss = dynamic_loss(prediction, train['target'][index])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite gate objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        gate.eval()
        metrics = evaluate(gate, data['validation'], device)
        chosen = improves(metrics, reference, best)
        if chosen:
            best, selected, stale = metrics['weighted_smooth_l1'], epoch, 0
        else:
            stale += 1
        state = {'schema': 'argus.observed-prior-gate/v1', 'seed': seed, 'epoch': epoch,
                 'model_state': gate.state_dict(), 'optimizer_state': optimizer.state_dict(),
                 'torch_rng': torch.get_rng_state(), 'sampler_rng': generator.get_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else [],
                 'binding': binding, 'validation': metrics}
        path = directory / f'epoch-{epoch:03d}.pt'
        atomic_save(state, path)
        atomic_save(state, directory / 'latest.pt')
        if chosen:
            atomic_save(state, directory / 'best.pt')
        with (directory / 'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'epoch': epoch, 'selected': chosen, 'validation': metrics, 'sha256': sha256(path)}, allow_nan=False) + '\n')
        print(json.dumps({'seed': seed, 'epoch': epoch, 'selected': chosen, 'validation_loss': metrics['weighted_smooth_l1']}), flush=True)
        if stale >= plan['patience']:
            break
    for path in directory.glob('epoch-*.pt'):
        state = torch.load(path, weights_only=True, map_location='cpu')
        restored = ObservedPriorGate()
        restored.load_state_dict(state['model_state'], strict=True)
        if not all(torch.isfinite(value).all() for value in state['model_state'].values()):
            raise ValueError('invalid emitted checkpoint')
    result = {'selected_epoch': selected, 'fallback': selected is None,
              'best_sha256': sha256(directory / 'best.pt') if selected else None,
              'persistence_reference': reference, 'runtime_admission': False}
    atomic_json(result, directory / 'RESULT.json')
    return result


def run(args):
    if args.output.exists():
        raise FileExistsError('immutable gate study exists')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    paths, hashes = validate_local_sources(plan, args.journals)
    previous = json.loads((args.study / 'FROZEN.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True)
    temporary = args.output / 'temporary-projections'
    temporary.mkdir()
    atomic_json({'plan': plan, 'plan_sha256': sha256(args.plan), 'source_sha256': hashes,
                 'prior_frozen_sha256': sha256(args.study / 'FROZEN.json'), 'assessment_parsed': False,
                 'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}, args.output / 'RUN.json')
    cache_hashes, selections, readers = {}, {}, {}
    def prepare(base, reader, role, seed):
        caches = []
        for index, source in enumerate(paths[role]):
            if sha256(source) != hashes[source.name]:
                raise ValueError('source changed after binding')
            destination = temporary / f'{seed}-{role}-{index}.pt'
            caches.append(prepare_cache(base, source, role, seed, plan, sha256(args.plan), destination,
                                        official=False, samples=plan['samples_per_source'][role]))
            cache_hashes[destination.name] = sha256(destination)
        return project_gate(reader, combine(caches, role)['tensors'], plan['batch'], args.device)
    for seed in plan['seeds']:
        base, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base = base.to(args.device).eval()
        selected = previous['selections'][f'{seed}/phase-soft']
        path = args.study / 'trials' / f'{seed}-phase-soft' / 'best.pt'
        if selected['fallback'] or sha256(path) != selected['checkpoint_sha256']:
            raise ValueError('frozen reader unavailable or changed')
        reader = AhemPhaseReader(base, phase_width=plan['phase_width'])
        reader.load_state_dict(torch.load(path, weights_only=True, map_location=args.device)['model_state'], strict=True)
        reader.eval().requires_grad_(False)
        data = {role: prepare(base, reader, role, seed) for role in ('train', 'validation')}
        selections[str(seed)] = fit(seed, data, plan, args.output / str(seed),
            {'plan_sha256': sha256(args.plan), 'reader_sha256': sha256(path), 'cache_sha256': dict(cache_hashes)}, args.device)
        readers[seed] = (base, reader)
        del data
    atomic_json({'selections': selections, 'assessment_parsed': False}, args.output / 'FROZEN.json')
    atomic_json({'frozen_sha256': sha256(args.output / 'FROZEN.json'), 'role': 'local_post_selection_followup'}, args.output / 'ACCESS.json')
    report = {}
    for seed, (base, reader) in readers.items():
        data = prepare(base, reader, 'assessment', seed)
        rows = {name: forecast_quality(data[field].numpy(), data['target'].numpy())
                for name, field in (('persistence', 'current'), ('raw_read', 'raw'), ('unchanged_reader_forecast', 'unchanged'))}
        selection = selections[str(seed)]
        if selection['fallback']:
            rows['selected_gate'] = rows['persistence'] | {'baseline_fallback': True}
        else:
            path = args.output / str(seed) / 'best.pt'
            if sha256(path) != selection['best_sha256']:
                raise ValueError('frozen gate changed')
            gate = ObservedPriorGate().to(args.device).eval()
            gate.load_state_dict(torch.load(path, weights_only=True, map_location=args.device)['model_state'], strict=True)
            rows['selected_gate'] = evaluate(gate, data, args.device)
        report[str(seed)] = rows
        atomic_json(rows, args.output / f'assessment-{seed}.json')
        del data
    atomic_json({'role': plan['role'], 'report': report, 'selections': selections,
                 'runtime_admission': False, 'publication_claim': False}, args.output / 'RESULT.json')
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

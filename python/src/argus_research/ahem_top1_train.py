"""Accuracy-first development repair; no screen or test CLI is provided."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_contextual_reader import AhemContextualReader
from .ahem_population_audit import sha256
from .ahem_reader_experiment import INPUTS, atomic_json, atomic_save, evaluate, infer, read_cache, resident_inputs
from .ahem_reader_ranking import centered_correction_penalty, tie_aware_regret_ranking
from .ahem_top1_reader import AhemAnchorReader, AhemConvexForecaster, best_oracle_margin, top1_selection
from .flow_v6_final_evaluate import _load_frozen_checkpoint
from .flow_v6_objective import dynamic_loss


def observed_infer(model, tensors, indices, device):
    if not isinstance(model, (AhemAnchorReader, AhemConvexForecaster)):
        return infer(model, tensors, indices, device)
    return model._read(*(tensors[name][indices].to(device) for name in INPUTS),
                       causal_anchor=tensors['causal_anchor'][indices].to(device))


def observed_inputs(cache, device):
    inputs = resident_inputs(cache, device)
    # Explicit current packet fields, not cached future targets or EWMA history.
    inputs['causal_anchor'] = cache['tensors']['anchor_features'][:, :2].to(device)
    if not torch.equal(cache['tensors']['anchor_features'][:, :2], cache['tensors']['persistence']):
        raise ValueError('observed anchor differs from causal persistence input')
    return inputs


def objective(output, original, target, plan):
    margin, details = best_oracle_margin(output.retrieval_logits, output.successor_dynamic, target,
                                        output.valid, tie_tolerance=plan['tie_tolerance'], margin=plan['margin'])
    regret, _ = tie_aware_regret_ranking(output.retrieval_logits, output.successor_dynamic, target,
                                        output.valid, minimum_regret=plan['minimum_regret'],
                                        tie_tolerance=plan['tie_tolerance'], margin=plan['margin'])
    penalty = centered_correction_penalty(output.retrieval_logits, original.retrieval_logits, output.valid)
    return (margin + plan['regret_weight'] * regret + plan['correction_penalty'] * penalty
            + plan.get('forecast_weight', 0.) * dynamic_loss(output.prediction, target)), details


def selection_score(metrics, reference, plan):
    score = top1_selection(metrics, reference, forecast_ceiling=plan['forecast_ceiling'],
                           regret_ceiling=plan['regret_ceiling'])
    priority = plan.get('selection_priority', 'top1')
    if priority not in ('top1', 'forecast'):
        raise ValueError('unknown selection priority')
    if priority == 'top1' or score is None:
        return score
    if score[0] < reference['retrieval']['multi_candidate_top1_tie_aware']:
        return None
    return (-metrics['forecast']['weighted_smooth_l1'], *score)


def fit(backbone, candidate, caches, plan, output, binding, device):
    if output.exists():
        raise FileExistsError('immutable top-1 run already exists')
    train_inputs = observed_inputs(caches['train'], device)
    observed_inputs(caches['validation'], device)
    reference = evaluate(backbone, caches['validation'], plan['batch'], device)
    best_score = selection_score(reference, reference, plan)
    if best_score is None:
        raise ValueError('validation has no meaningful multi-candidate reference')
    baseline_state = {key: value.detach().cpu().clone() for key, value in backbone.state_dict().items()}
    trainable = [p for p in candidate.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=plan['learning_rate'],
                                  weight_decay=plan['weight_decay'])
    # Independent RNG: architecture construction cannot alter paired minibatches.
    sampler = torch.Generator().manual_seed(binding['seed'])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(binding, output / 'RUN.json')
    atomic_json(reference, output / 'REFERENCE.json')
    stale, selected_epoch = 0, None
    for epoch in range(1, plan['epochs'] + 1):
        candidate.train()
        order = torch.randperm(len(train_inputs['target']), generator=sampler)
        total, informative = 0., 0
        for start in range(0, len(order), plan['batch']):
            indices = order[start:start + plan['batch']].to(device)
            actual = observed_infer(candidate, train_inputs, indices, device)
            with torch.no_grad():
                original = infer(backbone, train_inputs, indices, device)
            loss, details = objective(actual, original, train_inputs['target'][indices], plan)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite top-1 loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            optimizer.step()
            if hasattr(candidate, 'project_parameters'):
                candidate.project_parameters()
            total += float(loss.detach()) * len(indices)
            informative += details['informative_queries']
        candidate.eval()
        validation = evaluate(candidate, caches['validation'], plan['batch'], device, infer_fn=observed_infer)
        score = selection_score(validation, reference, plan)
        selected = score is not None and score > best_score
        if selected:
            best_score, stale, selected_epoch = score, 0, epoch
        else:
            stale += 1
        state = {'schema': 'argus.top1-reader-checkpoint/v3', 'binding': binding, 'epoch': epoch,
                 'model_state': candidate.state_dict(), 'optimizer_state': optimizer.state_dict(),
                 'torch_rng': torch.get_rng_state(), 'sampler_rng': sampler.get_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 'validation': validation, 'training_objective': total / len(order)}
        atomic_save(state, output / f'epoch-{epoch:03d}.pt')
        atomic_save(state, output / 'latest.pt')
        if selected:
            atomic_save(state, output / 'best.pt')
        row = {'epoch': epoch, 'training_objective': state['training_objective'], 'validation': validation,
               'informative_training_queries': informative, 'selected_best': selected,
               'checkpoint_sha256': sha256(output / f'epoch-{epoch:03d}.pt')}
        with (output / 'progress.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({'epoch': epoch, 'selected_best': selected, 'score': score}), flush=True)
        if stale >= plan['patience']:
            break
    for key, value in baseline_state.items():
        if not torch.equal(candidate.state_dict()[key].cpu(), value):
            raise ValueError('frozen backbone changed')
    result = {'status': 'validation_candidate_only' if selected_epoch else 'baseline_retained_no_improvement',
              'selected_epoch': selected_epoch, 'reference_validation': reference,
              'role': binding['role'], 'screen_accessed': False, 'runtime_admission': False,
              'publication_claim': False, 'backbone_unchanged': True}
    if selected_epoch:
        best = torch.load(output / 'best.pt', weights_only=True, map_location='cpu')
        result |= {'selected_validation': best['validation'], 'checkpoint_sha256': sha256(output / 'best.pt')}
    atomic_json(result, output / 'RESULT.json')
    return result


def run(args):
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if (plan['schema'] != 'argus.top1-reader-development/v3' or plan['screen_access'] is not False
            or plan['runtime_admission'] is not False or args.variant not in plan['variants']):
        raise ValueError('unplanned top-1 protocol or variant')
    parent_path = args.plan.parent / plan['bound_input_plan']
    if sha256(parent_path) != plan['bound_input_plan_sha256']:
        raise ValueError('bound input plan changed')
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    if args.seed not in parent['seeds']:
        raise ValueError('unplanned seed')
    caches = {role: read_cache(path, role, args.seed) for role, path in
              [('train', args.train), ('validation', args.validation)]}
    for role, path in [('train', args.train), ('validation', args.validation)]:
        cache = caches[role]
        if (sha256(path) != parent['expected_cache_sha256'][f'{args.seed}/{role}']
                or cache['source'] != parent['sources'][role] or not cache['official_manifest_verified']
                or len(cache['tensors']['target']) != parent['samples'][role]):
            raise ValueError('bound development inputs changed')
    if caches['train']['source_sha256'] == caches['validation']['source_sha256']:
        raise ValueError('source leakage')
    torch.manual_seed(args.seed)
    backbone, _, _ = _load_frozen_checkpoint(args.checkpoints, args.seed, AHEM_NAME,
                                             EXPECTED_CHECKPOINT_SHA256[args.seed][AHEM_NAME])
    backbone = backbone.to(args.device).eval()
    candidate = (AhemAnchorReader(backbone) if args.variant == 'anchor-top1' else AhemContextualReader(backbone))
    binding = {'seed': args.seed, 'variant': args.variant, 'role': 'exploratory_development_not_independent',
               'plan_sha256': sha256(args.plan), 'candidate': candidate.research_contract(),
               'backbone_sha256': EXPECTED_CHECKPOINT_SHA256[args.seed][AHEM_NAME],
               'cache_sha256': {role: sha256(path) for role, path in
                                [('train', args.train), ('validation', args.validation)]},
               'torch_version': str(torch.__version__), 'device': args.device,
               'source_code_sha256': {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    return fit(backbone, candidate, caches, plan, args.output, binding, args.device)


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'train', 'validation', 'checkpoints', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--variant', choices=('scalar-top1', 'anchor-top1'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args)


if __name__ == '__main__':
    main()

"""Bounded 2x2 development experiment; accepts no screen/test input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_contextual_reader import AhemContextualReader
from .ahem_forecast_audit import ProtocolMedian
from .ahem_population_audit import sha256
from .ahem_query_conditioned_reader import AhemQueryConditionedReader
from .ahem_reader_experiment import atomic_json, atomic_save, evaluate, infer, read_cache, resident_inputs
from .ahem_reader_ranking import centered_correction_penalty, tie_aware_regret_ranking
from .flow_v6_final_evaluate import _load_frozen_checkpoint
from .flow_v6_objective import dynamic_loss, frozen_utility_distribution, retrieval_cross_entropy


def objective(output, original, target, plan, variant):
    if variant['objective'] == 'mixed':
        utility, _ = frozen_utility_distribution(output.successor_dynamic, target, output.valid,
                                                temperature=plan['utility_temperature'])
        return dynamic_loss(output.prediction, target) + plan['retrieval_weight'] * retrieval_cross_entropy(
            output.retrieval_logits, utility, output.valid), {'objective': 'mixed'}
    if variant['objective'] != 'regret_ranking':
        raise ValueError('unplanned objective')
    ranking, details = tie_aware_regret_ranking(output.retrieval_logits, output.successor_dynamic, target,
                                               output.valid, minimum_regret=plan['minimum_regret'],
                                               tie_tolerance=plan['tie_tolerance'],
                                               duplicate_tolerance=plan['duplicate_tolerance'],
                                               margin=plan['ranking_margin'])
    penalty = centered_correction_penalty(output.retrieval_logits, original.retrieval_logits, output.valid)
    return ranking + plan['correction_penalty'] * penalty, details


def run(args):
    if args.output.exists():
        raise FileExistsError('existing immutable revision run')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if (plan['schema'] != 'argus.reader-revision-development/v2' or plan['screen_access']
            or plan['runtime_admission'] or plan['role'] != 'development_only_no_screen_or_final_access'):
        raise ValueError('wrong revision protocol')
    if args.variant not in plan['variants'] or args.seed not in plan['seeds']:
        raise ValueError('unplanned variant or seed')
    variant = plan['variants'][args.variant]
    caches = {role: read_cache(path, role, args.seed) for role, path in [('train', args.train), ('validation', args.validation)]}
    for role, path in [('train', args.train), ('validation', args.validation)]:
        cache = caches[role]
        if (sha256(path) != plan['expected_cache_sha256'][f'{args.seed}/{role}']
                or cache['source'] != plan['sources'][role] or not cache['official_manifest_verified']
                or len(cache['tensors']['target']) != plan['samples'][role]):
            raise ValueError('revision inputs differ from bound caches')
    if caches['train']['source_sha256'] == caches['validation']['source_sha256']:
        raise ValueError('source leakage')
    torch.manual_seed(args.seed)
    backbone, _, _ = _load_frozen_checkpoint(args.checkpoints, args.seed, AHEM_NAME,
                                             EXPECTED_CHECKPOINT_SHA256[args.seed][AHEM_NAME])
    backbone = backbone.to(args.device).eval()
    if variant['reader'] == 'scalar':
        candidate = AhemContextualReader(backbone)
    elif variant['reader'] == 'query_conditioned':
        candidate = AhemQueryConditionedReader(backbone, interaction_rank=plan['interaction_rank'],
                                               reader_width=plan['reader_width'])
    else:
        raise ValueError('unplanned reader')
    optimizer = torch.optim.AdamW([parameter for parameter in candidate.parameters() if parameter.requires_grad],
                                  lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
    train_inputs = resident_inputs(caches['train'], args.device)
    median = ProtocolMedian.fit(caches['train']['tensors']['anchor_features'].numpy(), caches['train']['tensors']['target'].numpy())
    reference = evaluate(backbone, caches['validation'], plan['batch'], args.device, median)
    args.output.mkdir(parents=True)
    binding = {'plan_sha256': sha256(args.plan), 'seed': args.seed, 'variant': args.variant,
               'backbone_sha256': EXPECTED_CHECKPOINT_SHA256[args.seed][AHEM_NAME],
               'role': plan['role'], 'screen_accessed': False, 'candidate': candidate.research_contract(),
               'source_code_sha256': {name: sha256(Path(__file__).parent / name) for name in
                                      ('ahem_reader_revision_train.py', 'ahem_query_conditioned_reader.py',
                                       'ahem_reader_ranking.py', 'ahem_contextual_reader.py', 'ahem_reader_experiment.py',
                                       'ahem_v6.py', 'flow_v6_objective.py')},
               'cache_sha256': {role: sha256(path) for role, path in [('train', args.train), ('validation', args.validation)]}}
    atomic_json(binding, args.output / 'RUN.json')
    best, stale = float('inf'), 0
    for epoch in range(1, plan['epochs'] + 1):
        candidate.train()
        order = torch.randperm(len(train_inputs['target']))
        total = 0.
        informative = 0
        for start in range(0, len(order), plan['batch']):
            selected = order[start:start + plan['batch']].to(args.device)
            output = infer(candidate, train_inputs, selected, args.device)
            with torch.no_grad():
                original = infer(backbone, train_inputs, selected, args.device)
            loss, details = objective(output, original, train_inputs['target'][selected], plan, variant)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite revision loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in candidate.parameters() if p.requires_grad], 1.,
                                           error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(selected)
            informative += details.get('informative_queries', 0)
        candidate.eval()
        validation = evaluate(candidate, caches['validation'], plan['batch'], args.device, median)
        forecast = validation['forecast']['weighted_smooth_l1']
        regret = validation['retrieval']['selected_oracle_regret']
        score = regret if regret is not None and forecast <= reference['forecast']['weighted_smooth_l1'] * plan['forecast_ceiling_ratio'] else float('inf')
        state = {'schema': 'argus.reader-revision-checkpoint/v2', 'binding': binding, 'epoch': epoch,
                 'model_state': candidate.state_dict(), 'optimizer_state': optimizer.state_dict(),
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 'validation': validation, 'training_objective': total / len(order)}
        atomic_save(state, args.output / f'epoch-{epoch:03d}.pt')
        atomic_save(state, args.output / 'latest.pt')
        selected_best = score < best
        if selected_best:
            best, stale = score, 0
            atomic_save(state, args.output / 'best.pt')
        else:
            stale += 1
        row = {'epoch': epoch, 'training_objective': state['training_objective'], 'validation': validation,
               'informative_training_queries': informative, 'selected_best': selected_best,
               'checkpoint_sha256': sha256(args.output / f'epoch-{epoch:03d}.pt')}
        with (args.output / 'progress.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
            handle.flush()
            import os
            os.fsync(handle.fileno())
        print(json.dumps({'variant': args.variant, 'seed': args.seed, 'epoch': epoch,
                          'forecast': forecast, 'regret': regret}), flush=True)
        if stale >= plan['patience']:
            break
    result = {'role': 'development_only_not_an_independent_test', 'reference_validation': reference,
              'screen_accessed': False, 'runtime_admission': False, 'publication_claim': False}
    if (args.output / 'best.pt').exists():
        checkpoint = torch.load(args.output / 'best.pt', weights_only=True, map_location=args.device)
        result |= {'status': 'validation_candidate_only', 'best_epoch': checkpoint['epoch'],
                   'selected_validation': checkpoint['validation'], 'checkpoint_sha256': sha256(args.output / 'best.pt')}
    else:
        result['status'] = 'failed_forecast_protection'
    atomic_json(result, args.output / 'RESULT.json')


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'train', 'validation', 'checkpoints', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--variant', choices=('v1-mixed', 'v2-mixed', 'v1-ranking', 'v2-ranking'), required=True)
    parser.add_argument('--seed', type=int, choices=(20260906, 20260907, 20260908), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError('threads must be positive')
    torch.set_num_threads(args.threads)
    run(args)


if __name__ == '__main__':
    main()

"""Real local-capture engineering smoke test, never a research comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_contextual_reader import AhemContextualReader
from .ahem_population_audit import sha256
from .ahem_query_conditioned_reader import AhemQueryConditionedReader
from .ahem_reader_experiment import atomic_json, atomic_save, evaluate, infer, read_cache, resident_inputs
from .ahem_reader_revision_train import objective
from .flow_v6_final_evaluate import _load_frozen_checkpoint


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'train', 'validation', 'checkpoints', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--steps', type=int, default=2)
    args = parser.parse_args()
    if not 2 <= args.steps <= 64:
        raise ValueError('engineering smoke must use 2 to 64 updates')
    if args.output.exists():
        raise FileExistsError('immutable engineering preflight already exists')
    torch.set_num_threads(4)
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if plan['schema'] != 'argus.reader-revision-development/v2' or plan['screen_access']:
        raise ValueError('wrong revision plan')
    seed = 20260906
    caches = {role: read_cache(path, role, seed) for role, path in
              [('train', args.train), ('validation', args.validation)]}
    for cache in caches.values():
        if cache['official_manifest_verified'] or len(cache['tensors']['target']) > 128:
            raise ValueError('this preflight only accepts small local engineering captures')
    if caches['train']['source_sha256'] == caches['validation']['source_sha256']:
        raise ValueError('engineering captures must be source-disjoint')
    backbone, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME,
                                             EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
    backbone = backbone.to(args.device).eval()
    frozen = {name: value.detach().clone() for name, value in backbone.state_dict().items()}
    inputs = resident_inputs(caches['train'], args.device)
    args.output.mkdir(parents=True)
    report = {'role': 'engineering_only_real_local_captures_not_scientific_evidence',
              'screen_accessed': False, 'runtime_admission': False, 'device': args.device,
              'engineering_updates': args.steps,
              'plan_sha256': sha256(args.plan), 'cache_sha256': {'train': sha256(args.train),
                                                               'validation': sha256(args.validation)},
              'variants': {}}
    for name, variant in plan['variants'].items():
        torch.manual_seed(seed)
        candidate = (AhemContextualReader(backbone) if variant['reader'] == 'scalar' else
                     AhemQueryConditionedReader(backbone, interaction_rank=plan['interaction_rank'],
                                                reader_width=plan['reader_width']))
        optimizer = torch.optim.AdamW([p for p in candidate.parameters() if p.requires_grad],
                                      lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
        progress = []
        for epoch in range(1, args.steps + 1):
            candidate.train()
            output = infer(candidate, inputs, slice(None), args.device)
            with torch.no_grad():
                original = infer(backbone, inputs, slice(None), args.device)
            loss, details = objective(output, original, inputs['target'], plan, variant)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                               for p in candidate.parameters()):
                raise ValueError('nonfinite engineering loss or gradient')
            optimizer.step()
            progress.append({'epoch': epoch, 'objective': float(loss.detach()), 'details': details})
        candidate.eval()
        for key, value in frozen.items():
            if not torch.equal(candidate.state_dict()[key], value):
                raise ValueError('frozen backbone changed')
        checkpoint_path = args.output / f'{name}.pt'
        atomic_save({'role': report['role'], 'model_state': candidate.state_dict(),
                     'optimizer_state': optimizer.state_dict(), 'epoch': args.steps,
                     'contract': candidate.research_contract()}, checkpoint_path)
        candidate.load_state_dict(torch.load(checkpoint_path, weights_only=True,
                                              map_location=args.device)['model_state'], strict=True)
        report['variants'][name] = {'contract': candidate.research_contract(), 'progress': progress,
                                   'checkpoint_sha256': sha256(checkpoint_path),
                                   'frozen_backbone_unchanged': True,
                                   'training_fit_only': evaluate(candidate, caches['train'], 64, args.device),
                                   'validation_smoke': evaluate(candidate, caches['validation'], 64, args.device)}
    atomic_json(report, args.output / 'ENGINEERING_ONLY.json')
    print(json.dumps({'status': 'engineering_preflight_passed', 'variants': list(report['variants']),
                      'device': args.device, 'scientific_improvement_proven': False}))


if __name__ == '__main__':
    main()

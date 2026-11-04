"""Small real-capture engineering check, explicitly not scientific evidence."""
import argparse
import json
from pathlib import Path

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_contextual_reader import AhemContextualReader
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json, evaluate, read_cache
from .ahem_top1_reader import AhemAnchorReader
from .ahem_top1_train import fit, observed_infer, observed_inputs
from .flow_v6_final_evaluate import _load_frozen_checkpoint


def main():
    parser = argparse.ArgumentParser()
    for name in ('caches', 'checkpoints', 'plan', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('immutable engineering run exists')
    torch.set_num_threads(4)
    seed = 20260906
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if plan['schema'] != 'argus.top1-reader-development/v3':
        raise ValueError('wrong preflight plan')
    plan = plan | {'epochs': 8, 'patience': 8, 'batch': 16}
    caches = {role: read_cache(args.caches / (role + '.pt'), role, seed) for role in ('train', 'validation')}
    if (any(len(c['tensors']['target']) != 64 for c in caches.values())
            or caches['train']['source_sha256'] == caches['validation']['source_sha256']
            or any('2018' in c['source'] for c in caches.values())):
        raise ValueError('expected separate 64-query laptop capture engineering inputs')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.output.mkdir(parents=True)
    results = {}
    for variant in ('scalar-top1', 'anchor-top1'):
        torch.manual_seed(seed)
        base, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME,
                                              EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base = base.to(device).eval()
        candidate = AhemAnchorReader(base) if variant == 'anchor-top1' else AhemContextualReader(base)
        observed_inputs(caches['train'], device)
        initial_train = evaluate(candidate, caches['train'], 16, device, infer_fn=observed_infer)
        binding = {'seed': seed, 'variant': variant, 'role': 'engineering_only_64_query_real_laptop_captures',
                   'candidate': candidate.research_contract(), 'device': device,
                   'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
                   'plan_sha256': sha256(args.plan), 'effective_engineering_plan': plan,
                   'cache_sha256': {role: sha256(args.caches / (role + '.pt')) for role in caches}}
        result = fit(base, candidate, caches, plan, args.output / variant, binding, device)
        final_train = evaluate(candidate, caches['train'], 16, device, infer_fn=observed_infer)
        result |= {'initial_training': initial_train, 'latest_training': final_train}
        results[variant] = result
    atomic_json({'role': 'engineering_only_not_a_generalization_claim', 'raw_data_downloaded': False,
                 'screen_accessed': False, 'results': results}, args.output / 'ENGINEERING_ONLY.json')


if __name__ == '__main__':
    main()

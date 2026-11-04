"""Exact train-only convex calibration, separate from the frozen local trial."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_forecast_audit import forecast_quality
from .ahem_local_readiness import predictions, summarize
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json, atomic_save
from .ahem_top1_reader import AhemAnchorBlendReader, AhemConvexForecaster
from .flow_v6_final_evaluate import _load_frozen_checkpoint


def fit_convex_blend(forecast, observed, target, *, iterations=64):
    """Minimize size/delay Smooth-L1 independently over coefficients in [0,1].

    e(lambda)=forecast + lambda*(observed-forecast) - target.
    dL/dlambda=mean(clip(e,-1,1)*(observed-forecast)), monotone in lambda.
    Positive frozen per-output weights do not change each minimizer. Targets
    belong exclusively to completed training queries, never online inference.
    """
    arrays = [np.asarray(value, dtype=np.float64) for value in (forecast, observed, target)]
    if (arrays[0].ndim != 2 or arrays[0].shape[1] != 2 or not len(arrays[0])
            or any(value.shape != arrays[0].shape or not np.isfinite(value).all() for value in arrays)
            or type(iterations) is not int or not 1 <= iterations <= 128):
        raise ValueError('finite aligned two-output training arrays and bounded iterations required')
    forecast, observed, target = arrays
    difference = observed - forecast
    def derivative(coefficient):
        error = forecast + coefficient * difference - target
        return np.mean(np.clip(error, -1, 1) * difference, axis=0)
    low, high = np.zeros(2), np.ones(2)
    at_zero, at_one = derivative(low), derivative(high)
    for _ in range(iterations):
        middle = (low + high) / 2
        negative = derivative(middle) < 0
        low, high = np.where(negative, middle, low), np.where(negative, high, middle)
    coefficient = (low + high) / 2
    coefficient = np.where(at_zero >= 0, 0., np.where(at_one <= 0, 1., coefficient))
    return coefficient, {'iterations': iterations, 'gradient_at_zero': at_zero.tolist(),
                         'gradient_at_one': at_one.tolist(), 'gradient_at_solution': derivative(coefficient).tolist(),
                         'samples': len(target), 'role': 'train_only_not_a_generalization_guarantee'}


def run(args):
    if args.output.exists():
        raise FileExistsError('immutable calibration directory exists')
    study = json.loads((args.study / 'STUDY.json').read_text(encoding='utf-8'))
    frozen = json.loads((args.study / 'FROZEN.json').read_text(encoding='utf-8'))
    if study['schema'] != 'argus.local-reader-readiness/v1':
        raise ValueError('only the completed local engineering study accepted')
    args.output.mkdir(parents=True)
    records = {}
    for seed in study['plan']['seeds']:
        selected = frozen['selections'][f'{seed}/anchor-top1']
        if selected['uses_original_backbone']:
            raise ValueError('no selected anchor checkpoint to calibrate')
        path = args.study / 'runs' / f'{seed}-anchor-top1' / 'best.pt'
        if sha256(path) != selected['checkpoint_sha256']:
            raise ValueError('selected parent hash changed')
        state = torch.load(path, map_location=args.device, weights_only=True)
        base, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base = base.to(args.device).eval()
        model = AhemConvexForecaster(base) if args.original_reader else AhemAnchorBlendReader(base)
        model.eval()
        parent_state = base.state_dict() if args.original_reader else state['model_state']
        model.load_state_dict(parent_state | {'persistence_blend': model.persistence_blend.detach().clone()}, strict=True)
        caches = {}
        for role in ('train', 'validation'):
            cache_path = args.study / 'caches' / str(seed) / (role + '.pt')
            if sha256(cache_path) != state['binding']['cache_sha256'][role]:
                raise ValueError('parent training/validation cache changed')
            caches[role] = torch.load(cache_path, map_location='cpu', weights_only=True)
        train_prediction, train_weights = predictions(model, caches['train'], args.device, 512)
        coefficients, details = fit_convex_blend(train_prediction.numpy(), caches['train']['tensors']['persistence'].numpy(),
                                                caches['train']['tensors']['target'].numpy())
        with torch.no_grad():
            model.persistence_blend.copy_(torch.as_tensor(coefficients, device=args.device, dtype=model.persistence_blend.dtype))
        results = {}
        for role, cache in caches.items():
            actual, weights = predictions(model, cache, args.device, 512)
            results[role] = summarize(actual, weights, cache)
            results[role]['persistence'] = forecast_quality(cache['tensors']['persistence'].numpy(), cache['tensors']['target'].numpy())
            if role == 'train' and not torch.equal(weights, train_weights):
                raise ValueError('convex calibration changed retrieval weights')
        for key, value in parent_state.items():
            if not torch.equal(model.state_dict()[key], value):
                raise ValueError('calibration changed reader or backbone')
        directory = args.output / str(seed)
        binding = {'role': 'train_fit_development_validation_only', 'seed': seed,
                   'parent_sha256': EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME] if args.original_reader else sha256(path),
                   'original_reader_preserved': args.original_reader, 'contract': model.research_contract(),
                   'study_sha256': sha256(args.study / 'STUDY.json'), 'assessment_reopened': False,
                   'source_code_sha256': sha256(Path(__file__)), 'details': details,
                   'coefficients': coefficients.tolist()}
        atomic_save({'schema': 'argus.convex-calibrated-reader/v1', 'binding': binding,
                     'model_state': model.state_dict()}, directory / 'calibrated.pt')
        record = {'binding': binding, 'metrics': results, 'checkpoint_sha256': sha256(directory / 'calibrated.pt'),
                  'reader_and_backbone_unchanged': True, 'runtime_admission': False, 'publication_claim': False}
        atomic_json(record, directory / 'RESULT.json')
        records[str(seed)] = record
    atomic_json({'role': 'development_only_not_independent', 'assessment_reopened': False,
                 'original_locked_test_opened': False, 'records': records}, args.output / 'RESULT.json')
    print(json.dumps({seed: {'coefficients': result['binding']['coefficients'],
                           'validation_forecast': result['metrics']['validation']['forecast']['weighted_smooth_l1'],
                           'persistence': result['metrics']['validation']['persistence']['weighted_smooth_l1']}
                      for seed, result in records.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('study', 'checkpoints', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--original-reader', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args)


if __name__ == '__main__':
    main()

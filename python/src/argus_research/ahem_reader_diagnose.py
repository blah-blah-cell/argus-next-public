"""Read-only diagnosis of pilot weights on an existing engineering cache.

Never reads the failed development screen or treats laptop traffic as new
scientific evidence. Measures movement, useful score differences and gradient
conflict instead of guessing from aggregate accuracy alone.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_contextual_reader import AhemContextualReader
from .ahem_reader_experiment import INPUTS, atomic_json
from .flow_v6_final_evaluate import _load_frozen_checkpoint
from .flow_v6_objective import dynamic_loss, frozen_utility_distribution, retrieval_cross_entropy


def gradient_vector(loss, parameters):
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(parameter) if value is None else value).flatten()
                      for value, parameter in zip(values, parameters)])


def diagnose(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    cache = torch.load(args.engineering_cache, map_location='cpu', weights_only=True)
    if cache['official_manifest_verified']:
        raise ValueError('use the designated engineering replay, not a study screen')
    inputs = cache['tensors']
    args_input = tuple(inputs[name] for name in INPUTS)
    rows = []
    for seed in (20260906, 20260907, 20260908):
        if cache['seed'] != seed:
            # Projections depend on the backbone: never reuse one seed's keys
            # with another seed. Weight/progress diagnostics remain possible.
            aligned = False
        else:
            aligned = True
        with zipfile.ZipFile(args.artifacts / f'{seed}-final.zip') as archive:
            best = torch.load(io.BytesIO(archive.read('best.pt')), map_location='cpu', weights_only=True)
            latest = torch.load(io.BytesIO(archive.read('latest.pt')), map_location='cpu', weights_only=True)
            progress = [json.loads(line) for line in archive.read('progress.jsonl').splitlines()]
        base, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME,
                                           EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        torch.manual_seed(seed)
        model = AhemContextualReader(base).eval()
        initial_last_layer = model.reader_head[-1].weight.clone()
        model.load_state_dict(best['model_state'], strict=True)
        row = {'seed': seed, 'best_epoch': best['epoch'], 'latest_epoch': latest['epoch'],
               'last_layer_weight_norm': float((model.reader_head[-1].weight - initial_last_layer).norm().detach()),
               'first_training_objective': progress[0]['training_objective'],
               'last_training_objective': progress[-1]['training_objective'],
               'first_validation_regret': progress[0]['validation']['retrieval']['selected_oracle_regret'],
               'last_validation_regret': progress[-1]['validation']['retrieval']['selected_oracle_regret'],
               'projection_seed_aligned': aligned}
        if aligned:
            parameters = list(model.reader_head.parameters())
            output = model._read(*args_input)
            with torch.no_grad():
                original = base._read(*args_input)
            valid = inputs['valid']
            counts = valid.sum(-1)
            multi = counts >= 2
            correction = output.retrieval_logits - original.retrieval_logits
            mean = (correction * valid).sum(-1) / counts.clamp_min(1)
            centered = (correction - mean[:, None]).masked_fill(~valid, 0)
            slot_std = (centered.square().sum(-1) / counts.clamp_min(1)).sqrt()
            sorted_scores = original.retrieval_logits.masked_fill(~valid, -torch.inf).sort(-1, descending=True).values
            margins = sorted_scores[:, 0] - sorted_scores[:, 1]
            changed = output.retrieval_weights.argmax(-1) != original.retrieval_weights.argmax(-1)
            utility, slot_loss = frozen_utility_distribution(output.successor_dynamic, inputs['target'], valid)
            useful = multi & ((slot_loss.masked_fill(~valid, -torch.inf).amax(-1) - slot_loss.amin(-1)) > .01)
            forecast = dynamic_loss(output.prediction, inputs['target'])
            retrieval = retrieval_cross_entropy(output.retrieval_logits, utility, valid)
            forecast_gradient = gradient_vector(forecast, parameters)
            retrieval_gradient = gradient_vector(.25 * retrieval, parameters)
            cosine = torch.nn.functional.cosine_similarity(forecast_gradient, retrieval_gradient, dim=0)
            row['engineering_replay'] = {
                'queries': len(valid), 'multi_candidate_queries': int(multi.sum()),
                'top1_changed': int((changed & multi).sum()),
                'centered_correction_std_median': float(slot_std[multi].median().detach()),
                'baseline_top2_margin_median': float(margins[multi].median()),
                'mean_attention_absolute_change': float((output.retrieval_weights - original.retrieval_weights).abs()[valid].mean().detach()),
                'forecast_gradient_norm': float(forecast_gradient.norm()),
                'weighted_retrieval_gradient_norm': float(retrieval_gradient.norm()),
                'gradient_cosine': float(cosine),
                'query_oracles_with_useful_loss_spread': int(useful.sum()),
                'utility_entropy_mean': float((-(utility * utility.clamp_min(1e-8).log()).sum(-1))[multi].mean())}
        rows.append(row)
    result = {'role': 'engineering_mechanism_diagnosis_not_new_test',
              'failed_screen_reaccessed': False, 'rows': rows,
              'warning': 'Engineering gradient/score observations do not establish a cause on the full pilot corpus.'}
    atomic_json(result, args.output)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    for name in ('artifacts', 'checkpoints', 'engineering-cache', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    torch.set_num_threads(1)
    diagnose(parser.parse_args())


if __name__ == '__main__':
    main()

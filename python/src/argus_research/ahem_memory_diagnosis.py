"""Post-hoc diagnostics of saved engineering outputs; no training or new access."""
import argparse
import json
from pathlib import Path

import torch

from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json
from .ahem_retrieval_audit import retrieval_quality
from .flow_v6_objective import dynamic_loss


def diagnose(tensors, weights):
    valid = tensors['valid']
    successor = tensors['successor_dynamic']
    target = tensors['target']
    if not torch.isfinite(successor[valid]).all() or not torch.isfinite(target).all():
        raise ValueError('nonfinite saved outcomes')
    retrieval = retrieval_quality(successor, target, valid, weights)
    losses = dynamic_loss(successor, target[:, None].expand_as(successor), reduction='none')
    losses = losses.masked_fill(~valid, torch.inf)
    available = valid.any(-1)
    multi = valid.sum(-1) >= 2
    oracle = losses.min(-1).values
    persistence = dynamic_loss(tensors['persistence'], target, reduction='none')
    entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1)
    selected = weights.masked_fill(~valid, -1).argmax(-1)
    selected_loss = losses.gather(1, selected[:, None]).squeeze(1)
    mixture = (weights[..., None] * successor).sum(1)
    mixture_loss = dynamic_loss(mixture, target, reduction='none')
    known_distances = dynamic_loss(successor, tensors['persistence'][:, None].expand_as(successor), reduction='none')
    nearest = known_distances.masked_fill(~valid, torch.inf).argmin(-1)
    recent = valid.long().argmax(-1)
    def mean(value, mask):
        return float(value[mask].double().mean()) if mask.any() else None
    return {
        'retrieval': retrieval,
        'memory_available_queries': int(available.sum()),
        'multi_candidate_queries': int(multi.sum()),
        'mean_valid_candidates_multi': mean(valid.sum(-1), multi),
        'effective_attended_candidates_multi': mean(entropy.exp(), multi),
        'largest_attention_weight_multi': mean(weights.max(-1).values, multi),
        'oracle_raw_successor_loss_available': mean(oracle, available),
        'persistence_loss_available': mean(persistence, available),
        'oracle_strictly_beats_persistence_fraction_available': mean((oracle < persistence - 1e-8).float(), available),
        'raw_soft_mixture_loss_available': mean(mixture_loss, available),
        'raw_top1_successor_loss_available': mean(selected_loss, available),
        'nearest_observed_anchor_equals_recency_fraction_multi': mean((nearest == recent).float(), multi),
        'interpretation': 'Oracle uses future outcomes offline and is not deployable. Raw successor controls are not the full neural forecast. Post-hoc descriptive evidence only.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('immutable diagnosis already exists')
    torch.set_num_threads(4)
    report, hashes = {}, {}
    for role, root, names in (
        ('later_captures', args.study / 'assessment', ('frozen_ahem', 'anchor-top1')),
        ('additional_captures', args.study.parent / 'additional-captures', ('frozen_ahem', 'parent_anchor')),
    ):
        report[role] = {}
        for seed in (20260906, 20260907, 20260908):
            directory = root / str(seed)
            cache_path = directory / 'CACHE.pt'
            hashes[str(cache_path)] = sha256(cache_path)
            cache = torch.load(cache_path, map_location='cpu', weights_only=True)
            report[role][str(seed)] = {}
            for name in names:
                output_path = directory / (name + '.pt')
                hashes[str(output_path)] = sha256(output_path)
                saved = torch.load(output_path, map_location='cpu', weights_only=True)
                report[role][str(seed)][name] = diagnose(cache['tensors'], saved['weights'])
    result = {'schema': 'argus.saved-memory-diagnosis/v1', 'training': False,
              'new_capture_access': False, 'publication_claim': False,
              'source_code_sha256': sha256(Path(__file__)), 'input_sha256': hashes,
              'report': report}
    atomic_json(result, args.output)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

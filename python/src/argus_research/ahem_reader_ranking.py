"""New development objective; leaves the v6.1 and first-pilot losses untouched."""
from __future__ import annotations

import math

import torch

from .flow_v6_objective import dynamic_loss


def tie_aware_regret_ranking(logits, successors, target, valid, *,
                           minimum_regret=.01, tie_tolerance=1e-8,
                           duplicate_tolerance=1e-6, margin=.1):
    """Rank an oracle-equivalent set above consequential observed alternatives.

    Equal historical outcomes share negative weight. All-tied, single-candidate
    and empty rows receive zero supervision instead of a uniformity objective.
    Targets label retrospective utility only and must never enter inference.
    """
    for value in (minimum_regret, tie_tolerance, duplicate_tolerance, margin):
        if not math.isfinite(value) or value < 0:
            raise ValueError('ranking settings must be finite and nonnegative')
    if (logits.ndim != 2 or not logits.shape[0] or not logits.shape[1]
            or successors.shape != (*logits.shape, 2) or target.shape != (len(logits), 2)
            or valid.shape != logits.shape or valid.dtype != torch.bool):
        raise ValueError('ranking inputs are not aligned')
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all() or not torch.isfinite(successors).all():
        raise ValueError('ranking inputs must be finite')
    with torch.no_grad():
        losses = dynamic_loss(successors, target[:, None].expand_as(successors), reduction='none')
        minimum = losses.masked_fill(~valid, torch.inf).amin(-1)
        minimum = torch.where(valid.any(-1), minimum, torch.zeros_like(minimum))
        regret = (losses - minimum[:, None]).clamp_min(0)
        positive = valid & (regret <= tie_tolerance)
        negative = valid & (regret > max(minimum_regret, tie_tolerance))
        equal_outcomes = ((successors[:, :, None] - successors[:, None, :]).abs().amax(-1)
                          <= duplicate_tolerance)
        multiplicity = (equal_outcomes & negative[:, None, :]).sum(-1).clamp_min(1)
        weights = regret.clamp_max(1) * negative / multiplicity
        informative = (valid.sum(-1) >= 2) & positive.any(-1) & negative.any(-1)
    # Exact masking preserves invariance to a common score shift. Empty rows
    # get a dummy slot only for numerical evaluation, never supervision; this
    # avoids the undefined backward of logsumexp over an all-negative-inf row.
    numerical_positive = positive.clone()
    numerical_positive[~positive.any(-1), 0] = True
    positive_score = logits.masked_fill(~numerical_positive, -torch.inf).logsumexp(-1)
    positive_score = positive_score - positive.sum(-1).clamp_min(1).to(logits.dtype).log()
    violation = torch.nn.functional.softplus(logits - positive_score[:, None] + margin)
    per_query = (violation * weights).sum(-1) / weights.sum(-1).clamp_min(1e-8)
    loss = per_query[informative].mean() if informative.any() else logits.sum() * 0
    return loss, {'informative_queries': int(informative.sum()),
                  'unsupervised_queries': int((~informative).sum()),
                  'positive_slots': int(positive.sum()), 'consequential_negative_slots': int(negative.sum())}


def centered_correction_penalty(logits, baseline_logits, valid):
    """Penalize changes in relative scores, not softmax-invariant shifts."""
    if logits.shape != baseline_logits.shape or valid.shape != logits.shape or valid.dtype != torch.bool:
        raise ValueError('correction penalty requires aligned score masks')
    if not torch.isfinite(logits).all() or not torch.isfinite(baseline_logits).all():
        raise ValueError('correction penalty scores must be finite')
    correction = (logits - baseline_logits.detach()).masked_fill(~valid, 0)
    mean = correction.sum(-1) / valid.sum(-1).clamp_min(1)
    centered = (correction - mean[:, None]).masked_fill(~valid, 0)
    multi = valid.sum(-1) >= 2
    if not multi.any():
        return logits.sum() * 0
    return (centered.square().sum(-1) / valid.sum(-1).clamp_min(1))[multi].mean()

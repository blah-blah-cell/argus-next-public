"""Target-free causal counterfactual forecast response features."""

from __future__ import annotations

import math

import torch
from torch import nn

from .ahem_forecast_teacher import from_weights


@torch.no_grad()
def responses(
    head: nn.Module,
    current: torch.Tensor,
    successors: torch.Tensor,
    valid: torch.Tensor,
    context: torch.Tensor,
    ordinal: torch.Tensor,
    startup_parent: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    cache = {
        "current": current,
        "successors": successors,
        "valid": valid,
        "context": context,
        "ordinal": ordinal,
        "startup_parent": startup_parent,
    }
    base_prediction, _ = from_weights(head, cache, weights)
    scale = current.new_tensor((math.log1p(65535), math.log1p(60)))
    trend = ((current - context[:, 14:16]) * context[:, 26:27]) / scale
    displacement = (base_prediction - current) / scale
    z = ((context - head.context_mean) / head.context_scale).clamp(-10, 10)
    contribution = (
        head.context_delta(z).clamp(-10, 10) / 10
        if head.context_enabled
        else torch.zeros_like(current)
    )
    columns = []
    for slot in range(weights.shape[1]):
        active = valid[:, slot]
        changed = weights.clone()
        changed[active] *= 0.9
        changed[active, slot] += 0.1
        prediction, _ = from_weights(head, cache, changed)
        d = (prediction - base_prediction) / 0.1 / scale
        physical = (
            (prediction[:, :1].expm1() - base_prediction[:, :1].expm1()) / 0.1 / 65535
        )
        outer = [
            (d[:, :, None] * value[:, None]).flatten(1)
            for value in (trend, displacement, contribution)
        ]
        features = torch.cat((d, d.abs(), *outer, physical), 1)
        features = torch.where((active & (ordinal >= 127))[:, None], features, 0)
        columns.append(features)
    result = torch.stack(columns, 1)
    if result.shape != (*weights.shape, 17) or not torch.isfinite(result).all():
        raise ValueError("finite17columncausalforecastresponse required")
    return result


def fit_scale(
    features: torch.Tensor, valid: torch.Tensor, ordinal: torch.Tensor
) -> torch.Tensor:
    rows = (ordinal >= 127) & (valid.sum(1) >= 2)
    slots = valid & rows[:, None]
    if (
        features.shape != (*valid.shape, 17)
        or not slots.any()
        or not torch.isfinite(features).all()
    ):
        raise ValueError("finite aligned fitting response slots required")
    return features[slots].double().square().mean(0).sqrt().clamp_min(1e-6)


def augment(
    hidden: torch.Tensor, features: torch.Tensor, scale: torch.Tensor, *, enabled: bool
) -> torch.Tensor:
    if (
        features.shape != (*hidden.shape[:2], 17)
        or scale.shape != (17,)
        or not torch.isfinite(scale).all()
        or (scale <= 0).any()
        or type(enabled) is not bool
    ):
        raise ValueError(
            "aligned response features, fixed scale and explicit arm required"
        )
    extra = (
        (features.double() / scale).clamp(-10, 10)
        if enabled
        else torch.zeros_like(features, dtype=torch.float64)
    )
    return torch.cat((hidden.double(), extra), -1)

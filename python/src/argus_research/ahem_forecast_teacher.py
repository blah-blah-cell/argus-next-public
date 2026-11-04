"""Training-only finite-attention-shift teachers; inference remains target-free."""

from __future__ import annotations

import math

import torch
from torch import nn

from .ahem_score_span import ScoreSpanReadout
from .ahem_transition_reranker import ExactSqrt
from .flow_v6_objective import dynamic_loss


def inputs(cache: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(
        cache[k]
        for k in (
            "features",
            "scores",
            "valid",
            "successors",
            "current",
            "context",
            "ordinal",
            "startup_parent",
        )
    )


def cost(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        dynamic_loss(prediction, target, reduction="none")
        + 0.1 * (prediction[:, 0].expm1() - target[:, 0].expm1()).abs() / 1500
    )


def eligible(cache: dict[str, torch.Tensor]) -> torch.Tensor:
    return (cache["ordinal"] >= 127) & (cache["valid"].sum(1) >= 2)


def from_weights(
    head: nn.Module, cache: dict[str, torch.Tensor], weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    valid, current = cache["valid"], cache["current"]
    if (
        weights.shape != valid.shape
        or not torch.isfinite(weights).all()
        or (weights < 0).any()
        or (weights[~valid] != 0).any()
    ):
        raise ValueError("finite nonnegative aligned valid weights required")
    available = valid.any(1, keepdim=True)
    if not torch.allclose(
        weights.sum(1, keepdim=True), available.to(weights.dtype), atol=1e-6, rtol=1e-5
    ):
        raise ValueError("attention mass does not match availability")
    raw = torch.where(
        available, (weights[..., None] * cache["successors"]).sum(1), current
    )
    variance = (weights[..., None] * (cache["successors"] - raw[:, None]).square()).sum(
        1
    )
    entropy = -(weights * weights.clamp_min(1e-8).log()).sum(1, keepdim=True)
    evidence = valid.sum(1, keepdim=True).to(current.dtype) / 16
    features = torch.cat(
        (
            current,
            raw,
            raw - current,
            ExactSqrt.apply(variance.clamp_min(0)),
            entropy,
            evidence,
        ),
        -1,
    )
    prediction = current + head.trust(features, cache["context"], available) * (
        raw - current
    )
    return torch.where(
        (cache["ordinal"] < 127)[:, None], cache["startup_parent"], prediction
    ), raw


@torch.no_grad()
def advantages(
    model: ScoreSpanReadout,
    cache: dict[str, torch.Tensor],
    *,
    forecast: bool,
    shift: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 < shift < 1:
        raise ValueError("interior finite attention shift required")
    prediction, weights, _ = model(*inputs(cache))
    replay, raw = from_weights(model.head, cache, weights)
    if not torch.equal(prediction, replay):
        raise ValueError("counterfactual forecast helper disagrees with original")
    base_cost = cost(prediction if forecast else raw, cache["target"])
    advantage = torch.zeros_like(weights)
    for slot in range(weights.shape[1]):
        # Only perturb rows where this candidate is valid. Others keep their bank.
        changed = weights.clone()
        active = cache["valid"][:, slot]
        changed[active] *= 1 - shift
        changed[active, slot] += shift
        counter_prediction, counter_raw = from_weights(model.head, cache, changed)
        counter_cost = cost(
            counter_prediction if forecast else counter_raw, cache["target"]
        )
        advantage[:, slot] = torch.where(active, (base_cost - counter_cost) / shift, 0)
    if not torch.isfinite(advantage).all():
        raise ValueError("nonfinite teacher cost differences")
    return weights.detach(), advantage.detach()


def distribution(
    base: torch.Tensor,
    advantage: torch.Tensor,
    valid: torch.Tensor,
    *,
    scale: float,
    exponent_clip: float = 4.0,
) -> torch.Tensor:
    if (
        base.shape != advantage.shape
        or base.shape != valid.shape
        or valid.dtype != torch.bool
        or not math.isfinite(scale)
        or scale <= 0
        or not math.isfinite(exponent_clip)
        or exponent_clip <= 0
    ):
        raise ValueError("aligned teacher and finite positive training scale required")
    if (
        not torch.isfinite(base).all()
        or not torch.isfinite(advantage).all()
        or (base < 0).any()
        or (base[~valid] != 0).any()
    ):
        raise ValueError("invalid teacher bank")
    if not torch.allclose(
        base.sum(1), valid.any(1).to(base.dtype), atol=1e-6, rtol=1e-5
    ):
        raise ValueError("invalid teacher base mass")
    unnormalized = base * (advantage / scale).clamp(-exponent_clip, exponent_clip).exp()
    teacher = unnormalized / unnormalized.sum(1, keepdim=True).clamp_min(1e-8)
    unchanged = (advantage.masked_fill(~valid, 0) == 0).all(1)
    return torch.where(unchanged[:, None], base, teacher).detach()


def distillation(
    weights: torch.Tensor, teacher: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    if (
        weights.shape != teacher.shape
        or rows.shape != weights.shape[:1]
        or rows.dtype != torch.bool
        or teacher.requires_grad
    ):
        raise ValueError("aligned detached teacher and explicit query mask required")
    if not rows.any():
        return weights.sum() * 0
    positive = teacher > 0
    point = torch.where(
        positive,
        teacher * (teacher.clamp_min(1e-30).log() - weights.clamp_min(1e-30).log()),
        0,
    ).sum(1)
    return point[rows].mean()

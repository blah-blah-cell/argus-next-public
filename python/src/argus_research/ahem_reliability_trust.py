"""Observed within-bank chronological skill, not future-target gate supervision.

Every backtest source state can use only older completed entries. Recomputing
inside the current phase-valid bank preserves its raw-information horizon.
It does not reproduce the larger bank that existed at a historical source.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def chronological_skill(
    keys: torch.Tensor,
    outcomes: torch.Tensor,
    source_current: torch.Tensor,
    source_ordinals: torch.Tensor,
    successor_ordinals: torch.Tensor,
    valid: torch.Tensor,
    query_weights: torch.Tensor,
    *,
    semantic_scale: torch.Tensor,
    recency_scale: torch.Tensor,
    beta: torch.Tensor,
    horizon: int = 10_000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Six causal features plus per-entry historical predictions/support.

    Keys are already normalized shared state+phase encodings. All outcomes
    belong to physically completed transitions, never the query's next target.
    Caller supplies the audited same-flow, phase-footprint-valid bank.
    """
    n, k, _ = keys.shape
    if (
        not n
        or not k
        or outcomes.shape != (n, k, 2)
        or source_current.shape != outcomes.shape
        or source_ordinals.shape != (n, k)
        or successor_ordinals.shape != (n, k)
        or valid.shape != (n, k)
        or valid.dtype != torch.bool
        or query_weights.shape != (n, k)
        or horizon <= 0
    ):
        raise ValueError("aligned causal skill bank required")
    if (
        any(
            not torch.isfinite(t).all()
            for t in (keys, outcomes, source_current, query_weights)
        )
        or (query_weights < 0).any()
    ):
        raise ValueError("finite nonnegative skill inputs required")
    if (query_weights[~valid] != 0).any() or not torch.allclose(
        query_weights.sum(1), valid.any(1).to(keys.dtype), atol=1e-6, rtol=1e-5
    ):
        raise ValueError("query mass must match bank availability")
    if (
        ((source_ordinals < 0) & valid).any()
        or ((successor_ordinals <= source_ordinals) & valid).any()
        or any(
            t.numel() != 1 or not torch.isfinite(t).all() or (t < 0).any()
            for t in (semantic_scale, recency_scale, beta)
        )
    ):
        raise ValueError("causal ordinals and finite scalar scales required")
    # i is a backtested source state, j a strictly older completed entry.
    mask = (
        valid[:, :, None]
        & valid[:, None, :]
        & (source_ordinals[:, None, :] < source_ordinals[:, :, None])
        & (successor_ordinals[:, None, :] <= source_ordinals[:, :, None])
    )
    age = (source_ordinals[:, :, None] - source_ordinals[:, None, :]).clamp_min(0)
    age = torch.log1p(age.to(keys.dtype)) / keys.new_tensor(float(horizon)).log1p()
    scores = beta * (
        semantic_scale * torch.bmm(keys, keys.transpose(1, 2))
        - recency_scale * age.clamp(max=1)
    )
    attention = scores.masked_fill(~mask, -1e4).softmax(-1) * mask.to(keys.dtype)
    attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
    supported = mask.any(-1)
    historical_read = torch.bmm(attention, outcomes)
    historical_read = torch.where(supported[..., None], historical_read, source_current)
    error = outcomes - historical_read
    advantage = (outcomes - source_current).abs() - error.abs()
    mass = query_weights * supported.to(keys.dtype)
    total = mass.sum(-1, keepdim=True)
    normalized = mass / total.clamp_min(1e-8)
    features = torch.cat(
        (
            (normalized[..., None] * advantage).sum(1),
            (normalized[..., None] * error).sum(1),
            total,
            supported.sum(-1, keepdim=True).to(keys.dtype) / 16,
        ),
        -1,
    )
    return features, historical_read, supported


class ReliabilityTrust(nn.Module):
    """Frozen parent + 14-parameter reliability or 2-parameter bias control.

    No target argument. Short-history and unsupported cases bypass the delta.
    Startup arithmetic exactly follows the parent reference operation order.
    """

    def __init__(
        self,
        head: nn.Module,
        startup_scale: torch.Tensor,
        feature_scale: torch.Tensor,
        *,
        reliability_enabled: bool,
    ) -> None:
        super().__init__()
        if (
            feature_scale.shape != (6,)
            or not torch.isfinite(feature_scale).all()
            or (feature_scale < 1e-6).any()
            or startup_scale.shape != (2,)
            or not torch.isfinite(startup_scale).all()
            or ((startup_scale < 0) | (startup_scale > 1)).any()
        ):
            raise ValueError("finite bounded startup and training RMS required")
        self.head = copy.deepcopy(head).requires_grad_(False)
        self.reliability_enabled = reliability_enabled
        self.register_buffer("startup_scale", startup_scale.detach().clone())
        self.register_buffer("feature_scale", feature_scale.detach().clone())
        self.bias = nn.Parameter(torch.zeros(2))
        self.weight = nn.Parameter(torch.zeros(2, 6), requires_grad=reliability_enabled)

    def forward(
        self,
        features: torch.Tensor,
        context: torch.Tensor,
        current: torch.Tensor,
        raw: torch.Tensor,
        available: torch.Tensor,
        ordinal: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        n = len(current)
        if (
            features.shape != (n, 10)
            or context.shape != (n, len(self.head.context_mean))
            or current.shape != (n, 2)
            or raw.shape != current.shape
            or available.shape != (n, 1)
            or available.dtype != torch.bool
            or ordinal.shape != (n,)
            or ordinal.dtype not in (torch.int32, torch.int64)
            or reliability.shape != (n, 6)
            or (ordinal < 0).any()
            or any(
                not torch.isfinite(t).all()
                for t in (features, context, current, raw, reliability)
            )
        ):
            raise ValueError("aligned finite trust inputs required")
        logits = self.head.network(features)
        if self.head.context_enabled:
            z = ((context - self.head.context_mean) / self.head.context_scale).clamp(
                -10, 10
            )
            logits = logits + self.head.context_delta(z)
        delta = self.bias.expand(n, -1)
        if self.reliability_enabled:
            z = (reliability / self.feature_scale).clamp(-10, 10)
            delta = delta + F.linear(z, self.weight)
        eligible = (ordinal >= 127)[:, None] & (reliability[:, 4:5] > 0)
        logits = logits + delta.tanh() * eligible.to(delta.dtype)
        coefficient = logits.sigmoid() * available.to(logits.dtype)
        result = current + coefficient * (raw - current)
        corrected = result + (1 - self.startup_scale) * (current - result)
        return torch.where((ordinal < 127)[:, None], corrected, result)


def predict(model: ReliabilityTrust, cache: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        *(
            cache[k]
            for k in (
                "features",
                "context",
                "current",
                "raw",
                "available",
                "ordinal",
                "reliability",
            )
        )
    )

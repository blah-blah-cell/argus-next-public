"""Derivative-matched causal score-range adaptation; research only."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from .ahem_transition_reranker import ExactSqrt


def score_inputs(
    weights: torch.Tensor, valid: torch.Tensor, beta: float
) -> torch.Tensor:
    if (
        weights.shape != valid.shape
        or valid.dtype != torch.bool
        or beta <= 0
        or not math.isfinite(beta)
    ):
        raise ValueError("aligned weights and finite positive beta required")
    if (
        not torch.isfinite(weights).all()
        or (weights[valid] <= 0).any()
        or (weights[~valid] != 0).any()
    ):
        raise ValueError("underflow/nonfinite original scores cannot be reconstructed")
    if not torch.allclose(
        weights.sum(-1), valid.any(-1).to(weights.dtype), atol=1e-6, rtol=1e-5
    ):
        raise ValueError("original probability mass does not match availability")
    safe = torch.where(valid, weights, torch.ones_like(weights))
    scores = safe.log() / beta
    peak = scores.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True)
    peak = torch.where(valid.any(-1, keepdim=True), peak, torch.zeros_like(peak))
    return torch.where(valid, scores - peak, torch.zeros_like(scores))


def score_cap(scores: torch.Tensor, valid: torch.Tensor, enabled: bool) -> torch.Tensor:
    maximum = scores.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True)
    minimum = scores.masked_fill(~valid, torch.inf).amin(-1, keepdim=True)
    span = torch.where(
        valid.any(-1, keepdim=True), maximum - minimum, torch.zeros_like(maximum)
    )
    return span.clamp_min(0.25) if enabled else torch.full_like(span, 0.25)


def causal_features(
    scores: torch.Tensor,
    ordinals: torch.Tensor,
    source: torch.Tensor,
    source_current: torch.Tensor,
    successors: torch.Tensor,
    current: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    scale = current.new_tensor((math.log1p(65535), math.log1p(60)))
    difference = (current[:, None] - source_current) / scale
    change = (successors - source_current) / scale
    trend = ((current - context[:, 14:16]) * context[:, 26:27])[:, None].expand_as(
        successors
    ) / scale
    relative = change - trend
    outcome = (successors - current[:, None]) / scale
    age = torch.log1p(
        (ordinals[:, None] - source).clamp_min(0).to(current.dtype)
    ) / math.log1p(10000)
    return torch.cat(
        (
            scores[..., None] / 8,
            age.clamp(max=1)[..., None],
            difference,
            difference.abs(),
            change,
            trend,
            relative,
            relative.abs(),
            outcome,
            outcome.abs(),
        ),
        -1,
    ).clamp(-4, 4)


class ScoreSpanReadout(nn.Module):
    def __init__(self, head: nn.Module, beta: float, *, span_enabled: bool) -> None:
        super().__init__()
        if beta <= 0 or not math.isfinite(beta):
            raise ValueError("positive finite frozen beta required")
        self.head = copy.deepcopy(head).requires_grad_(False).eval()
        self.beta = beta
        self.span_enabled = span_enabled
        self.reranker = nn.Sequential(nn.Linear(18, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.reranker[-1].weight)
        nn.init.zeros_(self.reranker[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        scores: torch.Tensor,
        valid: torch.Tensor,
        successors: torch.Tensor,
        current: torch.Tensor,
        context: torch.Tensor,
        ordinals: torch.Tensor,
        startup_parent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, k = scores.shape
        if (
            features.shape != (n, k, 18)
            or valid.shape != scores.shape
            or valid.dtype != torch.bool
            or successors.shape != (n, k, 2)
            or current.shape != (n, 2)
            or context.shape != (n, len(self.head.context_mean))
            or ordinals.shape != (n,)
            or ordinals.dtype not in (torch.int32, torch.int64)
            or (ordinals < 0).any()
            or startup_parent.shape != current.shape
        ):
            raise ValueError("aligned causal reranker inputs required")
        if any(
            not torch.isfinite(t).all()
            for t in (features, scores, successors, current, context, startup_parent)
        ):
            raise ValueError("nonfinite causal inputs")
        cap = score_cap(scores, valid, self.span_enabled)
        h = self.reranker(features).squeeze(-1)
        delta = cap * torch.tanh(h / cap)
        logits = (self.beta * (scores + delta)).masked_fill(~valid, -1e4)
        weights = logits.softmax(-1) * valid.to(scores.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        available = valid.any(-1, keepdim=True)
        raw = torch.where(available, (weights[..., None] * successors).sum(1), current)
        variance = (weights[..., None] * (successors - raw[:, None]).square()).sum(1)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1, keepdim=True)
        evidence = valid.sum(-1, keepdim=True).to(current.dtype) / 16
        gate_features = torch.cat(
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
        coefficient = self.head.trust(gate_features, context, available)
        prediction = current + coefficient * (raw - current)
        prediction = torch.where((ordinals < 127)[:, None], startup_parent, prediction)
        return prediction, weights, delta


def predict(model: ScoreSpanReadout, cache: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        *(
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
    )[0]

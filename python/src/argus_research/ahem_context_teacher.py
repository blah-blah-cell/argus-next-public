"""Causal query context for a teacher-trained reranker, with a masked control."""

from __future__ import annotations

import torch
from torch import nn

from .ahem_forecast_teacher import from_weights
from .ahem_score_span import ScoreSpanReadout, score_cap


class ContextTeacherReadout(ScoreSpanReadout):
    def __init__(self, head: nn.Module, beta: float, *, context_enabled: bool) -> None:
        if type(context_enabled) is not bool:
            raise ValueError("explicit context comparison arm required")
        super().__init__(head, beta, span_enabled=True)
        self.context_enabled = context_enabled
        original = self.reranker[0]
        expanded = nn.Linear(18 + len(self.head.context_mean), 16)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :18].copy_(original.weight)
            expanded.bias.copy_(original.bias)
        self.reranker = nn.Sequential(expanded, nn.GELU(), self.reranker[-1])

    def ranking_features(
        self, features: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        normalized = (
            (context - self.head.context_mean) / self.head.context_scale
        ).clamp(-10, 10)
        if not self.context_enabled:
            normalized = torch.zeros_like(normalized)
        repeated = normalized[:, None].expand(-1, features.shape[1], -1)
        return torch.cat((features, repeated), -1)

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
        if scores.ndim != 2:
            raise ValueError("matrix scores required")
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
            raise ValueError("aligned causal context-reranker inputs required")
        if any(
            not torch.isfinite(value).all()
            for value in (
                features,
                scores,
                successors,
                current,
                context,
                startup_parent,
            )
        ):
            raise ValueError("nonfinite causal inputs")
        cap = score_cap(scores, valid, True)
        hidden = self.reranker(self.ranking_features(features, context)).squeeze(-1)
        delta = cap * torch.tanh(hidden / cap)
        weights = (self.beta * (scores + delta)).masked_fill(~valid, -1e4).softmax(
            1
        ) * valid.to(scores.dtype)
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        cache = {
            "valid": valid,
            "successors": successors,
            "current": current,
            "context": context,
            "ordinal": ordinals,
            "startup_parent": startup_parent,
        }
        prediction, _ = from_weights(self.head, cache, weights)
        return prediction, weights, delta

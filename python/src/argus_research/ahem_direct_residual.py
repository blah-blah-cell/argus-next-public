"""Exact-identity native-bounded log residual using causal retained-bank summaries."""

from __future__ import annotations

import math

import torch
from torch import nn


@torch.no_grad()
def features(
    head: nn.Module, cache: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    valid = cache["valid"]
    weights = (2.5 * cache["scores"]).masked_fill(~valid, -1e4).softmax(1) * valid
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
    raw = torch.where(
        valid.any(1, keepdim=True),
        (weights[..., None] * cache["successors"]).sum(1),
        cache["current"],
    )
    bounds = cache["reference"].new_tensor((math.log1p(65535), math.log1p(60)))
    parent = cache["reference"]
    if not torch.isfinite(parent).all() or ((parent < 0) | (parent > bounds)).any():
        raise ValueError("bounded original forecast required")
    margin = torch.minimum(
        torch.minimum(parent, bounds - parent), torch.ones_like(parent)
    )
    z = ((cache["context"] - head.context_mean) / head.context_scale).clamp(-10, 10)
    x = torch.cat(
        (
            (weights[..., None] * cache["features"]).sum(1),
            z,
            cache["current"] / bounds,
            parent / bounds,
            raw / bounds,
        ),
        1,
    )
    return {
        "features": x,
        "parent": parent,
        "margin": margin,
        "active": (cache["ordinal"] >= 127) & valid.any(1),
    }


class DirectResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(164, 16), nn.GELU(), nn.Linear(16, 2))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        parent: torch.Tensor,
        margin: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        if (
            features.shape != (len(parent), 164)
            or parent.shape != margin.shape
            or parent.shape != (len(parent), 2)
            or active.shape != (len(parent),)
            or active.dtype != torch.bool
            or not all(torch.isfinite(v).all() for v in (features, parent, margin))
            or (margin < 0).any()
        ):
            raise ValueError("finite aligned causal residual inputs required")
        correction = 0.25 * self.network(features).tanh() * margin
        return torch.where(active[:, None], parent + correction, parent)


def apply(model: DirectResidual, data: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(*(data[k] for k in ("features", "parent", "margin", "active")))

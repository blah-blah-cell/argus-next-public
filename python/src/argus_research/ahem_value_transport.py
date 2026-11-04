"""A fixed causal log-delta value transform; no training or forecast targets."""

from __future__ import annotations

import math

import torch


def transport(
    current: torch.Tensor,
    source_current: torch.Tensor,
    successors: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        current.ndim != 2
        or current.shape[1] != 2
        or successors.ndim != 3
        or successors.shape != source_current.shape
        or successors.shape != (*valid.shape, 2)
        or len(current) != len(successors)
        or valid.dtype != torch.bool
        or any(
            not torch.isfinite(v).all() for v in (current, source_current, successors)
        )
    ):
        raise ValueError("finite aligned causal dynamic fields required")
    bounds = current.new_tensor((math.log1p(65535), math.log1p(60)))
    if any(
        ((v < 0) | (v > bounds + 1e-6)).any()
        for v in (current, source_current, successors)
    ):
        raise ValueError("observed fields violate physical bounds")
    unprojected = current[:, None] + successors - source_current
    clipped = ((unprojected < 0) | (unprojected > bounds)) & valid[..., None]
    transformed = torch.maximum(
        torch.minimum(unprojected, bounds), torch.zeros_like(unprojected)
    )
    transformed = torch.where(
        valid[..., None], transformed, torch.zeros_like(transformed)
    )
    return transformed, clipped

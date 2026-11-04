"""Causal baseline-inclusive full-forecast actions and a shared policy."""

from __future__ import annotations

import math

import torch
from torch import nn

from .ahem_forecast_teacher import from_weights


@torch.no_grad()
def actions(head: nn.Module, cache: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    valid = cache["valid"]
    weights = (2.5 * cache["scores"]).masked_fill(~valid, -1e4).softmax(1) * valid
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
    baseline = cache["reference"]
    predictions = [baseline]
    for slot in range(valid.shape[1]):
        changed = torch.zeros_like(weights)
        changed[:, slot] = valid[:, slot].to(weights.dtype)
        changed = torch.where(valid[:, slot, None], changed, weights)
        predictions.append(from_weights(head, cache, changed)[0])
    prediction = torch.stack(predictions, 1)
    mask = torch.cat((torch.ones(len(valid), 1, dtype=torch.bool), valid), 1)
    active = (cache["ordinal"] >= 127) & valid.any(1)
    mask[:, 1:] &= active[:, None]
    baseline_features = (weights[..., None] * cache["features"]).sum(1, keepdim=True)
    slot_features = torch.cat((baseline_features, cache["features"]), 1)
    context = ((cache["context"] - head.context_mean) / head.context_scale).clamp(
        -10, 10
    )
    indicator = torch.zeros(len(valid), 17, 1)
    indicator[:, 0] = 1
    bounds = prediction.new_tensor((math.log1p(65535), math.log1p(60)))
    features = torch.cat(
        (
            slot_features,
            context[:, None].expand(-1, 17, -1),
            indicator,
            prediction / bounds,
            (prediction - baseline[:, None]) / bounds,
        ),
        2,
    )
    return {
        "features": features,
        "prediction": prediction,
        "valid": mask,
        "active": active,
        "baseline": baseline,
    }


class ForecastActionPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(163, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if (
            features.ndim != 3
            or features.shape[1:] != (17, 163)
            or valid.shape != features.shape[:2]
            or valid.dtype != torch.bool
            or not valid[:, 0].all()
            or not torch.isfinite(features).all()
        ):
            raise ValueError("finite causal17action inputs with baseline required")
        logits = self.network(features).squeeze(-1)
        logits = logits + features.new_tensor([4.0] + [0.0] * 16)
        return logits.masked_fill(~valid, -torch.inf).softmax(1)


def apply(
    model: ForecastActionPolicy, data: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    probability = model(data["features"], data["valid"])
    prediction = (probability[..., None] * data["prediction"]).sum(1)
    prediction = torch.where(data["active"][:, None], prediction, data["baseline"])
    return prediction, probability

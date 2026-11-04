"""Hindsight-only reachability diagnostics. Never an online predictor."""

from __future__ import annotations

import torch
from torch import nn

from .ahem_forecast_teacher import from_weights
from .flow_v6_objective import dynamic_loss


@torch.no_grad()
def diagnose(
    head: nn.Module, cache: dict[str, torch.Tensor], beta: float = 2.5
) -> dict:
    valid, current, target = cache["valid"], cache["current"], cache["target"]
    base = (beta * cache["scores"]).masked_fill(~valid, -1e4).softmax(1) * valid
    base = base / base.sum(1, keepdim=True).clamp_min(1e-8)
    _, raw = from_weights(head, cache, base)
    reference = cache["reference"]
    active = (cache["ordinal"] >= 127) & valid.any(1)
    free = torch.maximum(
        torch.minimum(target, torch.maximum(current, raw)), torch.minimum(current, raw)
    )
    free = torch.where(active[:, None], free, reference)
    low = torch.minimum(
        current, cache["successors"].masked_fill(~valid[..., None], torch.inf).amin(1)
    )
    high = torch.maximum(
        current, cache["successors"].masked_fill(~valid[..., None], -torch.inf).amax(1)
    )
    box = torch.maximum(torch.minimum(target, high), low)
    box = torch.where(active[:, None], box, reference)
    best = reference.clone()
    best_loss = dynamic_loss(best, target, reduction="none")
    selected = torch.full((len(current),), -1, dtype=torch.long)
    vertex_predictions = []
    for slot in range(valid.shape[1]):
        weights = torch.zeros_like(base)
        weights[:, slot] = valid[:, slot].to(base.dtype)
        # Invalid rows still require supported mass for the complete head API.
        weights = torch.where(valid[:, slot, None], weights, base)
        prediction, _ = from_weights(head, cache, weights)
        loss = dynamic_loss(prediction, target, reduction="none")
        improve = active & valid[:, slot] & (loss < best_loss)
        best = torch.where(improve[:, None], prediction, best)
        best_loss = torch.where(improve, loss, best_loss)
        selected = torch.where(improve, slot, selected)
        vertex_predictions.append(prediction)
    return {
        "retained_fifo": reference,
        "best_tested_vertex_or_baseline": best,
        "fixed_raw_free_gate_oracle": free,
        "bank_box_relaxation": box,
        "raw": raw,
        "active": active,
        "selected_slot": selected,
        "vertex_predictions": torch.stack(vertex_predictions, 1),
    }

"""One fixed byte-auxiliary removal; otherwise identical bounded fitting."""

from __future__ import annotations

import copy

import torch
from torch import nn

from .ahem_fifo_teacher import centered_attention
from .ahem_forecast_teacher import eligible, from_weights
from .ahem_objective_alignment import BoundedFit
from .flow_v6_objective import dynamic_loss


def objective(
    theta: torch.Tensor, features: torch.Tensor, head: nn.Module, cache: dict
) -> torch.Tensor:
    weights, _ = centered_attention(
        cache["scores"], features, cache["valid"], theta, 2.5
    )
    prediction, _ = from_weights(head, cache, weights)
    rows = eligible(cache)
    if not rows.any():
        raise ValueError("eligible fitting coordinates required")
    return (
        dynamic_loss(prediction, cache["target"], reduction="none")[rows].mean()
        + 0.5e-6 * theta.square().sum()
    )


def fit(
    features: torch.Tensor,
    head: nn.Module,
    cache: dict,
    *,
    max_iter: int = 80,
    max_eval: int = 120,
) -> BoundedFit:
    if (
        features.shape != (*cache["scores"].shape, 33)
        or not torch.isfinite(features).all()
    ):
        raise ValueError("finite fixed33columnfeatures required")
    frozen = copy.deepcopy(head).requires_grad_(False).eval()
    theta = nn.Parameter(torch.zeros(33, dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [theta],
        lr=1,
        max_iter=max_iter,
        max_eval=max_eval,
        tolerance_grad=1e-7,
        tolerance_change=1e-12,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    trace = []

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        value = objective(theta, features, frozen, cache)
        if not torch.isfinite(value):
            raise ValueError("nonfinite primary objective")
        value.backward()
        if theta.grad is None or not torch.isfinite(theta.grad).all():
            raise ValueError("nonfinite primary gradient")
        trace.append(float(value.detach()))
        return value

    initial = float(objective(theta, features, frozen, cache).detach())
    optimizer.step(closure)
    final = objective(theta, features, frozen, cache)
    gradient = torch.autograd.grad(final, theta)[0]
    return BoundedFit(
        theta.detach(),
        initial,
        float(final.detach()),
        float(gradient.abs().max()),
        int(optimizer.state[theta]["n_iter"]),
        len(trace),
        trace,
    )

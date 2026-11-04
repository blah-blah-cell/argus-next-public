"""Paired bounded readout objectives with identical full-batch optimization."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from .ahem_fifo_teacher import centered_attention
from .ahem_forecast_teacher import cost, distillation, eligible, from_weights


@dataclass
class BoundedFit:
    theta: torch.Tensor
    initial_objective: float
    final_objective: float
    gradient_inf: float
    iterations: int
    closure_evaluations: int
    trace: list[float]


def objective(
    theta: torch.Tensor,
    features: torch.Tensor,
    head: nn.Module,
    cache: dict,
    teacher: torch.Tensor,
    arm: str,
) -> torch.Tensor:
    weights, _ = centered_attention(
        cache["scores"], features, cache["valid"], theta, 2.5
    )
    rows = eligible(cache)
    if not rows.any() or arm not in ("bounded_kl", "forecast_cost"):
        raise ValueError("explicit objective and eligible fitting queries required")
    if arm == "bounded_kl":
        value = distillation(weights, teacher, rows)
    else:
        prediction, _ = from_weights(head, cache, weights)
        value = cost(prediction, cache["target"])[rows].mean()
    return value + 0.5e-6 * theta.square().sum()


def fit(
    features: torch.Tensor,
    head: nn.Module,
    cache: dict,
    teacher: torch.Tensor,
    arm: str,
    *,
    max_iter: int = 80,
    max_eval: int = 120,
) -> BoundedFit:
    if (
        features.shape != (*cache["scores"].shape, 33)
        or not torch.isfinite(features).all()
        or max_iter < 1
        or max_eval < 1
    ):
        raise ValueError(
            "fixed finite33feature design and bounded optimizer budget required"
        )
    frozen_head = copy.deepcopy(head).requires_grad_(False).eval()
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
        value = objective(theta, features, frozen_head, cache, teacher, arm)
        if not torch.isfinite(value):
            raise ValueError("nonfinite bounded objective")
        value.backward()
        if theta.grad is None or not torch.isfinite(theta.grad).all():
            raise ValueError("nonfinite bounded objective gradient")
        trace.append(float(value.detach()))
        return value

    initial = float(
        objective(theta, features, frozen_head, cache, teacher, arm).detach()
    )
    optimizer.step(closure)
    final = objective(theta, features, frozen_head, cache, teacher, arm)
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

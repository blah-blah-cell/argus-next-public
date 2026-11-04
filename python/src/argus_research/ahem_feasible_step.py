"""Small exact-active-set projection and finite-step checks, not an optimizer claim."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Callable

import torch


@dataclass
class Projection:
    displacement: torch.Tensor
    active: list[int]
    multipliers: torch.Tensor


def project(
    proposal: torch.Tensor, matrix: torch.Tensor, slack: torch.Tensor
) -> Projection:
    if (
        proposal.ndim != 1
        or matrix.ndim != 2
        or matrix.shape[1] != len(proposal)
        or slack.shape != matrix.shape[:1]
        or len(slack) > 6
        or any(not torch.isfinite(v).all() for v in (proposal, matrix, slack))
    ):
        raise ValueError("finite aligned projection with at most6constraints required")
    v, a, b = proposal.double(), matrix.double(), slack.double()
    if (b < 0).any():
        raise ValueError("zero must remain feasible")
    best = None
    distance = torch.inf
    for size in range(len(b) + 1):
        for subset in itertools.combinations(range(len(b)), size):
            if subset:
                selected = a[list(subset)]
                lam = torch.linalg.pinv(selected @ selected.T, rtol=1e-12) @ (
                    selected @ v - b[list(subset)]
                )
                d = v - selected.T @ lam
                if (lam < -1e-9).any() or (
                    (selected @ d - b[list(subset)]).abs() > 1e-8
                ).any():
                    continue
            else:
                lam, d = torch.empty(0, dtype=torch.float64), v.clone()
            if (a @ d > b + 1e-9).any():
                continue
            value = float((d - v).square().sum())
            if value < distance:
                distance = value
                best = Projection(d, list(subset), lam)
    if best is None:
        raise ValueError("no numerically certified projection")
    return best


def backtrack(
    objective: float,
    directional: float,
    evaluate: Callable[[float], tuple[float, bool]],
) -> tuple[float, list[dict]]:
    trials = []
    if directional >= 0 or not torch.isfinite(torch.tensor(directional)):
        return 0.0, trials
    for attempt in range(12):
        alpha = 0.5**attempt
        value, feasible = evaluate(alpha)
        accepted = (
            bool(torch.isfinite(torch.tensor(value)))
            and feasible
            and value <= objective + 1e-4 * alpha * directional
        )
        trials.append(
            {
                "alpha": alpha,
                "objective": value,
                "feasible": feasible,
                "accepted": accepted,
            }
        )
        if accepted:
            return alpha, trials
    return 0.0, trials

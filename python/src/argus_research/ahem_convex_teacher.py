"""Strictly convex frozen-feature teacher probe, distinct from bounded student."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .ahem_score_span import score_cap


@dataclass
class ConvexFit:
    theta: torch.Tensor
    trace: list[dict[str, float]]
    converged: bool
    gradient_inf: float


def attention(
    scores: torch.Tensor,
    hidden: torch.Tensor,
    valid: torch.Tensor,
    theta: torch.Tensor,
    *,
    beta: float,
    bounded: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Original uncentered hidden features: centering is solver conditioning only."""
    correction = hidden.double() @ theta.double()
    if bounded:
        cap = score_cap(scores, valid, True).double()
        correction = cap * torch.tanh(correction / cap)
    weights = (beta * (scores.double() + correction)).masked_fill(~valid, -1e4).softmax(
        1
    ) * valid
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-30)
    return weights.to(scores.dtype), correction


def prepare(
    scores: torch.Tensor,
    hidden: torch.Tensor,
    valid: torch.Tensor,
    teacher: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        scores.ndim != 2
        or hidden.ndim != 3
        or hidden.shape[:2] != scores.shape
        or valid.shape != scores.shape
        or valid.dtype != torch.bool
        or teacher.shape != scores.shape
        or not len(scores)
        or hidden.shape[-1] < 1
        or not math.isfinite(beta)
        or beta <= 0
    ):
        raise ValueError(
            "aligned nonempty bank features and positive finite beta required"
        )
    if (
        any(not torch.isfinite(value).all() for value in (scores, hidden, teacher))
        or (teacher < 0).any()
        or (teacher[~valid] != 0).any()
        or not valid.any(1).all()
        or not torch.allclose(
            teacher.sum(1), torch.ones_like(teacher[:, 0]), atol=1e-6, rtol=1e-5
        )
    ):
        raise ValueError("finite supported probability teacher required")
    s, h, q = scores.double(), hidden.double(), teacher.double()
    q = q / q.sum(1, keepdim=True)
    base = (beta * s).masked_fill(~valid, -torch.inf).softmax(1)
    centered = h - (base[..., None] * h).sum(1, keepdim=True)
    return s, centered, q


def geometry(
    theta: torch.Tensor,
    scores: torch.Tensor,
    hidden: torch.Tensor,
    valid: torch.Tensor,
    teacher: torch.Tensor,
    *,
    beta: float,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic KL+ridge value, gradient and Hessian for prepared float64 tensors."""
    logits = (beta * (scores + hidden @ theta)).masked_fill(~valid, -torch.inf)
    log_p = logits.log_softmax(1).masked_fill(~valid, 0)
    p = logits.softmax(1)
    entropy_term = teacher * teacher.clamp_min(1e-300).log()
    objective = (entropy_term - teacher * log_p).sum(
        1
    ).mean() + ridge * theta.square().sum() / 2
    gradient = beta * ((p - teacher)[..., None] * hidden).sum(1).mean(0) + ridge * theta
    mean_h = (p[..., None] * hidden).sum(1)
    centered = hidden - mean_h[:, None]
    hessian = beta**2 * torch.einsum("nk,nkd,nke->de", p, centered, centered) / len(
        scores
    ) + ridge * torch.eye(len(theta), dtype=theta.dtype, device=theta.device)
    return objective, gradient, hessian


def fit(
    scores: torch.Tensor,
    hidden: torch.Tensor,
    valid: torch.Tensor,
    teacher: torch.Tensor,
    *,
    beta: float = 2.5,
    ridge: float = 1e-6,
    max_steps: int = 50,
    tolerance: float = 1e-8,
) -> ConvexFit:
    if (
        not math.isfinite(ridge)
        or ridge <= 0
        or type(max_steps) is not int
        or max_steps < 1
        or not math.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("positive ridge, tolerance and bounded step count required")
    scores, centered, teacher = prepare(scores, hidden, valid, teacher, beta)
    theta = torch.zeros(centered.shape[-1], dtype=torch.float64, device=scores.device)
    trace = []
    for iteration in range(max_steps + 1):
        objective, gradient, hessian = geometry(
            theta, scores, centered, valid, teacher, beta=beta, ridge=ridge
        )
        if any(
            not torch.isfinite(value).all() for value in (objective, gradient, hessian)
        ):
            raise ValueError("nonfinite convex geometry")
        norm = float(gradient.abs().max())
        trace.append(
            {"step": iteration, "objective": float(objective), "gradient_inf": norm}
        )
        if norm <= tolerance or iteration == max_steps:
            break
        direction = torch.cholesky_solve(
            gradient[:, None], torch.linalg.cholesky(hessian)
        ).squeeze(1)
        directional = float(gradient @ direction)
        accepted = False
        for backtrack in range(30):
            alpha = 0.5**backtrack
            candidate = theta - alpha * direction
            proposed = geometry(
                candidate, scores, centered, valid, teacher, beta=beta, ridge=ridge
            )[0]
            if proposed <= objective - 1e-4 * alpha * directional:
                theta, accepted = candidate, True
                trace[-1]["accepted_step_size"] = alpha
                break
        if not accepted:
            break
    objective, gradient, _ = geometry(
        theta, scores, centered, valid, teacher, beta=beta, ridge=ridge
    )
    norm = float(gradient.abs().max())
    return ConvexFit(theta.detach(), trace, norm <= tolerance, norm)

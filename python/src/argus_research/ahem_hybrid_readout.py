"""Small bank-matched scorer/gate adaptation; frozen encoders and old evidence."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .ahem_score_span import causal_features, score_inputs
from .ahem_transition_reranker import ExactSqrt
from .flow_v6_objective import dynamic_loss


INPUTS = (
    "features",
    "scores",
    "valid",
    "successors",
    "current",
    "context",
    "ordinal",
    "reference",
)


def make_data(
    common: dict, arm: dict, context: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Observed prefix and completed bank only; outcome targets are not inputs."""
    x, anchors = common["features"], common["anchors"]
    src, nxt, valid = (arm[k][anchors] for k in ("source", "successor", "valid"))
    now = anchors[:, None].expand_as(valid)
    flows = common["flow_ids"]
    if (
        valid.dtype != torch.bool
        or valid.shape != (len(anchors), 16)
        or context.shape != (len(x), 140)
        or ((src < 0) & valid).any()
        or ((nxt <= src) & valid).any()
        or ((nxt > now) & valid).any()
    ):
        raise ValueError("unaligned or future candidate bank")
    if (
        not torch.equal(common["previous"][nxt[valid]], src[valid])
        or not torch.equal(flows[src[valid]], flows[nxt[valid]])
        or not torch.equal(flows[src[valid]], flows[now[valid]])
    ):
        raise ValueError("candidate is not the completed same-flow successor")
    values = x[nxt.clamp_min(0), :2] * valid[..., None]
    source_current = x[src.clamp_min(0), :2] * valid[..., None]
    scores = score_inputs(arm["weights"][anchors], valid, 2.5)
    current, phase = x[anchors, :2], context[anchors]
    features = causal_features(
        scores, anchors, src, source_current, values, current, phase
    )
    return {
        "features": features * valid[..., None],
        "scores": scores,
        "valid": valid,
        "successors": values,
        "current": current,
        "context": phase,
        "ordinal": anchors,
        "reference": arm["prediction"][anchors],
    }


class BankReadout(nn.Module):
    """Identity-at-init within score reconstruction tolerance; bounded logit deltas."""

    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        if len(head.context_mean) != 140:
            raise ValueError("retained140-dimensional causal context required")
        self.head = copy.deepcopy(head).requires_grad_(False).eval()
        self.slot = nn.Linear(18, 16)
        self.query = nn.Linear(140, 16)
        self.score = nn.Linear(16, 1)
        self.gate = nn.Linear(150, 2)
        for module in (self.score, self.gate):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        self.register_buffer(
            "gate_scale",
            torch.tensor([math.log1p(65535), math.log1p(60)] * 4 + [math.log(16), 1.0]),
        )

    def train(self, mode: bool = True) -> Any:
        super().train(mode)
        self.head.eval()
        return self

    def forward(
        self, features, scores, valid, successors, current, context, ordinal, reference
    ):
        n = len(current)
        if (
            scores.shape != (n, 16)
            or valid.shape != scores.shape
            or valid.dtype != torch.bool
            or features.shape != (n, 16, 18)
            or successors.shape != (n, 16, 2)
            or current.shape != (n, 2)
            or context.shape != (n, 140)
            or ordinal.shape != (n,)
            or ordinal.dtype != torch.long
            or reference.shape != current.shape
            or (ordinal < 0).any()
            or any(
                not torch.isfinite(t).all()
                for t in (features, scores, successors, current, context, reference)
            )
        ):
            raise ValueError("finite aligned causal readout inputs required")
        z = ((context - self.head.context_mean) / self.head.context_scale).clamp(
            -10, 10
        )
        hidden = F.gelu(self.slot(features) + self.query(z)[:, None])
        delta = 2 * torch.tanh(self.score(hidden).squeeze(-1) / 2)
        weights = (2.5 * (scores + delta)).masked_fill(~valid, -1e4).softmax(1) * valid
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        available = valid.any(1, keepdim=True)
        raw = torch.where(available, (weights[..., None] * successors).sum(1), current)
        variance = (weights[..., None] * (successors - raw[:, None]).square()).sum(1)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(1, keepdim=True)
        evidence = valid.sum(1, keepdim=True).to(current.dtype) / 16
        gate_features = torch.cat(
            (
                current,
                raw,
                raw - current,
                ExactSqrt.apply(variance.clamp_min(0)),
                entropy,
                evidence,
            ),
            1,
        )
        gate_logits = self.head.network(gate_features)
        if self.head.context_enabled:
            gate_logits = gate_logits + self.head.context_delta(z)
        adjustment = 2 * torch.tanh(
            self.gate(torch.cat((gate_features / self.gate_scale, z), 1)) / 2
        )
        coefficient = (gate_logits + adjustment).sigmoid() * available.to(current.dtype)
        prediction = current + coefficient * (raw - current)
        prediction = torch.where((ordinal < 127)[:, None], reference, prediction)
        return prediction, weights


def predict(
    model: BankReadout, data: dict[str, torch.Tensor], index: torch.Tensor | None = None
):
    return model(*(data[k] if index is None else data[k][index] for k in INPUTS))


def objective(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    physical = (prediction[:, 0].expm1() - target[:, 0].expm1()) / 1500
    return dynamic_loss(prediction, target) + 0.1 * F.smooth_l1_loss(
        physical, torch.zeros_like(physical)
    )

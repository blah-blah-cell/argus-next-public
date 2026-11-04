"""AHEM-v6.3 frozen-backbone, proposal-aligned retrieval candidate.

This module is an executable research specification, not a trained result or
an Argus runtime artifact.  The v6.1 backbone is immutable by default.  Slot
zero is its exact forecast and every memory action is a bounded correction
around that forecast.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ahem_v6 import AhemV6StateTransitionNextPacket, rotate_observed_values
from .ahem_v6 import transition_slots
from .flow_v2_features import FEATURE_CONTRACT_SHA256
from .flow_v6_objective import dynamic_loss


def masked_sparsemax(logits: Tensor, valid: Tensor) -> Tensor:
    """Project valid logits onto the simplex, assigning invalid slots zero."""
    if logits.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError("sparsemax logits and validity mask are incompatible")
    if not valid.any(dim=-1).all():
        raise ValueError("sparsemax requires at least one valid action per row")
    masked = logits.masked_fill(~valid, -torch.inf)
    ordered = masked.sort(dim=-1, descending=True).values
    cumulative = ordered.masked_fill(~ordered.isfinite(), 0).cumsum(dim=-1)
    ranks = torch.arange(
        1, logits.shape[-1] + 1, device=logits.device, dtype=logits.dtype
    )
    support = (1 + ranks * ordered) > cumulative
    support_size = support.sum(-1, keepdim=True).clamp_min(1)
    threshold = (
        cumulative.gather(1, support_size - 1) - 1
    ) / support_size.to(logits.dtype)
    weights = (masked - threshold).clamp_min(0).masked_fill(~valid, 0)
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)


@dataclass(frozen=True)
class AhemV63Config:
    reader_width: int = 32
    key_dim: int = 16
    capacity: int = 16
    horizon: int = 10_000
    radius: float = 1.0
    transport: float = 0.5
    utility_temperature: float = 0.04
    selection_temperature: float = 0.25
    selection_transform: str = "straight_through_sparsemax"
    retrieval_weight: float = 0.75
    regret_weight: float = 0.10
    reader: str = "learned"

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value <= 0
            for value in (self.reader_width, self.key_dim, self.capacity, self.horizon)
        ):
            raise ValueError("v6.3 dimensions must be positive integers")
        if self.capacity < 2:
            raise ValueError("v6.3 requires at least two transition slots")
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("v6.3 radius must be finite and positive")
        if not math.isfinite(self.transport) or not 0 <= self.transport <= 1:
            raise ValueError("v6.3 transport must be in [0, 1]")
        if (
            not math.isfinite(self.utility_temperature)
            or self.utility_temperature <= 0
        ):
            raise ValueError("v6.3 utility temperature must be finite and positive")
        if (
            not math.isfinite(self.selection_temperature)
            or not 0 < self.selection_temperature <= 1
        ):
            raise ValueError("v6.3 selection temperature must be in (0, 1]")
        if any(
            not math.isfinite(value) or value < 0
            for value in (self.retrieval_weight, self.regret_weight)
        ):
            raise ValueError("v6.3 objective weights must be finite and nonnegative")
        if self.reader not in {"learned", "recency", "none"}:
            raise ValueError("v6.3 reader must be learned, recency, or none")
        if self.selection_transform not in {
            "softmax",
            "sparsemax",
            "straight_through_sparsemax",
        }:
            raise ValueError(
                "v6.3 selection transform must be softmax, sparsemax, or "
                "straight_through_sparsemax"
            )

    def contract(self) -> dict[str, object]:
        return {
            "schema": "argus.ahem-v6.3-candidate/v1",
            "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
            **asdict(self),
            "task": "next_same_flow_dynamic_forecast",
            "backbone": "frozen_ahem_v6.1",
            "null_action": "exact_backbone_forecast",
            "security_detector": False,
            "trained_result": False,
        }


@dataclass(frozen=True)
class AhemV63Output:
    prediction: Tensor
    backbone_prediction: Tensor
    raw_proposals: Tensor
    candidates: Tensor
    gains: Tensor
    weights: Tensor
    logits: Tensor
    valid: Tensor
    source_ordinals: Tensor


class AhemV63FrozenBackbone(nn.Module):
    """Proposal-aligned reader around an immutable v6.1 forecast.

    The reference path intentionally recomputes transition slots.  A future
    streaming kernel may cache projections only after equivalence tests prove
    it returns the same values and decisions.
    """

    schema = "argus.ahem-v6.3-candidate/v1"

    def __init__(
        self,
        backbone: AhemV6StateTransitionNextPacket | None = None,
        config: AhemV63Config | None = None,
        *,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or AhemV63Config()
        self.backbone = backbone or AhemV6StateTransitionNextPacket()
        if self.backbone.packet_dim != 12 or self.backbone.output_dim != 2:
            raise ValueError("v6.3 requires the 12-field, two-target v6.1 backbone")
        if self.config.horizon < self.backbone.local_packets:
            raise ValueError("v6.3 horizon cannot be shorter than the backbone lane")
        self.freeze_backbone = freeze_backbone
        state_width = self.backbone.packet_dim * 2 + 1
        c = self.config
        self.state_signature = nn.Sequential(
            nn.LayerNorm(state_width),
            nn.Linear(state_width, c.reader_width),
            nn.GELU(),
            nn.Linear(c.reader_width, c.key_dim),
        )
        self.value_signature = nn.Sequential(
            nn.LayerNorm(state_width),
            nn.Linear(state_width, c.reader_width),
            nn.GELU(),
        )
        proposal_width = c.key_dim * 2 + c.reader_width
        self.proposal_head = nn.Sequential(
            nn.Linear(proposal_width, c.reader_width),
            nn.GELU(),
            nn.Linear(c.reader_width, 2),
        )
        self.gain_head = nn.Sequential(
            nn.Linear(proposal_width, c.reader_width),
            nn.GELU(),
            nn.Linear(c.reader_width, 2),
        )
        self.null_head = nn.Linear(c.key_dim, 1)
        self.semantic_log_scale = nn.Parameter(torch.tensor(math.log(0.1)))
        self.recency_log_scale = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        nn.init.zeros_(self.gain_head[-1].weight)
        nn.init.zeros_(self.gain_head[-1].bias)
        nn.init.zeros_(self.null_head.weight)
        nn.init.constant_(self.null_head.bias, -1.0)
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> AhemV63FrozenBackbone:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _normalized_gap(self, earlier: Tensor, later: Tensor) -> Tensor:
        return (
            torch.log1p((later - earlier).clamp_min(0).to(torch.float32))
            / math.log1p(self.config.horizon)
        ).clamp_max(1.0)

    def _state_key(self, previous: Tensor, current: Tensor, gap: Tensor) -> Tensor:
        state = torch.cat((current, current - previous, gap[..., None]), dim=-1)
        return F.normalize(self.state_signature(state), dim=-1)

    def _proposal_values(
        self, source: Tensor, successor: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        del valid
        return source, successor

    def _slot_logits(self, query: Tensor, keys: Tensor, age: Tensor) -> Tensor:
        semantic_scale = self.semantic_log_scale.exp().clamp(0.01, 8.0)
        recency_scale = F.softplus(self.recency_log_scale)
        return semantic_scale * torch.einsum("bd,bkd->bk", query, keys) - (
            recency_scale * age
        )

    def forward_diagnostics(self, packets: Tensor) -> AhemV63Output:
        c = self.config
        if packets.ndim != 3 or packets.shape[0] < 1 or packets.shape[-1] != 13:
            raise ValueError("v6.3 expects [batch, context, 13]")
        if packets.shape[1] < self.backbone.local_packets:
            raise ValueError("v6.3 context is shorter than the backbone lane")
        if not torch.isfinite(packets).all() or (packets[..., :2] < 0).any():
            raise ValueError("v6.3 requires finite features and nonnegative dynamics")
        if not ((packets[..., -1] == 0) | (packets[..., -1] == 1)).all():
            raise ValueError("v6.3 flow relation must be binary")
        packets = packets[:, -c.horizon :]
        with torch.set_grad_enabled(not self.freeze_backbone and torch.is_grad_enabled()):
            backbone = self.backbone.forward_diagnostics(packets)
        base = backbone.prediction
        previous, current, slots = transition_slots(
            packets, max_transitions=c.capacity
        )

        def clean(value: Tensor) -> Tensor:
            return value.masked_fill(~slots.valid[..., None], 0)

        source = clean(slots.source)
        successor = clean(slots.successor)
        slot_gap = self._normalized_gap(
            slots.predecessor_ordinals, slots.source_ordinals
        ).to(packets.dtype)
        keys = self._state_key(clean(slots.predecessor), source, slot_gap)
        positions = torch.arange(packets.shape[1], device=packets.device)[None]
        prior = (
            positions[:, :-1]
            .expand(packets.shape[0], -1)
            .masked_fill(packets[:, :-1, -1] == 0, -1)
            .amax(-1)
        )
        anchor = torch.full_like(prior, packets.shape[1] - 1)
        query_gap = torch.where(
            prior >= 0,
            self._normalized_gap(prior, anchor).to(packets.dtype),
            0,
        )
        query = self._state_key(previous, current, query_gap)
        age = self._normalized_gap(slots.source_ordinals, anchor[:, None]).to(
            packets.dtype
        )
        slot_logits = self._slot_logits(query, keys, age)
        valid = torch.cat((torch.ones_like(slots.valid[:, :1]), slots.valid), dim=1)
        if c.reader == "recency":
            memory_age = age.masked_fill(~slots.valid, torch.inf)
            newest = memory_age.argmin(-1)
            has_memory = slots.valid.any(-1)
            weights = torch.zeros_like(valid, dtype=packets.dtype)
            weights[:, 0] = (~has_memory).to(packets.dtype)
            weights.scatter_(
                1,
                (newest + 1)[:, None],
                has_memory[:, None].to(packets.dtype),
            )
            logits = torch.where(
                weights > 0,
                torch.zeros_like(weights),
                torch.full_like(weights, -torch.inf),
            )
        elif c.reader == "none":
            valid[:, 1:] = False
            weights = torch.zeros_like(valid, dtype=packets.dtype)
            weights[:, 0] = 1
            logits = torch.where(
                weights > 0,
                torch.zeros_like(weights),
                torch.full_like(weights, -torch.inf),
            )
        else:
            logits = (
                torch.cat((self.null_head(query), slot_logits), dim=1)
                / c.selection_temperature
            ).masked_fill(~valid, -torch.inf)
            if c.selection_transform == "softmax":
                weights = logits.softmax(dim=-1)
            else:
                soft_weights = masked_sparsemax(logits, valid)
                if c.selection_transform == "sparsemax":
                    weights = soft_weights
                else:
                    hard_weights = F.one_hot(
                        soft_weights.argmax(-1),
                        num_classes=soft_weights.shape[-1],
                    ).to(soft_weights.dtype)
                    weights = hard_weights + soft_weights - soft_weights.detach()

        value_source, value_successor = self._proposal_values(
            source, successor, slots.valid
        )
        transition_gap = self._normalized_gap(
            slots.source_ordinals, slots.successor_ordinals
        ).to(packets.dtype)
        value_state = torch.cat(
            (
                value_source,
                value_successor - value_source,
                transition_gap[..., None],
            ),
            dim=-1,
        )
        values = self.value_signature(value_state)
        query_expanded = query[:, None].expand(-1, c.capacity, -1)
        proposal_state = torch.cat((query_expanded, keys, values), dim=-1)
        observed = value_successor[..., :2]
        transported = base[:, None] + observed - value_source[..., :2]
        raw = (
            (1 - c.transport) * observed
            + c.transport * transported
            + c.radius * self.proposal_head(proposal_state).tanh()
        )
        gains = self.gain_head(proposal_state)
        actions = base[:, None] + c.radius * gains.tanh() * (
            (raw - base[:, None]) / c.radius
        ).tanh()
        actions = torch.where(slots.valid[..., None], actions, base[:, None])
        candidates = torch.cat((base[:, None], actions), dim=1)
        prediction = base + (
            weights[..., None] * (candidates - base[:, None])
        ).sum(dim=1)
        return AhemV63Output(
            prediction=prediction,
            backbone_prediction=base,
            raw_proposals=raw,
            candidates=candidates,
            gains=gains,
            weights=weights,
            logits=logits,
            valid=valid,
            source_ordinals=slots.source_ordinals,
        )

    def forward(self, packets: Tensor) -> Tensor:
        return self.forward_diagnostics(packets).prediction


class AhemV63QueryRotation(AhemV63FrozenBackbone):
    """Mechanism intervention: rotate queries while retaining memory slots."""

    def _slot_logits(self, query: Tensor, keys: Tensor, age: Tensor) -> Tensor:
        if query.size(0) < 2:
            raise ValueError("v6.3 query rotation requires a batch of at least two")
        return super()._slot_logits(query.roll(1, dims=0), keys, age)


class AhemV63ValuePermutation(AhemV63FrozenBackbone):
    """Mechanism intervention: break key-to-successor association."""

    def _proposal_values(
        self, source: Tensor, successor: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        return source, rotate_observed_values(successor, valid)


def ahem_v63_loss(
    output: AhemV63Output,
    target: Tensor,
    config: AhemV63Config,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Score and supervise the exact bounded proposals used by prediction."""
    if (
        target.shape != output.prediction.shape
        or not torch.isfinite(target).all()
        or (target < 0).any()
    ):
        raise ValueError("v6.3 target must be finite nonnegative [batch, 2]")
    costs = dynamic_loss(
        output.candidates.detach(),
        target[:, None].expand_as(output.candidates),
        reduction="none",
    ).masked_fill(~output.valid, torch.inf)
    utility_logits = (-costs / config.utility_temperature).masked_fill(
        ~output.valid, -torch.inf
    )
    utility = utility_logits.softmax(dim=-1)
    log_weights = output.logits.log_softmax(dim=-1)
    retrieval = -(utility * log_weights).masked_fill(~output.valid, 0).sum(-1).mean()
    best = costs.amin(-1, keepdim=True)
    regret = (costs - best).masked_fill(~output.valid, 0).detach()
    expected_regret = (output.weights * regret).sum(-1).mean()
    forecast = dynamic_loss(output.prediction, target)
    backbone = dynamic_loss(output.backbone_prediction, target)
    total = (
        forecast
        + config.retrieval_weight * retrieval
        + config.regret_weight * expected_regret
    )
    return total, {
        "forecast": forecast.detach(),
        "backbone": backbone.detach(),
        "proposal_cross_entropy": retrieval.detach(),
        "expected_proposal_regret": expected_regret.detach(),
    }

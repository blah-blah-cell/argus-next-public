"""Pro v2.1: bounded recency prior with reachable long-range memory.

No trained weights or superiority claim are implied by constructing this model.
The fixed ordinal horizon is part of the key/value projection contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .ahem_v6 import AhemV6Output, AhemV6StateTransitionNextPacket, transition_slots
from .flow_v2_features import FEATURE_CONTRACT_SHA256
from .flow_v6_objective import dynamic_loss


def objective_contract() -> dict[str, object]:
    return {
        "schema": "argus.ahem-pro-objective/v2",
        "dynamic_loss": "smooth_l1_beta1_weighted_length1_time4_div5",
        "forecast_weight": 1.0,
        "local_weight": 0.10,
        "memory_weight": 0.10,
        "retrieval_weight": 0.35,
        "gate_weight": 0.10,
        "retrieval_target": "hard_minimum_observed_successor_utility",
        "retrieval_override_margin": 0.02,
        "retrieval_override_weight": 4.0,
        "gate_temperature": 0.05,
        "gate_improvement_margin": 0.01,
        "retrieval_normalization": "log(max(valid_slots,2))",
        "auxiliary_masking": "memory_gate_nonempty_retrieval_at_least_two",
    }


@dataclass(frozen=True)
class ProConfig:
    width: int = 192
    key_dim: int = 64
    max_transitions: int = 64
    horizon_packets: int = 30_000
    local_packets: int = 128

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("Pro dimensions must be positive integers")
        # Four kernel-3 causal layers at dilations 1,2,4,8 see 31 events.
        if not 31 <= self.local_packets <= self.horizon_packets:
            raise ValueError("Pro requires 31 <= local packets <= horizon")
        if self.max_transitions < 2:
            raise ValueError("Pro requires at least two transition slots")

    def contract(self) -> dict[str, object]:
        return {
            "schema": "argus.ahem-pro-architecture/v2.1",
            "dimensions": asdict(self),
            "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
            "packet_dim": 12,
            "output_dim": 2,
            "gap_normalization": "log1p(ordinal_gap)/log1p(fixed_horizon)",
            "retention": "newest_completed_triples_with_predecessor_in_horizon",
            "retrieval_prior": "bounded_log_rank_gap_div_log_capacity_plus_one",
            "prior_penalty_supremum": 8.0,
            "initial_prior_strength": 6.0,
            "semantic_scale_absolute_supremum": 8.0,
            "output_transform": "softplus_length_and_zero_inflated_time",
            "task": "next_same_flow_dynamic_event_forecast",
            "security_detector": False,
        }

    def contract_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.contract(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class AhemProNextPacket(AhemV6StateTransitionNextPacket):
    """Scaled transition memory with a matching batched and streaming contract."""

    def __init__(self, config: ProConfig | None = None) -> None:
        self.config = config or ProConfig()
        super().__init__(
            width=self.config.width,
            key_dim=self.config.key_dim,
            local_packets=self.config.local_packets,
            max_transitions=self.config.max_transitions,
        )
        # Pro-v2 starts exactly at its strong recency ablation. Training must
        # earn a semantic override instead of destroying that prior at step 0.
        with torch.no_grad():
            self.semantic_log_scale.zero_()
            self.recency_log_scale.fill_(math.log(3.0))

    def _recency_prior(self, recency: Tensor) -> Tensor:
        rank = torch.arange(
            recency.size(1), device=recency.device, dtype=recency.dtype
        )[None]
        # Rank is zero-based and gaps are in [0,1]. Using configured capacity
        # (not allocated slot count) preserves compact/padded read equivalence.
        # The total prior range is <=8 for ANY supported cache capacity, below
        # the attainable semantic pairwise advantage approaching 16.
        relative_age = torch.log1p(rank + recency) / math.log1p(self.max_transitions)
        return -8.0 * self.recency_log_scale.sigmoid() * relative_age

    def _scores(self, query: Tensor, keys: Tensor, recency: Tensor) -> Tensor:
        # Bounded signed residual; alpha=0 is exactly the recency prior.
        semantic = (
            8.0
            * torch.tanh(self.semantic_log_scale)
            * torch.einsum("bd,bkd->bk", query, keys)
        )
        return self._recency_prior(recency) + semantic

    @staticmethod
    def _replace_time(prediction: Tensor, parameters: Tensor) -> Tensor:
        # Pro is trained with this transform from initialization onward.
        time = parameters[:, :1].sigmoid() * F.softplus(parameters[:, 1:2])
        return torch.cat((F.softplus(prediction[:, :1]), time), dim=1)

    def gap(self, earlier: Tensor, later: Tensor) -> Tensor:
        span = torch.full(
            (earlier.shape[0],),
            float(self.config.horizon_packets),
            device=earlier.device,
            dtype=next(self.parameters()).dtype,
        )
        return self._normalized_gap(earlier, later, span)

    def read_projected(
        self,
        local_state: Tensor,
        predecessor: Tensor,
        current: Tensor,
        query_gap: Tensor,
        keys: Tensor,
        values: Tensor,
        successors: Tensor,
        source_ordinals: Tensor,
        valid: Tensor,
        anchor_ordinals: Tensor,
    ) -> AhemV6Output:
        """Batch independent queries; tensors must share model device/dtype.

        Callers own causal ordering, snapshot isolation and horizon expiration.
        All floating inputs must be finite, including invalid padded slots.
        At least one padded slot is required even when no transitions exist.
        This is the fixed-work tensor boundary for a future fused kernel.
        """
        if keys.ndim != 3:
            raise ValueError("Pro keys must be [batch, slots, key dimension]")
        batch, slots = keys.shape[:2]
        if not 1 <= slots <= self.max_transitions:
            raise ValueError("Pro projected slot count is outside capacity")
        shapes = (
            (local_state, (batch, self.width)),
            (predecessor, (batch, 12)),
            (current, (batch, 12)),
            (query_gap, (batch,)),
            (keys, (batch, slots, self.key_dim)),
            (values, (batch, slots, self.width)),
            (successors, (batch, slots, 2)),
            (source_ordinals, (batch, slots)),
            (valid, (batch, slots)),
            (anchor_ordinals, (batch,)),
        )
        if batch < 1 or any(tuple(tensor.shape) != shape for tensor, shape in shapes):
            raise ValueError("Pro projected read has incompatible shapes")
        if valid.dtype != torch.bool:
            raise ValueError("Pro validity must be boolean")
        if source_ordinals.dtype != torch.long or anchor_ordinals.dtype != torch.long:
            raise ValueError("Pro ordinals must be int64")
        parameter = next(self.parameters())
        if parameter.dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "Pro reference supports FP32/FP64; mixed precision is not validated"
            )
        if any(tensor.device != parameter.device for tensor, _ in shapes):
            raise ValueError("Pro projected tensors must share model device")
        if any(tensor.dtype != parameter.dtype for tensor, _ in shapes[:7]):
            raise ValueError("Pro projected floating tensors must share model dtype")
        span = local_state.new_full((batch,), float(self.config.horizon_packets))
        query = self.project_state(predecessor, current, query_gap)
        return self._read(
            local_state,
            query,
            keys,
            values,
            successors,
            source_ordinals,
            valid,
            anchor_ordinals,
            span,
        )

    def forward_from_projected_cache(self, *args, **kwargs) -> AhemV6Output:
        raise ValueError(
            "Pro uses read_projected with its fixed-horizon contract, not the v6 cache API"
        )

    def forward_diagnostics(self, packets: Tensor) -> AhemV6Output:
        if packets.ndim != 3 or packets.size(-1) != 13:
            raise ValueError("Pro expects [batch, context, 12 features + relation]")
        if packets.size(1) < self.local_packets:
            raise ValueError("Pro context is shorter than the local warmup")
        if not torch.isfinite(packets).all():
            raise ValueError("Pro features must be finite")
        if not ((packets[..., -1] == 0) | (packets[..., -1] == 1)).all():
            raise ValueError("Pro query-flow relation must be binary")
        packets = packets[:, -self.config.horizon_packets :]
        predecessor, current, slots = transition_slots(
            packets, max_transitions=self.max_transitions
        )
        batch, count, _ = slots.source.shape
        keys = self.project_state(
            slots.predecessor.reshape(-1, 12),
            slots.source.reshape(-1, 12),
            self.gap(slots.predecessor_ordinals, slots.source_ordinals).reshape(-1),
        ).reshape(batch, count, self.key_dim)
        values = self.project_value(
            slots.source.reshape(-1, 12),
            slots.successor.reshape(-1, 12),
            self.gap(slots.source_ordinals, slots.successor_ordinals).reshape(-1),
        ).reshape(batch, count, self.width)
        return self.read_projected(
            self.local(packets[:, -self.local_packets :, :12]),
            predecessor,
            current,
            torch.where(
                slots.source_ordinals[:, 0] >= 0,
                self.gap(slots.source_ordinals[:, 0], slots.successor_ordinals[:, 0]),
                packets.new_zeros(batch),
            ),
            keys,
            values,
            slots.successor[..., :2],
            slots.source_ordinals,
            slots.valid,
            torch.full(
                (batch,), packets.size(1) - 1, device=packets.device, dtype=torch.long
            ),
        )

    def resource_budget(self, flow_capacity: int = 1024) -> dict[str, int]:
        if type(flow_capacity) is not int or flow_capacity <= 0:
            raise ValueError("flow capacity must be a positive integer")
        scalar_bytes = next(self.parameters()).element_size()
        parameters = sum(parameter.numel() for parameter in self.parameters())
        # Each triple holds projected key/value/successor and 3 int64 ordinals.
        per_flow = self.max_transitions * (
            scalar_bytes * (self.key_dim + self.width + 2) + 3 * 8
        ) + 2 * (scalar_bytes * 12 + 8)
        local_bytes = (
            sum(2 * d for d in self.local.dilations) * self.width * scalar_bytes
        )
        return {
            "parameters": parameters,
            "weight_bytes": parameters * scalar_bytes,
            "max_persistent_payload_bytes_per_flow": per_flow,
            "flow_capacity": flow_capacity,
            "local_state_bytes": local_bytes,
            "max_persistent_payload_bytes": flow_capacity * per_flow + local_bytes,
        }


class AhemProLocalOnlyNextPacket(AhemProNextPacket):
    """Capacity-matched ablation that cannot read transition memory."""

    uses_transition_memory = False


class AhemProRecencyNextPacket(AhemProNextPacket):
    """Capacity-matched ablation whose cache order is fixed by recency."""

    def _scores(self, query: Tensor, keys: Tensor, recency: Tensor) -> Tensor:
        del query, keys
        return self._recency_prior(recency)


def pro_training_loss(
    output: AhemV6Output, target: Tensor
) -> tuple[Tensor, dict[str, Tensor]]:
    """Pro-v2 objective: targets are used only after causal forward prediction.

    Auxiliary memory and gate losses exclude empty-cache rows. Retrieval
    supervision requires two alternatives. All labels are detached.
    """
    contract = objective_contract()
    if target.shape != output.prediction.shape or not torch.isfinite(target).all():
        raise ValueError(
            "Pro targets must be finite and match the two-field prediction"
        )
    if (target < 0).any():
        raise ValueError("Pro targets are nonnegative log1p dynamic features")
    forecast = dynamic_loss(output.prediction, target)
    local_rows = dynamic_loss(output.local_prediction, target, reduction="none")
    memory_rows = dynamic_loss(output.memory_candidate, target, reduction="none")
    has_memory = output.valid.any(dim=-1)
    informative = output.valid.sum(dim=-1) >= 2
    zero = output.prediction.sum() * 0.0
    memory = memory_rows[has_memory].mean() if has_memory.any() else zero
    gate = zero
    if has_memory.any():
        gate_target = (
            (
                local_rows.detach()
                - memory_rows.detach()
                - contract["gate_improvement_margin"]
            )
            / contract["gate_temperature"]
        ).sigmoid()
        gate = F.binary_cross_entropy(
            output.memory_gate[has_memory, 0].clamp(1e-6, 1 - 1e-6),
            gate_target[has_memory],
        )
    retrieval = zero
    if informative.any():
        with torch.no_grad():
            utility = dynamic_loss(
                output.successor_dynamic,
                target[:, None].expand_as(output.successor_dynamic),
                reduction="none",
            ).masked_fill(~output.valid, torch.inf)
            best = utility.argmin(-1)
            recency = output.valid.to(torch.int64).argmax(-1)
            best_loss = utility.gather(1, best[:, None]).squeeze(1)
            recency_loss = utility.gather(1, recency[:, None]).squeeze(1)
            override = recency_loss - best_loss >= contract["retrieval_override_margin"]
            row_weight = torch.where(
                override,
                target.new_tensor(contract["retrieval_override_weight"]),
                target.new_tensor(1.0),
            )
        log_weights = output.retrieval_logits.masked_fill(
            ~output.valid, -1e4
        ).log_softmax(-1)
        normalizer = output.valid.sum(-1).to(target.dtype).clamp_min(2).log()
        selected_log_weight = log_weights.gather(1, best[:, None]).squeeze(1)
        retrieval = (
            -row_weight[informative]
            * selected_log_weight[informative]
            / normalizer[informative]
        ).mean()
    total = (
        forecast * contract["forecast_weight"]
        + contract["local_weight"] * local_rows.mean()
        + contract["memory_weight"] * memory
        + contract["retrieval_weight"] * retrieval
        + contract["gate_weight"] * gate
    )
    return total, {
        "forecast": forecast.detach(),
        "local": local_rows.mean().detach(),
        "memory": memory.detach(),
        "retrieval": retrieval.detach(),
        "gate": gate.detach(),
    }

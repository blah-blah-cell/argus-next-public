"""AHEM-v6 causal state-transition retrieval candidate.

V5 compared a current packet with a cached source packet.  V6 instead compares
the same causal object on both sides: a two-event state ending at the query or
at the cached transition source.  The successor remains value-only and is
therefore unavailable to the key until it has actually been observed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .ahem_v4 import StreamableLocalCausalBranch


@dataclass(frozen=True)
class AhemV6Output:
    prediction: Tensor
    local_prediction: Tensor
    memory_candidate: Tensor
    memory_gate: Tensor
    source_ordinals: Tensor
    retrieval_weights: Tensor
    retrieval_logits: Tensor
    valid: Tensor
    successor_dynamic: Tensor


@dataclass(frozen=True)
class AhemV6Slots:
    predecessor: Tensor
    source: Tensor
    successor: Tensor
    predecessor_ordinals: Tensor
    source_ordinals: Tensor
    successor_ordinals: Tensor
    valid: Tensor


def transition_slots(
    packets: Tensor, *, max_transitions: int
) -> tuple[Tensor, Tensor, AhemV6Slots]:
    """Extract query pair and completed same-flow transitions, newest first.

    ``packets`` contains twelve causal packet fields and the query-flow relation
    bit.  A cache key uses ``predecessor -> source`` while its value uses the
    already completed ``source -> successor`` transition.  Thus no successor
    information can enter a key.
    """
    if packets.ndim != 3 or packets.size(-1) < 2:
        raise ValueError("v6 expects [batch, context, packet+relation]")
    if max_transitions <= 0:
        raise ValueError("max_transitions must be positive")
    batch, length, feature_width = packets.shape
    if batch == 0 or length < 2:
        raise ValueError("v6 requires non-empty batches and at least two events")
    packet_dim = feature_width - 1
    relation = packets[..., -1] > 0.5
    if not relation[:, -1].all():
        raise ValueError("the final event must belong to the query flow")
    positions = torch.arange(length, device=packets.device)[None].expand(batch, -1)
    count = min(length, max_transitions + 2)
    selected = positions.masked_fill(~relation, -1).topk(count, dim=1).values
    lane = packets.gather(
        1, selected.clamp_min(0)[..., None].expand(-1, -1, feature_width)
    )[..., :packet_dim]
    lane_valid = selected >= 0

    query = lane[:, 0]
    query_predecessor = lane[:, 1]
    query_valid = lane_valid[:, 1]
    # If a query has no prior same-flow event (the first packet of a flow,
    # single-packet flows, port scans, DNS lookups), degrade gracefully:
    # return a zero predecessor and all-invalid transition slots so the
    # model falls back to its local-only forecast instead of crashing.
    no_predecessor = ~query_valid
    if no_predecessor.any():
        query_predecessor = torch.where(
            no_predecessor[:, None],
            torch.zeros_like(query_predecessor),
            query_predecessor,
        )

    available = max(0, count - 2)
    predecessor = lane[:, 2 : 2 + available]
    source = lane[:, 1 : 1 + available]
    successor = lane[:, :available]
    predecessor_ordinals = selected[:, 2 : 2 + available]
    source_ordinals = selected[:, 1 : 1 + available]
    successor_ordinals = selected[:, :available]
    valid = (
        lane_valid[:, 2 : 2 + available]
        & lane_valid[:, 1 : 1 + available]
        & lane_valid[:, :available]
    )
    # Rows without a predecessor have no usable transitions.
    if no_predecessor.any():
        valid = valid & ~no_predecessor[:, None]
    if available < max_transitions:
        pad = max_transitions - available

        def pad_features(value: Tensor) -> Tensor:
            return F.pad(value, (0, 0, 0, pad))

        def pad_ordinals(value: Tensor) -> Tensor:
            return F.pad(value, (0, pad), value=-1)

        predecessor = pad_features(predecessor)
        source = pad_features(source)
        successor = pad_features(successor)
        predecessor_ordinals = pad_ordinals(predecessor_ordinals)
        source_ordinals = pad_ordinals(source_ordinals)
        successor_ordinals = pad_ordinals(successor_ordinals)
        valid = F.pad(valid, (0, pad), value=False)
    return (
        query_predecessor,
        query,
        AhemV6Slots(
            predecessor=predecessor,
            source=source,
            successor=successor,
            predecessor_ordinals=predecessor_ordinals,
            source_ordinals=source_ordinals,
            successor_ordinals=successor_ordinals,
            valid=valid,
        ),
    )


class AhemV6StateTransitionNextPacket(nn.Module):
    """Bounded online memory keyed by shared causal two-event states."""

    uses_transition_memory = True

    def __init__(
        self,
        packet_dim: int = 12,
        output_dim: int = 2,
        width: int = 60,
        key_dim: int = 32,
        local_packets: int = 128,
        max_transitions: int = 16,
        state_input_normalization: bool = False,
        state_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if (
            output_dim > packet_dim
            or min(
                packet_dim, output_dim, width, key_dim, local_packets, max_transitions
            )
            <= 0
        ):
            raise ValueError("v6 dimensions are invalid")
        self.packet_dim = packet_dim
        self.output_dim = output_dim
        self.width = width
        self.key_dim = key_dim
        self.local_packets = local_packets
        self.max_transitions = max_transitions
        if not 0.0 <= state_dropout < 1.0:
            raise ValueError("state dropout must be in [0, 1)")
        self.local = StreamableLocalCausalBranch(packet_dim, width)
        self.local_predict = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, output_dim)
        )
        self.local_time = nn.Linear(width, 2)
        state_width = packet_dim * 2 + 1
        self.state_input_normalization = (
            nn.LayerNorm(state_width) if state_input_normalization else nn.Identity()
        )
        self.state_dropout = nn.Dropout(state_dropout)
        self.state_signature = nn.Sequential(
            nn.Linear(state_width, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, key_dim),
        )
        self.transition_value = nn.Sequential(
            nn.Linear(packet_dim * 2 + 1, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.memory_delta = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, output_dim)
        )
        self.memory_time = nn.Linear(width * 2, 2)
        self.utility_gate = nn.Sequential(
            nn.Linear(width * 2 + 2, 32), nn.GELU(), nn.Linear(32, 1)
        )
        self.semantic_log_scale = nn.Parameter(torch.tensor(math.log(2.0)))
        self.recency_log_scale = nn.Parameter(torch.tensor(-1.0))
        nn.init.zeros_(self.local_time.weight)
        nn.init.zeros_(self.memory_time.weight)
        nn.init.constant_(self.local_time.bias, -2.0)
        nn.init.constant_(self.memory_time.bias, -2.0)
        nn.init.zeros_(self.utility_gate[-1].weight)
        nn.init.constant_(self.utility_gate[-1].bias, -1.5)

    @staticmethod
    def _normalized_gap(earlier: Tensor, later: Tensor, span: Tensor) -> Tensor:
        gap = (later - earlier).clamp_min(0).to(span.dtype)
        denominator = torch.log1p(span.clamp_min(2.0))
        if gap.ndim == 2:
            denominator = denominator[:, None]
        # If the retained cache span exceeds the context window (e.g. an
        # LRU entry surviving across >10k global packets), the raw ratio can
        # exceed 1.0 and inject out-of-distribution values. Clamp to [0, 1].
        return (torch.log1p(gap) / denominator).clamp(max=1.0)

    @staticmethod
    def _replace_time(prediction: Tensor, parameters: Tensor) -> Tensor:
        # Frozen v6.1 checkpoints were trained and evaluated with an
        # unconstrained log-length coordinate. Changing this transform after
        # training changes the model and invalidates the frozen result.
        nonzero_probability = torch.sigmoid(parameters[:, :1])
        positive_magnitude = F.softplus(parameters[:, 1:2])
        time = nonzero_probability * positive_magnitude
        return torch.cat((prediction[:, :1], time, prediction[:, 2:]), dim=1)

    def project_state(
        self, predecessor: Tensor, current: Tensor, normalized_gap: Tensor
    ) -> Tensor:
        if predecessor.ndim != 2 or predecessor.shape != current.shape:
            raise ValueError("state endpoints must be matching batches")
        if predecessor.size(1) != self.packet_dim:
            raise ValueError("state endpoints have the wrong feature width")
        if normalized_gap.shape != (predecessor.size(0),):
            raise ValueError("state gap has an incompatible shape")
        state = torch.cat(
            (current, current - predecessor, normalized_gap[:, None]), dim=-1
        )
        encoded = self.state_dropout(
            self.state_signature(self.state_input_normalization(state))
        )
        return F.normalize(encoded, dim=-1)

    def project_value(
        self, source: Tensor, successor: Tensor, normalized_gap: Tensor
    ) -> Tensor:
        if source.ndim != 2 or source.shape != successor.shape:
            raise ValueError("transition endpoints must be matching batches")
        if source.size(1) != self.packet_dim:
            raise ValueError("transition endpoints have the wrong feature width")
        if normalized_gap.shape != (source.size(0),):
            raise ValueError("transition gap has an incompatible shape")
        return self.transition_value(
            torch.cat((source, successor - source, normalized_gap[:, None]), dim=-1)
        )

    def _scores(self, query: Tensor, keys: Tensor, recency: Tensor) -> Tensor:
        semantic_scale = self.semantic_log_scale.exp().clamp(1.0, 8.0)
        return (
            semantic_scale * torch.einsum("bd,bkd->bk", query, keys)
            - F.softplus(self.recency_log_scale) * recency
        )

    def _memory_values(
        self, values: Tensor, successors: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        return values, successors

    def _read(
        self,
        local_state: Tensor,
        query: Tensor,
        keys: Tensor,
        values: Tensor,
        successor_dynamic: Tensor,
        source_ordinals: Tensor,
        valid: Tensor,
        anchor_ordinals: Tensor,
        context_spans: Tensor,
    ) -> AhemV6Output:
        local_prediction = self._replace_time(
            self.local_predict(local_state), self.local_time(local_state)
        )
        if not self.uses_transition_memory:
            return AhemV6Output(
                prediction=local_prediction,
                local_prediction=local_prediction,
                memory_candidate=local_prediction,
                memory_gate=local_state.new_zeros((local_state.size(0), 1)),
                source_ordinals=source_ordinals,
                retrieval_weights=local_state.new_zeros(valid.shape),
                retrieval_logits=local_state.new_zeros(valid.shape),
                valid=valid,
                successor_dynamic=successor_dynamic,
            )
        recency = self._normalized_gap(
            source_ordinals, anchor_ordinals[:, None], context_spans
        )
        logits = self._scores(query, keys, recency).masked_fill(~valid, -1e4)
        weights = logits.softmax(dim=-1) * valid.to(logits.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        read_values, read_successors = self._memory_values(
            values, successor_dynamic, valid
        )
        memory = torch.einsum("bk,bkd->bd", weights, read_values)
        retrieved_successor = torch.einsum("bk,bko->bo", weights, read_successors)
        mixed = torch.cat((local_state, memory), dim=-1)
        raw = (
            0.20 * local_prediction
            + 0.80 * retrieved_successor
            + 0.20 * self.memory_delta(mixed)
        )
        memory_candidate = self._replace_time(raw, self.memory_time(mixed))
        # Training always uses max_transitions slots. Compact inference must
        # retain that denominator, including for a nonzero learned gate.
        evidence = (
            valid.to(local_state.dtype).sum(dim=-1, keepdim=True) / self.max_transitions
        )
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(
            dim=-1, keepdim=True
        ) / math.log(max(self.max_transitions, 2))
        gate = torch.sigmoid(
            self.utility_gate(torch.cat((mixed, evidence, entropy), dim=-1))
        ) * valid.any(dim=-1, keepdim=True).to(local_state.dtype)
        prediction = local_prediction + gate * (memory_candidate - local_prediction)
        return AhemV6Output(
            prediction=prediction,
            local_prediction=local_prediction,
            memory_candidate=memory_candidate,
            memory_gate=gate,
            source_ordinals=source_ordinals,
            retrieval_weights=weights,
            retrieval_logits=logits,
            valid=valid,
            successor_dynamic=successor_dynamic,
        )

    def forward_from_projected_cache(
        self,
        local_packets: Tensor,
        query_predecessor: Tensor,
        query_current: Tensor,
        keys: Tensor,
        values: Tensor,
        successor_dynamic: Tensor,
        source_ordinals: Tensor,
        valid: Tensor,
        anchor_ordinals: Tensor,
        context_spans: Tensor,
        query_gap: Tensor,
    ) -> AhemV6Output:
        batch, count = keys.shape[:2]
        if local_packets.shape != (batch, self.local_packets, self.packet_dim):
            raise ValueError("local packets have an incompatible v6 shape")
        if query_predecessor.shape != (
            batch,
            self.packet_dim,
        ) or query_current.shape != (
            batch,
            self.packet_dim,
        ):
            raise ValueError("query state has an incompatible v6 shape")
        if not 1 <= count <= self.max_transitions:
            raise ValueError("projected cache has an incompatible transition count")
        if keys.shape != (batch, count, self.key_dim) or values.shape != (
            batch,
            count,
            self.width,
        ):
            raise ValueError("projected cache tensors have incompatible v6 shapes")
        if successor_dynamic.shape != (batch, count, self.output_dim):
            raise ValueError("successor values have an incompatible v6 shape")
        if source_ordinals.shape != (batch, count) or valid.shape != (batch, count):
            raise ValueError("cache evidence metadata has an incompatible v6 shape")
        if valid.dtype != torch.bool:
            raise ValueError("cache validity must be boolean")
        if anchor_ordinals.shape != (batch,) or context_spans.shape != (batch,):
            raise ValueError("query ordinal metadata has an incompatible shape")
        query = self.project_state(query_predecessor, query_current, query_gap)
        return self._read(
            self.local(local_packets),
            query,
            keys,
            values,
            successor_dynamic,
            source_ordinals,
            valid,
            anchor_ordinals,
            context_spans,
        )

    def forward_diagnostics(self, packets: Tensor) -> AhemV6Output:
        if packets.ndim != 3 or packets.size(-1) != self.packet_dim + 1:
            raise ValueError("v6 reference input expects packet features plus relation")
        if packets.size(1) < self.local_packets:
            raise ValueError("v6 context is shorter than its local lane")
        predecessor, query_current, slots = transition_slots(
            packets, max_transitions=self.max_transitions
        )
        batch, count, _ = slots.source.shape
        span = packets.new_full((batch,), float(packets.size(1)))
        state_gap = self._normalized_gap(
            slots.predecessor_ordinals, slots.source_ordinals, span
        )
        transition_gap = self._normalized_gap(
            slots.source_ordinals, slots.successor_ordinals, span
        )
        keys = self.project_state(
            slots.predecessor.reshape(-1, self.packet_dim),
            slots.source.reshape(-1, self.packet_dim),
            state_gap.reshape(-1),
        ).reshape(batch, count, self.key_dim)
        values = self.project_value(
            slots.source.reshape(-1, self.packet_dim),
            slots.successor.reshape(-1, self.packet_dim),
            transition_gap.reshape(-1),
        ).reshape(batch, count, self.width)
        query_gap = self._normalized_gap(
            slots.source_ordinals[:, 0],
            slots.successor_ordinals[:, 0],
            span,
        )
        return self.forward_from_projected_cache(
            packets[:, -self.local_packets :, : self.packet_dim],
            predecessor,
            query_current,
            keys,
            values,
            slots.successor[..., : self.output_dim],
            slots.source_ordinals,
            slots.valid,
            torch.full(
                (batch,), packets.size(1) - 1, dtype=torch.int64, device=packets.device
            ),
            span,
            query_gap,
        )

    def forward(self, packets: Tensor) -> Tensor:
        return self.forward_diagnostics(packets).prediction


class AhemV6LocalOnlyNextPacket(AhemV6StateTransitionNextPacket):
    uses_transition_memory = False


class AhemV6RecencyNextPacket(AhemV6StateTransitionNextPacket):
    def _scores(self, query: Tensor, keys: Tensor, recency: Tensor) -> Tensor:
        del query, keys
        return -recency


class AhemV6QueryRotationNextPacket(AhemV6StateTransitionNextPacket):
    def _scores(self, query: Tensor, keys: Tensor, recency: Tensor) -> Tensor:
        if query.size(0) < 2:
            raise ValueError("query rotation requires a batch of at least two")
        return super()._scores(query.roll(1, dims=0), keys, recency)


def rotate_observed_values(values: Tensor, valid: Tensor) -> Tensor:
    """Cycle observed slots only, retaining padding and singletons in place."""
    result = values.clone()
    for row in range(values.size(0)):
        indices = valid[row].nonzero(as_tuple=True)[0]
        result[row, indices] = values[row, indices.roll(1)]
    return result


class AhemV6ValuePermutationNextPacket(AhemV6StateTransitionNextPacket):
    def _memory_values(
        self, values: Tensor, successors: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        return rotate_observed_values(values, valid), rotate_observed_values(
            successors, valid
        )


class AhemV62StateTransitionNextPacket(AhemV6StateTransitionNextPacket):
    """Cross-day-regularized v6 state signatures.

    Per-example LayerNorm is a hypothesis about robustness, not a fitted
    cross-day normalization or a guarantee of scale invariance. Dropout
    regularizes only neural signature learning; online
    inference remains deterministic in evaluation mode.
    """

    def __init__(self) -> None:
        super().__init__(state_input_normalization=True, state_dropout=0.10)

    @staticmethod
    def _replace_time(prediction: Tensor, parameters: Tensor) -> Tensor:
        """Constrain both v6.2 targets; v6.2 requires new training weights."""
        frame_length = F.softplus(prediction[:, :1])
        nonzero_probability = torch.sigmoid(parameters[:, :1])
        positive_magnitude = F.softplus(parameters[:, 1:2])
        time = nonzero_probability * positive_magnitude
        return torch.cat((frame_length, time, prediction[:, 2:]), dim=1)


class AhemV62LocalOnlyNextPacket(AhemV62StateTransitionNextPacket):
    uses_transition_memory = False


class AhemV62RecencyNextPacket(AhemV62StateTransitionNextPacket):
    def _scores(self, query: Tensor, keys: Tensor, recency: Tensor) -> Tensor:
        del query, keys
        return -recency


class AhemV62QueryRotationNextPacket(AhemV62StateTransitionNextPacket):
    def _scores(self, query: Tensor, keys: Tensor, recency: Tensor) -> Tensor:
        if query.size(0) < 2:
            raise ValueError("query rotation requires a batch of at least two")
        return super()._scores(query.roll(1, dims=0), keys, recency)


class AhemV62ValuePermutationNextPacket(AhemV62StateTransitionNextPacket):
    def _memory_values(
        self, values: Tensor, successors: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        return rotate_observed_values(values, valid), rotate_observed_values(
            successors, valid
        )


class FlowLaneLstmNextPacket(nn.Module):
    """Stateful-LSTM comparator over the exact query-flow event lane.

    The relation bit is used only to reconstruct the lane in offline batches.
    Deployment maintains one ordinary recurrent state per opaque flow generation;
    no query-relative feature is supplied to the recurrent model.
    """

    def __init__(self, packet_dim: int = 12, hidden_dim: int = 85) -> None:
        super().__init__()
        if min(packet_dim, hidden_dim) <= 0:
            raise ValueError("flow LSTM dimensions must be positive")
        self.packet_dim = packet_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(packet_dim, hidden_dim, batch_first=True)
        self.predict = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2)
        )

    def forward(self, packets: Tensor) -> Tensor:
        if packets.ndim != 3 or packets.size(-1) != self.packet_dim + 1:
            raise ValueError("flow LSTM expects packet features plus relation")
        relation = packets[..., -1] > 0.5
        lengths = relation.sum(dim=-1)
        if (lengths < 1).any():
            raise ValueError("flow LSTM received an empty query lane")
        lanes = [
            row[mask, : self.packet_dim]
            for row, mask in zip(packets, relation, strict=True)
        ]
        padded = nn.utils.rnn.pad_sequence(lanes, batch_first=True)
        packed = nn.utils.rnn.pack_padded_sequence(
            padded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.encoder(packed)
        return self.predict(hidden[-1])

    def forward_step(
        self,
        packet: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if packet.ndim == 2:
            packet = packet[:, None]
        if packet.ndim != 3 or packet.shape[1:] != (1, self.packet_dim):
            raise ValueError("flow LSTM step expects [batch, 1, packet]")
        encoded, next_state = self.encoder(packet, state)
        return self.predict(encoded[:, -1]), next_state

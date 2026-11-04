"""AHEM-v4 online projected transition memory candidate.

This module implements a falsifiable forecasting mechanism, not an intrusion
detector and not a novelty claim. Cache entries are projected once only after
their successor packet has been observed. Opaque flow-generation identifiers
route cache lanes but never enter learned coordinates.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .ahem_v3 import AhemV3NextPacket, AhemV3Output, LocalCausalBranch

LocalState = tuple[Tensor, ...]


class StreamableLocalCausalBranch(LocalCausalBranch):
    """Depthwise causal stack with an exact one-event stateful path."""

    dilations = (1, 2, 4, 8)

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> LocalState:
        if batch_size <= 0:
            raise ValueError("local-state batch size must be positive")
        width = self.input.out_features
        return tuple(
            torch.zeros(batch_size, 2 * dilation, width, device=device, dtype=dtype)
            for dilation in self.dilations
        )

    def forward_step(
        self, packet: Tensor, state: LocalState | None = None
    ) -> tuple[Tensor, LocalState]:
        if packet.ndim != 2 or packet.size(1) != self.input.in_features:
            raise ValueError("local streaming step expects [batch, feature] input")
        if state is None:
            state = self.initial_state(
                packet.size(0), device=packet.device, dtype=packet.dtype
            )
        if len(state) != len(self.filters):
            raise ValueError("local streaming state has the wrong depth")

        current = self.input(packet)
        next_state: list[Tensor] = []
        for dilation, filter_, mix, history in zip(
            self.dilations, self.filters, self.mix, state, strict=True
        ):
            expected = (packet.size(0), 2 * dilation, self.input.out_features)
            if history.shape != expected:
                raise ValueError("local streaming state has an incompatible shape")
            window = torch.cat((history, current[:, None]), dim=1)
            # A one-output depthwise convolution has exactly three taps. Use
            # the same weights without dispatching a general convolution kernel.
            taps = filter_.weight[:, 0, :]
            causal = (
                history[:, 0] * taps[:, 0]
                + history[:, dilation] * taps[:, 1]
                + current * taps[:, 2]
            )
            if filter_.bias is not None:
                causal = causal + filter_.bias
            next_state.append(window[:, 1:].clone())
            current = current + F.gelu(mix(causal))
        return current, tuple(next_state)


@dataclass(frozen=True)
class AhemV4TransitionLane:
    source_ordinals: Tensor
    successor_ordinals: Tensor
    keys: Tensor
    values: Tensor
    successor_dynamic: Tensor


class AhemV4ProjectedCache:
    """Bounded LRU store of write-time projected completed transitions."""

    def __init__(
        self,
        flow_capacity: int,
        transitions_per_flow: int,
        key_dim: int = 32,
        value_dim: int = 60,
        output_dim: int = 2,
    ) -> None:
        if (
            min(flow_capacity, transitions_per_flow, key_dim, value_dim, output_dim)
            <= 0
        ):
            raise ValueError("v4 cache dimensions must be positive")
        self.flow_capacity = flow_capacity
        self.transitions_per_flow = transitions_per_flow
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_dim = output_dim
        self.version: str | None = None
        self._lanes: OrderedDict[
            Hashable, list[tuple[int, int, Tensor, Tensor, Tensor]]
        ] = OrderedDict()

    @property
    def flow_count(self) -> int:
        return len(self._lanes)

    @property
    def transition_count(self) -> int:
        return sum(len(lane) for lane in self._lanes.values())

    def append(
        self,
        flow_generation: Hashable,
        *,
        source_ordinal: int,
        successor_ordinal: int,
        key: Tensor,
        value: Tensor,
        successor_dynamic: Tensor,
        version: str,
    ) -> None:
        if not version:
            raise ValueError("cache version must not be empty")
        if source_ordinal < 0 or successor_ordinal <= source_ordinal:
            raise ValueError("a transition requires increasing non-negative ordinals")
        expected = (
            (key, (self.key_dim,), "key"),
            (value, (self.value_dim,), "value"),
            (successor_dynamic, (self.output_dim,), "successor"),
        )
        for tensor, shape, name in expected:
            if tensor.shape != shape or not torch.isfinite(tensor).all():
                raise ValueError(f"cache {name} has an incompatible value")
        if version != self.version:
            self.version = version
            self._lanes.clear()
        lane = self._lanes.pop(flow_generation, [])
        if lane and source_ordinal <= lane[-1][0]:
            raise ValueError("transition source ordinals must increase within a lane")
        lane.append(
            (
                source_ordinal,
                successor_ordinal,
                key.detach().cpu().clone(),
                value.detach().cpu().clone(),
                successor_dynamic.detach().cpu().clone(),
            )
        )
        self._lanes[flow_generation] = lane[-self.transitions_per_flow :]
        while len(self._lanes) > self.flow_capacity:
            self._lanes.popitem(last=False)

    def snapshot(
        self, flow_generation: Hashable, *, version: str
    ) -> AhemV4TransitionLane:
        if version != self.version or flow_generation not in self._lanes:
            raise ValueError("no compatible projected transition lane is available")
        lane = self._lanes.pop(flow_generation)
        self._lanes[flow_generation] = lane
        newest_first = list(reversed(lane))
        return AhemV4TransitionLane(
            source_ordinals=torch.tensor(
                [item[0] for item in newest_first], dtype=torch.int64
            ),
            successor_ordinals=torch.tensor(
                [item[1] for item in newest_first], dtype=torch.int64
            ),
            keys=torch.stack([item[2].clone() for item in newest_first]),
            values=torch.stack([item[3].clone() for item in newest_first]),
            successor_dynamic=torch.stack([item[4].clone() for item in newest_first]),
        )


class AhemV4ProjectedStreamNextPacket(nn.Module):
    """Streaming local state plus projected completed-transition retrieval."""

    uses_transition_memory = True

    def __init__(
        self,
        packet_dim: int = 12,
        output_dim: int = 2,
        width: int = 60,
        key_dim: int = 32,
        local_packets: int = 128,
        max_transitions: int = 16,
    ) -> None:
        super().__init__()
        if (
            output_dim > packet_dim
            or min(
                packet_dim, output_dim, width, key_dim, local_packets, max_transitions
            )
            <= 0
        ):
            raise ValueError("v4 dimensions are invalid")
        self.packet_dim = packet_dim
        self.output_dim = output_dim
        self.width = width
        self.key_dim = key_dim
        self.local_packets = local_packets
        self.max_transitions = max_transitions
        self.recency_log_scale = nn.Parameter(torch.tensor(0.0))
        self.local = StreamableLocalCausalBranch(packet_dim, width)
        self.local_predict = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, output_dim)
        )
        self.local_time = nn.Linear(width, 2)
        self.transition = nn.Sequential(
            nn.Linear(packet_dim * 2 + 1, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.query = nn.Linear(width + packet_dim, key_dim)
        self.key = nn.Linear(width, key_dim)
        self.value = nn.Linear(width, width)
        self.memory_delta = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, output_dim)
        )
        self.memory_time = nn.Linear(width * 2, 2)
        self.utility_gate = nn.Sequential(
            nn.Linear(width * 2 + 2, 32), nn.GELU(), nn.Linear(32, 1)
        )
        nn.init.zeros_(self.local_time.weight)
        nn.init.zeros_(self.memory_time.weight)
        nn.init.constant_(self.local_time.bias, -2.0)
        nn.init.constant_(self.memory_time.bias, -2.0)
        nn.init.zeros_(self.utility_gate[-1].weight)
        nn.init.constant_(self.utility_gate[-1].bias, -1.5)

    def _local_prediction(self, local_state: Tensor) -> Tensor:
        return AhemV3NextPacket._replace_time(
            self.local_predict(local_state), self.local_time(local_state)
        )

    def project_transition(
        self, source: Tensor, successor: Tensor, normalized_step_gap: Tensor
    ) -> tuple[Tensor, Tensor]:
        if source.ndim != 2 or source.shape != successor.shape:
            raise ValueError("transition endpoints must be matching batches")
        if source.size(1) != self.packet_dim:
            raise ValueError("transition endpoints have the wrong feature width")
        if normalized_step_gap.shape != (source.size(0),):
            raise ValueError("transition gap has an incompatible shape")
        encoded = self.transition(
            torch.cat(
                (source, successor - source, normalized_step_gap[:, None]), dim=-1
            )
        )
        return F.normalize(self.key(encoded), dim=-1), self.value(encoded)

    def _project_query(self, local_state: Tensor, anchor: Tensor) -> Tensor:
        return F.normalize(self.query(torch.cat((local_state, anchor), dim=-1)), dim=-1)

    def _read_projected(
        self,
        local_state: Tensor,
        anchor: Tensor,
        keys: Tensor,
        values: Tensor,
        successor_dynamic: Tensor,
        source_ordinals: Tensor,
        valid: Tensor,
        anchor_ordinals: Tensor,
        context_spans: Tensor,
    ) -> AhemV3Output:
        local_prediction = self._local_prediction(local_state)
        query = self._project_query(local_state, anchor)
        scale = context_spans.to(local_state.dtype).clamp_min(2.0)
        recency = torch.log1p(
            (anchor_ordinals[:, None] - source_ordinals)
            .clamp_min(0)
            .to(local_state.dtype)
        ) / torch.log1p(scale[:, None])
        scores = self._retrieval_scores(query, keys, recency)
        scores = scores.masked_fill(~valid, -1e4)
        weights = scores.softmax(dim=-1) * valid.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        read_values, read_successors = self._retrieval_values(values, successor_dynamic)
        memory = torch.einsum("bk,bkd->bd", weights, read_values)
        retrieved_successor = torch.einsum("bk,bko->bo", weights, read_successors)
        mixed = torch.cat((local_state, memory), dim=-1)
        raw_candidate = (
            0.20 * local_prediction
            + 0.80 * retrieved_successor
            + 0.20 * self.memory_delta(mixed)
        )
        memory_candidate = AhemV3NextPacket._replace_time(
            raw_candidate, self.memory_time(mixed)
        )
        evidence = valid.to(local_state.dtype).mean(dim=-1, keepdim=True)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(
            dim=-1, keepdim=True
        ) / math.log(max(weights.size(1), 2))
        gate = torch.sigmoid(
            self.utility_gate(torch.cat((mixed, evidence, entropy), dim=-1))
        ) * valid.any(dim=-1, keepdim=True).to(local_state.dtype)
        prediction = local_prediction + gate * (memory_candidate - local_prediction)
        return AhemV3Output(
            prediction=prediction,
            local_prediction=local_prediction,
            memory_candidate=memory_candidate,
            memory_gate=gate,
            selected_chunks=source_ordinals,
            selected_weights=weights,
        )

    def _retrieval_scores(self, query: Tensor, keys: Tensor, recency: Tensor) -> Tensor:
        return (
            torch.einsum("bd,bkd->bk", query, keys) / math.sqrt(self.key_dim)
            - F.softplus(self.recency_log_scale) * recency
        )

    def _retrieval_values(
        self, values: Tensor, successor_dynamic: Tensor
    ) -> tuple[Tensor, Tensor]:
        return values, successor_dynamic

    def forward_from_projected_cache(
        self,
        local_packets: Tensor,
        anchor: Tensor,
        keys: Tensor,
        values: Tensor,
        successor_dynamic: Tensor,
        source_ordinals: Tensor,
        valid: Tensor,
        anchor_ordinals: Tensor,
        context_spans: Tensor,
    ) -> AhemV3Output:
        batch = local_packets.size(0)
        count = keys.size(1)
        if local_packets.shape != (batch, self.local_packets, self.packet_dim):
            raise ValueError("local packets have an incompatible v4 shape")
        if anchor.shape != (batch, self.packet_dim):
            raise ValueError("anchor packets have an incompatible v4 shape")
        if not 1 <= count <= self.max_transitions:
            raise ValueError("projected cache has an incompatible transition count")
        if keys.shape != (batch, count, self.key_dim):
            raise ValueError("projected keys have an incompatible v4 shape")
        if values.shape != (batch, count, self.width):
            raise ValueError("projected values have an incompatible v4 shape")
        if successor_dynamic.shape != (batch, count, self.output_dim):
            raise ValueError("successor values have an incompatible v4 shape")
        if source_ordinals.shape != (batch, count):
            raise ValueError("source ordinals have an incompatible v4 shape")
        if valid.shape != (batch, count) or valid.dtype != torch.bool:
            raise ValueError("projected validity mask has an incompatible v4 shape")
        if anchor_ordinals.shape != (batch,) or context_spans.shape != (batch,):
            raise ValueError(
                "anchor ordinals or context spans have an incompatible shape"
            )
        return self._read_projected(
            self.local(local_packets),
            anchor,
            keys,
            values,
            successor_dynamic,
            source_ordinals,
            valid,
            anchor_ordinals,
            context_spans,
        )

    def forward_stream_step(
        self,
        packet: Tensor,
        local_state: LocalState | None,
        *,
        keys: Tensor,
        values: Tensor,
        successor_dynamic: Tensor,
        source_ordinals: Tensor,
        valid: Tensor,
        anchor_ordinals: Tensor,
        context_spans: Tensor,
    ) -> tuple[AhemV3Output, LocalState]:
        state, next_local_state = self.local.forward_step(packet, local_state)
        output = self._read_projected(
            state,
            packet,
            keys,
            values,
            successor_dynamic,
            source_ordinals,
            valid,
            anchor_ordinals,
            context_spans,
        )
        return output, next_local_state

    def forward_diagnostics(self, packets: Tensor) -> AhemV3Output:
        if packets.ndim != 3 or packets.size(2) != self.packet_dim + 1:
            raise ValueError("v4 reference input expects packet features plus relation")
        if packets.size(1) < self.local_packets:
            raise ValueError("v4 context is shorter than its local lane")
        batch, length, _ = packets.shape
        positions = torch.arange(length, device=packets.device)[None].expand(batch, -1)
        relation = packets[..., -1] > 0.5
        selected = (
            positions.masked_fill(~relation, -1)
            .topk(min(length, self.max_transitions + 1), dim=1)
            .values
        )
        lane = packets.gather(
            1,
            selected.clamp_min(0)[..., None].expand(-1, -1, self.packet_dim + 1),
        )[..., : self.packet_dim]
        lane_valid = selected >= 0
        source = lane[:, 1:]
        successor = lane[:, :-1]
        source_ordinals = selected[:, 1:]
        successor_ordinals = selected[:, :-1]
        valid = lane_valid[:, 1:] & lane_valid[:, :-1] & relation[:, -1:]
        span = packets.new_full((batch,), float(length))
        gap = torch.log1p(
            (successor_ordinals - source_ordinals).clamp_min(0).to(packets.dtype)
        ) / torch.log1p(span[:, None])
        keys, values = self.project_transition(
            source.reshape(-1, self.packet_dim),
            successor.reshape(-1, self.packet_dim),
            gap.reshape(-1),
        )
        count = source.size(1)
        return self.forward_from_projected_cache(
            packets[:, -self.local_packets :, : self.packet_dim],
            packets[:, -1, : self.packet_dim],
            keys.reshape(batch, count, self.key_dim),
            values.reshape(batch, count, self.width),
            successor[..., : self.output_dim],
            source_ordinals,
            valid,
            positions[:, -1],
            span,
        )

    def forward(self, packets: Tensor) -> Tensor:
        return self.forward_diagnostics(packets).prediction


class AhemV4LocalOnlyNextPacket(AhemV4ProjectedStreamNextPacket):
    """Exact v4 local-only control with identical learned parameters."""

    uses_transition_memory = False

    def _read_projected(
        self,
        local_state: Tensor,
        anchor: Tensor,
        keys: Tensor,
        values: Tensor,
        successor_dynamic: Tensor,
        source_ordinals: Tensor,
        valid: Tensor,
        anchor_ordinals: Tensor,
        context_spans: Tensor,
    ) -> AhemV3Output:
        del (
            anchor,
            keys,
            values,
            successor_dynamic,
            valid,
            anchor_ordinals,
            context_spans,
        )
        local_prediction = self._local_prediction(local_state)
        return AhemV3Output(
            prediction=local_prediction,
            local_prediction=local_prediction,
            memory_candidate=local_prediction,
            memory_gate=local_state.new_zeros((local_state.size(0), 1)),
            selected_chunks=source_ordinals,
            selected_weights=local_state.new_zeros(source_ordinals.shape),
        )


class AhemV4RecencyNextPacket(AhemV4ProjectedStreamNextPacket):
    """Exact v4 control using deterministic newest-transition retrieval."""

    def _retrieval_scores(self, query: Tensor, keys: Tensor, recency: Tensor) -> Tensor:
        del query, keys
        return -recency


class AhemV4QueryRotationNextPacket(AhemV4ProjectedStreamNextPacket):
    """Evaluation-only negative control using another example's query."""

    def _retrieval_scores(self, query: Tensor, keys: Tensor, recency: Tensor) -> Tensor:
        if query.size(0) < 2:
            raise ValueError(
                "query rotation requires an evaluation batch of at least two"
            )
        return super()._retrieval_scores(query.roll(1, dims=0), keys, recency)


class AhemV4ValuePermutationNextPacket(AhemV4ProjectedStreamNextPacket):
    """Evaluation-only negative control preserving scores but moving evidence."""

    def _retrieval_values(
        self, values: Tensor, successor_dynamic: Tensor
    ) -> tuple[Tensor, Tensor]:
        if values.size(1) < 2:
            raise ValueError("value permutation requires at least two cache entries")
        return values.roll(1, dims=1), successor_dynamic.roll(1, dims=1)


class AhemV41ProjectedStreamNextPacket(AhemV4ProjectedStreamNextPacket):
    """Single predeclared v4 revision with bounded cosine temperature.

    The parent normalizes both queries and keys and then applies the raw-vector
    ``sqrt(key_dim)`` divisor a second time. That compresses semantic logits to
    at most +/-0.1768 for the frozen key width. V4.1 changes only that scoring
    scale; every writer, value, gate, local path, cache bound, and output remains
    identical to v4.
    """

    semantic_scale_min = 1.0
    semantic_scale_max = 8.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.semantic_log_scale = nn.Parameter(torch.tensor(math.log(2.0)))

    def _semantic_scale(self) -> Tensor:
        return self.semantic_log_scale.exp().clamp(
            self.semantic_scale_min, self.semantic_scale_max
        )

    def _retrieval_scores(self, query: Tensor, keys: Tensor, recency: Tensor) -> Tensor:
        semantic = torch.einsum("bd,bkd->bk", query, keys)
        return (
            self._semantic_scale() * semantic
            - F.softplus(self.recency_log_scale) * recency
        )


class AhemV41LocalOnlyNextPacket(AhemV41ProjectedStreamNextPacket):
    """Exact v4.1 local-only control with an identical parameter budget."""

    uses_transition_memory = False
    _read_projected = AhemV4LocalOnlyNextPacket._read_projected


class AhemV41RecencyNextPacket(AhemV41ProjectedStreamNextPacket):
    """Exact v4.1 control replacing semantic scores with fixed recency."""

    def _retrieval_scores(self, query: Tensor, keys: Tensor, recency: Tensor) -> Tensor:
        del query, keys
        return -recency


class AhemV41QueryRotationNextPacket(AhemV41ProjectedStreamNextPacket):
    """Evaluation-only v4.1 query corruption control."""

    def _retrieval_scores(self, query: Tensor, keys: Tensor, recency: Tensor) -> Tensor:
        if query.size(0) < 2:
            raise ValueError(
                "query rotation requires an evaluation batch of at least two"
            )
        return super()._retrieval_scores(query.roll(1, dims=0), keys, recency)


class AhemV41ValuePermutationNextPacket(AhemV41ProjectedStreamNextPacket):
    """Evaluation-only v4.1 value corruption control."""

    _retrieval_values = AhemV4ValuePermutationNextPacket._retrieval_values


class AhemV5SourceAlignedStreamNextPacket(AhemV41ProjectedStreamNextPacket):
    """V5 uses one past-only signature space for queries and transition keys.

    V4/v4.1 used independent query and key projections, and the key encoder
    included the already-observed successor. Real-capture diagnostics showed
    that both spaces collapsed. V5 keeps successor information in the value
    only and ties current anchors and cached transition sources through one
    small projection.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        del self.query
        del self.key
        self.signature = nn.Linear(self.packet_dim, self.key_dim)
        bound = 1.0 / math.sqrt(self.width + self.packet_dim)
        nn.init.uniform_(self.signature.weight, -bound, bound)
        nn.init.uniform_(self.signature.bias, -bound, bound)

    def _project_query(self, local_state: Tensor, anchor: Tensor) -> Tensor:
        del local_state
        return F.normalize(self.signature(anchor), dim=-1)

    def project_transition(
        self, source: Tensor, successor: Tensor, normalized_step_gap: Tensor
    ) -> tuple[Tensor, Tensor]:
        if source.ndim != 2 or source.shape != successor.shape:
            raise ValueError("transition endpoints must be matching batches")
        if source.size(1) != self.packet_dim:
            raise ValueError("transition endpoints have the wrong feature width")
        if normalized_step_gap.shape != (source.size(0),):
            raise ValueError("transition gap has an incompatible shape")
        encoded = self.transition(
            torch.cat(
                (source, successor - source, normalized_step_gap[:, None]), dim=-1
            )
        )
        return F.normalize(self.signature(source), dim=-1), self.value(encoded)


class AhemV5LocalOnlyNextPacket(AhemV5SourceAlignedStreamNextPacket):
    """Exact v5 local-only control with the same parameter budget."""

    uses_transition_memory = False
    _read_projected = AhemV4LocalOnlyNextPacket._read_projected


class AhemV5RecencyNextPacket(AhemV5SourceAlignedStreamNextPacket):
    """Exact v5 fixed-recency control."""

    def _retrieval_scores(self, query: Tensor, keys: Tensor, recency: Tensor) -> Tensor:
        del query, keys
        return -recency


class AhemV5QueryRotationNextPacket(AhemV5SourceAlignedStreamNextPacket):
    """Evaluation-only v5 query corruption control."""

    def _retrieval_scores(self, query: Tensor, keys: Tensor, recency: Tensor) -> Tensor:
        if query.size(0) < 2:
            raise ValueError(
                "query rotation requires an evaluation batch of at least two"
            )
        return super()._retrieval_scores(query.roll(1, dims=0), keys, recency)


class AhemV5ValuePermutationNextPacket(AhemV5SourceAlignedStreamNextPacket):
    """Evaluation-only v5 value corruption control."""

    _retrieval_values = AhemV4ValuePermutationNextPacket._retrieval_values

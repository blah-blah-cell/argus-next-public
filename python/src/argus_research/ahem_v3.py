"""AHEM-v3: counterfactual utility-gated hierarchical event memory.

The model is a next-packet forecaster, not an intrusion detector. Completed
chunks are compressed without knowledge of future queries and can therefore be
cached once. Retrieval first selects coarse groups, then a bounded number of
fine chunks. A separate local branch remains useful when memory is irrelevant.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Hashable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class AhemV3Output:
    prediction: Tensor
    local_prediction: Tensor
    memory_candidate: Tensor
    memory_gate: Tensor
    selected_chunks: Tensor
    selected_weights: Tensor


class AhemV3MemoryCache:
    """Bounded query-independent chunk states with version invalidation."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("AHEM-v3 cache capacity must be positive")
        self.capacity = capacity
        self.version: str | None = None
        self.states: Tensor | None = None

    @property
    def chunk_count(self) -> int:
        return 0 if self.states is None else self.states.size(0)

    def append(self, state: Tensor, *, version: str) -> None:
        if not version:
            raise ValueError("cache version must not be empty")
        if state.ndim != 1 or not torch.isfinite(state).all():
            raise ValueError("cache accepts one finite projected chunk state")
        if version != self.version:
            self.version = version
            self.states = None
        value = state.detach().to(device="cpu").clone()[None]
        if self.states is not None and value.shape[1:] != self.states.shape[1:]:
            raise ValueError("cache chunk shape changed without a version change")
        self.states = value if self.states is None else torch.cat((self.states, value))
        self.states = self.states[-self.capacity :]


@dataclass(frozen=True)
class AhemV34FlowLane:
    ordinals: Tensor
    packets: Tensor


class AhemV34FlowCache:
    """Bounded LRU cache keyed by an opaque flow-generation identity."""

    def __init__(
        self, flow_capacity: int, events_per_flow: int, packet_dim: int = 12
    ) -> None:
        if min(flow_capacity, events_per_flow, packet_dim) <= 0:
            raise ValueError("v3.4 flow-cache dimensions must be positive")
        self.flow_capacity = flow_capacity
        self.events_per_flow = events_per_flow
        self.packet_dim = packet_dim
        self.version: str | None = None
        self._lanes: OrderedDict[Hashable, list[tuple[int, Tensor]]] = OrderedDict()

    @property
    def flow_count(self) -> int:
        return len(self._lanes)

    @property
    def event_count(self) -> int:
        return sum(len(lane) for lane in self._lanes.values())

    def append(
        self,
        flow_generation: Hashable,
        packet: Tensor,
        *,
        ordinal: int,
        version: str,
    ) -> None:
        if not version:
            raise ValueError("cache version must not be empty")
        if packet.shape != (self.packet_dim,) or not torch.isfinite(packet).all():
            raise ValueError("cache accepts one finite packet feature vector")
        if ordinal < 0:
            raise ValueError("packet ordinal must not be negative")
        if version != self.version:
            self.version = version
            self._lanes.clear()
        lane = self._lanes.pop(flow_generation, [])
        if lane and ordinal <= lane[-1][0]:
            raise ValueError("flow-cache ordinals must increase within a generation")
        lane.append((ordinal, packet.detach().to(device="cpu").clone()))
        self._lanes[flow_generation] = lane[-self.events_per_flow :]
        while len(self._lanes) > self.flow_capacity:
            self._lanes.popitem(last=False)

    def snapshot(self, flow_generation: Hashable, *, version: str) -> AhemV34FlowLane:
        if version != self.version or flow_generation not in self._lanes:
            raise ValueError("no compatible flow-generation lane is available")
        lane = self._lanes.pop(flow_generation)
        self._lanes[flow_generation] = lane
        return AhemV34FlowLane(
            ordinals=torch.tensor([ordinal for ordinal, _ in lane], dtype=torch.int64),
            packets=torch.stack([packet.clone() for _, packet in lane]),
        )


class LocalCausalBranch(nn.Module):
    def __init__(self, feature_dim: int, width: int) -> None:
        super().__init__()
        self.input = nn.Linear(feature_dim, width)
        self.filters = nn.ModuleList(
            nn.Conv1d(width, width, 3, groups=width, dilation=dilation)
            for dilation in (1, 2, 4, 8)
        )
        self.mix = nn.ModuleList(nn.Linear(width, width) for _ in self.filters)

    def forward_sequence(self, packets: Tensor) -> Tensor:
        state = self.input(packets)
        for dilation, filter_, mix in zip(
            (1, 2, 4, 8), self.filters, self.mix, strict=True
        ):
            causal = filter_(F.pad(state.transpose(1, 2), (2 * dilation, 0))).transpose(
                1, 2
            )
            state = state + F.gelu(mix(causal))
        return state

    def forward(self, packets: Tensor) -> Tensor:
        return self.forward_sequence(packets)[:, -1]


class AhemV3NextPacket(nn.Module):
    """Sparse two-stage memory with an auditable per-query utility gate.

    Chunk writing is O(N) and amortized by caching. For M cached chunks, each
    query scores O(M / group_size + coarse_top_k * group_size) entries instead
    of all packets or all chunks. Only local packets and selected values enter
    the final prediction path.
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int | None = None,
        width: int = 60,
        key_dim: int = 32,
        chunk_size: int = 64,
        local_packets: int = 128,
        group_size: int = 16,
        coarse_top_k: int = 4,
        fine_top_k: int = 8,
        hierarchical_retrieval: bool = True,
        use_age_prior: bool = True,
        use_utility_gate: bool = True,
        local_only: bool = False,
        flat_retrieval_until_chunks: int = 0,
    ) -> None:
        super().__init__()
        output_dim = feature_dim if output_dim is None else output_dim
        dimensions = (
            feature_dim,
            output_dim,
            width,
            key_dim,
            chunk_size,
            local_packets,
            group_size,
            coarse_top_k,
            fine_top_k,
        )
        if min(dimensions) <= 0:
            raise ValueError("AHEM-v3 dimensions must be positive")
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.width = width
        self.chunk_size = chunk_size
        self.local_packets = local_packets
        self.group_size = group_size
        self.coarse_top_k = coarse_top_k
        self.fine_top_k = fine_top_k
        self.hierarchical_retrieval = hierarchical_retrieval
        self.use_age_prior = use_age_prior
        self.use_utility_gate = use_utility_gate
        self.local_only = local_only
        if flat_retrieval_until_chunks < 0:
            raise ValueError("flat retrieval threshold must not be negative")
        self.flat_retrieval_until_chunks = flat_retrieval_until_chunks
        self.local = LocalCausalBranch(feature_dim, width)
        # Mean, standard deviation, extrema, last value, and linear trend are
        # query-independent sufficient statistics for the completed chunk.
        self.chunk_encoder = nn.Sequential(
            nn.Linear(feature_dim * 6, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.query = nn.Linear(width, key_dim)
        self.coarse_key = nn.Linear(width, key_dim)
        self.fine_key = nn.Linear(width, key_dim)
        self.value = nn.Linear(width, width)
        self.coarse_age = nn.Sequential(nn.Linear(1, 8), nn.GELU(), nn.Linear(8, 1))
        self.fine_age = nn.Sequential(nn.Linear(1, 8), nn.GELU(), nn.Linear(8, 1))
        self.local_predict = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, output_dim)
        )
        self.memory_delta = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, output_dim)
        )
        self.local_time = nn.Linear(width, 2)
        self.memory_time = nn.Linear(width * 2, 2)
        nn.init.zeros_(self.local_time.weight)
        nn.init.zeros_(self.memory_time.weight)
        nn.init.constant_(self.local_time.bias, -2.0)
        nn.init.constant_(self.memory_time.bias, -2.0)
        self.utility_gate = (
            nn.Sequential(nn.Linear(width * 2 + 2, 32), nn.GELU(), nn.Linear(32, 1))
            if use_utility_gate
            else None
        )

    @staticmethod
    def _replace_time(prediction: Tensor, parameters: Tensor) -> Tensor:
        # Packet gaps have a large point mass at zero. A Bernoulli-like
        # non-zero probability times a positive magnitude models that fact
        # without allowing physically impossible negative log-interarrival.
        # The raw length coordinate is retained for checkpoint compatibility;
        # adding a positive transform requires a newly trained model version.
        nonzero_probability = torch.sigmoid(parameters[:, :1])
        positive_magnitude = F.softplus(parameters[:, 1:2])
        time = nonzero_probability * positive_magnitude
        return torch.cat((prediction[:, :1], time, prediction[:, 2:]), dim=1)

    def encode_completed_chunks(self, packets: Tensor) -> Tensor:
        if packets.ndim != 3 or packets.size(-1) != self.feature_dim:
            raise ValueError("completed packets have an incompatible shape")
        chunks = packets.size(1) // self.chunk_size
        if chunks < 1:
            raise ValueError("at least one completed chunk is required")
        values = packets[:, : chunks * self.chunk_size].reshape(
            packets.size(0), chunks, self.chunk_size, self.feature_dim
        )
        mean = values.mean(dim=2)
        standard_deviation = values.std(dim=2, unbiased=False)
        maximum = values.amax(dim=2)
        minimum = values.amin(dim=2)
        last = values[:, :, -1]
        positions = torch.linspace(
            -1.0, 1.0, self.chunk_size, device=values.device, dtype=values.dtype
        )
        trend = (values * positions[None, None, :, None]).mean(dim=2) * 3.0
        statistics = torch.cat(
            (mean, standard_deviation, maximum, minimum, last, trend), dim=-1
        )
        return self.chunk_encoder(statistics)

    def write_completed_chunk(
        self, packets: Tensor, cache: AhemV3MemoryCache, *, version: str
    ) -> None:
        if packets.shape != (self.chunk_size, self.feature_dim):
            raise ValueError("completed chunk does not match AHEM-v3 configuration")
        state = self.encode_completed_chunks(packets[None])[0, 0]
        cache.append(state, version=version)

    @staticmethod
    def _normalized_age(count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        age = torch.arange(count - 1, -1, -1, device=device, dtype=dtype)
        return torch.log1p(age) / math.log1p(max(count, 1))

    def _retrieve(
        self,
        query_state: Tensor,
        chunk_states: Tensor,
        local_packets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not self.hierarchical_retrieval or (
            self.flat_retrieval_until_chunks > 0
            and chunk_states.size(1) <= self.flat_retrieval_until_chunks
        ):
            return self._retrieve_flat(query_state, chunk_states, local_packets)
        batch, chunks, _ = chunk_states.shape
        groups = math.ceil(chunks / self.group_size)
        padded_count = groups * self.group_size
        if padded_count != chunks:
            padding = chunk_states.new_zeros((batch, padded_count - chunks, self.width))
            padded = torch.cat((chunk_states, padding), dim=1)
        else:
            padded = chunk_states
        valid = torch.arange(padded_count, device=chunk_states.device) < chunks
        grouped = padded.reshape(batch, groups, self.group_size, self.width)
        group_valid = valid.reshape(groups, self.group_size)
        coarse = grouped.sum(dim=2) / group_valid.sum(dim=1).clamp_min(1)[None, :, None]
        query = self.query(query_state)
        coarse_scores = torch.einsum(
            "bd,bgd->bg", query, self.coarse_key(coarse)
        ) / math.sqrt(query.size(-1))
        coarse_age = self._normalized_age(
            groups, chunk_states.device, chunk_states.dtype
        )
        if self.use_age_prior:
            coarse_scores = coarse_scores + self.coarse_age(coarse_age[:, None]).T
        selected_groups = coarse_scores.topk(
            min(self.coarse_top_k, groups), dim=1
        ).indices
        offsets = torch.arange(self.group_size, device=chunk_states.device)
        candidates = (
            selected_groups[:, :, None] * self.group_size + offsets[None, None]
        ).flatten(1)
        candidate_valid = candidates < chunks
        candidates = candidates.clamp_max(chunks - 1)
        gathered = chunk_states.gather(
            1, candidates[:, :, None].expand(-1, -1, self.width)
        )
        fine_scores = torch.einsum(
            "bd,bkd->bk", query, self.fine_key(gathered)
        ) / math.sqrt(query.size(-1))
        age = self._normalized_age(chunks, chunk_states.device, chunk_states.dtype)
        gathered_age = age[candidates]
        if self.use_age_prior:
            fine_scores = fine_scores + self.fine_age(gathered_age[:, :, None]).squeeze(
                -1
            )
        fine_scores = fine_scores.masked_fill(~candidate_valid, float("-inf"))
        selected_scores, selected_in_candidates = fine_scores.topk(
            min(self.fine_top_k, chunks), dim=1
        )
        selected_chunks = candidates.gather(1, selected_in_candidates)
        selected_states = gathered.gather(
            1,
            selected_in_candidates[:, :, None].expand(-1, -1, self.width),
        )
        weights = selected_scores.softmax(dim=1)
        retrieved = torch.einsum("bk,bkd->bd", weights, self.value(selected_states))
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        confidence = selected_scores[:, :1]
        return retrieved, confidence, entropy, selected_chunks, weights

    def _retrieve_flat(
        self,
        query_state: Tensor,
        chunk_states: Tensor,
        local_packets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Reference ablation that scans every cached chunk per query."""
        _, chunks, _ = chunk_states.shape
        query = self.query(query_state)
        scores = torch.einsum(
            "bd,bkd->bk", query, self.fine_key(chunk_states)
        ) / math.sqrt(query.size(-1))
        if self.use_age_prior:
            age = self._normalized_age(chunks, chunk_states.device, chunk_states.dtype)
            scores = scores + self.fine_age(age[None, :, None]).squeeze(-1)
        selected_scores, selected_chunks = scores.topk(
            min(self.fine_top_k, chunks), dim=1
        )
        selected_states = chunk_states.gather(
            1, selected_chunks[:, :, None].expand(-1, -1, self.width)
        )
        weights = selected_scores.softmax(dim=1)
        retrieved = torch.einsum("bk,bkd->bd", weights, self.value(selected_states))
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        confidence = selected_scores[:, :1]
        return retrieved, confidence, entropy, selected_chunks, weights

    def forward_from_memory(
        self, local_packets: Tensor, chunk_states: Tensor
    ) -> AhemV3Output:
        if local_packets.ndim != 3 or local_packets.size(-1) != self.feature_dim:
            raise ValueError("local packets have an incompatible shape")
        if chunk_states.ndim != 3 or chunk_states.size(-1) != self.width:
            raise ValueError("chunk memory has an incompatible shape")
        if local_packets.size(0) != chunk_states.size(0):
            raise ValueError("local packets and memory batch sizes differ")
        local_state = self.local(local_packets[:, -self.local_packets :])
        local_prediction = self._replace_time(
            self.local_predict(local_state), self.local_time(local_state)
        )
        if self.local_only:
            batch = local_packets.size(0)
            empty_indices = torch.empty(
                (batch, 0), device=local_packets.device, dtype=torch.long
            )
            empty_weights = local_packets.new_empty((batch, 0))
            return AhemV3Output(
                local_prediction,
                local_prediction,
                local_prediction,
                local_packets.new_zeros((batch, 1)),
                empty_indices,
                empty_weights,
            )
        retrieved, confidence, entropy, selected, weights = self._retrieve(
            local_state, chunk_states, local_packets
        )
        mixed = torch.cat((local_state, retrieved), dim=1)
        memory_candidate = self._replace_time(
            local_prediction + self.memory_delta(mixed), self.memory_time(mixed)
        )
        if self.use_utility_gate:
            if self.utility_gate is None:
                raise RuntimeError("utility gate configuration is inconsistent")
            gate = torch.sigmoid(
                self.utility_gate(torch.cat((mixed, confidence, entropy), dim=1))
            )
            prediction = local_prediction + gate * (memory_candidate - local_prediction)
        else:
            gate = local_packets.new_ones((local_packets.size(0), 1))
            prediction = memory_candidate
        return AhemV3Output(
            prediction,
            local_prediction,
            memory_candidate,
            gate,
            selected,
            weights,
        )

    def forward(self, packets: Tensor) -> Tensor:
        if packets.ndim != 3 or packets.size(-1) != self.feature_dim:
            raise ValueError("packets have an incompatible shape")
        if packets.size(1) < self.local_packets + self.chunk_size:
            raise ValueError("context needs local evidence and one completed chunk")
        memory_packets = packets[:, : -self.local_packets]
        chunk_states = self.encode_completed_chunks(memory_packets)
        return self.forward_from_memory(
            packets[:, -self.local_packets :], chunk_states
        ).prediction

    def forward_diagnostics(self, packets: Tensor) -> AhemV3Output:
        if packets.size(1) < self.local_packets + self.chunk_size:
            raise ValueError("context needs local evidence and one completed chunk")
        states = self.encode_completed_chunks(packets[:, : -self.local_packets])
        return self.forward_from_memory(packets[:, -self.local_packets :], states)

    def forward_from_cache(
        self, local_packets: Tensor, cache: AhemV3MemoryCache, *, version: str
    ) -> AhemV3Output:
        if cache.version != version or cache.states is None:
            raise ValueError("no compatible AHEM-v3 memory is available")
        states = cache.states.to(local_packets.device)[None].expand(
            local_packets.size(0), -1, -1
        )
        return self.forward_from_memory(local_packets, states)


class AhemV31NextPacket(AhemV3NextPacket):
    """Evidence-selected revision after the controlled v3 component ablation.

    At the target 10k--30k history, scoring at most 467 cached chunks is faster
    and no worse in quality than Python's hierarchical gather. The hierarchy is
    retained above 512 chunks. The soft utility blend is removed because its
    controlled ablation reduced forecast quality and did not avoid retrieval.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        output_dim: int | None = None,
        width: int = 60,
        key_dim: int = 32,
        chunk_size: int = 64,
        local_packets: int = 128,
        group_size: int = 16,
        coarse_top_k: int = 4,
        fine_top_k: int = 8,
        flat_retrieval_until_chunks: int = 512,
    ) -> None:
        super().__init__(
            feature_dim,
            output_dim=output_dim,
            width=width,
            key_dim=key_dim,
            chunk_size=chunk_size,
            local_packets=local_packets,
            group_size=group_size,
            coarse_top_k=coarse_top_k,
            fine_top_k=fine_top_k,
            use_utility_gate=False,
            flat_retrieval_until_chunks=flat_retrieval_until_chunks,
        )


class AhemV32NextPacket(AhemV31NextPacket):
    """Causal temporal transition memory with utility-ranked retrieval."""

    temporal_signature_dim = 8
    uses_transition_memory = True

    def __init__(
        self,
        feature_dim: int,
        *,
        output_dim: int | None = None,
        width: int = 60,
        key_dim: int = 32,
        chunk_size: int = 64,
        local_packets: int = 128,
        group_size: int = 16,
        coarse_top_k: int = 4,
        fine_top_k: int = 8,
        flat_retrieval_until_chunks: int = 512,
    ) -> None:
        super().__init__(
            feature_dim,
            output_dim=output_dim,
            width=width,
            key_dim=key_dim,
            chunk_size=chunk_size,
            local_packets=local_packets,
            group_size=group_size,
            coarse_top_k=coarse_top_k,
            fine_top_k=fine_top_k,
            flat_retrieval_until_chunks=flat_retrieval_until_chunks,
        )
        self.temporal_chunk = nn.Sequential(
            nn.Linear(self.temporal_signature_dim, 16),
            nn.GELU(),
            nn.Linear(16, self.width),
        )
        self.temporal_query = nn.Sequential(
            nn.Linear(self.temporal_signature_dim, 16),
            nn.GELU(),
            nn.Linear(16, self.query.out_features),
        )
        self.temporal_key = nn.Linear(self.width, self.query.out_features)
        self.retrieval_log_scale = nn.Parameter(torch.tensor(math.log(4.0)))
        self.successor_embed = nn.Linear(self.feature_dim, self.width)
        self.successor_projection: nn.Module = (
            nn.Identity()
            if self.output_dim == self.feature_dim
            else nn.Linear(self.feature_dim, self.output_dim)
        )
        self.memory_mix = 0.4

    @property
    def memory_width(self) -> int:
        return self.width + self.feature_dim

    @staticmethod
    def temporal_signature(packets: Tensor) -> Tensor:
        if packets.size(-1) < 2:
            raise ValueError("temporal packets lack an interarrival feature")
        gaps = packets[..., 1]
        length = gaps.size(-1)
        positions = torch.linspace(
            -1.0, 1.0, length, device=gaps.device, dtype=gaps.dtype
        )
        mean = gaps.mean(dim=-1)
        deviation = gaps.std(dim=-1, unbiased=False)
        maximum = gaps.amax(dim=-1)
        minimum = gaps.amin(dim=-1)
        last = gaps[..., -1]
        trend = (gaps * positions).mean(dim=-1) * 3.0
        nonzero = (gaps > 1e-7).to(gaps.dtype).mean(dim=-1)
        duration = torch.log1p(torch.expm1(gaps.clamp(0.0, 8.0)).sum(dim=-1))
        return torch.stack(
            (mean, deviation, maximum, minimum, last, trend, nonzero, duration),
            dim=-1,
        )

    def encode_completed_chunks(self, packets: Tensor) -> Tensor:
        if packets.ndim != 3 or packets.size(-1) != self.feature_dim:
            raise ValueError("completed packets have an incompatible shape")
        chunks = (packets.size(1) - 1) // self.chunk_size
        if chunks < 1:
            raise ValueError("transition memory needs one chunk and its successor")
        contexts = packets[:, : chunks * self.chunk_size]
        values = contexts.reshape(
            packets.size(0), chunks, self.chunk_size, self.feature_dim
        )
        keys = AhemV31NextPacket.encode_completed_chunks(self, contexts)
        keys = keys + self.temporal_chunk(self.temporal_signature(values))
        successor_indices = torch.arange(
            self.chunk_size,
            (chunks + 1) * self.chunk_size,
            self.chunk_size,
            device=packets.device,
        )
        successors = packets.index_select(1, successor_indices)
        return torch.cat((keys, successors), dim=-1)

    def write_completed_transition(
        self,
        context: Tensor,
        successor: Tensor,
        cache: AhemV3MemoryCache,
        *,
        version: str,
    ) -> None:
        if context.shape != (self.chunk_size, self.feature_dim):
            raise ValueError("transition context does not match AHEM-v3.2")
        if successor.shape != (self.feature_dim,):
            raise ValueError("transition successor does not match AHEM-v3.2")
        state = self.encode_completed_chunks(
            torch.cat((context, successor[None]))[None]
        )[0, 0]
        cache.append(state, version=version)

    def retrieval_scores(
        self,
        query_state: Tensor,
        chunk_states: Tensor,
        local_packets: Tensor,
    ) -> Tensor:
        keys = chunk_states[..., : self.width]
        content_query = F.normalize(self.query(query_state), dim=-1)
        content_keys = F.normalize(self.fine_key(keys), dim=-1)
        time_query = F.normalize(
            self.temporal_query(self.temporal_signature(local_packets)), dim=-1
        )
        time_keys = F.normalize(self.temporal_key(keys), dim=-1)
        content = torch.einsum("bd,bkd->bk", content_query, content_keys)
        temporal = torch.einsum("bd,bkd->bk", time_query, time_keys)
        scale = self.retrieval_log_scale.clamp(math.log(1.0), math.log(30.0)).exp()
        scores = (content + temporal) * (0.5 * scale)
        if self.use_age_prior:
            age = self._normalized_age(
                chunk_states.size(1), chunk_states.device, chunk_states.dtype
            )
            scores = scores + self.fine_age(age[None, :, None]).squeeze(-1)
        return scores

    def memory_candidate(
        self,
        local_state: Tensor,
        local_prediction: Tensor,
        retrieved_key: Tensor,
        retrieved_successor: Tensor,
    ) -> Tensor:
        retrieved = self.value(retrieved_key) + self.successor_embed(
            retrieved_successor
        )
        mixed = torch.cat((local_state, retrieved), dim=-1)
        projected_successor = self.successor_projection(retrieved_successor)
        raw = (
            0.15 * local_prediction
            + 0.85 * projected_successor
            + 0.25 * self.memory_delta(mixed)
        )
        head_prediction = self._replace_time(raw, self.memory_time(mixed))
        time = (
            0.75 * projected_successor[:, 1:2].clamp_min(0.0)
            + 0.25 * (head_prediction[:, 1:2])
        )
        return torch.cat((raw[:, :1], time, raw[:, 2:]), dim=-1)

    def _retrieve_transition(
        self,
        query_state: Tensor,
        chunk_states: Tensor,
        local_packets: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        scores = self.retrieval_scores(query_state, chunk_states, local_packets)
        selected_scores, selected_chunks = scores.topk(
            min(self.fine_top_k, chunk_states.size(1)), dim=1
        )
        selected_states = chunk_states.gather(
            1,
            selected_chunks[:, :, None].expand(-1, -1, self.memory_width),
        )
        weights = selected_scores.softmax(dim=1)
        keys = selected_states[..., : self.width]
        successors = selected_states[..., self.width :]
        retrieved_key = torch.einsum("bk,bkd->bd", weights, keys)
        retrieved_successor = torch.einsum("bk,bkf->bf", weights, successors)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        return (
            retrieved_key,
            retrieved_successor,
            selected_scores[:, :1],
            entropy,
            selected_chunks,
            weights,
        )

    def forward_from_memory(
        self, local_packets: Tensor, chunk_states: Tensor
    ) -> AhemV3Output:
        if local_packets.ndim != 3 or local_packets.size(-1) != self.feature_dim:
            raise ValueError("local packets have an incompatible shape")
        if chunk_states.ndim != 3 or chunk_states.size(-1) != self.memory_width:
            raise ValueError("transition memory has an incompatible shape")
        if local_packets.size(0) != chunk_states.size(0):
            raise ValueError("local packets and memory batch sizes differ")
        local_state = self.local(local_packets[:, -self.local_packets :])
        local_prediction = self._replace_time(
            self.local_predict(local_state), self.local_time(local_state)
        )
        (
            retrieved_key,
            retrieved_successor,
            _confidence,
            _entropy,
            selected,
            weights,
        ) = self._retrieve_transition(local_state, chunk_states, local_packets)
        memory_candidate = self.memory_candidate(
            local_state,
            local_prediction,
            retrieved_key,
            retrieved_successor,
        )
        gate = local_packets.new_full((local_packets.size(0), 1), self.memory_mix)
        prediction = local_prediction + gate * (memory_candidate - local_prediction)
        return AhemV3Output(
            prediction,
            local_prediction,
            memory_candidate,
            gate,
            selected,
            weights,
        )

    def _retrieve_flat(
        self,
        query_state: Tensor,
        chunk_states: Tensor,
        local_packets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if local_packets is None:
            raise ValueError("AHEM-v3.2 retrieval requires local temporal evidence")
        key, successor, confidence, entropy, selected, weights = (
            self._retrieve_transition(query_state, chunk_states, local_packets)
        )
        retrieved = self.value(key) + self.successor_embed(successor)
        return retrieved, confidence, entropy, selected, weights


class AhemV32LocalOnlyNextPacket(AhemV32NextPacket):
    """Pre-registered ablation that removes the transition-memory read path."""

    uses_transition_memory = False

    def forward(self, packets: Tensor) -> Tensor:
        if packets.ndim != 3 or packets.size(-1) != self.feature_dim:
            raise ValueError("packets have an incompatible shape")
        local_packets = packets[:, -self.local_packets :]
        local_state = self.local(local_packets)
        return self._replace_time(
            self.local_predict(local_state), self.local_time(local_state)
        )

    def forward_diagnostics(self, packets: Tensor) -> AhemV3Output:
        prediction = self.forward(packets)
        batch = packets.size(0)
        return AhemV3Output(
            prediction=prediction,
            local_prediction=prediction,
            memory_candidate=prediction,
            memory_gate=packets.new_zeros((batch, 1)),
            selected_chunks=torch.empty(
                (batch, 0), dtype=torch.long, device=packets.device
            ),
            selected_weights=packets.new_empty((batch, 0)),
        )


class AhemV32RecencyNextPacket(AhemV32NextPacket):
    """Pre-registered ablation that replaces learned retrieval with recency."""

    def retrieval_scores(
        self,
        query_state: Tensor,
        chunk_states: Tensor,
        local_packets: Tensor,
    ) -> Tensor:
        del query_state, local_packets
        chunks = chunk_states.size(1)
        return torch.arange(
            chunks, device=chunk_states.device, dtype=chunk_states.dtype
        )[None].expand(chunk_states.size(0), -1)


class AhemV33NextPacket(AhemV32NextPacket):
    """Flow-conditioned global memory with relation-aware inputs."""

    def __init__(self, input_dim: int = 13, output_dim: int = 12, **kwargs: object):
        super().__init__(input_dim, output_dim=output_dim, **kwargs)


class AhemV33LocalOnlyNextPacket(AhemV32LocalOnlyNextPacket):
    """Local-only v3.3 ablation under the identical input contract."""

    def __init__(self, input_dim: int = 13, output_dim: int = 12, **kwargs: object):
        super().__init__(input_dim, output_dim=output_dim, **kwargs)


class AhemV33RecencyNextPacket(AhemV32RecencyNextPacket):
    """Fixed-recency v3.3 ablation under the identical input contract."""

    def __init__(self, input_dim: int = 13, output_dim: int = 12, **kwargs: object):
        super().__init__(input_dim, output_dim=output_dim, **kwargs)


class AhemV34FlowIndexedNextPacket(nn.Module):
    """Bounded flow-indexed episodic transition memory.

    The final input coordinate is a runtime relation bit, not a learned flow
    identifier. It marks packets belonging to the query flow. The production
    cache can therefore fetch a bounded lane by exact flow-generation key in
    O(1), while the neural model only learns how much each observed transition
    helps the current forecast.
    """

    uses_transition_memory = True

    def __init__(
        self,
        input_dim: int = 13,
        output_dim: int = 2,
        width: int = 60,
        key_dim: int = 32,
        local_packets: int = 128,
        max_flow_packets: int = 17,
    ) -> None:
        super().__init__()
        if input_dim < 2 or output_dim > input_dim - 1:
            raise ValueError("v3.4 expects packet features plus one relation bit")
        if min(width, key_dim, local_packets) <= 0 or max_flow_packets < 2:
            raise ValueError("v3.4 dimensions are invalid")
        self.feature_dim = input_dim
        self.packet_dim = input_dim - 1
        self.output_dim = output_dim
        self.width = width
        self.local_packets = local_packets
        self.max_flow_packets = max_flow_packets
        self.local = LocalCausalBranch(input_dim, width)
        self.local_predict = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, output_dim)
        )
        self.local_time = nn.Linear(width, 2)
        self.transition = nn.Sequential(
            nn.Linear(self.packet_dim * 2 + 2, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.query = nn.Linear(width + self.packet_dim, key_dim)
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

    def _flow_lane(self, packets: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, length, _ = packets.shape
        count = min(length, self.max_flow_packets)
        positions = torch.arange(length, device=packets.device)[None].expand(batch, -1)
        relation = packets[..., -1] > 0.5
        anchor_valid = relation[:, -1:]
        marked = positions.masked_fill(~relation, -1)
        selected = marked.topk(count, dim=1).values
        gathered = packets.gather(
            1, selected.clamp_min(0)[..., None].expand(-1, -1, self.feature_dim)
        )[..., : self.packet_dim]
        valid = (selected >= 0) & anchor_valid
        spans = packets.new_full((batch,), float(length))
        return gathered, selected, valid, spans

    def _forward_indexed_lane(
        self,
        local_packets: Tensor,
        flow_packets: Tensor,
        flow_ordinals: Tensor,
        flow_valid: Tensor,
        context_spans: Tensor,
    ) -> AhemV3Output:
        local_state = self.local(local_packets)
        local_prediction = self._local_prediction(local_state)
        source = flow_packets[:, 1:]
        successor = flow_packets[:, :-1]
        source_position = flow_ordinals[:, 1:]
        successor_position = flow_ordinals[:, :-1]
        valid = flow_valid[:, 1:] & flow_valid[:, :-1]
        scale = context_spans.to(dtype=local_packets.dtype).clamp_min(2.0)
        step_gap = torch.log1p(
            (successor_position - source_position).clamp_min(0).to(local_packets.dtype)
        ) / torch.log1p(scale[:, None])
        anchor_position = flow_ordinals[:, :1]
        recency = torch.log1p(
            (anchor_position - source_position).clamp_min(0).to(local_packets.dtype)
        ) / torch.log1p(scale[:, None])
        inputs = torch.cat(
            (source, successor - source, step_gap[..., None], recency[..., None]),
            dim=-1,
        )
        transitions = self.transition(inputs)
        anchor = flow_packets[:, 0]
        query = F.normalize(
            self.query(torch.cat((local_state, anchor), dim=-1)), dim=-1
        )
        keys = F.normalize(self.key(transitions), dim=-1)
        scores = self._retrieval_scores(query, keys, recency)
        scores = scores.masked_fill(~valid, -1e4)
        weights = scores.softmax(dim=-1) * valid.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        memory = torch.einsum("bk,bkd->bd", weights, self.value(transitions))
        retrieved_successor = torch.einsum(
            "bk,bkf->bf", weights, successor[..., : self.output_dim]
        )
        mixed = torch.cat((local_state, memory), dim=-1)
        raw_candidate = (
            0.20 * local_prediction
            + 0.80 * retrieved_successor
            + 0.20 * self.memory_delta(mixed)
        )
        memory_candidate = AhemV3NextPacket._replace_time(
            raw_candidate, self.memory_time(mixed)
        )
        evidence = valid.to(local_packets.dtype).mean(dim=-1, keepdim=True)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(
            dim=-1, keepdim=True
        ) / math.log(max(weights.size(1), 2))
        gate = torch.sigmoid(
            self.utility_gate(torch.cat((mixed, evidence, entropy), dim=-1))
        ) * valid.any(dim=-1, keepdim=True).to(local_packets.dtype)
        prediction = local_prediction + gate * (memory_candidate - local_prediction)
        return AhemV3Output(
            prediction=prediction,
            local_prediction=local_prediction,
            memory_candidate=memory_candidate,
            memory_gate=gate,
            selected_chunks=source_position,
            selected_weights=weights,
        )

    def _local_prediction(self, local_state: Tensor) -> Tensor:
        return AhemV3NextPacket._replace_time(
            self.local_predict(local_state), self.local_time(local_state)
        )

    def _retrieval_scores(self, query: Tensor, keys: Tensor, recency: Tensor) -> Tensor:
        del recency
        return torch.einsum("bd,bkd->bk", query, keys) / math.sqrt(query.size(-1))

    def forward_from_indexed_lane(
        self,
        local_packets: Tensor,
        flow_packets: Tensor,
        flow_ordinals: Tensor,
        flow_valid: Tensor,
        context_spans: Tensor,
    ) -> AhemV3Output:
        """Infer from a bounded local ring and descending cached flow lane.

        ``flow_packets[:, 0]`` is the current anchor. Remaining valid entries
        are older same-flow packets in descending ordinal order. Invalid padded
        entries are masked by ``flow_valid`` and never contribute to retrieval.
        """
        if (
            local_packets.ndim != 3
            or local_packets.size(1) != self.local_packets
            or local_packets.size(2) != self.feature_dim
        ):
            raise ValueError("local packets have an incompatible v3.4 shape")
        if (
            flow_packets.ndim != 3
            or flow_packets.size(2) != self.packet_dim
            or not 2 <= flow_packets.size(1) <= self.max_flow_packets
        ):
            raise ValueError("flow packets have an incompatible v3.4 shape")
        batch, count, _ = flow_packets.shape
        if flow_ordinals.shape != (batch, count):
            raise ValueError("flow ordinals have an incompatible v3.4 shape")
        if flow_valid.shape != (batch, count) or flow_valid.dtype != torch.bool:
            raise ValueError("flow validity mask has an incompatible v3.4 shape")
        if context_spans.shape != (batch,):
            raise ValueError("context spans have an incompatible v3.4 shape")
        if local_packets.size(0) != batch:
            raise ValueError("local and flow lanes have different batch sizes")
        if not torch.all(flow_valid[:, 0]):
            raise ValueError("every indexed flow lane must contain its anchor")
        return self._forward_indexed_lane(
            local_packets,
            flow_packets,
            flow_ordinals,
            flow_valid,
            context_spans,
        )

    def forward_diagnostics(self, packets: Tensor) -> AhemV3Output:
        if packets.ndim != 3 or packets.size(-1) != self.feature_dim:
            raise ValueError("packets have an incompatible v3.4 shape")
        if packets.size(1) < self.local_packets:
            raise ValueError("v3.4 context is shorter than its local lane")
        flow_packets, flow_ordinals, flow_valid, spans = self._flow_lane(packets)
        return self._forward_indexed_lane(
            packets[:, -self.local_packets :],
            flow_packets,
            flow_ordinals,
            flow_valid,
            spans,
        )

    def forward(self, packets: Tensor) -> Tensor:
        return self.forward_diagnostics(packets).prediction


class AhemV34LocalOnlyNextPacket(AhemV34FlowIndexedNextPacket):
    """Exact v3.4 ablation with the episodic read disabled."""

    uses_transition_memory = False

    def _forward_indexed_lane(
        self,
        local_packets: Tensor,
        flow_packets: Tensor,
        flow_ordinals: Tensor,
        flow_valid: Tensor,
        context_spans: Tensor,
    ) -> AhemV3Output:
        del flow_packets, flow_valid, context_spans
        local_prediction = self._local_prediction(self.local(local_packets))
        batch = local_packets.size(0)
        selected = flow_ordinals[:, 1:]
        return AhemV3Output(
            prediction=local_prediction,
            local_prediction=local_prediction,
            memory_candidate=local_prediction,
            memory_gate=local_packets.new_zeros((batch, 1)),
            selected_chunks=selected,
            selected_weights=local_packets.new_zeros(selected.shape),
        )


class AhemV34RecencyNextPacket(AhemV34FlowIndexedNextPacket):
    """Exact v3.4 ablation using fixed recency instead of learned retrieval."""

    def _retrieval_scores(self, query: Tensor, keys: Tensor, recency: Tensor) -> Tensor:
        del query, keys
        return -recency

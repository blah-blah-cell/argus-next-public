"""AHEM-v2: causal, sparse, multi-slot hierarchical event memory.

This is a research model, not a deployed detector.  Its central constraint is
that it must retrieve from a long packet history without quadratic attention or
an unbounded recurrent state.  A production implementation will keep completed
chunk slots in Rust; this reference computes them from a supplied context so it
can be trained and audited in PyTorch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Auditable retrieval information; no packet payload is emitted here."""

    selected_chunks: Tensor
    selected_weights: Tensor


class AhemV2MemoryCache:
    """Per-flow completed-chunk cache with explicit version invalidation.

    This deliberately stores only projected chunk keys and slot values. The
    packet history itself is not retained after a completed chunk is written.
    The caller owns flow identity and feeds chunks in observed order.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("memory cache capacity must be positive")
        self.capacity = capacity
        self.version: str | None = None
        self.keys: Tensor | None = None
        self.slot_keys: Tensor | None = None
        self.slot_values: Tensor | None = None

    @property
    def chunk_count(self) -> int:
        return 0 if self.keys is None else self.keys.size(0)

    def clear_for_version(self, version: str) -> None:
        if not version:
            raise ValueError("cache version must not be empty")
        if version != self.version:
            self.version = version
            self.keys = self.slot_keys = self.slot_values = None

    def append(
        self, key: Tensor, slot_key: Tensor, slot_value: Tensor, *, version: str
    ) -> None:
        self.clear_for_version(version)
        if key.ndim != 1 or slot_key.ndim != 2 or slot_value.ndim != 2:
            raise ValueError(
                "cache entries must be one chunk key and two slot matrices"
            )
        if slot_key.size(0) != slot_value.size(0):
            raise ValueError("slot key/value counts must match")
        key, slot_key, slot_value = (
            value.detach().clone() for value in (key, slot_key, slot_value)
        )
        if self.keys is None:
            self.keys, self.slot_keys, self.slot_values = (
                key[None],
                slot_key[None],
                slot_value[None],
            )
        else:
            if (
                key.shape != self.keys.shape[1:]
                or slot_key.shape != self.slot_keys.shape[1:]
                or slot_value.shape != self.slot_values.shape[1:]
            ):
                raise ValueError("incompatible cache entry shape")
            self.keys = torch.cat((self.keys, key[None]), dim=0)[-self.capacity :]
            self.slot_keys = torch.cat((self.slot_keys, slot_key[None]), dim=0)[
                -self.capacity :
            ]
            self.slot_values = torch.cat((self.slot_values, slot_value[None]), dim=0)[
                -self.capacity :
            ]


class CausalLocalEncoder(nn.Module):
    """A short causal receptive field, used before long-memory retrieval."""

    def __init__(self, input_dim: int, width: int) -> None:
        super().__init__()
        self.project = nn.Linear(input_dim, width)
        self.filters = nn.ModuleList(
            nn.Conv1d(width, width, 3, groups=width, dilation=dilation)
            for dilation in (1, 2, 4, 8)
        )
        self.mix = nn.ModuleList(nn.Linear(width, width) for _ in self.filters)

    def forward(self, packets: Tensor) -> Tensor:
        if packets.ndim != 3:
            raise ValueError("packets must have shape [batch, packets, features]")
        state = self.project(packets)
        for dilation, filter_, mix in zip(
            (1, 2, 4, 8), self.filters, self.mix, strict=True
        ):
            causal = filter_(F.pad(state.transpose(1, 2), (2 * dilation, 0))).transpose(
                1, 2
            )
            state = state + F.gelu(mix(causal))
        return state


class AhemV2NextPacket(nn.Module):
    """Hierarchical Event Memory with bounded multi-slot chunk retrieval.

    For a context of ``N`` packets this uses O(N) chunk writing and O(N/chunk)
    scoring, then reads only ``top_k * slots_per_chunk`` values.  It does not
    form an N x N attention matrix and its writer is query-independent, which
    makes completed chunks safe to cache before a later query exists.
    """

    def __init__(
        self,
        feature_dim: int,
        width: int = 96,
        key_dim: int = 48,
        chunk_size: int = 64,
        slots_per_chunk: int = 4,
        top_k: int = 8,
    ) -> None:
        super().__init__()
        if min(feature_dim, width, key_dim, chunk_size, slots_per_chunk, top_k) <= 0:
            raise ValueError("AHEM-v2 dimensions must be positive")
        self.chunk_size = chunk_size
        self.slots_per_chunk = slots_per_chunk
        self.top_k = top_k
        self.local = CausalLocalEncoder(feature_dim, width)
        self.slot_queries = nn.Parameter(torch.empty(slots_per_chunk, width))
        nn.init.normal_(self.slot_queries, std=width**-0.5)
        self.chunk_key = nn.Linear(width, key_dim)
        self.slot_key = nn.Linear(width, key_dim)
        self.slot_value = nn.Linear(width, width)
        self.query = nn.Linear(width, key_dim)
        self.fuse = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, width)
        )
        self.gate = nn.Linear(width * 2, width)
        self.predict = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, feature_dim)
        )

    def encode_packets(self, packets: Tensor) -> Tensor:
        """Expose causal local states for causality tests and diagnostics."""
        return self.local(packets)

    def _write_chunks(self, history: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        chunks = history.size(1) // self.chunk_size
        if chunks < 1:
            raise ValueError(
                "context needs one completed memory chunk plus a query packet"
            )
        history = history[:, : chunks * self.chunk_size]
        packet_slots = history.reshape(history.size(0), chunks, self.chunk_size, -1)
        scores = torch.einsum(
            "bmtd,sd->bmts", packet_slots, self.slot_queries
        ) / math.sqrt(packet_slots.size(-1))
        weights = scores.softmax(dim=2)
        slots = torch.einsum("bmts,bmtd->bmsd", weights, packet_slots)
        chunk_summary = slots.mean(dim=2)
        return (
            self.chunk_key(chunk_summary),
            self.slot_key(slots),
            self.slot_value(slots),
        )

    def write_completed_chunk(
        self, encoded_chunk: Tensor, cache: AhemV2MemoryCache, *, version: str
    ) -> None:
        """Project one causal completed chunk into the bounded runtime cache."""
        if encoded_chunk.shape != (self.chunk_size, self.local.project.out_features):
            raise ValueError("encoded chunk does not match AHEM-v2 configuration")
        key, slot_key, slot_value = self._write_chunks(encoded_chunk[None])
        cache.append(key[0, 0], slot_key[0, 0], slot_value[0, 0], version=version)

    def forward_from_cache(
        self,
        query_state: Tensor,
        cache: AhemV2MemoryCache,
        *,
        version: str,
        diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, RetrievalDiagnostics]:
        """Fuse one already-causal local state with cached completed chunks."""
        if (
            query_state.ndim != 2
            or query_state.size(1) != self.local.project.out_features
        ):
            raise ValueError("query state does not match AHEM-v2 configuration")
        if (
            cache.version != version
            or cache.keys is None
            or cache.slot_keys is None
            or cache.slot_values is None
        ):
            raise ValueError("no compatible completed chunks are available")
        keys, slot_keys, slot_values = (
            value.to(query_state.device)
            for value in (cache.keys, cache.slot_keys, cache.slot_values)
        )
        query = self.query(query_state)
        chunk_scores = torch.einsum("bd,md->bm", query, keys) / math.sqrt(
            query.size(-1)
        )
        count = min(self.top_k, keys.size(0))
        selected_scores, selected_chunks = chunk_scores.topk(count, dim=1)
        selected_keys = slot_keys[selected_chunks].flatten(1, 2)
        selected_values = slot_values[selected_chunks].flatten(1, 2)
        slot_scores = torch.einsum("bd,bkd->bk", query, selected_keys) / math.sqrt(
            query.size(-1)
        )
        slot_weights = slot_scores.softmax(dim=1)
        retrieved = torch.einsum("bk,bkd->bd", slot_weights, selected_values)
        mixed = torch.cat((query_state, retrieved), dim=-1)
        candidate = self.fuse(mixed)
        gate = torch.sigmoid(self.gate(mixed))
        output = self.predict(gate * candidate + (1 - gate) * query_state)
        if diagnostics:
            return output, RetrievalDiagnostics(
                selected_chunks, selected_scores.softmax(dim=1)
            )
        return output

    def forward(
        self, packets: Tensor, *, diagnostics: bool = False
    ) -> Tensor | tuple[Tensor, RetrievalDiagnostics]:
        encoded = self.encode_packets(packets)
        query_state = encoded[:, -1]
        chunk_keys, slot_keys, slot_values = self._write_chunks(encoded[:, :-1])
        query = self.query(query_state)
        chunk_scores = torch.einsum("bd,bmd->bm", query, chunk_keys) / math.sqrt(
            query.size(-1)
        )
        count = min(self.top_k, chunk_keys.size(1))
        selected_scores, selected_chunks = chunk_scores.topk(count, dim=1)
        key_gather = selected_chunks[:, :, None, None].expand(
            -1, -1, self.slots_per_chunk, slot_keys.size(-1)
        )
        value_gather = selected_chunks[:, :, None, None].expand(
            -1, -1, self.slots_per_chunk, slot_values.size(-1)
        )
        selected_keys = slot_keys.gather(1, key_gather).flatten(1, 2)
        selected_values = slot_values.gather(1, value_gather).flatten(1, 2)
        slot_scores = torch.einsum("bd,bkd->bk", query, selected_keys) / math.sqrt(
            query.size(-1)
        )
        slot_weights = slot_scores.softmax(dim=1)
        retrieved = torch.einsum("bk,bkd->bd", slot_weights, selected_values)
        mixed = torch.cat((query_state, retrieved), dim=-1)
        candidate = self.fuse(mixed)
        gate = torch.sigmoid(self.gate(mixed))
        output = self.predict(gate * candidate + (1 - gate) * query_state)
        if diagnostics:
            return output, RetrievalDiagnostics(
                selected_chunks, selected_scores.softmax(dim=1)
            )
        return output

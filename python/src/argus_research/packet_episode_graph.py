"""Deterministic causal packet episodes and bounded router neighborhoods.

Identity and packet membership are deliberately not learned.  A model may later
score the relationships emitted here, but it cannot move packets between flow
episodes or use observations later than the bundle decision time.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterable, Sequence

from torch import Tensor

PACKET_ID_SCHEMA = "argus.packet-identity/v1"
EPISODE_SCHEMA = "argus.causal-flow-episode/v1"
BUNDLE_SCHEMA = "argus.causal-episode-graph/v1"


def _stable_id(domain: str, value: object) -> str:
    encoded = json.dumps(
        {"domain": domain, "value": value},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def privacy_token(capture_salt: str, endpoint: str) -> str:
    """Create a capture-local relationship token without retaining an endpoint."""
    if not capture_salt or not endpoint:
        raise ValueError("capture salt and endpoint must be non-empty")
    return _stable_id("argus.endpoint-token/v1", [capture_salt, endpoint])


@dataclass(frozen=True)
class PacketRecord:
    capture_id: str
    interface_id: str
    flow_generation_id: str
    timestamp_ns: int
    source_ordinal: int
    original_length: int
    captured_length: int
    features: tuple[float, ...]
    conversation_id: str | None = None
    endpoint_tokens: tuple[str, ...] = ()
    protocol: str = "other"
    packets_lost_before: int = 0
    truncated: bool = False
    reordered: bool = False

    def __post_init__(self) -> None:
        if not self.capture_id or not self.interface_id or not self.flow_generation_id:
            raise ValueError("packet identity fields must be non-empty")
        if self.timestamp_ns < 0 or self.source_ordinal < 0:
            raise ValueError("packet timestamp and ordinal must be non-negative")
        if self.original_length < 0 or self.captured_length < 0:
            raise ValueError("packet lengths must be non-negative")
        if self.captured_length > self.original_length:
            raise ValueError("captured length exceeds original length")
        if len(self.features) != 12 or not all(math.isfinite(value) for value in self.features):
            raise ValueError("packet requires twelve finite causal features")
        if self.packets_lost_before < 0:
            raise ValueError("packets_lost_before must be non-negative")
        if len(set(self.endpoint_tokens)) != len(self.endpoint_tokens):
            raise ValueError("endpoint relationship tokens must be unique")

    @cached_property
    def packet_id(self) -> str:
        return _stable_id(
            PACKET_ID_SCHEMA,
            [
                self.capture_id,
                self.interface_id,
                self.source_ordinal,
                self.timestamp_ns,
                self.original_length,
                self.captured_length,
            ],
        )

    @property
    def closes_tcp_generation(self) -> bool:
        # Both supported 12-field packet contracts use FIN=8 and RST=9.
        return bool(self.features[8] >= 0.5 or self.features[9] >= 0.5)


@dataclass(frozen=True)
class EpisodeConfig:
    min_packets: int = 16
    max_packets: int = 256
    idle_timeout_ns: int = 2_000_000_000
    max_duration_ns: int = 15_000_000_000
    change_threshold: float = 4.5
    variance_floor: float = 0.05
    ema_alpha: float = 0.125
    max_open_flows: int = 16_384

    def __post_init__(self) -> None:
        if self.min_packets < 2 or self.max_packets < self.min_packets:
            raise ValueError("episode packet bounds are invalid")
        if self.idle_timeout_ns <= 0 or self.max_duration_ns <= 0:
            raise ValueError("episode time bounds must be positive")
        if self.change_threshold <= 0 or self.variance_floor <= 0:
            raise ValueError("change detector bounds must be positive")
        if not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.max_open_flows <= 0:
            raise ValueError("max_open_flows must be positive")

    @cached_property
    def fingerprint(self) -> str:
        return _stable_id(EPISODE_SCHEMA, self.__dict__)


@dataclass(frozen=True)
class CausalEpisode:
    episode_id: str
    capture_id: str
    interface_id: str
    flow_generation_id: str
    conversation_id: str | None
    endpoint_tokens: tuple[str, ...]
    protocol: str
    packet_ids: tuple[str, ...]
    packet_ordinals: tuple[int, ...]
    start_timestamp_ns: int
    end_timestamp_ns: int
    boundary_reason: str
    membership_sha256: str
    identity_health: str
    packets_lost_before: int
    truncated_packets: int
    reordered_packets: int
    summary: dict[str, float | int]

    @property
    def packet_count(self) -> int:
        return len(self.packet_ids)


@dataclass
class _OpenEpisode:
    packets: list[PacketRecord] = field(default_factory=list)
    mean: list[float] = field(default_factory=lambda: [0.0] * 12)
    variance: list[float] = field(default_factory=lambda: [0.0] * 12)

    def update_distribution(self, signature: Sequence[float], alpha: float) -> None:
        if len(self.packets) == 1:
            self.mean = list(signature)
            return
        for index, value in enumerate(signature):
            delta = value - self.mean[index]
            self.mean[index] += alpha * delta
            self.variance[index] = (1.0 - alpha) * (
                self.variance[index] + alpha * delta * delta
            )


class CausalEpisodeBuilder:
    """Assign every accepted packet to exactly one causal flow episode."""

    def __init__(self, config: EpisodeConfig | None = None) -> None:
        self.config = config or EpisodeConfig()
        self._open: OrderedDict[tuple[str, str, str], _OpenEpisode] = OrderedDict()
        self._seen_packet_ids: set[str] = set()
        self._last_ordinal: dict[tuple[str, str], int] = {}
        self.packets_seen = 0
        self.packets_finalized = 0

    @staticmethod
    def _signature(packet: PacketRecord) -> tuple[float, ...]:
        # The full payload-free packet-causal-flow-v2 vector is retained.  The
        # cumulative capture ordinal is removed from change detection because it
        # increases by construction and would create false boundaries.
        values = list(packet.features)
        values[11] = 0.0
        return tuple(values)

    def _change_score(self, state: _OpenEpisode, packet: PacketRecord) -> float:
        if len(state.packets) < self.config.min_packets:
            return 0.0
        signature = self._signature(packet)
        residuals = []
        for index, value in enumerate(signature):
            scale = math.sqrt(max(state.variance[index], self.config.variance_floor**2))
            residuals.append(min(10.0, abs(value - state.mean[index]) / scale))
        # A single flag transition should not split an episode; use the mean of
        # the four strongest residuals as a bounded multivariate change score.
        return sum(sorted(residuals, reverse=True)[:4]) / 4.0

    def _boundary_before(self, state: _OpenEpisode, packet: PacketRecord) -> str | None:
        previous = state.packets[-1]
        if packet.timestamp_ns < previous.timestamp_ns:
            raise ValueError("per-flow packet time moved backwards")
        if packet.source_ordinal <= previous.source_ordinal:
            raise ValueError("per-flow packet ordinal did not increase")
        if previous.closes_tcp_generation:
            return "tcp_generation_closed"
        if packet.timestamp_ns - previous.timestamp_ns > self.config.idle_timeout_ns:
            return "idle_timeout"
        if len(state.packets) >= self.config.max_packets:
            return "packet_limit"
        if packet.timestamp_ns - state.packets[0].timestamp_ns > self.config.max_duration_ns:
            return "duration_limit"
        if self._change_score(state, packet) >= self.config.change_threshold:
            return "causal_change_point"
        return None

    def ingest(self, packet: PacketRecord) -> list[CausalEpisode]:
        packet_id = packet.packet_id
        if packet_id in self._seen_packet_ids:
            raise ValueError("duplicate packet identity")
        capture_lane = (packet.capture_id, packet.interface_id)
        prior_ordinal = self._last_ordinal.get(capture_lane)
        if prior_ordinal is not None and packet.source_ordinal <= prior_ordinal:
            raise ValueError("capture source ordinal did not increase")
        self._last_ordinal[capture_lane] = packet.source_ordinal
        self._seen_packet_ids.add(packet_id)
        self.packets_seen += 1

        key = (packet.capture_id, packet.interface_id, packet.flow_generation_id)
        finalized: list[CausalEpisode] = []
        state = self._open.get(key)
        if state is not None:
            boundary = self._boundary_before(state, packet)
            if boundary is not None:
                finalized.append(self._finalize(key, state, boundary))
                state = None
        if state is None:
            while len(self._open) >= self.config.max_open_flows:
                oldest_key, oldest = self._open.popitem(last=False)
                finalized.append(self._finish(oldest_key, oldest, "flow_capacity"))
            state = _OpenEpisode()
            self._open[key] = state
        else:
            self._open.move_to_end(key)

        state.packets.append(packet)
        state.update_distribution(self._signature(packet), self.config.ema_alpha)
        if packet.closes_tcp_generation:
            finalized.append(self._finalize(key, state, "tcp_fin_or_rst"))
        return finalized

    def _finalize(
        self,
        key: tuple[str, str, str],
        state: _OpenEpisode,
        reason: str,
    ) -> CausalEpisode:
        self._open.pop(key, None)
        return self._finish(key, state, reason)

    def _finish(
        self,
        key: tuple[str, str, str],
        state: _OpenEpisode,
        reason: str,
    ) -> CausalEpisode:
        if not state.packets:
            raise RuntimeError("cannot finalize an empty episode")
        packet_ids = tuple(packet.packet_id for packet in state.packets)
        ordinals = tuple(packet.source_ordinal for packet in state.packets)
        membership = _stable_id("argus.episode-membership/v1", list(packet_ids))
        first = state.packets[0]
        last = state.packets[-1]
        episode_id = _stable_id(
            EPISODE_SCHEMA,
            [first.capture_id, first.interface_id, first.flow_generation_id, packet_ids[0], self.config.fingerprint],
        )
        lost = sum(packet.packets_lost_before for packet in state.packets)
        truncated = sum(packet.truncated for packet in state.packets)
        reordered = sum(packet.reordered for packet in state.packets)
        health = "nominal" if lost == truncated == reordered == 0 else "degraded"
        endpoints = tuple(sorted({token for packet in state.packets for token in packet.endpoint_tokens}))
        lengths = [packet.original_length for packet in state.packets]
        duration_ns = last.timestamp_ns - first.timestamp_ns
        self.packets_finalized += len(state.packets)
        return CausalEpisode(
            episode_id=episode_id,
            capture_id=key[0],
            interface_id=key[1],
            flow_generation_id=key[2],
            conversation_id=first.conversation_id,
            endpoint_tokens=endpoints,
            protocol=first.protocol,
            packet_ids=packet_ids,
            packet_ordinals=ordinals,
            start_timestamp_ns=first.timestamp_ns,
            end_timestamp_ns=last.timestamp_ns,
            boundary_reason=reason,
            membership_sha256=membership,
            identity_health=health,
            packets_lost_before=lost,
            truncated_packets=truncated,
            reordered_packets=reordered,
            summary={
                "packets": len(state.packets),
                "duration_seconds": duration_ns / 1_000_000_000.0,
                "mean_packet_bytes": sum(lengths) / len(lengths),
                "maximum_packet_bytes": max(lengths),
            },
        )

    def flush(self, reason: str = "capture_end") -> list[CausalEpisode]:
        finalized = [self._finish(key, state, reason) for key, state in self._open.items()]
        self._open.clear()
        return finalized

    def assert_complete(self) -> None:
        if self._open:
            raise RuntimeError("open episodes must be flushed before completeness validation")
        if self.packets_seen != self.packets_finalized:
            raise RuntimeError("packet accounting differs between ingestion and finalized episodes")


@dataclass(frozen=True)
class BundleConfig:
    maximum_neighbors: int = 8
    history_horizon_ns: int = 30_000_000_000
    temporal_decay_ns: int = 5_000_000_000

    def __post_init__(self) -> None:
        if self.maximum_neighbors < 0:
            raise ValueError("maximum_neighbors must be non-negative")
        if self.history_horizon_ns <= 0 or self.temporal_decay_ns <= 0:
            raise ValueError("bundle time bounds must be positive")

    @property
    def fingerprint(self) -> str:
        return _stable_id(BUNDLE_SCHEMA, self.__dict__)


@dataclass(frozen=True)
class EpisodeRelation:
    episode_id: str
    score: float
    kinds: tuple[str, ...]
    age_ns: int


@dataclass(frozen=True)
class RouterBundle:
    bundle_id: str
    anchor_episode_id: str
    decision_timestamp_ns: int
    decision_source_ordinal: int
    episode_ids: tuple[str, ...]
    packet_ids: tuple[str, ...]
    membership_sha256: str
    relations: tuple[EpisodeRelation, ...]
    identity_health: str


class CausalEpisodeGraph:
    """Build a bounded, past-only neighborhood for one finalized episode."""

    def __init__(self, config: BundleConfig | None = None) -> None:
        self.config = config or BundleConfig()

    def _relation(self, anchor: CausalEpisode, candidate: CausalEpisode) -> EpisodeRelation | None:
        if candidate.capture_id != anchor.capture_id or candidate.interface_id != anchor.interface_id:
            return None
        if candidate.packet_ordinals[-1] >= anchor.packet_ordinals[-1]:
            raise ValueError("future episode supplied to causal graph by source ordinal")
        if candidate.end_timestamp_ns > anchor.end_timestamp_ns:
            raise ValueError("future episode supplied to causal graph")
        age = anchor.end_timestamp_ns - candidate.end_timestamp_ns
        if age > self.config.history_horizon_ns:
            return None
        kinds = ["temporal"]
        score = math.exp(-age / self.config.temporal_decay_ns)
        if anchor.conversation_id and anchor.conversation_id == candidate.conversation_id:
            kinds.append("same_conversation")
            score += 2.0
        shared = set(anchor.endpoint_tokens).intersection(candidate.endpoint_tokens)
        if shared:
            kinds.append("shared_endpoint")
            score += 1.0 + min(0.5, 0.1 * len(shared))
        if anchor.protocol == candidate.protocol:
            kinds.append("same_protocol")
            score += 0.25
        return EpisodeRelation(candidate.episode_id, score, tuple(kinds), age)

    def build(self, anchor: CausalEpisode, history: Iterable[CausalEpisode]) -> RouterBundle:
        candidates = []
        by_id: dict[str, CausalEpisode] = {}
        for candidate in history:
            if candidate.episode_id == anchor.episode_id:
                continue
            relation = self._relation(anchor, candidate)
            if relation is not None:
                candidates.append(relation)
                by_id[candidate.episode_id] = candidate
        candidates.sort(key=lambda value: (-value.score, value.age_ns, value.episode_id))
        selected = tuple(candidates[: self.config.maximum_neighbors])
        episodes = [by_id[relation.episode_id] for relation in selected] + [anchor]
        episodes.sort(key=lambda value: (value.end_timestamp_ns, value.episode_id))
        episode_ids = tuple(episode.episode_id for episode in episodes)
        packet_ids = tuple(packet_id for episode in episodes for packet_id in episode.packet_ids)
        if len(packet_ids) != len(set(packet_ids)):
            raise ValueError("episode graph contains duplicate packet membership")
        membership = _stable_id("argus.bundle-membership/v1", list(packet_ids))
        bundle_id = _stable_id(
            BUNDLE_SCHEMA,
            [anchor.episode_id, anchor.end_timestamp_ns, list(episode_ids), self.config.fingerprint],
        )
        health = (
            "nominal"
            if all(episode.identity_health == "nominal" for episode in episodes)
            else "degraded"
        )
        return RouterBundle(
            bundle_id=bundle_id,
            anchor_episode_id=anchor.episode_id,
            decision_timestamp_ns=anchor.end_timestamp_ns,
            decision_source_ordinal=anchor.packet_ordinals[-1],
            episode_ids=episode_ids,
            packet_ids=packet_ids,
            membership_sha256=membership,
            relations=selected,
            identity_health=health,
        )


def records_from_tensor(
    *,
    capture_id: str,
    interface_id: str,
    flow_ids: Tensor,
    timestamps_ns: Tensor,
    source_ordinals: Tensor,
    features: Tensor,
) -> list[PacketRecord]:
    """Adapt a checksum-verified ARGUSLR2 record without inventing endpoints."""
    count = len(features)
    if features.shape != (count, 12):
        raise ValueError("features must have shape [packets, 12]")
    if not (len(flow_ids) == len(timestamps_ns) == len(source_ordinals) == count):
        raise ValueError("ARGUSLR2 metadata length differs")
    output = []
    for index in range(count):
        row = tuple(float(value) for value in features[index].tolist())
        length = max(0, round(math.expm1(max(0.0, row[0]))))
        if row[4] >= 0.5:
            protocol = "tcp"
        elif row[5] >= 0.5:
            protocol = "udp"
        else:
            protocol = "other"
        output.append(
            PacketRecord(
                capture_id=capture_id,
                interface_id=interface_id,
                flow_generation_id=f"{capture_id}:flow:{int(flow_ids[index])}",
                timestamp_ns=int(timestamps_ns[index]),
                source_ordinal=int(source_ordinals[index]),
                original_length=length,
                captured_length=length,
                features=row,
                protocol=protocol,
            )
        )
    return output

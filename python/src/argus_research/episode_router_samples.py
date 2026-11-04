"""Create causal episode-to-episode routing samples from packet records."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Sequence

from .packet_episode_graph import (
    BundleConfig,
    CausalEpisode,
    CausalEpisodeBuilder,
    CausalEpisodeGraph,
    EpisodeConfig,
    PacketRecord,
    RouterBundle,
)

SAMPLE_SCHEMA = "argus.episode-router-sample/v2"
_NO_SUCCESSOR = {"capture_end", "tcp_fin_or_rst", "tcp_generation_closed"}


@dataclass(frozen=True)
class EpisodeRoutingSample:
    """One online decision and its future-only supervision interval."""

    anchor: CausalEpisode
    bundle: RouterBundle
    bundle_episodes: tuple[CausalEpisode, ...]
    target_episode: CausalEpisode
    target_packets: tuple[PacketRecord, ...]


def build_episode_samples(
    packets: Sequence[PacketRecord],
    *,
    episode_config: EpisodeConfig | None = None,
    bundle_config: BundleConfig | None = None,
) -> tuple[EpisodeRoutingSample, ...]:
    """Build samples without exposing target packets to bundle construction.

    A routing decision is made after an episode ends. Its supervision interval
    is the immediately following episode from the same deterministic flow
    generation. Episodes ending at capture termination or TCP closure have no
    valid successor and are excluded.
    """
    if not packets:
        raise ValueError("episode sample source is empty")
    builder = CausalEpisodeBuilder(episode_config)
    packet_by_id: dict[str, PacketRecord] = {}
    episodes: list[CausalEpisode] = []
    for packet in packets:
        if packet.packet_id in packet_by_id:
            raise ValueError("episode sample source contains a duplicate packet")
        packet_by_id[packet.packet_id] = packet
        episodes.extend(builder.ingest(packet))
    episodes.extend(builder.flush())
    builder.assert_complete()

    by_flow: dict[tuple[str, str, str], list[CausalEpisode]] = defaultdict(list)
    for episode in episodes:
        by_flow[
            (episode.capture_id, episode.interface_id, episode.flow_generation_id)
        ].append(episode)

    successors: dict[str, CausalEpisode] = {}
    for values in by_flow.values():
        values.sort(key=lambda value: value.packet_ordinals[0])
        for anchor, target in zip(values, values[1:], strict=False):
            if anchor.boundary_reason in _NO_SUCCESSOR:
                continue
            if target.packet_ordinals[0] <= anchor.packet_ordinals[-1]:
                raise ValueError("episode successor does not follow its anchor")
            successors[anchor.episode_id] = target

    graph = CausalEpisodeGraph(bundle_config)
    config = graph.config
    samples: list[EpisodeRoutingSample] = []
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    by_lane: dict[tuple[str, str], list[CausalEpisode]] = defaultdict(list)
    for episode in episodes:
        by_lane[(episode.capture_id, episode.interface_id)].append(episode)
    for values in by_lane.values():
        values.sort(key=lambda value: value.packet_ordinals[-1])
        history: deque[CausalEpisode] = deque()
        for anchor in values:
            while (
                history
                and anchor.end_timestamp_ns - history[0].end_timestamp_ns
                > config.history_horizon_ns
            ):
                history.popleft()
            target = successors.get(anchor.episode_id)
            if target is not None:
                bundle = graph.build(anchor, tuple(history))
                targets = tuple(packet_by_id[packet_id] for packet_id in target.packet_ids)
                if not targets or targets[0].source_ordinal <= bundle.decision_source_ordinal:
                    raise RuntimeError("episode supervision is not future-only")
                samples.append(
                    EpisodeRoutingSample(
                        anchor=anchor,
                        bundle=bundle,
                        bundle_episodes=tuple(
                            episode_by_id[episode_id] for episode_id in bundle.episode_ids
                        ),
                        target_episode=target,
                        target_packets=targets,
                    )
                )
            history.append(anchor)
    samples.sort(key=lambda value: value.bundle.decision_source_ordinal)
    return tuple(samples)

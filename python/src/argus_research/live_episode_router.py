"""Build one causal v2 routing decision from a real runtime feature segment.

The feature exporter numbers flows within each segment.  We therefore keep
episode identity segment-local and only route when the current packet has a
completed earlier episode of the same flow in that segment.  No inferred
cross-segment flow identity is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address

import torch

from .packet_episode_graph import (
    CausalEpisodeBuilder,
    CausalEpisodeGraph,
    PacketRecord,
)

_TERMINAL = {"capture_end", "tcp_fin_or_rst", "tcp_generation_closed"}
_IDENTITY_FIELDS = (
    "flow_initiator_ip", "flow_initiator_port", "flow_responder_ip",
    "flow_responder_port", "flow_generation",
)


def runtime_flow_identity_status(rows: list[dict[str, str]]) -> dict[str, object]:
    """Audit observed tuple/generation metadata without claiming cross-segment identity.

    The PCAP parser restarts its generation counter for every file, so even a
    well-formed tuple is not a durable flow ID across live segments.
    """
    if not rows:
        return {"state": "no_packets", "cross_segment_ready": False}
    if all(all(field not in row for field in _IDENTITY_FIELDS) for row in rows):
        return {
            "state": "legacy_segment_only",
            "cross_segment_ready": False,
            "reason": "export has no observed endpoint tuple or generation metadata",
        }
    by_ordinal: dict[int, tuple[object, ...]] = {}
    by_identity: dict[tuple[object, ...], int] = {}
    try:
        for row in rows:
            if any(not row.get(field) for field in _IDENTITY_FIELDS):
                raise ValueError("partial observed flow identity")
            initiator = ip_address(row["flow_initiator_ip"])
            responder = ip_address(row["flow_responder_ip"])
            if initiator.version != responder.version:
                raise ValueError("mixed IP versions in one flow")
            if int(row["ip_version"]) != initiator.version:
                raise ValueError("flow IP version disagrees with observed addresses")
            if row["transport_protocol"] not in {"tcp", "udp", "other"}:
                raise ValueError("unsupported flow transport metadata")
            ports = (int(row["flow_initiator_port"]), int(row["flow_responder_port"]))
            if any(port < 0 or port > 65535 for port in ports):
                raise ValueError("flow port outside 0..65535")
            generation = int(row["flow_generation"])
            opaque = int(row["opaque_flow_id"])
            if generation < 0 or opaque < 0:
                raise ValueError("negative flow generation or ordinal")
            identity = (
                initiator, ports[0], responder, ports[1],
                row["transport_protocol"], generation,
            )
            previous = by_ordinal.setdefault(opaque, identity)
            if previous != identity:
                raise ValueError("one opaque flow ID maps to multiple identities")
            previous_ordinal = by_identity.setdefault(identity, opaque)
            if previous_ordinal != opaque:
                raise ValueError("one flow identity maps to multiple opaque IDs")
    except (KeyError, TypeError, ValueError) as error:
        return {
            "state": "invalid",
            "cross_segment_ready": False,
            "reason": str(error),
        }
    return {
        "state": "observed_segment_identity",
        "flow_instances": len(by_identity),
        "cross_segment_ready": False,
        "reason": "parser generation and opaque IDs restart for each capture segment",
    }


@dataclass(frozen=True)
class RuntimeEpisodeDecision:
    evidence: dict[str, object]
    context_indices: tuple[int, ...]
    decision_ordinal: int
    target_ordinal: int
    bundle_id: str


def build_runtime_episode_decision(
    features: torch.Tensor,
    rows: list[dict[str, str]],
    history_scope: str,
    *,
    context_packets: int = 128,
) -> RuntimeEpisodeDecision | None:
    """Find the latest completed same-flow anchor strictly before the target.

    The last row is the target for the desktop worker's retrospective forecast.
    Its feature vector never enters the episode graph or Laya evidence.
    """
    from .episode_router_supervision import episode_evidence

    if features.ndim != 2 or features.shape != (len(rows), 12):
        raise ValueError("runtime episode features require [rows, 12]")
    if not history_scope or not rows:
        return None
    if not torch.isfinite(features).all():
        raise ValueError("runtime episode features are non-finite")
    identity_status = runtime_flow_identity_status(rows)
    if identity_status["state"] == "invalid":
        raise ValueError(f"invalid runtime flow identity: {identity_status['reason']}")
    required = {
        "source_capture_ordinal", "monotonic_timestamp_ns", "opaque_flow_id",
        "packet_length_bytes", "captured_length_bytes", "transport_protocol",
        "packets_lost_before", "is_truncated", "is_reordered",
    }
    if any(not required.issubset(row) for row in rows):
        raise ValueError("runtime episode health or identity fields are absent")
    ordinals = [int(row["source_capture_ordinal"]) for row in rows]
    if any(right <= left for left, right in zip(ordinals, ordinals[1:])):
        raise ValueError("runtime episode ordinals are not strictly increasing")
    target_flow = rows[-1]["opaque_flow_id"]
    prior = tuple(
        index for index, row in enumerate(rows[:-1])
        if row["opaque_flow_id"] == target_flow
    )
    if len(prior) < context_packets:
        return None

    capture_id = f"{history_scope}:segment:{ordinals[0]}"
    feature_rows = features.tolist()
    builder = CausalEpisodeBuilder()
    finalized = []
    for index, row in enumerate(rows[:-1]):
        length = int(row["packet_length_bytes"])
        record = PacketRecord(
            capture_id=capture_id,
            interface_id="runtime-segment",
            flow_generation_id=f"segment-flow:{row['opaque_flow_id']}",
            timestamp_ns=int(row["monotonic_timestamp_ns"]),
            source_ordinal=ordinals[index],
            original_length=length,
            captured_length=int(row["captured_length_bytes"]),
            features=tuple(feature_rows[index]),
            protocol=row["transport_protocol"],
            packets_lost_before=int(row["packets_lost_before"]),
            truncated=bool(int(row["is_truncated"])),
            reordered=bool(int(row["is_reordered"])),
        )
        finalized.extend(builder.ingest(record))
    anchors = [
        episode for episode in finalized
        if episode.flow_generation_id == f"segment-flow:{target_flow}"
        and episode.boundary_reason not in _TERMINAL
        and episode.packet_ordinals[-1] < ordinals[-1]
    ]
    if not anchors:
        return None
    anchor = max(anchors, key=lambda episode: episode.packet_ordinals[-1])
    history = [
        episode for episode in finalized
        if episode.episode_id != anchor.episode_id
        and episode.packet_ordinals[-1] < anchor.packet_ordinals[-1]
    ]
    bundle = CausalEpisodeGraph().build(anchor, history)
    by_id = {episode.episode_id: episode for episode in [*history, anchor]}
    evidence = episode_evidence(anchor, bundle, by_id)
    return RuntimeEpisodeDecision(
        evidence=evidence,
        context_indices=prior[-context_packets:],
        decision_ordinal=bundle.decision_source_ordinal,
        target_ordinal=ordinals[-1],
        bundle_id=bundle.bundle_id,
    )

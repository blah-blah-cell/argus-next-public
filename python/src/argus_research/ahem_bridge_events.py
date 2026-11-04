"""Build causal, label-audited packet events for the frozen AHEM bridge.

This is an extraction command, not a detector trainer. It replays every
supported packet through the three frozen AHEM/LSTM pairs and writes aligned
event evidence only after the next same-flow target actually arrives.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import BinaryIO, Any

import dpkt
import numpy as np
import torch

from .ahem_detection_bridge import causal_base_features, load_frozen_plan
from .ahem_v6 import AhemV6StateTransitionNextPacket, FlowLaneLstmNextPacket
from .ahem_v6_stream import AhemV6Stream, FlowLaneLstmStream
from .cic_attack_schedule import match_attack_episode, normalize_attack_label
from .cic_flow_labels import FlowLabelIndex, infer_clock_offset, read_labeled_flows
from .cse_pcap_to_journal import (
    PCAP_MAGIC,
    PCAPNG_MAGIC,
    FlowGenerationTracker,
    ParsedPacket,
    _features,
    _parse_packet,
    _reader,
)
from .flow_v6_final_evaluate import _load_frozen_checkpoint

SEEDS = (20260906, 20260907, 20260908)
AHEM_NAME = "ahem_v6_state_transition"
LSTM_NAME = "flow_lane_lstm"
HORIZON_PACKETS = 10_000
EVENT_SCHEMA = "argus.ahem-v61-bridge-events/v1"
MANIFEST_SCHEMA = "argus.ahem-v61-bridge-event-manifest/v1"
EXPECTED_CHECKPOINT_SHA256 = {
    20260906: {
        AHEM_NAME: "0edff7f3be0045d82f6ae3e682393965a8942501102ed4fd2d84cdad5e56ad89",
        LSTM_NAME: "58ea4e5bc50156504036492e7ad3a5a20d0959949771f05ee9c28dcb1cc52c8e",
    },
    20260907: {
        AHEM_NAME: "04b0cc5ce778ac41c4ebab8887ee83b5bfee3287906eb3997dcdc248b0e4ed79",
        LSTM_NAME: "458ff246a6d89046e2daa184dc8c70d201ada06d35b12287f54fae454452798f",
    },
    20260908: {
        AHEM_NAME: "389b35bad8d69d83eff7e996c9480a0f249e3a42d1c39dc62c5765a0d61fd61b",
        LSTM_NAME: "7c77e078c06e32fd66e7fd9bca5c181a2c57a92ac6b8376c1f686369fa748c8b",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ExtractionStats:
    source_records: int = 0
    accepted_packets: int = 0
    unsupported_packets: int = 0
    malformed_packets: int = 0
    clock_rollbacks: int = 0
    warmup_packets: int = 0
    awaiting_same_flow_target: int = 0
    label_unmatched: int = 0
    label_ambiguous: int = 0
    schedule_disagreement: int = 0
    schedule_endpoint_disagreement: int = 0
    schedule_port_disagreement: int = 0
    schedule_ambiguous: int = 0
    emitted_events: int = 0
    discarded_truncated_capture_events: int = 0
    discarded_rollback_capture_events: int = 0
    truncated_capture_members: int = 0


@dataclass
class _FlowEvidence:
    count: int
    first_timestamp_ns: int
    previous_timestamp_ns: int


class _SeedResidualTracker:
    """Keep all three seed forecasts on-device until an event is retained."""

    def __init__(self, capture_id: str) -> None:
        self.capture_id = capture_id
        self.pending: dict[object, torch.Tensor] = {}

    def observe(
        self,
        flow: object,
        observed: torch.Tensor,
        predictions: torch.Tensor,
        *,
        capture_id: str,
    ) -> torch.Tensor | None:
        if capture_id != self.capture_id:
            raise ValueError("capture changed; explicitly reset residual state")
        if predictions.shape != (3, 2) or observed.shape != (2,):
            raise ValueError("seed residual inputs have incompatible shapes")
        previous = self.pending.pop(flow, None)
        self.pending[flow] = predictions.detach().clone()
        if previous is None:
            return None
        signed = observed[None] - previous
        absolute = signed.abs()
        smooth_l1 = torch.where(
            absolute < 1.0, 0.5 * signed.square(), absolute - 0.5
        )
        weighted = (smooth_l1[:, :1] + 4.0 * smooth_l1[:, 1:2]) / 5.0
        return torch.cat((signed, absolute, smooth_l1, weighted), dim=1)


class BridgeEventBuilder:
    """Stateful exact-stream event builder with no completed-flow features."""

    def __init__(
        self,
        ahem_models: Sequence[AhemV6StateTransitionNextPacket],
        lstm_models: Sequence[FlowLaneLstmNextPacket],
        label_index: FlowLabelIndex,
        *,
        expected_date: date,
        clock_offset_ns: int,
        device: torch.device,
        horizon_packets: int = HORIZON_PACKETS,
    ) -> None:
        if len(ahem_models) != 3 or len(lstm_models) != 3:
            raise ValueError("the bridge requires exactly three AHEM/LSTM seeds")
        if horizon_packets <= 0:
            raise ValueError("horizon_packets must be positive")
        if any(model.training for model in (*ahem_models, *lstm_models)):
            raise ValueError("bridge forecasters must be immutable eval-mode models")
        self.ahem_models = tuple(ahem_models)
        self.lstm_models = tuple(lstm_models)
        self.label_index = label_index
        self.expected_date = expected_date
        self.clock_offset_ns = clock_offset_ns
        self.device = device
        self.horizon_packets = horizon_packets
        self.stats = ExtractionStats()
        self.base: list[np.ndarray] = []
        self.ahem_residuals: list[np.ndarray] = []
        self.lstm_residuals: list[np.ndarray] = []
        self.labels: list[int] = []
        self.attack_families: list[str] = []
        self.episode_ids: list[str] = []
        self.capture_ids: list[str] = []
        self.flow_ids: list[str] = []
        self.timestamps_ns: list[int] = []
        self.source_ordinals: list[int] = []

    def reset_capture(self, capture_id: str, *, capture_group: str | None = None) -> None:
        if not capture_id:
            raise ValueError("capture_id must be non-empty")
        self.capture_id = capture_id
        self.capture_group = capture_group or capture_id
        self.ordinal = 0
        self.last_raw_timestamp_ns: int | None = None
        self.flows = FlowGenerationTracker()
        self.flow_evidence: dict[object, _FlowEvidence] = {}
        self.flow_identifiers: dict[object, str] = {}
        self.label_matcher = self.label_index.matcher(
            clock_offset_ns=self.clock_offset_ns
        )
        self.ahem_streams = tuple(
            AhemV6Stream(model, capture_id, horizon_packets=self.horizon_packets)
            for model in self.ahem_models
        )
        self.lstm_streams = tuple(
            FlowLaneLstmStream(model, capture_id) for model in self.lstm_models
        )
        self.ahem_tracker = _SeedResidualTracker(capture_id)
        self.lstm_tracker = _SeedResidualTracker(capture_id)

    def checkpoint(self) -> int:
        return len(self.labels)

    def discard_since(self, checkpoint: int) -> int:
        if not 0 <= checkpoint <= len(self.labels):
            raise ValueError("invalid event checkpoint")
        discarded = len(self.labels) - checkpoint
        for values in (
            self.base,
            self.ahem_residuals,
            self.lstm_residuals,
            self.labels,
            self.attack_families,
            self.episode_ids,
            self.capture_ids,
            self.flow_ids,
            self.timestamps_ns,
            self.source_ordinals,
        ):
            del values[checkpoint:]
        self.stats.emitted_events -= discarded
        return discarded

    def process(
        self, packet: ParsedPacket, raw_timestamp_ns: int, source_ordinal: int
    ) -> bool:
        if not hasattr(self, "capture_id"):
            raise ValueError("reset_capture must be called before processing")
        if type(raw_timestamp_ns) is not int or raw_timestamp_ns < 0:
            raise ValueError("raw packet timestamp must be non-negative integer ns")
        if (
            self.last_raw_timestamp_ns is not None
            and raw_timestamp_ns < self.last_raw_timestamp_ns
        ):
            raise ValueError("capture clock rollback requires a new capture segment")
        self.last_raw_timestamp_ns = raw_timestamp_ns
        flow = self.flows.observe(
            packet.flow_key, raw_timestamp_ns, closes_flow=packet.closes_flow
        )
        evidence = self.flow_evidence.get(flow)
        if evidence is None:
            evidence = _FlowEvidence(0, raw_timestamp_ns, raw_timestamp_ns)
            self.flow_evidence[flow] = evidence
        interarrival = max(0.0, (raw_timestamp_ns - evidence.previous_timestamp_ns) / 1e9)
        evidence.count += 1
        evidence.previous_timestamp_ns = raw_timestamp_ns
        packet_features = np.asarray(_features(packet, interarrival), dtype=np.float32)
        tensor = torch.as_tensor(packet_features, device=self.device)

        ahem_outputs = [
            stream.step(tensor, flow, self.ordinal, capture_id=self.capture_id)
            for stream in self.ahem_streams
        ]
        lstm_outputs = [
            stream.step(tensor, flow, self.ordinal, capture_id=self.capture_id)
            for stream in self.lstm_streams
        ]
        self.ordinal += 1
        self.stats.accepted_packets += 1
        label_match = self.label_matcher.observe(packet.flow_key, raw_timestamp_ns)

        if any(output is None for output in ahem_outputs):
            self.stats.warmup_packets += 1
            return False
        ahem_predictions = torch.stack(
            [output.prediction[0] for output in ahem_outputs]  # type: ignore[union-attr]
        )
        lstm_predictions = torch.stack(lstm_outputs)
        ahem_residuals = self.ahem_tracker.observe(
            flow,
            tensor[:2],
            ahem_predictions,
            capture_id=self.capture_id,
        )
        lstm_residuals = self.lstm_tracker.observe(
            flow,
            tensor[:2],
            lstm_predictions,
            capture_id=self.capture_id,
        )
        if ahem_residuals is None or lstm_residuals is None:
            self.stats.awaiting_same_flow_target += 1
            return False
        if label_match.status == "unmatched":
            self.stats.label_unmatched += 1
            return False
        if label_match.status == "ambiguous":
            self.stats.label_ambiguous += 1
            return False
        assert label_match.flow is not None
        wall_timestamp_ns = raw_timestamp_ns + self.clock_offset_ns
        episode_match = match_attack_episode(
            label_match.flow.label,
            wall_timestamp_ns,
            self.expected_date,
            packet_key=packet.flow_key,
        )
        if episode_match.status == "schedule_disagreement":
            self.stats.schedule_disagreement += 1
            return False
        if episode_match.status == "schedule_endpoint_disagreement":
            self.stats.schedule_endpoint_disagreement += 1
            return False
        if episode_match.status == "schedule_port_disagreement":
            self.stats.schedule_port_disagreement += 1
            return False
        if episode_match.status == "schedule_ambiguous":
            self.stats.schedule_ambiguous += 1
            return False

        age = (raw_timestamp_ns - evidence.first_timestamp_ns) / 1e9
        base = causal_base_features(
            packet_features[None],
            np.asarray([evidence.count], dtype=np.int64),
            np.asarray([age], dtype=np.float64),
        )[0]
        flow_id = self.flow_identifiers.setdefault(
            flow, f"{self.capture_id}:flow-{len(self.flow_identifiers) + 1}"
        )
        attack = int(episode_match.status == "attack")
        self.base.append(base.astype(np.float32))
        residual_pair = (
            torch.stack((ahem_residuals, lstm_residuals)).detach().cpu().numpy()
        )
        self.ahem_residuals.append(residual_pair[0].astype(np.float32))
        self.lstm_residuals.append(residual_pair[1].astype(np.float32))
        self.labels.append(attack)
        self.attack_families.append(
            normalize_attack_label(label_match.flow.label) if attack else "benign"
        )
        self.episode_ids.append(
            episode_match.episode.episode_id if episode_match.episode else ""
        )
        self.capture_ids.append(self.capture_group)
        self.flow_ids.append(flow_id)
        self.timestamps_ns.append(wall_timestamp_ns)
        self.source_ordinals.append(source_ordinal)
        self.stats.emitted_events += 1
        return True

    def arrays(self) -> dict[str, np.ndarray]:
        count = len(self.labels)
        if count == 0:
            raise ValueError("no eligible bridge events were emitted")
        capture_ids = np.asarray(self.capture_ids)
        timestamps_ns = np.asarray(self.timestamps_ns, dtype=np.int64)
        order = np.lexsort((timestamps_ns, capture_ids))
        return {
            "schema": np.asarray(EVENT_SCHEMA),
            "base": np.stack(self.base).astype(np.float32)[order],
            "ahem_residuals": np.stack(self.ahem_residuals).astype(np.float32)[order],
            "lstm_residuals": np.stack(self.lstm_residuals).astype(np.float32)[order],
            "labels": np.asarray(self.labels, dtype=np.int8)[order],
            "attack_families": np.asarray(self.attack_families)[order],
            "episode_ids": np.asarray(self.episode_ids)[order],
            "capture_ids": capture_ids[order],
            "flow_ids": np.asarray(self.flow_ids)[order],
            "timestamps_ns": timestamps_ns[order],
            "source_ordinals": np.asarray(self.source_ordinals, dtype=np.int64)[order],
        }


def load_forecasters(
    checkpoint_root: Path, device: torch.device
) -> tuple[
    tuple[AhemV6StateTransitionNextPacket, ...],
    tuple[FlowLaneLstmNextPacket, ...],
    dict[str, dict[str, Any]],
]:
    ahem: list[AhemV6StateTransitionNextPacket] = []
    lstm: list[FlowLaneLstmNextPacket] = []
    identities: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        for name, destination in ((AHEM_NAME, ahem), (LSTM_NAME, lstm)):
            model, digest, identity = _load_frozen_checkpoint(
                checkpoint_root,
                seed,
                name,
                EXPECTED_CHECKPOINT_SHA256[seed][name],
            )
            model = model.to(device)
            destination.append(model)  # type: ignore[arg-type]
            identities[f"{seed}/{name}"] = {"sha256": digest, **identity}
    return tuple(ahem), tuple(lstm), identities


def _archive_members(path: Path, member_pattern: str) -> tuple[str, list[str]]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as bundle:
            names = [
                item.filename
                for item in bundle.infolist()
                if not item.is_dir() and item.file_size >= 24
            ]
        format_ = "zip"
    elif path.suffix.casefold() == ".rar":
        unrar = shutil.which("unrar")
        if unrar is None:
            raise RuntimeError("RAR extraction requires the unrar executable")
        result = subprocess.run(
            [unrar, "lb", "-c-", str(path)], check=True, capture_output=True, text=True
        )
        names = [line.strip() for line in result.stdout.splitlines()]
        format_ = "rar"
    else:
        with path.open("rb") as source:
            magic = source.read(4)
        if magic not in PCAP_MAGIC | {PCAPNG_MAGIC}:
            raise ValueError("input must be a PCAP/PCAPNG file or ZIP/RAR archive")
        return "raw", [path.name]
    matches = sorted(
        name
        for name in names
        if Path(name).suffix.casefold() in {".pcap", ".pcapng"}
        and fnmatch.fnmatch(name.casefold(), member_pattern.casefold())
    )
    if not matches:
        raise ValueError("no capture members match the requested pattern")
    return format_, matches


@contextmanager
def _open_member(
    path: Path, format_: str, member: str
) -> Iterator[BinaryIO]:
    if format_ == "raw":
        with path.open("rb") as source:
            yield source
        return
    if format_ == "zip":
        with zipfile.ZipFile(path) as bundle, bundle.open(member) as source:
            yield source
        return
    unrar = shutil.which("unrar")
    if unrar is None:
        raise RuntimeError("RAR extraction requires the unrar executable")
    process = subprocess.Popen(
        [unrar, "p", "-inul", "-c-", str(path), member],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        process.kill()
        raise RuntimeError("unrar did not expose a capture stream")
    try:
        yield process.stdout
    finally:
        process.stdout.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        returncode = process.wait(timeout=30)
        if returncode != 0:
            raise RuntimeError(f"unrar failed for {member!r}: {stderr.strip()}")


def _clock_sample(
    archive: Path, format_: str, member: str, limit: int
) -> list[tuple[tuple[bytes, bytes, int, int, int], int]]:
    packets: list[tuple[tuple[bytes, bytes, int, int, int], int]] = []
    prior = -1
    with _open_member(archive, format_, member) as source:
        for timestamp, frame in _reader(source):
            try:
                packet = _parse_packet(frame)
            except ValueError:
                continue
            if packet is None or not math.isfinite(timestamp) or timestamp < 0:
                continue
            timestamp_ns = round(float(timestamp) * 1_000_000_000)
            if timestamp_ns < prior:
                break
            prior = timestamp_ns
            packets.append((packet.flow_key, timestamp_ns))
            if len(packets) >= limit:
                break
    if not packets:
        raise ValueError("capture produced no clock-alignment packet sample")
    return packets


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"event output already exists: {path}")
    with temporary.open("xb") as destination:
        np.savez_compressed(destination, **arrays)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)
    return _sha256(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"manifest output already exists: {path}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as destination:
        destination.write(encoded)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def extract(args: argparse.Namespace) -> dict[str, Any]:
    load_frozen_plan(args.plan)
    if args.max_packets < 0 or args.clock_sample_packets <= 0:
        raise ValueError("packet limits must be non-negative")
    expected_date = date.fromisoformat(args.date)
    label_paths = tuple(args.labels)
    external_test_dates = {
        date(2017, 7, 4),
        date(2017, 7, 5),
        date(2017, 7, 6),
        date(2017, 7, 7),
    }
    access_ledger = getattr(args, "access_ledger", None)
    if expected_date in external_test_dates:
        if access_ledger is None:
            raise ValueError("external-test extraction requires --access-ledger")
        # This immutable marker is intentionally written before reading an
        # archive byte, hashing an archive, or parsing a labeled-flow row.
        _atomic_json(
            access_ledger,
            {
                "schema": "argus.ahem-v61-detection-bridge-access/v1",
                "state": "external_test_access_committed",
                "date": expected_date.isoformat(),
                "plan_sha256": _sha256(args.plan),
                "archive_name": args.archive.name,
                "archive_bytes": args.archive.stat().st_size,
                "label_files": [
                    {"name": path.name, "bytes": path.stat().st_size}
                    for path in label_paths
                ],
                "command": [Path(sys.executable).name, *sys.argv],
                "warning": "A crash after this marker spends the frozen attempt and must be reported.",
            },
        )
    labeled_flows = [
        flow
        for path in label_paths
        for flow in read_labeled_flows(path, expected_date=expected_date)
    ]
    label_index = FlowLabelIndex(labeled_flows)
    format_, members = _archive_members(args.archive, args.member_pattern)
    clock = infer_clock_offset(
        label_index,
        _clock_sample(
            args.archive, format_, members[0], args.clock_sample_packets
        ),
    )
    device = torch.device(args.device)
    ahem, lstm, checkpoint_identities = load_forecasters(args.checkpoints, device)
    builder = BridgeEventBuilder(
        ahem,
        lstm,
        label_index,
        expected_date=expected_date,
        clock_offset_ns=int(clock["clock_offset_ns"]),
        device=device,
    )
    processed = 0
    monitored_seconds = 0.0
    member_results: list[dict[str, Any]] = []
    for member in members:
        if args.max_packets and processed >= args.max_packets:
            break
        capture_id = f"{expected_date.isoformat()}:{member}"
        builder.reset_capture(capture_id, capture_group=expected_date.isoformat())
        event_checkpoint = builder.checkpoint()
        truncated = False
        rollback_affected = False
        segment = 0
        prior_raw_ns: int | None = None
        segment_start_ns: int | None = None
        segment_end_ns: int | None = None
        member_monitored_ns = 0
        with _open_member(args.archive, format_, member) as source:
            try:
                for source_ordinal, (timestamp, frame) in enumerate(_reader(source)):
                    builder.stats.source_records += 1
                    try:
                        packet = _parse_packet(frame)
                    except ValueError:
                        builder.stats.malformed_packets += 1
                        continue
                    if packet is None:
                        builder.stats.unsupported_packets += 1
                        continue
                    if not math.isfinite(timestamp) or timestamp < 0:
                        builder.stats.malformed_packets += 1
                        continue
                    raw_ns = round(float(timestamp) * 1_000_000_000)
                    if prior_raw_ns is not None and raw_ns < prior_raw_ns:
                        builder.stats.clock_rollbacks += 1
                        rollback_affected = True
                        if segment_start_ns is not None and segment_end_ns is not None:
                            member_monitored_ns += segment_end_ns - segment_start_ns
                        segment += 1
                        capture_id = f"{expected_date.isoformat()}:{member}#rollback-{segment}"
                        builder.reset_capture(
                            capture_id, capture_group=expected_date.isoformat()
                        )
                        segment_start_ns = None
                    if segment_start_ns is None:
                        segment_start_ns = raw_ns
                    segment_end_ns = raw_ns
                    prior_raw_ns = raw_ns
                    builder.process(packet, raw_ns, source_ordinal)
                    processed += 1
                    if args.max_packets and processed >= args.max_packets:
                        break
            except dpkt.dpkt.NeedData:
                truncated = True
                builder.stats.truncated_capture_members += 1
        if segment_start_ns is not None and segment_end_ns is not None:
            member_monitored_ns += segment_end_ns - segment_start_ns
        discarded = 0
        if truncated or rollback_affected:
            discarded = builder.discard_since(event_checkpoint)
            if truncated:
                builder.stats.discarded_truncated_capture_events += discarded
            else:
                builder.stats.discarded_rollback_capture_events += discarded
        else:
            monitored_seconds += member_monitored_ns / 1e9
        member_results.append(
            {
                "member": member,
                "truncated": truncated,
                "clock_rollback_affected": rollback_affected,
                "monitored_seconds": (
                    0.0 if truncated or rollback_affected else member_monitored_ns / 1e9
                ),
                "events_discarded": discarded,
            }
        )
    arrays = builder.arrays()
    digest = _atomic_npz(args.output, arrays)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "event_schema": EVENT_SCHEMA,
        "scientific_eligible": args.max_packets == 0,
        "scientific_ineligibility_reason": (
            None if args.max_packets == 0 else "preflight packet cap was active"
        ),
        "frozen_plan_sha256": _sha256(args.plan),
        "archive": {"path": args.archive.name, "sha256": _sha256(args.archive)},
        "labels": [
            {"path": path.name, "sha256": _sha256(path)} for path in label_paths
        ],
        "expected_date": expected_date.isoformat(),
        "archive_format": format_,
        "members": member_results,
        "complete_monitored_capture_seconds": monitored_seconds,
        "clock_alignment": clock,
        "checkpoints": checkpoint_identities,
        "event_npz": {
            "path": args.output.name,
            "sha256": digest,
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in arrays.items()
            },
        },
        "stats": asdict(builder.stats),
        "contracts": {
            "per_flow_interarrival": True,
            "future_or_completed_flow_features": False,
            "model_training_or_finetuning": False,
            "lstm_state": "ordinary full per-flow recurrent state reset at capture boundary",
            "ahem_horizon_packets": HORIZON_PACKETS,
            "attack_label_requires_schedule_and_identity_flow_agreement": True,
        },
    }
    _atomic_json(args.output.with_suffix(args.output.suffix + ".manifest.json"), manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, nargs="+")
    parser.add_argument("--date", required=True, help="exact capture date, YYYY-MM-DD")
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--access-ledger",
        type=Path,
        help="required fresh immutable path for CIC-IDS2017 Tuesday-Friday",
    )
    parser.add_argument("--member-pattern", default="*")
    parser.add_argument("--clock-sample-packets", type=int, default=50_000)
    parser.add_argument(
        "--max-packets",
        type=int,
        default=0,
        help="engineering preflight cap; any nonzero value marks output ineligible",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def main() -> None:
    manifest = extract(build_parser().parse_args())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Bounded shadow learning from real daily packet features.

This module never assigns intrusion labels and never replaces the active model.
It learns only the causal next-packet forecasting task and writes a challenger
artifact whose promotion remains an explicit, reviewed operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import struct
import subprocess
import sys
import time
from array import array
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .ahem_pretrain import CausalCnnNextPacket

MAGIC = b"ARGUSLR1"
MAGIC_V2 = b"ARGUSLR2"
FEATURE_DIM = 12
V2_PACKET_BYTES = FEATURE_DIM * 4 + 8 + 8 + 8
MAX_DAYS = 7
MAX_DAY_BYTES = 32 * 1024 * 1024
MAX_HOUR_BYTES = MAX_DAY_BYTES // 24
MIN_TRAIN_PACKETS = 8_192
MIN_RETRY_NEW_PACKETS = 4_096
TRAIN_INTERVAL_SECONDS = 20 * 60 * 60
CONTEXT_PACKETS = 31


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _pack_u64(values: Sequence[int]) -> bytes:
    packed = array("Q", values)
    if packed.itemsize != 8:
        raise RuntimeError(
            "the current Python runtime lacks an eight-byte unsigned array"
        )
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def _journal_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.arguslr"))


def iter_segments_with_source(
    directory: Path,
) -> Iterator[tuple[str, str, Tensor]]:
    """Read complete journal records with their source-file identity."""
    for path in _journal_files(directory):
        with path.open("rb") as source:
            if source.read(len(MAGIC)) != MAGIC:
                continue
            while True:
                count_bytes = source.read(4)
                if not count_bytes:
                    break
                digest = source.read(32)
                if len(count_bytes) != 4 or len(digest) != 32:
                    break
                count = struct.unpack("<I", count_bytes)[0]
                if count == 0 or count > 1_000_000:
                    break
                payload = source.read(count * FEATURE_DIM * 4)
                if len(payload) != count * FEATURE_DIM * 4:
                    break
                values = array("f")
                values.frombytes(payload)
                if sys.byteorder != "little":
                    values.byteswap()
                yield (
                    path.stem,
                    digest.hex(),
                    torch.tensor(values, dtype=torch.float32).reshape(
                        count, FEATURE_DIM
                    ),
                )


def iter_segments(directory: Path) -> Iterator[tuple[str, Tensor]]:
    """Read complete journal records; a partial tail is ignored fail-closed."""
    for _, identifier, values in iter_segments_with_source(directory):
        yield identifier, values


@dataclass(frozen=True)
class SequenceRecordV2:
    """One globally ordered segment with explicit segment-local flow identity."""

    source_file: str
    identifier: str
    features: Tensor
    flow_ids: Tensor
    timestamps_ns: Tensor
    source_ordinals: Tensor


@dataclass(frozen=True)
class FlowSequenceV2:
    """One record-bounded flow lane with its split-group provenance."""

    source_file: str
    segment_id: str
    flow_id: int
    features: Tensor


def iter_sequence_records_v2(directory: Path) -> Iterator[SequenceRecordV2]:
    """Read checksum-verified v2 records without treating v1 as v2 data."""
    for path in _journal_files(directory):
        with path.open("rb") as source:
            if source.read(len(MAGIC_V2)) != MAGIC_V2:
                continue
            while True:
                count_bytes = source.read(4)
                if not count_bytes:
                    break
                identifier_digest = source.read(32)
                payload_digest = source.read(32)
                if (
                    len(count_bytes) != 4
                    or len(identifier_digest) != 32
                    or len(payload_digest) != 32
                ):
                    break
                count = struct.unpack("<I", count_bytes)[0]
                if count == 0 or count > 1_000_000:
                    break
                payload = source.read(count * V2_PACKET_BYTES)
                if len(payload) != count * V2_PACKET_BYTES:
                    break
                if hashlib.sha256(payload).digest() != payload_digest:
                    break

                feature_bytes = count * FEATURE_DIM * 4
                integer_bytes = count * 8
                features_raw = payload[:feature_bytes]
                flow_start = feature_bytes
                timestamp_start = flow_start + integer_bytes
                ordinal_start = timestamp_start + integer_bytes

                values = array("f")
                values.frombytes(features_raw)
                if sys.byteorder != "little":
                    values.byteswap()
                flow_ids = [
                    value[0]
                    for value in struct.iter_unpack(
                        "<Q", payload[flow_start:timestamp_start]
                    )
                ]
                timestamps = [
                    value[0]
                    for value in struct.iter_unpack(
                        "<Q", payload[timestamp_start:ordinal_start]
                    )
                ]
                ordinals = [
                    value[0]
                    for value in struct.iter_unpack("<Q", payload[ordinal_start:])
                ]
                if any(right < left for left, right in zip(timestamps, timestamps[1:])):
                    break
                if any(right <= left for left, right in zip(ordinals, ordinals[1:])):
                    break
                yield SequenceRecordV2(
                    source_file=path.stem,
                    identifier=identifier_digest.hex(),
                    features=torch.tensor(values, dtype=torch.float32).reshape(
                        count, FEATURE_DIM
                    ),
                    flow_ids=torch.tensor(flow_ids, dtype=torch.int64),
                    timestamps_ns=torch.tensor(timestamps, dtype=torch.int64),
                    source_ordinals=torch.tensor(ordinals, dtype=torch.int64),
                )


def iter_global_sequences_v2(
    directory: Path,
) -> Iterator[tuple[str, Tensor]]:
    """Yield the explicit global-context lane in capture order."""
    for record in iter_sequence_records_v2(directory):
        yield record.identifier, record.features


def flow_sequences_from_record(record: SequenceRecordV2) -> list[FlowSequenceV2]:
    """Derive causal per-flow lanes without crossing the parent record."""
    grouped: dict[int, list[int]] = {}
    for index, flow_id in enumerate(record.flow_ids.tolist()):
        grouped.setdefault(flow_id, []).append(index)
    sequences = []
    for flow_id, indices in grouped.items():
        stream = record.features[indices].clone()
        prior_timestamp: int | None = None
        for local_index, packet_index in enumerate(indices):
            timestamp = int(record.timestamps_ns[packet_index])
            delta_seconds = (
                0.0
                if prior_timestamp is None
                else max(
                    0.0,
                    min((timestamp - prior_timestamp) / 1_000_000_000.0, 60.0),
                )
            )
            stream[local_index, 1] = math.log1p(delta_seconds)
            prior_timestamp = timestamp
        sequences.append(
            FlowSequenceV2(
                source_file=record.source_file,
                segment_id=record.identifier,
                flow_id=flow_id,
                features=stream,
            )
        )
    return sequences


def iter_flow_sequences_v2(
    directory: Path,
) -> Iterator[tuple[str, Tensor]]:
    """Yield isolated per-flow streams with causal per-flow interarrival time."""
    for record in iter_sequence_records_v2(directory):
        for sequence in flow_sequences_from_record(record):
            yield (
                f"{sequence.segment_id}:flow:{sequence.flow_id}",
                sequence.features,
            )


def _group_flow_records_by_segment(
    records: Sequence[tuple[str, Tensor]],
) -> list[tuple[str, list[Tensor]]]:
    grouped: dict[str, list[Tensor]] = {}
    for stream_id, values in records:
        segment_id, separator, _ = stream_id.partition(":flow:")
        if not separator or not segment_id:
            raise ValueError("v2 flow stream lacks its segment identity")
        grouped.setdefault(segment_id, []).append(values)
    return list(grouped.items())


class DailyReplayJournal:
    """Append-only v2 replay journal with separate flow and global lanes."""

    def __init__(self, directory: Path, *, spawn_training: bool = True) -> None:
        self.directory = directory
        self.status_path = directory / "status.json"
        self.spawn_training = spawn_training
        self.directory.mkdir(parents=True, exist_ok=True)
        self._prune()
        records = list(iter_sequence_records_v2(directory))
        self._seen = {record.identifier for record in records}
        self.retained_packets = sum(len(record.features) for record in records)

    def _prune(self) -> None:
        files = _journal_files(self.directory)
        for stale in files[: -(MAX_DAYS * 24)]:
            stale.unlink(missing_ok=True)

    def append(
        self,
        segment_key: str,
        features: Tensor,
        packet_rows: Sequence[Mapping[str, str]],
    ) -> dict[str, Any]:
        digest = hashlib.sha256(segment_key.encode("utf-8")).digest()
        identifier = digest.hex()
        if identifier in self._seen:
            return self.status()
        values = features.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if (
            values.ndim != 2
            or values.shape[1] != FEATURE_DIM
            or not values.isfinite().all()
        ):
            raise ValueError(
                "daily replay accepts only finite packet-causal-v2 tensors"
            )
        if len(packet_rows) != len(values):
            raise ValueError("v2 journal metadata count differs from feature count")
        try:
            flow_ids = [int(row["opaque_flow_id"]) for row in packet_rows]
            timestamps = [int(row["monotonic_timestamp_ns"]) for row in packet_rows]
            ordinals = [int(row["source_capture_ordinal"]) for row in packet_rows]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("v2 journal metadata is incomplete or invalid") from error
        maximum_i64 = (1 << 63) - 1
        if any(value < 0 or value > maximum_i64 for value in flow_ids):
            raise ValueError("v2 journal flow identity exceeds the runtime contract")
        if any(value < 0 or value > maximum_i64 for value in timestamps + ordinals):
            raise ValueError(
                "v2 journal timestamp or ordinal exceeds the runtime contract"
            )
        if any(right < left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("v2 journal timestamps must be globally monotonic")
        if any(right <= left for left, right in zip(ordinals, ordinals[1:])):
            raise ValueError("v2 journal source ordinals must be strictly increasing")
        hour = datetime.now(UTC).strftime("%Y-%m-%dT%H")
        path = self.directory / f"{hour}.v2.arguslr"
        feature_payload = values.numpy().astype("<f4", copy=False).tobytes()
        payload = b"".join(
            (
                feature_payload,
                _pack_u64(flow_ids),
                _pack_u64(timestamps),
                _pack_u64(ordinals),
            )
        )
        payload_digest = hashlib.sha256(payload).digest()
        record_size = 4 + len(digest) + len(payload_digest) + len(payload)
        existing_size = path.stat().st_size if path.exists() else len(MAGIC_V2)
        if existing_size + record_size > MAX_HOUR_BYTES:
            state = self.status()
            state["state"] = "collection_capped"
            state["detail"] = (
                "hourly feature budget reached; collection resumes next UTC hour"
            )
            state["dailyBytesLimit"] = MAX_DAY_BYTES
            state["hourlyBytesLimit"] = MAX_HOUR_BYTES
            _atomic_json(self.status_path, state)
            return state
        with path.open("ab") as output:
            if output.tell() == 0:
                output.write(MAGIC_V2)
            output.write(struct.pack("<I", len(values)))
            output.write(digest)
            output.write(payload_digest)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        self._seen.add(identifier)
        self.retained_packets += len(values)
        state = self.status()
        total = int(state.get("totalCollectedPackets", 0)) + len(values)
        update = {
            "schema": "argus.daily-learning-status/v2",
            "task": "self_supervised_next_packet_forecast_only",
            "sequenceContract": "argus.packet-sequence/v2",
            "featureSchema": "packet-causal-flow-v2",
            "flowIdentityScope": "segment_local_generation",
            "globalContextLane": True,
            "securityLabelsLearned": False,
            "autoPromotion": False,
            "retainedPackets": self.retained_packets,
            "totalCollectedPackets": total,
            "newPacketsSinceTraining": total
            - int(state.get("trainedThroughPackets", 0)),
            "retentionDays": MAX_DAYS,
            "dailyBytesLimit": MAX_DAY_BYTES,
            "hourlyBytesLimit": MAX_HOUR_BYTES,
            "lastCollectionUtc": datetime.now(UTC).isoformat(),
        }
        if state.get("state") not in {
            "queued",
            "training",
            "review_ready",
            "challenger_rejected",
        }:
            update["state"] = "collecting"
            update["detail"] = "collecting bounded real causal features"
        state.update(update)
        _atomic_json(self.status_path, state)
        if self.spawn_training:
            self._maybe_spawn(state)
        return state

    def status(self) -> dict[str, Any]:
        state = _read_status(self.status_path)
        if state.get("schema") != "argus.daily-learning-status/v2":
            legacy_schema = state.get("schema")
            state = {"legacyStatusSchema": legacy_schema} if legacy_schema else {}
        state.setdefault("schema", "argus.daily-learning-status/v2")
        state.setdefault("state", "collecting")
        state.setdefault("task", "self_supervised_next_packet_forecast_only")
        state.setdefault("sequenceContract", "argus.packet-sequence/v2")
        state.setdefault("featureSchema", "packet-causal-flow-v2")
        state.setdefault("flowIdentityScope", "segment_local_generation")
        state.setdefault("globalContextLane", True)
        state.setdefault("securityLabelsLearned", False)
        state.setdefault("autoPromotion", False)
        state["retainedPackets"] = self.retained_packets
        state["retentionDays"] = MAX_DAYS
        state["dailyBytesLimit"] = MAX_DAY_BYTES
        state["hourlyBytesLimit"] = MAX_HOUR_BYTES
        return state

    def _maybe_spawn(self, state: dict[str, Any]) -> None:
        if state.get("state") in {"queued", "training"}:
            return
        if int(state.get("newPacketsSinceTraining", 0)) < MIN_TRAIN_PACKETS:
            return
        v2_incumbent = self.directory.parent / "models" / "cnn_local_flow_v2.pt"
        if not v2_incumbent.is_file():
            state["state"] = "collecting"
            state["detail"] = (
                "v2 per-flow journal ready; waiting for a reviewed "
                "flow-v2 base artifact"
            )
            state["trainingBlocked"] = "missing_reviewed_flow_v2_base_artifact"
            _atomic_json(self.status_path, state)
            return
        state.pop("trainingBlocked", None)
        last_attempt_packets = int(state.get("lastAttemptPacketCount", 0))
        total = int(state.get("totalCollectedPackets", 0))
        if (
            last_attempt_packets
            and total - last_attempt_packets < MIN_RETRY_NEW_PACKETS
        ):
            return
        last = float(state.get("lastTrainingCompletedUnix", 0.0))
        if last and time.time() - last < TRAIN_INTERVAL_SECONDS:
            return
        state["state"] = "queued"
        state["detail"] = "shadow challenger queued; live monitoring continues"
        state["lastAttemptPacketCount"] = total
        _atomic_json(self.status_path, state)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.Popen(  # noqa: S603 - fixed local module and interpreter
                [
                    sys.executable,
                    "-m",
                    "argus_research.continual_learning",
                    "--journal",
                    str(self.directory),
                    "--model-dir",
                    str(self.directory.parent / "models"),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=flags,
            )
        except OSError as error:
            state["state"] = "error"
            state["detail"] = f"challenger process could not start: {error}"
            _atomic_json(self.status_path, state)


def _sample_batch(
    streams: list[Tensor], batch_size: int, rng: random.Random
) -> tuple[Tensor, Tensor]:
    contexts = []
    targets = []
    eligible = [stream for stream in streams if len(stream) > CONTEXT_PACKETS]
    if not eligible:
        raise ValueError("no journal segment is long enough for causal training")
    for _ in range(batch_size):
        stream = rng.choice(eligible)
        target = rng.randrange(CONTEXT_PACKETS, len(stream))
        contexts.append(stream[target - CONTEXT_PACKETS : target])
        targets.append(stream[target])
    return torch.stack(contexts), torch.stack(targets)


def _evaluate(model: nn.Module, streams: list[Tensor], samples: int = 256) -> float:
    model.eval()
    rng = random.Random(20260814)
    losses = []
    with torch.inference_mode():
        for _ in range(max(1, math.ceil(samples / 64))):
            x, y = _sample_batch(streams, min(64, samples), rng)
            losses.append(float(nn.functional.smooth_l1_loss(model(x), y)))
    return sum(losses) / len(losses)


def promotion_gate(
    incumbent_frozen: float,
    challenger_frozen: float,
    incumbent_recent: float,
    challenger_recent: float,
) -> tuple[bool, list[str]]:
    reasons = []
    values = (
        incumbent_frozen,
        challenger_frozen,
        incumbent_recent,
        challenger_recent,
    )
    if not all(math.isfinite(value) and value >= 0 for value in values):
        reasons.append("non-finite or negative evaluation metric")
    if challenger_frozen > incumbent_frozen * 1.02:
        reasons.append("frozen holdout regressed by more than 2%")
    if challenger_recent >= incumbent_recent * 0.995:
        reasons.append("recent holdout did not improve by at least 0.5%")
    return not reasons, reasons


def train_challenger(journal: Path, model_dir: Path) -> dict[str, Any]:
    status_path = journal / "status.json"
    lock = journal / "training.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(descriptor)
    except FileExistsError:
        return _read_status(status_path)
    state = _read_status(status_path)
    try:
        state.update(
            {
                "state": "training",
                "detail": "training a shadow CNN forecaster from real daily features",
                "trainingStartedUtc": datetime.now(UTC).isoformat(),
            }
        )
        _atomic_json(status_path, state)
        records = list(iter_flow_sequences_v2(journal))
        grouped_records = _group_flow_records_by_segment(records)
        streams = [values for _, values in records if len(values) > CONTEXT_PACKETS]
        if (
            sum(map(len, streams)) < MIN_TRAIN_PACKETS
            or len(streams) < 3
            or len(grouped_records) < 3
        ):
            raise ValueError(
                "insufficient bounded daily features for a causal challenger"
            )
        frozen_path = journal / "frozen-holdout-v2.pt"
        if frozen_path.is_file():
            frozen_payload = torch.load(
                frozen_path, map_location="cpu", weights_only=True
            )
            frozen = frozen_payload["streams"]
            frozen_ids = set(frozen_payload["segment_ids"])
        else:
            frozen_count = max(1, len(grouped_records) // 10)
            frozen_groups = grouped_records[:frozen_count]
            frozen_ids = {identifier for identifier, _ in frozen_groups}
            frozen = [
                values
                for _, group in frozen_groups
                for values in group
                if len(values) > CONTEXT_PACKETS
            ]
            temporary = frozen_path.with_suffix(".tmp")
            torch.save(
                {"segment_ids": sorted(frozen_ids), "streams": frozen}, temporary
            )
            os.replace(temporary, frozen_path)
        adaptable_groups = [
            (identifier, group)
            for identifier, group in grouped_records
            if identifier not in frozen_ids
        ]
        if not frozen or len(adaptable_groups) < 2:
            raise ValueError(
                "daily replay cannot form disjoint frozen/train/recent splits"
            )
        recent_count = max(1, len(adaptable_groups) // 5)
        train_groups = adaptable_groups[:-recent_count]
        recent_groups = adaptable_groups[-recent_count:]
        train_streams = [
            values
            for _, group in train_groups
            for values in group
            if len(values) > CONTEXT_PACKETS
        ]
        recent = [
            values
            for _, group in recent_groups
            for values in group
            if len(values) > CONTEXT_PACKETS
        ]
        if not train_streams:
            raise ValueError("daily replay has no training split after holdouts")
        incumbent_path = model_dir / "cnn_local_flow_v2.pt"
        payload = torch.load(incumbent_path, map_location="cpu", weights_only=True)
        if (
            payload.get("feature_schema") != "packet-causal-flow-v2"
            or payload.get("supervision") != "next_packet_only"
        ):
            raise ValueError("active CNN artifact has an incompatible contract")
        incumbent = CausalCnnNextPacket(FEATURE_DIM)
        incumbent.load_state_dict(payload["state_dict"], strict=True)
        challenger = CausalCnnNextPacket(FEATURE_DIM)
        challenger.load_state_dict(payload["state_dict"], strict=True)
        torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
        incumbent_frozen = _evaluate(incumbent, frozen)
        incumbent_recent = _evaluate(incumbent, recent)
        optimizer = torch.optim.AdamW(
            challenger.parameters(), lr=2e-4, weight_decay=1e-4
        )
        rng = random.Random(20260814)
        challenger.train()
        steps = max(20, min(100, sum(map(len, train_streams)) // 256))
        for _ in range(3):
            for _ in range(steps):
                x, y = _sample_batch(train_streams, 64, rng)
                loss = nn.functional.smooth_l1_loss(challenger(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite continual training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(challenger.parameters(), 1.0)
                optimizer.step()
        challenger_frozen = _evaluate(challenger, frozen)
        challenger_recent = _evaluate(challenger, recent)
        eligible, reasons = promotion_gate(
            incumbent_frozen,
            challenger_frozen,
            incumbent_recent,
            challenger_recent,
        )
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        destination = model_dir / "challengers" / f"cnn_local_daily_{stamp}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "artifact_type": "continual_temporal_forecaster_challenger",
            "feature_schema": "packet-causal-flow-v2",
            "sequence_contract": "argus.packet-sequence/v2",
            "supervision": "next_packet_only",
            "realDailyTraffic": True,
            "securityDetector": False,
            "securityLabelsLearned": False,
            "autoPromotion": False,
            "promotionEligible": eligible,
            "promotionBlockers": reasons,
            "metrics": {
                "incumbentFrozenLoss": incumbent_frozen,
                "challengerFrozenLoss": challenger_frozen,
                "incumbentRecentLoss": incumbent_recent,
                "challengerRecentLoss": challenger_recent,
            },
            "training": {"epochs": 3, "stepsPerEpoch": steps, "threads": 2},
            "createdUtc": datetime.now(UTC).isoformat(),
        }
        temporary = destination.with_suffix(".tmp")
        torch.save(
            {"state_dict": challenger.state_dict(), "manifest": manifest}, temporary
        )
        os.replace(temporary, destination)
        total = int(state.get("totalCollectedPackets", 0))
        state.update(
            {
                "state": "review_ready" if eligible else "challenger_rejected",
                "detail": "challenger passed gates; manual promotion required"
                if eligible
                else "; ".join(reasons),
                "trainedThroughPackets": total,
                "newPacketsSinceTraining": 0,
                "lastTrainingCompletedUnix": time.time(),
                "lastTrainingCompletedUtc": datetime.now(UTC).isoformat(),
                "challenger": str(destination),
                "promotionEligible": eligible,
                "metrics": manifest["metrics"],
                "autoPromotion": False,
            }
        )
    except Exception as error:  # noqa: BLE001 - persisted fail-closed status boundary
        detail = str(error)
        insufficient = (
            detail.startswith("insufficient bounded daily features")
            or detail.startswith("daily replay cannot form")
            or detail.startswith("daily replay has no training split")
        )
        state.update(
            {
                "state": "collecting" if insufficient else "error",
                "detail": detail,
                "promotionEligible": False,
            }
        )
        if not insufficient:
            state["lastTrainingCompletedUnix"] = time.time()
            state["lastTrainingCompletedUtc"] = datetime.now(UTC).isoformat()
    finally:
        _atomic_json(status_path, state)
        lock.unlink(missing_ok=True)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(train_challenger(args.journal, args.model_dir), indent=2))


if __name__ == "__main__":
    main()

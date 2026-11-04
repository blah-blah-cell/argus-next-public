"""Compile a leakage-safe ARGUSLR2 model-comparison protocol.

This module does not train models. It freezes exact source-disjoint partitions,
eligible causal windows, model capacities, and evaluation coordinates before a
GPU run can begin. Retrieval accuracy remains unavailable until a dataset has
an explicit causal evidence target; next-packet loss is not relabeled as
retrieval quality.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .ahem_pretrain import (
    CausalCnnNextPacket,
    GruNextPacket,
    LongTcnNextPacket,
    LstmNextPacket,
)
from .ahem_v3 import (
    AhemV32LocalOnlyNextPacket,
    AhemV32NextPacket,
    AhemV32RecencyNextPacket,
)
from .continual_learning import (
    FEATURE_DIM,
    FlowSequenceV2,
    flow_sequences_from_record,
    iter_sequence_records_v2,
)
from .flow_v2_features import (
    DATASET_SOURCES,
    FEATURE_CONTRACT,
    FEATURE_CONTRACT_SHA256,
    FEATURE_SCHEMA,
)
from .s4d_baseline import S4DNextPacket

PROTOCOL_SCHEMA = "argus.flow-v2-comparison-protocol/v2"
SEQUENCE_CONTRACT = "argus.packet-sequence/v2"
PREREGISTRATION_SHA256 = (
    "f70840f929520a0c04cb4b1d9850a44f9a683aa2828b3872f60313679a372d8b"
)
# V3.2 retrieval training needs two completed 64-packet chunks, each with a
# causally observed successor, plus the 128-packet local lane.  One memory
# entry is enough for inference but cannot define a ranking objective.
MIN_AHEM_CONTEXT = 257
PRIMARY_TRAIN_SOURCES = (
    "Wednesday-14-02-2018",
    "Thursday-15-02-2018",
    "Friday-16-02-2018",
    "Tuesday-20-02-2018",
    "Wednesday-21-02-2018",
    "Wednesday-28-02-2018",
    "Thursday-01-03-2018",
    "Friday-02-03-2018",
)
PRIMARY_VALIDATION_SOURCE = "Thursday-22-02-2018"
PRIMARY_TEST_SOURCE = "Friday-23-02-2018"
PRIMARY_SOURCES = (
    *PRIMARY_TRAIN_SOURCES,
    PRIMARY_VALIDATION_SOURCE,
    PRIMARY_TEST_SOURCE,
)


@dataclass(frozen=True)
class ProtocolConfig:
    context_packets: int = 10_000
    train_samples: int = 2_048
    validation_samples: int = 512
    test_samples: int = 512
    seed: int = 20260821

    def validate(self) -> None:
        if self.context_packets < MIN_AHEM_CONTEXT:
            raise ValueError(
                f"context must be at least {MIN_AHEM_CONTEXT} packets for AHEM"
            )
        if (
            min(
                self.train_samples,
                self.validation_samples,
                self.test_samples,
            )
            <= 0
        ):
            raise ValueError("every comparison partition requires positive samples")
        if self.seed < 0:
            raise ValueError("comparison seed must be non-negative")


@dataclass(frozen=True)
class WindowCoordinate:
    stream_index: int
    target_index: int


@dataclass(frozen=True)
class CompiledProtocol:
    partitions: dict[str, list[FlowSequenceV2]]
    coordinates: dict[str, list[WindowCoordinate]]
    manifest: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _journal_manifest(directory: Path, accepted_sources: set[str]) -> dict[str, Any]:
    members = []
    for path in sorted(directory.glob("*.arguslr")):
        if path.stem not in accepted_sources:
            continue
        journal_sha256 = _sha256(path)
        provenance_path = path.with_suffix(path.suffix + ".manifest.json")
        provenance: dict[str, Any] = {
            "complete": False,
            "manifest": provenance_path.name,
            "manifest_sha256": None,
            "archive_sha256": None,
            "problems": [],
        }
        try:
            sidecar = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
            provenance["problems"].append(
                f"missing or invalid converter sidecar: {type(error).__name__}"
            )
        else:
            provenance["manifest_sha256"] = _sha256(provenance_path)
            archive_sha256 = sidecar.get("archive", {}).get("sha256")
            dataset = sidecar.get("dataset")
            provenance["archive_sha256"] = archive_sha256
            provenance["dataset"] = dataset
            checks = {
                "converter schema": sidecar.get("schema")
                == "argus.real-pcap-journal/v3",
                "admitted dataset": dataset in DATASET_SOURCES,
                "official source": sidecar.get("official_source")
                == DATASET_SOURCES.get(dataset),
                "real dataset": sidecar.get("synthetic_data") is False,
                "journal checksum": sidecar.get("journal", {}).get("sha256")
                == journal_sha256,
                "feature schema": sidecar.get("journal", {}).get("feature_schema")
                == FEATURE_SCHEMA,
                "feature dimension": sidecar.get("journal", {}).get("feature_dimension")
                == FEATURE_CONTRACT["dimension"],
                "feature contract": sidecar.get("journal", {}).get("feature_contract")
                == FEATURE_CONTRACT,
                "feature contract hash": sidecar.get("journal", {}).get(
                    "feature_contract_sha256"
                )
                == FEATURE_CONTRACT_SHA256,
                "raw archive checksum": isinstance(archive_sha256, str)
                and len(archive_sha256) == 64
                and all(value in "0123456789abcdef" for value in archive_sha256),
            }
            provenance["problems"].extend(
                name for name, passed in checks.items() if not passed
            )
            provenance["complete"] = not provenance["problems"]
        members.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": journal_sha256,
                "provenance": provenance,
            }
        )
    if not members:
        raise ValueError("journal directory contains no accepted ARGUSLR2 files")
    combined = hashlib.sha256(
        json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "members": members,
        "sha256": combined,
        "feature_contract": FEATURE_CONTRACT,
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "datasets": sorted(
            {
                member["provenance"].get("dataset")
                for member in members
                if member["provenance"].get("dataset") is not None
            }
        ),
        "provenance_complete": all(
            member["provenance"]["complete"] for member in members
        ),
    }


def _model_inventory() -> dict[str, dict[str, Any]]:
    constructors: dict[str, tuple[str, Any]] = {
        "ahem_v3.2": ("long_memory", AhemV32NextPacket),
        "ahem_v3.2_local": ("local_ablation", AhemV32LocalOnlyNextPacket),
        "ahem_v3.2_recency": ("recency_ablation", AhemV32RecencyNextPacket),
        "lstm": ("recurrent", LstmNextPacket),
        "gru": ("recurrent", GruNextPacket),
        "causal_cnn": ("local_convolution", CausalCnnNextPacket),
        "long_tcn": ("dilated_convolution", LongTcnNextPacket),
        "s4d": ("diagonal_state_space", S4DNextPacket),
    }
    inventory = {}
    for name, (family, constructor) in constructors.items():
        model: nn.Module = constructor(FEATURE_DIM)
        parameters = sum(value.numel() for value in model.parameters())
        state_bytes = sum(
            value.numel() * value.element_size()
            for value in model.state_dict().values()
        )
        inventory[name] = {
            "family": family,
            "parameters": parameters,
            "state_bytes_float32": state_bytes,
            "random_initialization_only": True,
            "v1_weights_reused": False,
        }
    return inventory


def deterministic_coordinates(
    streams: list[FlowSequenceV2],
    context_packets: int,
    samples: int,
    seed: int,
) -> list[WindowCoordinate]:
    """Sample uniformly from all eligible causal windows without enumeration."""
    cumulative = []
    total = 0
    for stream in streams:
        total += max(0, len(stream.features) - context_packets)
        cumulative.append(total)
    if total < samples:
        raise ValueError(
            f"partition has {total} eligible windows but requires {samples}"
        )
    rng = random.Random(seed)
    chosen = rng.sample(range(total), samples)
    coordinates = []
    for flat_index in chosen:
        stream_index = bisect.bisect_right(cumulative, flat_index)
        previous = cumulative[stream_index - 1] if stream_index else 0
        target_index = context_packets + flat_index - previous
        coordinates.append(WindowCoordinate(stream_index, target_index))
    return coordinates


def materialize_batch(
    streams: list[FlowSequenceV2],
    coordinates: list[WindowCoordinate],
    context_packets: int,
) -> tuple[Tensor, Tensor]:
    contexts = []
    targets = []
    for coordinate in coordinates:
        stream = streams[coordinate.stream_index].features
        target = coordinate.target_index
        if target < context_packets or target >= len(stream):
            raise ValueError("comparison coordinate is outside its causal stream")
        contexts.append(stream[target - context_packets : target])
        targets.append(stream[target])
    return torch.stack(contexts), torch.stack(targets)


def _partition_summary(
    sequences: list[FlowSequenceV2], context_packets: int
) -> dict[str, Any]:
    lengths = sorted(len(sequence.features) for sequence in sequences)
    windows = sum(max(0, length - context_packets) for length in lengths)

    def percentile(fraction: float) -> int:
        if not lengths:
            return 0
        return lengths[min(len(lengths) - 1, int((len(lengths) - 1) * fraction))]

    return {
        "sources": sorted({sequence.source_file for sequence in sequences}),
        "segments": len({sequence.segment_id for sequence in sequences}),
        "flow_streams": len(sequences),
        "packets": sum(lengths),
        "eligible_windows": windows,
        "flow_length": {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "max": lengths[-1] if lengths else 0,
        },
    }


def compile_protocol(
    journal: Path,
    validation_source: str,
    test_source: str,
    config: ProtocolConfig,
) -> CompiledProtocol:
    config.validate()
    if not validation_source or not test_source:
        raise ValueError("validation and test source names are required")
    if validation_source == test_source:
        raise ValueError("validation and test sources must differ")
    records = list(iter_sequence_records_v2(journal))
    if not records:
        raise ValueError("no checksum-valid ARGUSLR2 records were found")
    available = {record.source_file for record in records}
    missing = {validation_source, test_source} - available
    if missing:
        raise ValueError(f"requested holdout sources are absent: {sorted(missing)}")
    train_sources = available - {validation_source, test_source}
    if not train_sources:
        raise ValueError("source-disjoint comparison requires a training source")

    partitions: dict[str, list[FlowSequenceV2]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for record in records:
        partition = (
            "validation"
            if record.source_file == validation_source
            else "test"
            if record.source_file == test_source
            else "train"
        )
        partitions[partition].extend(flow_sequences_from_record(record))

    segment_sets = {
        name: {sequence.segment_id for sequence in sequences}
        for name, sequences in partitions.items()
    }
    if (
        segment_sets["train"] & segment_sets["validation"]
        or segment_sets["train"] & segment_sets["test"]
        or segment_sets["validation"] & segment_sets["test"]
    ):
        raise ValueError("a record segment appears in more than one partition")

    requested_samples = {
        "train": config.train_samples,
        "validation": config.validation_samples,
        "test": config.test_samples,
    }
    coordinates = {
        name: deterministic_coordinates(
            sequences,
            config.context_packets,
            requested_samples[name],
            config.seed + index,
        )
        for index, (name, sequences) in enumerate(partitions.items())
    }
    summaries = {
        name: _partition_summary(sequences, config.context_packets)
        for name, sequences in partitions.items()
    }
    coordinate_digest = hashlib.sha256(
        json.dumps(
            {
                name: [asdict(value) for value in values]
                for name, values in coordinates.items()
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest = {
        "schema": PROTOCOL_SCHEMA,
        "sequence_contract": SEQUENCE_CONTRACT,
        "feature_schema": FEATURE_SCHEMA,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "task": "causal_next_packet_forecast_only",
        "not_intrusion_detection_efficacy": True,
        "real_data_required": True,
        "synthetic_training_allowed": False,
        "split": {
            "method": "journal_source_file_disjoint",
            "validation_source": validation_source,
            "test_source": test_source,
            "train_sources": sorted(train_sources),
        },
        "config": asdict(config),
        "journal": _journal_manifest(journal, available),
        "partitions": summaries,
        "coordinate_sha256": coordinate_digest,
        "models": _model_inventory(),
        "retrieval_evaluation": {
            "available": False,
            "accuracy": None,
            "reason": (
                "ARGUSLR2 has no causal evidence-pair ground truth; forecast loss "
                "must not be presented as retrieval accuracy"
            ),
            "required_next": "immutable evidence-pair labels on held-out real traffic",
        },
    }
    return CompiledProtocol(partitions, coordinates, manifest)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(json.dumps(manifest, indent=2) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--validation-source", required=True)
    parser.add_argument("--test-source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=10_000)
    parser.add_argument("--train-samples", type=int, default=2_048)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    config = ProtocolConfig(
        context_packets=args.context,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        seed=args.seed,
    )
    compiled = compile_protocol(
        args.journal,
        args.validation_source,
        args.test_source,
        config,
    )
    write_manifest(args.output, compiled.manifest)
    print(json.dumps(compiled.manifest, indent=2))


if __name__ == "__main__":
    main()

"""Causal flow-conditioned windows over globally ordered packet history."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .continual_learning import FEATURE_DIM, SequenceRecordV2, iter_sequence_records_v2
from .flow_v2_protocol import _journal_manifest

PROTOCOL_SCHEMA = "argus.flow-conditioned-global/v2"
PREREGISTRATION_SHA256 = (
    "95ecd0ffd174550958811fa94a7b2947f7a56187a5acd7c41e2fc241ff404283"
)
INPUT_DIM = FEATURE_DIM + 1
RELATION_FEATURE = "same_flow_as_query"
TARGET_INDICES = (0, 1)
TARGET_FIELDS = ("log_frame_length", "log_per_flow_interarrival_seconds")


@dataclass(frozen=True)
class ProtocolConfig:
    context_packets: int = 10_000
    local_packets: int = 128
    train_samples: int = 2_048
    validation_samples: int = 512
    test_samples: int = 512
    seed: int = 20260828
    require_memory_match: bool = True
    changed_event_fraction: float = 0.5
    sampling_policy: str = "balanced"
    minimum_completed_transitions: int = 0

    def validate(self) -> None:
        if self.sampling_policy not in {
            "balanced",
            "natural",
            "source_balanced_natural",
        }:
            raise ValueError("unknown sampling policy")
        if self.context_packets < self.local_packets + 129:
            raise ValueError("context must leave at least 129 packets for memory")
        if (
            min(
                self.local_packets,
                self.train_samples,
                self.validation_samples,
                self.test_samples,
            )
            <= 0
        ):
            raise ValueError("protocol dimensions and sample counts must be positive")
        if not 0.0 < self.changed_event_fraction < 1.0:
            raise ValueError("changed event fraction must be between zero and one")
        if self.minimum_completed_transitions < 0:
            raise ValueError("minimum completed transitions cannot be negative")


@dataclass(frozen=True)
class FlowConditionedRecord:
    source_file: str
    segment_id: str
    features: Tensor
    flow_ids: Tensor
    next_same_flow: Tensor
    previous_same_flow: Tensor


@dataclass(frozen=True)
class WindowCoordinate:
    record_index: int
    anchor_index: int
    target_index: int


@dataclass(frozen=True)
class CompiledProtocol:
    partitions: dict[str, list[FlowConditionedRecord]]
    coordinates: dict[str, list[WindowCoordinate]]
    manifest: dict[str, Any]


def condition_record(record: SequenceRecordV2) -> FlowConditionedRecord:
    """Reconstruct per-flow time and causal recurrence links in global order."""
    features = record.features.clone()
    count = len(features)
    previous = [-1] * count
    following = [-1] * count
    deltas = [0.0] * count
    last_seen: dict[int, int] = {}
    flow_ids = record.flow_ids.tolist()
    timestamps = record.timestamps_ns.tolist()
    for index, flow_id in enumerate(flow_ids):
        prior = last_seen.get(flow_id)
        if prior is not None:
            previous[index] = prior
            following[prior] = index
            delta = max(
                0.0,
                min(
                    (timestamps[index] - timestamps[prior]) / 1_000_000_000.0,
                    60.0,
                ),
            )
            deltas[index] = math.log1p(delta)
        last_seen[flow_id] = index
    features[:, 1] = torch.tensor(deltas, dtype=features.dtype)
    return FlowConditionedRecord(
        source_file=record.source_file,
        segment_id=record.identifier,
        features=features,
        flow_ids=record.flow_ids,
        next_same_flow=torch.tensor(following, dtype=torch.int64),
        previous_same_flow=torch.tensor(previous, dtype=torch.int64),
    )


def eligible_anchors(
    record: FlowConditionedRecord,
    context_packets: int,
    local_packets: int,
    *,
    require_memory_match: bool = True,
    minimum_completed_transitions: int = 0,
) -> Tensor:
    """Require a future same-flow target and older same-flow memory evidence."""
    if minimum_completed_transitions < 0:
        raise ValueError("minimum completed transitions cannot be negative")
    indices = torch.arange(len(record.features), dtype=torch.int64)
    context_start = indices - context_packets + 1
    older_than_local = indices - local_packets
    eligible = (
        (indices >= context_packets - 1)
        & (record.next_same_flow > indices)
        & (record.previous_same_flow >= context_start)
    )
    if require_memory_match:
        eligible &= record.previous_same_flow <= older_than_local
    # One completed cache transition needs two predecessors of the query:
    # predecessor -> source -> query. K alternatives therefore need K+1
    # predecessor hops, all known inside the causal context.
    cursor = indices
    for _ in range(minimum_completed_transitions + 1):
        cursor = record.previous_same_flow[cursor.clamp_min(0)]
        eligible &= cursor >= context_start
    return indices[eligible]


def deterministic_coordinates(
    records: list[FlowConditionedRecord],
    *,
    context_packets: int,
    local_packets: int,
    samples: int,
    seed: int,
    require_memory_match: bool = True,
    changed_event_fraction: float = 0.5,
    sampling_policy: str = "balanced",
    minimum_completed_transitions: int = 0,
) -> list[WindowCoordinate]:
    if sampling_policy not in {
        "balanced",
        "natural",
        "source_balanced_natural",
    }:
        raise ValueError("unknown sampling policy")
    anchors = [
        eligible_anchors(
            record,
            context_packets,
            local_packets,
            require_memory_match=require_memory_match,
            minimum_completed_transitions=minimum_completed_transitions,
        )
        for record in records
    ]

    def sample_pool(
        pool: list[Tensor], count: int, pool_seed: int | str
    ) -> list[WindowCoordinate]:
        cumulative: list[int] = []
        total = 0
        for values in pool:
            total += len(values)
            cumulative.append(total)
        if total < count:
            raise ValueError(
                f"partition has {total} windows in a required stratum but needs {count}"
            )
        coordinates = []
        for flat_index in random.Random(pool_seed).sample(range(total), count):
            record_index = bisect.bisect_right(cumulative, flat_index)
            previous_total = cumulative[record_index - 1] if record_index else 0
            anchor = int(pool[record_index][flat_index - previous_total])
            target = int(records[record_index].next_same_flow[anchor])
            coordinates.append(WindowCoordinate(record_index, anchor, target))
        return coordinates

    if sampling_policy == "natural":
        return sample_pool(anchors, samples, seed)
    if sampling_policy == "source_balanced_natural":
        capacities: dict[str, int] = {}
        for record_index, (record, values) in enumerate(
            zip(records, anchors, strict=True)
        ):
            capacities[record.source_file] = capacities.get(
                record.source_file, 0
            ) + len(values)
        if sum(capacities.values()) < samples:
            raise ValueError("partition has fewer eligible windows than requested")
        ordered_sources = sorted(capacities)
        quotas = {source: 0 for source in ordered_sources}
        allocated = 0
        while allocated < samples:
            for source in ordered_sources:
                if quotas[source] >= capacities[source]:
                    continue
                quotas[source] += 1
                allocated += 1
                if allocated == samples:
                    break
        selected: list[WindowCoordinate] = []
        for source in ordered_sources:
            # Sample integer offsets before constructing coordinates. Building
            # a Python object for every eligible packet scales with corpus size.
            pool = [
                values if record.source_file == source else values[:0]
                for record, values in zip(records, anchors, strict=True)
            ]
            selected.extend(sample_pool(pool, quotas[source], f"{seed}:{source}"))
        random.Random(seed).shuffle(selected)
        return selected
    # Only the explicitly target-stratified policy inspects future feature
    # values while sampling. Natural policies above use causal eligibility.
    changed: list[Tensor] = []
    unchanged: list[Tensor] = []
    for record, values in zip(records, anchors, strict=True):
        targets = record.next_same_flow[values]
        delta = (record.features[targets, :2] - record.features[values, :2]).abs()
        mask = delta.amax(dim=1) > 1e-6
        changed.append(values[mask])
        unchanged.append(values[~mask])
    changed_count = round(samples * changed_event_fraction)
    coordinates = sample_pool(changed, changed_count, seed)
    coordinates.extend(sample_pool(unchanged, samples - changed_count, seed + 1))
    random.Random(seed + 2).shuffle(coordinates)
    return coordinates


def materialize_batch(
    records: list[FlowConditionedRecord],
    coordinates: list[WindowCoordinate],
    context_packets: int,
) -> tuple[Tensor, Tensor]:
    contexts = []
    targets = []
    for coordinate in coordinates:
        record = records[coordinate.record_index]
        start = coordinate.anchor_index - context_packets + 1
        if start < 0 or coordinate.target_index <= coordinate.anchor_index:
            raise ValueError("flow-conditioned coordinate violates causality")
        features = record.features[start : coordinate.anchor_index + 1]
        query_flow = record.flow_ids[coordinate.anchor_index]
        relation = record.flow_ids[start : coordinate.anchor_index + 1] == query_flow
        context = torch.cat((features, relation.to(features.dtype)[:, None]), dim=1)
        if context.shape != (context_packets, INPUT_DIM) or context[-1, -1] != 1:
            raise ValueError("flow-conditioned materialization contract failed")
        contexts.append(context)
        targets.append(record.features[coordinate.target_index, list(TARGET_INDICES)])
    return torch.stack(contexts), torch.stack(targets)


def _partition_summary(
    records: list[FlowConditionedRecord], config: ProtocolConfig
) -> dict[str, Any]:
    counts = [
        len(
            eligible_anchors(
                record,
                config.context_packets,
                config.local_packets,
                require_memory_match=config.require_memory_match,
                minimum_completed_transitions=config.minimum_completed_transitions,
            )
        )
        for record in records
    ]
    return {
        "sources": sorted({record.source_file for record in records}),
        "records": len(records),
        "segments": len({record.segment_id for record in records}),
        "packets": sum(len(record.features) for record in records),
        "flow_conditioned_windows": sum(counts),
        "records_with_windows": sum(count > 0 for count in counts),
    }


def _manifest(
    partitions: dict[str, list[FlowConditionedRecord]],
    coordinates: dict[str, list[WindowCoordinate]],
    config: ProtocolConfig,
    split: str | dict[str, Any],
) -> dict[str, Any]:
    coordinate_payload = {
        name: [asdict(value) for value in values]
        for name, values in coordinates.items()
    }
    return {
        "schema": PROTOCOL_SCHEMA,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "study_role": "exploratory",
        "publication_evidence": False,
        "task": "predict_next_dynamic_event_of_query_flow_from_global_history",
        "causal_query": "last observed packet of query flow",
        "target": {
            "event": "next packet of the same segment-local flow generation",
            "fields": list(TARGET_FIELDS),
            "dimension": len(TARGET_INDICES),
            "excluded_shortcut_fields": "static IP and transport indicators",
        },
        "input": {
            "packet_features": FEATURE_DIM,
            "relation_features": [RELATION_FEATURE],
            "dimension": INPUT_DIM,
            "raw_flow_identity_learned": False,
        },
        "sampling": {
            "method": "deterministic_changed_event_stratification"
            if config.sampling_policy == "balanced"
            else "uniform_eligible_without_replacement"
            if config.sampling_policy == "natural"
            else "source_balanced_uniform_eligible_without_replacement",
            "changed_event_fraction": config.changed_event_fraction
            if config.sampling_policy == "balanced"
            else None,
            "persistence_baseline_required": True,
        },
        "config": asdict(config),
        "split": split,
        "partitions": {
            name: _partition_summary(values, config)
            for name, values in partitions.items()
        },
        "coordinate_sha256": hashlib.sha256(
            json.dumps(
                coordinate_payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }


def compile_local_development(
    journal: Path, config: ProtocolConfig
) -> CompiledProtocol:
    """Split real local records chronologically; never publication evidence."""
    config.validate()
    raw_records = list(iter_sequence_records_v2(journal))
    if len(raw_records) < 5:
        raise ValueError("local development requires at least five complete records")
    records = [condition_record(record) for record in raw_records]
    train_end = max(1, int(len(records) * 0.6))
    validation_end = max(train_end + 1, int(len(records) * 0.8))
    partitions = {
        "train": records[:train_end],
        "validation": records[train_end:validation_end],
        "test": records[validation_end:],
    }
    requested = {
        "train": config.train_samples,
        "validation": config.validation_samples,
        "test": config.test_samples,
    }
    coordinates = {
        name: deterministic_coordinates(
            values,
            context_packets=config.context_packets,
            local_packets=config.local_packets,
            samples=requested[name],
            seed=config.seed + index,
            require_memory_match=config.require_memory_match,
            changed_event_fraction=config.changed_event_fraction,
            sampling_policy=config.sampling_policy,
            minimum_completed_transitions=config.minimum_completed_transitions,
        )
        for index, (name, values) in enumerate(partitions.items())
    }
    manifest = _manifest(
        partitions,
        coordinates,
        config,
        "chronological_record_split_for_local_development_only",
    )
    return CompiledProtocol(partitions, coordinates, manifest)


def compile_source_disjoint(
    journal: Path,
    validation_source: str,
    test_source: str,
    config: ProtocolConfig,
) -> CompiledProtocol:
    """Compile strict real-source holdouts; remains exploratory until frozen."""
    config.validate()
    if not validation_source or not test_source:
        raise ValueError("validation and test source names are required")
    if validation_source == test_source:
        raise ValueError("validation and test sources must differ")
    raw_records = list(iter_sequence_records_v2(journal))
    if not raw_records:
        raise ValueError("no checksum-valid ARGUSLR2 records were found")
    available = {record.source_file for record in raw_records}
    missing = {validation_source, test_source} - available
    if missing:
        raise ValueError(f"requested holdout sources are absent: {sorted(missing)}")
    train_sources = available - {validation_source, test_source}
    if not train_sources:
        raise ValueError("source-disjoint comparison requires a training source")
    journal_manifest = _journal_manifest(journal, available)
    partitions: dict[str, list[FlowConditionedRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for raw in raw_records:
        partition = (
            "validation"
            if raw.source_file == validation_source
            else "test"
            if raw.source_file == test_source
            else "train"
        )
        partitions[partition].append(condition_record(raw))
    segment_sets = {
        name: {record.segment_id for record in records}
        for name, records in partitions.items()
    }
    if (
        segment_sets["train"] & segment_sets["validation"]
        or segment_sets["train"] & segment_sets["test"]
        or segment_sets["validation"] & segment_sets["test"]
    ):
        raise ValueError("a record segment appears in more than one partition")
    requested = {
        "train": config.train_samples,
        "validation": config.validation_samples,
        "test": config.test_samples,
    }
    coordinates = {
        name: deterministic_coordinates(
            records,
            context_packets=config.context_packets,
            local_packets=config.local_packets,
            samples=requested[name],
            seed=config.seed + index,
            require_memory_match=config.require_memory_match,
            changed_event_fraction=config.changed_event_fraction,
            sampling_policy=config.sampling_policy,
            minimum_completed_transitions=config.minimum_completed_transitions,
        )
        for index, (name, records) in enumerate(partitions.items())
    }
    manifest = _manifest(
        partitions,
        coordinates,
        config,
        {
            "method": "journal_source_file_disjoint",
            "validation_source": validation_source,
            "test_source": test_source,
            "train_sources": sorted(train_sources),
        },
    )
    manifest["journal"] = journal_manifest
    return CompiledProtocol(partitions, coordinates, manifest)

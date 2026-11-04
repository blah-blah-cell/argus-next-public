"""Aggregate independent fresh-process AHEM-v5 runtime reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from .flow_v2_protocol import write_manifest
from .flow_v3_train import V4_PREREGISTRATION_SHA256
from .flow_v4_runtime_benchmark import BENCHMARK_SCHEMA

AGGREGATE_SCHEMA = "argus.ahem-v5-runtime-replicates/v2"


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def aggregate(
    reports: list[Path], output: Path, *, expected_processes: int = 5
) -> dict[str, Any]:
    if len(reports) != expected_processes or expected_processes < 2:
        raise ValueError("runtime aggregation requires the frozen process count")
    if len({path.resolve() for path in reports}) != len(reports):
        raise ValueError("runtime aggregation repeats a report path")
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in reports]
    if any(value.get("schema") != BENCHMARK_SCHEMA for value in loaded):
        raise ValueError("runtime aggregation received an incompatible report")
    process_ids = [value.get("process_instance_id") for value in loaded]
    if (
        any(not isinstance(value, str) or not value for value in process_ids)
        or len(set(process_ids)) != expected_processes
    ):
        raise ValueError("runtime aggregation requires distinct fresh processes")
    for field in ("environment", "source_provenance", "implementation_sha256"):
        if not isinstance(loaded[0].get(field), dict) or not loaded[0][field]:
            raise ValueError(f"runtime report lacks {field}")
        if any(value.get(field) != loaded[0][field] for value in loaded[1:]):
            raise ValueError(f"runtime aggregation mixes {field}")
    provenance_fields = (
        "study_role",
        "publication_evidence",
        "preregistration_sha256",
        "protocol_sha256",
        "coordinate_sha256",
        "training_seed",
    )
    provenance = {field: loaded[0].get(field) for field in provenance_fields}
    if any(
        any(value.get(field) != provenance[field] for field in provenance_fields)
        for value in loaded[1:]
    ):
        raise ValueError("runtime aggregation mixes provenance")
    if provenance["publication_evidence"] is True and (
        provenance["study_role"] != "confirmatory_runtime_audit"
        or provenance["preregistration_sha256"] != V4_PREREGISTRATION_SHA256
        or not isinstance(provenance["protocol_sha256"], str)
        or not isinstance(provenance["coordinate_sha256"], str)
        or not isinstance(provenance["training_seed"], int)
    ):
        raise ValueError("confirmatory runtime provenance is incomplete")
    checkpoint_sets = [value.get("checkpoints") for value in loaded]
    if any(value != checkpoint_sets[0] for value in checkpoint_sets[1:]):
        raise ValueError("runtime aggregation mixes checkpoints")
    settings = [value.get("settings") for value in loaded]
    immutable_settings = [
        {
            "warmup": value.get("warmup"),
            "iterations": value.get("iterations"),
            "torch_threads": value.get("torch_threads"),
            "measurement_order": value.get("measurement_order"),
        }
        for value in settings
    ]
    if any(value != immutable_settings[0] for value in immutable_settings[1:]):
        raise ValueError("runtime aggregation mixes measurement settings")
    lengths = [list(value.get("measurements", {})) for value in loaded]
    if any(len(value) != 1 for value in lengths):
        raise ValueError("every report must contain one fresh-process length")
    if any(value != lengths[0] for value in lengths[1:]):
        raise ValueError("runtime aggregation mixes context lengths")
    length = lengths[0][0]
    settings_contract = immutable_settings[0]
    sample_count = settings_contract["iterations"]
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("runtime sample count must be positive")
    if provenance["publication_evidence"] is True and (
        expected_processes != 5
        or settings_contract
        != {
            "warmup": 50,
            "iterations": 200,
            "torch_threads": 1,
            "measurement_order": "alternating_forward_reverse",
        }
        or loaded[0]["environment"].get("device") != "cpu"
        or not loaded[0]["environment"].get("platform", "").startswith("Windows-")
        or loaded[0]["source_provenance"].get("available") is not True
        or loaded[0]["source_provenance"].get("git_dirty") is not False
        or int(length) not in (257, 10_000, 30_000)
    ):
        raise ValueError("confirmatory timing differs from the frozen runtime contract")

    process_p95: dict[str, list[float]] = {
        "ahem_v5_stream_step": [],
        "ahem_v5_transition_write": [],
        "lstm_stateful_step": [],
    }
    all_samples: dict[str, list[float]] = {name: [] for name in process_p95}
    equivalence_errors = []
    for report in loaded:
        measurement = report["measurements"][length]
        error = measurement["stream_reference_max_abs_error"]
        if not isinstance(error, (int, float)) or not math.isfinite(error) or error < 0:
            raise ValueError("runtime equivalence error is invalid")
        equivalence_errors.append(error)
        for name, values in process_p95.items():
            samples = measurement[name]["samples_ms"]
            if (
                not isinstance(samples, list)
                or len(samples) != sample_count
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                    for value in samples
                )
            ):
                raise ValueError("runtime raw samples are incomplete or invalid")
            p95 = _percentile(samples, 0.95)
            reported = measurement[name]["p95_ms"]
            if not isinstance(reported, (int, float)) or not math.isclose(
                reported, p95, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("runtime p95 does not match its raw samples")
            values.append(p95)
            all_samples[name].extend(samples)

    summaries = {}
    for name, values in process_p95.items():
        summaries[name] = {
            "process_p95_ms": values,
            "median_process_p95_ms": median(values),
            "min_process_p95_ms": min(values),
            "max_process_p95_ms": max(values),
            "pooled_p95_ms_secondary": _percentile(all_samples[name], 0.95),
            "pooled_samples": len(all_samples[name]),
        }
    result = {
        "schema": AGGREGATE_SCHEMA,
        "measurement_scope": "conditional_neural_step_with_prebuilt_query_state",
        "end_to_end_streaming_evidence": False,
        **provenance,
        "context_packets": int(length),
        "processes": expected_processes,
        "process_instance_ids": process_ids,
        "environment": loaded[0]["environment"],
        "source_provenance": loaded[0]["source_provenance"],
        "implementation_sha256": loaded[0]["implementation_sha256"],
        "reports": [str(path) for path in reports],
        "checkpoints": checkpoint_sets[0],
        "settings": immutable_settings[0],
        "summaries": summaries,
        "max_stream_reference_abs_error": max(equivalence_errors),
        "feasibility_gate": {
            "metric": "median_process_p95_ms",
            "threshold_ms": 2.0,
            "passed": summaries["ahem_v5_stream_step"]["median_process_p95_ms"] <= 2.0,
        },
        "lstm_comparison": {
            "ahem_median_p95_divided_by_lstm_median_p95": (
                summaries["ahem_v5_stream_step"]["median_process_p95_ms"]
                / summaries["lstm_stateful_step"]["median_process_p95_ms"]
            ),
            "faster_than_lstm_claimed": False,
        },
    }
    write_manifest(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-processes", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            aggregate(
                args.reports,
                args.output,
                expected_processes=args.expected_processes,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

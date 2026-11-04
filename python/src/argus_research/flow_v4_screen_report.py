"""Fail-closed development-screen verdict for AHEM-v5."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from .flow_v2_protocol import write_manifest
from .flow_v2_train import _canonical_sha256
from .flow_v3_train import V4_PREREGISTRATION_SHA256, V4_RUN_SCHEMA
from .flow_v4_mechanism_eval import MECHANISM_SCHEMA, summarize
from .flow_v4_runtime_benchmark import BENCHMARK_SCHEMA
from .flow_v4_runtime_replicates import _percentile

SCREEN_REPORT_SCHEMA = "argus.ahem-v5-development-screen/v1"
SCREEN_SEEDS = (20260831, 20260901, 20260902)
SCREEN_MODELS = (
    "ahem_v5_source_aligned",
    "ahem_v5_local_only",
    "ahem_v5_recency",
    "lstm",
)


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _hash(value: Any, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _losses(test: dict[str, Any]) -> None:
    coordinates = test.get("sample_coordinates")
    if (
        not isinstance(coordinates, list)
        or not coordinates
        or not all(isinstance(value, dict) and value for value in coordinates)
        or len({_canonical_sha256(value) for value in coordinates}) != len(coordinates)
        or type(test.get("samples")) is not int
        or test["samples"] != len(coordinates)
    ):
        raise ValueError("test coordinate/sample inventory is invalid")
    for raw, scalar in (
        ("per_sample_weighted_smooth_l1", "weighted_smooth_l1"),
        ("per_sample_persistence_weighted_smooth_l1", "persistence_weighted_smooth_l1"),
    ):
        values = test.get(raw)
        reported = test.get(scalar)
        if (
            not isinstance(values, list)
            or len(values) != len(coordinates)
            or not all(_number(value) for value in values)
            or not _number(reported)
            # Training records a float32 tensor mean, not Python's double mean.
            or not math.isclose(reported, fmean(values), rel_tol=1e-6, abs_tol=1e-7)
        ):
            raise ValueError("reported loss does not match finite nonnegative samples")


def _metric(path: Path, *, study_role: str = "diagnostic") -> dict[str, Any]:
    if study_role not in ("diagnostic", "confirmatory"):
        raise ValueError("unknown evidence study role")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != V4_RUN_SCHEMA
        or value.get("study_role") != study_role
        or value.get("publication_eligible") is not (study_role == "confirmatory")
        or value.get("preregistration_sha256") != V4_PREREGISTRATION_SHA256
    ):
        raise ValueError(f"{path.name} is not a v5 {study_role} result")
    test = value.get("test", {})
    access = value.get("test_access", {})
    _losses(test)
    protocol = value.get("protocol", {})
    source = value.get("source_provenance", {})
    if (
        protocol.get("config", {}).get("sampling_policy") not in ("balanced", "natural")
        or protocol.get("preregistration_sha256") != V4_PREREGISTRATION_SHA256
        or not _hash(value.get("coordinate_sha256"))
        or protocol.get("coordinate_sha256") != value.get("coordinate_sha256")
        or value.get("protocol_sha256") != _canonical_sha256(protocol)
        or source.get("available") is not True
        or source.get("git_dirty") is not False
        or not _hash(source.get("git_revision"), 40)
        or not _hash(access.get("best_checkpoint_sha256"))
        or access.get("protocol_sha256") != value.get("protocol_sha256")
        or access.get("coordinate_sha256") != value.get("coordinate_sha256")
        or access.get("model") != value.get("model", {}).get("name")
        or access.get("training_seed") != value.get("training", {}).get("seed")
        or access.get("test_samples") != test["samples"]
    ):
        raise ValueError("diagnostic protocol/source/checkpoint binding is invalid")
    checks = {
        "per_sample_sha256": test.get("per_sample_weighted_smooth_l1"),
        "persistence_per_sample_sha256": test.get(
            "per_sample_persistence_weighted_smooth_l1"
        ),
        "sample_coordinates_sha256": test.get("sample_coordinates"),
    }
    if access.get("status") != "completed" or any(
        access.get(field) != _canonical_sha256(payload)
        for field, payload in checks.items()
    ):
        raise ValueError(f"{path.name} has an invalid diagnostic ledger")
    return value


def _mechanism(value: dict[str, Any], runs: dict[str, dict[str, Any]]) -> None:
    candidate = runs[SCREEN_MODELS[0]]
    if (
        value.get("preregistration_sha256") != V4_PREREGISTRATION_SHA256
        or value.get("candidate_checkpoint_sha256")
        != candidate["test_access"]["best_checkpoint_sha256"]
    ):
        raise ValueError("mechanism report does not bind the selected checkpoint")
    for name in ("query_rotation", "value_permutation"):
        result = value.get(name, {})
        _losses(result)
        if any(
            result[field] != candidate["test"][field]
            for field in (
                "sample_coordinates",
                "per_sample_persistence_weighted_smooth_l1",
            )
        ):
            raise ValueError("mechanism report has unpaired test evidence")
    expected = summarize(
        candidate["test"]["weighted_smooth_l1"],
        runs["ahem_v5_local_only"]["test"]["weighted_smooth_l1"],
        runs["ahem_v5_recency"]["test"]["weighted_smooth_l1"],
        value["query_rotation"]["weighted_smooth_l1"],
        value["value_permutation"]["weighted_smooth_l1"],
    )
    if (
        value.get("summary") != expected
        or any(type(gate) is not bool for gate in value["summary"]["gates"].values())
        or value.get("all_mechanism_gates_pass") is not all(expected["gates"].values())
    ):
        raise ValueError("mechanism summary/pass flag disagrees with observed losses")


def _runtime(value: dict[str, Any], runs: dict[str, dict[str, Any]]) -> None:
    if (
        value.get("study_role") != "exploratory_runtime_audit"
        or value.get("preregistration_sha256") != V4_PREREGISTRATION_SHA256
        or value.get("training_seed") != SCREEN_SEEDS[0]
        or value.get("source_provenance") != runs[SCREEN_MODELS[0]]["source_provenance"]
        or not isinstance(value.get("process_instance_id"), str)
        or not value["process_instance_id"]
        or not value.get("implementation_sha256")
        or not all(_hash(item) for item in value["implementation_sha256"].values())
        or value.get("settings")
        != {
            "warmup": 20,
            "iterations": 100,
            "torch_threads": 1,
            "measurement_order": "alternating_forward_reverse",
        }
        or len(value.get("measurements", {})) != 1
    ):
        raise ValueError("runtime does not follow the diagnostic measurement contract")
    for name, model in (("ahem_v5", SCREEN_MODELS[0]), ("lstm", "lstm")):
        if (
            value.get("checkpoints", {}).get(name, {}).get("sha256")
            != runs[model]["test_access"]["best_checkpoint_sha256"]
        ):
            raise ValueError("runtime uses a different selected checkpoint")
    measurement = next(iter(value["measurements"].values()))
    if not _number(measurement.get("stream_reference_max_abs_error")):
        raise ValueError("runtime equivalence error is invalid")
    for name in (
        "ahem_v5_stream_step",
        "ahem_v5_transition_write",
        "lstm_stateful_step",
    ):
        record = measurement.get(name, {})
        samples = record.get("samples_ms")
        if (
            not isinstance(samples, list)
            or len(samples) != 100
            or not all(_number(sample) for sample in samples)
            or not _number(record.get("p95_ms"))
            or not math.isclose(
                record["p95_ms"], _percentile(samples, 0.95), rel_tol=0, abs_tol=1e-12
            )
        ):
            raise ValueError("runtime p95 does not match its raw measurement samples")


def build(
    metrics: list[Path], mechanisms: list[Path], runtimes: list[Path]
) -> dict[str, Any]:
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for path in metrics:
        value = _metric(path)
        model = value["model"]["name"]
        seed = int(value["training"]["seed"])
        if seed in grouped.setdefault(model, {}):
            raise ValueError("development screen contains a duplicate model seed")
        grouped[model][seed] = value
    if set(grouped) != set(SCREEN_MODELS) or any(
        tuple(sorted(values)) != SCREEN_SEEDS for values in grouped.values()
    ):
        raise ValueError("development screen lacks a frozen model/seed run")
    all_metrics = [value for values in grouped.values() for value in values.values()]
    protocol_hashes = {value["protocol_sha256"] for value in all_metrics}
    coordinate_hashes = {value["coordinate_sha256"] for value in all_metrics}
    if len(protocol_hashes) != 1 or len(coordinate_hashes) != 1:
        raise ValueError("development screen mixes protocols or coordinates")
    for value in all_metrics[1:]:
        if value["source_provenance"] != all_metrics[0]["source_provenance"] or any(
            value["test"][field] != all_metrics[0]["test"][field]
            for field in (
                "sample_coordinates",
                "per_sample_persistence_weighted_smooth_l1",
            )
        ):
            raise ValueError("development screen mixes source or paired test evidence")

    candidate_losses = {
        seed: float(
            grouped["ahem_v5_source_aligned"][seed]["test"]["weighted_smooth_l1"]
        )
        for seed in SCREEN_SEEDS
    }
    comparisons = {}
    for reference in SCREEN_MODELS[1:]:
        differences = [
            float(grouped[reference][seed]["test"]["weighted_smooth_l1"])
            - candidate_losses[seed]
            for seed in SCREEN_SEEDS
        ]
        comparisons[reference] = {
            "reference_minus_candidate_by_seed": differences,
            "mean_reference_minus_candidate": fmean(differences),
            "positive_seed_count": sum(value > 0 for value in differences),
            "directional_gate": fmean(differences) > 0
            and sum(value > 0 for value in differences) >= 2,
        }

    mechanism_values = [
        json.loads(path.read_text(encoding="utf-8")) for path in mechanisms
    ]
    if len(mechanism_values) != len(SCREEN_SEEDS) or any(
        value.get("schema") != MECHANISM_SCHEMA
        or value.get("study_role") != "diagnostic"
        or value.get("publication_evidence") is not False
        or value.get("protocol_sha256") not in protocol_hashes
        or value.get("coordinate_sha256") not in coordinate_hashes
        for value in mechanism_values
    ):
        raise ValueError("development screen has invalid mechanism evidence")
    if sorted(int(value["training_seed"]) for value in mechanism_values) != list(
        SCREEN_SEEDS
    ):
        raise ValueError("development screen lacks one mechanism result per seed")
    for value in mechanism_values:
        _mechanism(
            value,
            {name: grouped[name][value["training_seed"]] for name in SCREEN_MODELS},
        )

    runtime_values = [json.loads(path.read_text(encoding="utf-8")) for path in runtimes]
    expected_contexts = {257, 10_000, 30_000}
    if len(runtime_values) != len(expected_contexts) or any(
        value.get("schema") != BENCHMARK_SCHEMA
        or value.get("publication_evidence") is not False
        or value.get("environment", {}).get("device") != "cpu"
        or not value.get("environment", {}).get("platform", "").startswith("Windows-")
        or value.get("protocol_sha256") not in protocol_hashes
        or value.get("coordinate_sha256") not in coordinate_hashes
        for value in runtime_values
    ):
        raise ValueError("development screen has invalid runtime evidence")
    for value in runtime_values:
        _runtime(
            value, {name: grouped[name][SCREEN_SEEDS[0]] for name in SCREEN_MODELS}
        )
    if len({value["process_instance_id"] for value in runtime_values}) != 3 or any(
        value[field] != runtime_values[0][field]
        for value in runtime_values[1:]
        for field in ("environment", "implementation_sha256")
    ):
        raise ValueError(
            "runtime mixes implementations/environments or repeats processes"
        )
    context_values = {
        int(next(iter(value["measurements"]))): value for value in runtime_values
    }
    if set(context_values) != expected_contexts:
        raise ValueError("development screen runtime contexts are incomplete")
    runtime_gate = all(
        value["measurements"][str(context)]["ahem_v5_stream_step"]["p95_ms"] <= 2.0
        and value["measurements"][str(context)]["stream_reference_max_abs_error"]
        <= 1e-6
        for context, value in context_values.items()
    )

    gates = {
        "original_sampling_protocol": all(
            value.get("protocol", {}).get("config", {}).get("sampling_policy")
            == "balanced"
            for value in all_metrics
        ),
        "directional_quality": all(
            value["directional_gate"] for value in comparisons.values()
        ),
        "mechanism_controls": all(
            value["all_mechanism_gates_pass"] for value in mechanism_values
        ),
        "runtime_feasibility_and_equivalence": runtime_gate,
    }
    return {
        "schema": SCREEN_REPORT_SCHEMA,
        "study_role": "development_falsification_screen",
        "publication_evidence": False,
        "preregistration_sha256": V4_PREREGISTRATION_SHA256,
        "protocol_sha256": next(iter(protocol_hashes)),
        "coordinate_sha256": next(iter(coordinate_hashes)),
        "seeds": list(SCREEN_SEEDS),
        "comparisons": comparisons,
        "mechanism_results": mechanism_values,
        "runtime_results": runtime_values,
        "gates": gates,
        "screen_passed": all(gates.values()),
        "decision": "freeze_for_confirmation"
        if all(gates.values())
        else "reject_or_narrow_without_holdout_tuning",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, nargs=12, required=True)
    parser.add_argument("--mechanisms", type=Path, nargs=3, required=True)
    parser.add_argument("--runtimes", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_manifest(
        args.output,
        build(args.metrics, args.mechanisms, args.runtimes),
    )


if __name__ == "__main__":
    main()

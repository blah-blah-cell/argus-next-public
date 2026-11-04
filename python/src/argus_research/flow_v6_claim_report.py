"""Render the fail-closed verdict for the frozen AHEM-v6.1 final evaluation.

The verdict is re-derived here from the run artifact's stored numbers and
predeclared gates.  Missing, schema-incompatible, or inconsistent evidence is
an error, never a silent pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .flow_v6_final_evaluate import (
    CANDIDATE,
    MODELS,
    PRIMARY_REFERENCE,
    SCHEMA,
    SEEDS,
    _final_interval,
    _validate_plan,
)

CLAIM_SCHEMA = "argus.ahem-v6-final-claim-report/v2"
REQUIRED_MODELS = frozenset(MODELS)


def _read(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema:
        raise ValueError(f"{path} has an incompatible claim-evidence schema")
    return value


def build_claim_report(result: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    if result.get("schema") != SCHEMA:
        raise ValueError("final claim requires a final-evaluation run artifact")
    if result.get("study_role") != "confirmatory_final_evaluation":
        raise ValueError("final claim rejects non-confirmatory evidence")
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes.decode("utf-8"))
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    _validate_plan(plan, plan_sha256)
    if result.get("plan_sha256") != plan_sha256:
        raise ValueError("final result was produced against a different plan file")
    if set(result.get("models", {})) != REQUIRED_MODELS:
        raise ValueError("final result does not cover every predeclared model")
    for name in MODELS:
        seeds = [row["seed"] for row in result["models"][name]]
        if seeds != list(SEEDS):
            raise ValueError(f"final result for {name} lacks the three frozen seeds")
    expected_hashes = plan["checkpoints"]["expected_sha256"]
    if set(result.get("checkpoints", {})) != set(expected_hashes):
        raise ValueError("final result lacks the exact frozen checkpoint inventory")
    for key, checkpoint in result["checkpoints"].items():
        if checkpoint.get("sha256") != expected_hashes[key]:
            raise ValueError(f"final result has the wrong checkpoint hash for {key}")
    primary = result.get("primary", {})
    if primary.get("candidate") != CANDIDATE or primary.get("reference") != (
        PRIMARY_REFERENCE
    ):
        raise ValueError("final result names an unexpected primary comparison")
    interval = primary.get("interval", {})
    if interval.get("analysis_contract") != (
        "confirmatory_crossed_seed_whole_flow_cluster_ratio/v2"
    ):
        raise ValueError("final claim requires the frozen crossed-cluster interval")
    if int(interval.get("resamples", 0)) < 10_000:
        raise ValueError("final claim requires at least 10,000 bootstrap resamples")
    if result.get("protocol", {}).get("split", {}).get("train_sources") != []:
        raise ValueError("final evidence must be evaluate-only with no training split")
    if result.get("coordinate_sha256") != result.get("protocol", {}).get(
        "coordinate_sha256"
    ):
        raise ValueError("final result coordinate binding changed")
    for name in MODELS:
        for row in result["models"][name]:
            for partition in ("validation", "test"):
                losses = row[partition].get("per_sample_weighted_smooth_l1", [])
                samples = row[partition].get("samples")
                if samples != len(losses):
                    raise ValueError(
                        f"{name} seed {row['seed']} {partition} lost per-sample alignment"
                    )
                mean = statistics.fmean(losses)
                if not math.isclose(
                    mean,
                    row[partition].get("weighted_smooth_l1", math.nan),
                    rel_tol=1e-6,
                    abs_tol=1e-8,
                ):
                    raise ValueError(
                        f"{name} seed {row['seed']} {partition} mean changed"
                    )

    candidate_rows = result["models"][CANDIDATE]
    reference_rows = result["models"][PRIMARY_REFERENCE]
    candidate_means = [
        statistics.fmean(row["test"]["per_sample_weighted_smooth_l1"])
        for row in candidate_rows
    ]
    reference_means = [
        statistics.fmean(row["test"]["per_sample_weighted_smooth_l1"])
        for row in reference_rows
    ]
    reductions = [
        (reference - candidate) / reference
        for candidate, reference in zip(candidate_means, reference_means, strict=True)
    ]
    mean_reduction = statistics.fmean(reductions)
    for field, expected in (
        ("per_seed_candidate_test_loss", candidate_means),
        ("per_seed_reference_test_loss", reference_means),
        ("per_seed_relative_reduction", reductions),
    ):
        actual = primary.get(field)
        if (
            actual is None
            or len(actual) != len(expected)
            or any(
                not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
                for left, right in zip(actual, expected, strict=True)
            )
        ):
            raise ValueError(f"final result primary field {field} changed")
    if not math.isclose(
        primary.get("mean_relative_reduction", math.nan),
        mean_reduction,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("final result primary mean changed")
    clusters = primary.get("bootstrap_cluster_ids", [])
    rebuilt_interval = _final_interval(
        [row["test"]["per_sample_weighted_smooth_l1"] for row in candidate_rows],
        [row["test"]["per_sample_weighted_smooth_l1"] for row in reference_rows],
        clusters,
        int(plan["primary_endpoint"]["uncertainty"]["bootstrap_seed"]),
    )
    for field in (
        "analysis_contract",
        "mean_relative_reduction",
        "interval_95pct_low",
        "interval_95pct_high",
        "seed_count",
        "cluster_count",
        "coordinates_per_seed",
        "resamples",
        "bootstrap_seed",
        "estimand",
    ):
        if interval.get(field) != rebuilt_interval[field]:
            raise ValueError(f"final result interval field {field} changed")
    rederived = {
        "mean_relative_reduction_at_least_5pct": mean_reduction >= 0.05,
        "interval_lower_bound_strictly_positive": rebuilt_interval["interval_95pct_low"]
        > 0.0,
    }
    controls = result.get("mechanism_controls", [])
    if len(controls) != len(SEEDS):
        raise ValueError("final result lacks mechanism controls for every seed")
    candidate_aggregate = [row["test"]["weighted_smooth_l1"] for row in candidate_rows]
    rotation_harm = []
    permutation_harm = []
    for index, control in enumerate(controls):
        for name in ("query_rotation", "value_permutation"):
            values = control.get(name, {}).get("per_sample_weighted_smooth_l1", [])
            samples = control.get(name, {}).get("samples")
            if samples != len(values) or not values:
                raise ValueError(f"final result has incomplete {name} evidence")
            if not math.isclose(
                statistics.fmean(values),
                control[name].get("weighted_smooth_l1", math.nan),
                rel_tol=1e-6,
                abs_tol=1e-8,
            ):
                raise ValueError(f"final result {name} mean changed")
        rotation_harm.append(
            control["query_rotation"]["weighted_smooth_l1"] - candidate_aggregate[index]
        )
        permutation_harm.append(
            control["value_permutation"]["weighted_smooth_l1"]
            - candidate_aggregate[index]
        )
    rotation_ok = all(value > 0.0 for value in rotation_harm)
    permutation_ok = all(value > 0.0 for value in permutation_harm)
    secondary = result.get("secondary", {})
    stored_gates = secondary.get("gates", {})
    if stored_gates.get("query_rotation_hurts_every_seed") is not rotation_ok:
        raise ValueError("stored query-rotation gate changed")
    if stored_gates.get("value_permutation_hurts_every_seed") is not permutation_ok:
        raise ValueError("stored value-permutation gate changed")
    for field, expected in (
        ("rotation_harm_per_seed", rotation_harm),
        ("permutation_harm_per_seed", permutation_harm),
    ):
        actual = secondary.get(field)
        if (
            actual is None
            or len(actual) != len(expected)
            or any(
                not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
                for left, right in zip(actual, expected, strict=True)
            )
        ):
            raise ValueError(f"stored {field} changed")
    primary_pass = (
        rederived["mean_relative_reduction_at_least_5pct"]
        and rederived["interval_lower_bound_strictly_positive"]
    )
    verdict = "supported" if primary_pass else "rejected"
    mechanism_interpretation = (
        "corroborated" if rotation_ok and permutation_ok else "inconclusive"
    )
    if secondary.get("mechanism_interpretation") != mechanism_interpretation:
        raise ValueError("stored mechanism interpretation changed")
    runtime = secondary.get("runtime_evidence", {})
    expected_runtime = plan["secondary_endpoints"]["runtime_evidence"]
    if (
        runtime.get("sha256") != expected_runtime["expected_sha256"]
        or runtime.get("schema") != expected_runtime["expected_schema"]
    ):
        raise ValueError("stored runtime evidence binding changed")
    if result.get("decision") != verdict:
        raise ValueError(
            f"final result verdict {result.get('decision')!r} disagrees with the "
            f"re-derived verdict {verdict!r}"
        )
    failed = sorted(name for name, passed in rederived.items() if not passed)
    return {
        "schema": CLAIM_SCHEMA,
        "claim": (
            "The frozen v6.1 AHEM state-transition forecaster improves causal "
            "10k-packet next-same-flow forecast loss over its parameter-matched "
            "flow-lane LSTM on the untouched CSE-CIC-IDS2018 holdout day by at "
            "least five percent with a strictly positive crossed flow-cluster "
            "interval"
        ),
        "scope": "forecasting_only",
        "not_retrieval_novelty_claim": True,
        "not_intrusion_detection_efficacy": True,
        "superseded_compound_claim": "the v3.4 H1-H5 preregistration was never executed and remains retired",
        "development_selection_disclosed": True,
        "candidate": CANDIDATE,
        "plan_sha256": plan_sha256,
        "coordinate_sha256": result.get("coordinate_sha256"),
        "checkpoints": result.get("checkpoints"),
        "gates": {
            **rederived,
            "mechanism_controls_pass": rotation_ok and permutation_ok,
        },
        "verdict": verdict,
        "mechanism_interpretation": mechanism_interpretation,
        "failed_gates": failed,
        "ends_study_on_rejection": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = _read(args.result, SCHEMA)
    report = build_claim_report(result, args.plan)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

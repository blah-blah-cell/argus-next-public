"""Hierarchical paired comparison for frozen flow-v3 confirmatory runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from .flow_v2_multiseed import hierarchical_bootstrap
from .flow_v2_protocol import write_manifest
from .flow_v3_protocol import PREREGISTRATION_SHA256
from .flow_v3_train import RUN_SCHEMA
from .flow_v2_train import _canonical_sha256

MULTISEED_SCHEMA = "argus.flow-v3-multiseed-comparison/v1"
MINIMUM_RELATIVE_IMPROVEMENT = 0.05
CANDIDATE = "ahem_v3.4_flow_indexed"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != RUN_SCHEMA:
        raise ValueError(f"{path.name} is not a flow-v3 model run")
    if value.get("test_access", {}).get("status") != "completed":
        raise ValueError(f"{path.name} has no completed one-attempt test result")
    if value.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError(f"{path.name} belongs to another preregistration")
    if (
        value.get("study_role") != "confirmatory"
        or value.get("publication_eligible") is not True
    ):
        raise ValueError(f"{path.name} is not publication-eligible")
    access = value["test_access"]
    test = value.get("test", {})
    if access.get("per_sample_sha256") != _canonical_sha256(
        test.get("per_sample_weighted_smooth_l1")
    ):
        raise ValueError(f"{path.name} per-sample losses do not match the ledger")
    if access.get("sample_coordinates_sha256") != _canonical_sha256(
        test.get("sample_coordinates")
    ):
        raise ValueError(f"{path.name} sample coordinates do not match the ledger")
    return value


def compare(
    metric_paths: list[Path],
    *,
    reference: str,
    resamples: int = 10_000,
    seed: int = 20260829,
) -> dict[str, Any]:
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for path in metric_paths:
        run = _load(path)
        model = run["model"]["name"]
        training_seed = int(run["training"]["seed"])
        runs = grouped.setdefault(model, {})
        if training_seed in runs:
            raise ValueError(f"duplicate {model} run for seed {training_seed}")
        runs[training_seed] = run
    if set(grouped) != {CANDIDATE, reference}:
        raise ValueError(
            "comparison requires exactly the candidate and requested reference"
        )
    candidate = grouped[CANDIDATE]
    baseline = grouped[reference]
    if set(candidate) != set(baseline):
        raise ValueError("candidate and reference training seeds differ")
    training_seeds = sorted(candidate)
    if training_seeds != [20260829, 20260830, 20260831, 20260901, 20260902]:
        raise ValueError("comparison does not contain the five preregistered seeds")
    all_runs = list(candidate.values()) + list(baseline.values())
    if {run["study_role"] for run in all_runs} != {"confirmatory"}:
        raise ValueError("comparison includes non-confirmatory runs")
    if len({run["protocol_sha256"] for run in all_runs}) != 1:
        raise ValueError("runs do not share one frozen protocol")
    if len({run["coordinate_sha256"] for run in all_runs}) != 1:
        raise ValueError("runs do not share frozen coordinates")

    differences_by_seed = []
    cluster_ids_by_seed = []
    baseline_means = []
    candidate_means = []
    for training_seed in training_seeds:
        candidate_test = candidate[training_seed]["test"]
        baseline_test = baseline[training_seed]["test"]
        coordinates = candidate_test["sample_coordinates"]
        if coordinates != baseline_test["sample_coordinates"]:
            raise ValueError("paired sample coordinates differ across models")
        candidate_losses = candidate_test["per_sample_weighted_smooth_l1"]
        baseline_losses = baseline_test["per_sample_weighted_smooth_l1"]
        if len(candidate_losses) != len(baseline_losses) or not candidate_losses:
            raise ValueError("paired test losses have incompatible lengths")
        differences_by_seed.append(
            [
                float(base) - float(ahem)
                for base, ahem in zip(baseline_losses, candidate_losses, strict=True)
            ]
        )
        identities = [
            (
                coordinate["source_file"],
                coordinate["segment_id"],
                coordinate["flow_generation"],
            )
            for coordinate in coordinates
        ]
        mapping = {
            identity: index for index, identity in enumerate(sorted(set(identities)))
        }
        cluster_ids_by_seed.append([mapping[identity] for identity in identities])
        baseline_means.append(fmean(float(value) for value in baseline_losses))
        candidate_means.append(fmean(float(value) for value in candidate_losses))

    paired = hierarchical_bootstrap(
        differences_by_seed,
        cluster_ids_by_seed=cluster_ids_by_seed,
        resamples=resamples,
        seed=seed,
    )
    relative = paired["mean_baseline_minus_candidate"] / max(
        fmean(baseline_means), 1e-12
    )
    result = {
        "schema": MULTISEED_SCHEMA,
        "study_role": "confirmatory",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "protocol_sha256": all_runs[0]["protocol_sha256"],
        "coordinate_sha256": all_runs[0]["coordinate_sha256"],
        "candidate": CANDIDATE,
        "reference": reference,
        "training_seeds": training_seeds,
        "hierarchical_bootstrap_resamples": resamples,
        "bootstrap_contract": "seeds_then_flow_generations_then_coordinates",
        "paired_weighted_forecast_loss": paired,
        "mean_candidate_loss": fmean(candidate_means),
        "mean_reference_loss": fmean(baseline_means),
        "mean_relative_improvement": relative,
        "gates": {
            "paired_interval_strictly_positive": paired["bootstrap_95pct_low"] > 0,
            "relative_improvement_at_least_five_percent": (
                relative >= MINIMUM_RELATIVE_IMPROVEMENT
            ),
        },
    }
    result["gates"]["comparison_passed"] = all(result["gates"].values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--reference",
        choices=("lstm", "ahem_v3.4_local_only", "ahem_v3.4_recency"),
        required=True,
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_manifest(
        args.output,
        compare(
            args.metrics,
            reference=args.reference,
            resamples=args.resamples,
            seed=args.seed,
        ),
    )


if __name__ == "__main__":
    main()

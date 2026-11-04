"""Hierarchical paired comparison for frozen AHEM-v5 runs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

from .flow_v2_multiseed import hierarchical_bootstrap
from .flow_v2_protocol import write_manifest
from .flow_v2_train import _canonical_sha256
from .flow_v3_protocol import ProtocolConfig
from .flow_v3_train import (
    PRIMARY_TEST_SOURCE,
    PRIMARY_TRAIN_SOURCES,
    PRIMARY_VALIDATION_SOURCE,
    TrainingConfig,
    V4_PREREGISTRATION_SHA256,
)
from .flow_v4_screen_report import _metric

MULTISEED_SCHEMA = "argus.ahem-v5-multiseed-comparison/v1"
CANDIDATE = "ahem_v5_source_aligned"
TRAINING_SEEDS = (20260904, 20260905, 20260906, 20260907, 20260908)
LONG_GAP_BUCKETS = ("129_to_1023", "1024_to_4095", "4096_to_9999")
PERSISTENCE = "last_event_persistence"
REFERENCES = (
    "lstm",
    "ahem_v5_local_only",
    "ahem_v5_recency",
    "gru",
    "long_tcn",
    "s4d",
    PERSISTENCE,
)


def _load(path: Path) -> dict[str, Any]:
    value = _metric(path, study_role="confirmatory")
    training_seed = value.get("training", {}).get("seed")
    if (
        type(training_seed) is not int
        or training_seed not in TRAINING_SEEDS
        or value["model"].get("name") not in (CANDIDATE, *REFERENCES[:-1])
        or value["model"].get("fresh_random_initialization") is not True
    ):
        raise ValueError(
            "comparison requires fresh frozen-model runs and training seeds"
        )
    budget = asdict(TrainingConfig(seed=training_seed))
    protocol = value["protocol"]
    if (
        any(
            value["training"].get(field) != expected
            for field, expected in budget.items()
        )
        or value["training"].get("config_sha256") != _canonical_sha256(budget)
        or protocol.get("config") != asdict(ProtocolConfig(seed=20260830))
        or protocol.get("split")
        != {
            "validation_source": PRIMARY_VALIDATION_SOURCE,
            "test_source": PRIMARY_TEST_SOURCE,
            "train_sources": sorted(PRIMARY_TRAIN_SOURCES),
        }
        or protocol.get("journal", {}).get("provenance_complete") is not True
        or protocol.get("journal", {}).get("datasets") != ["CSE-CIC-IDS2018"]
        or value["test"]["samples"] != 512
        or any(
            coordinate.get("source_file") != PRIMARY_TEST_SOURCE
            for coordinate in value["test"]["sample_coordinates"]
        )
    ):
        raise ValueError(
            "comparison inputs differ from the frozen training/data contract"
        )
    return value


def _paired(
    candidate: dict[int, dict[str, Any]],
    baseline: dict[int, dict[str, Any]],
    *,
    resamples: int,
    bootstrap_seed: int,
    bucket: str | None = None,
) -> dict[str, Any]:
    differences_by_seed = []
    cluster_ids_by_seed = []
    candidate_means = []
    baseline_means = []
    sample_counts = []
    for training_seed in TRAINING_SEEDS:
        candidate_test = candidate[training_seed]["test"]
        baseline_test = baseline[training_seed]["test"]
        coordinates = candidate_test["sample_coordinates"]
        if coordinates != baseline_test["sample_coordinates"]:
            raise ValueError("paired sample coordinates differ across models")
        indices = [
            index
            for index, coordinate in enumerate(coordinates)
            if bucket is None or coordinate.get("long_gap_bucket") == bucket
        ]
        if not indices:
            raise ValueError("a requested long-gap bucket has no paired samples")
        candidate_losses = [
            float(candidate_test["per_sample_weighted_smooth_l1"][index])
            for index in indices
        ]
        baseline_losses = [
            float(baseline_test["per_sample_weighted_smooth_l1"][index])
            for index in indices
        ]
        identities = [
            (
                coordinates[index]["source_file"],
                coordinates[index]["segment_id"],
                coordinates[index]["flow_generation"],
            )
            for index in indices
        ]
        mapping = {
            identity: index for index, identity in enumerate(sorted(set(identities)))
        }
        differences_by_seed.append(
            [
                baseline_loss - candidate_loss
                for baseline_loss, candidate_loss in zip(
                    baseline_losses, candidate_losses, strict=True
                )
            ]
        )
        cluster_ids_by_seed.append([mapping[identity] for identity in identities])
        candidate_means.append(fmean(candidate_losses))
        baseline_means.append(fmean(baseline_losses))
        sample_counts.append(len(indices))
    paired = hierarchical_bootstrap(
        differences_by_seed,
        cluster_ids_by_seed=cluster_ids_by_seed,
        resamples=resamples,
        seed=bootstrap_seed,
    )
    relative = paired["mean_baseline_minus_candidate"] / max(
        fmean(baseline_means), 1e-12
    )
    return {
        "bucket": bucket,
        "samples_per_seed": sample_counts,
        "paired_weighted_forecast_loss": paired,
        "mean_candidate_loss": fmean(candidate_means),
        "mean_reference_loss": fmean(baseline_means),
        "mean_relative_improvement": relative,
    }


def compare(
    metric_paths: list[Path],
    *,
    reference: str,
    resamples: int = 10_000,
    seed: int = 20260904,
) -> dict[str, Any]:
    if reference not in REFERENCES:
        raise ValueError("unknown frozen comparison reference")
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for path in metric_paths:
        run = _load(path)
        model = run["model"]["name"]
        training_seed = int(run["training"]["seed"])
        if training_seed in grouped.setdefault(model, {}):
            raise ValueError(f"duplicate {model} run for seed {training_seed}")
        grouped[model][training_seed] = run
    if reference == PERSISTENCE:
        if set(grouped) != {CANDIDATE}:
            raise ValueError("persistence comparison requires only candidate runs")
        persistence_runs = {}
        for training_seed, candidate_run in grouped[CANDIDATE].items():
            persistence_run = deepcopy(candidate_run)
            persistence_run["test"]["per_sample_weighted_smooth_l1"] = list(
                candidate_run["test"]["per_sample_persistence_weighted_smooth_l1"]
            )
            persistence_runs[training_seed] = persistence_run
        grouped[reference] = persistence_runs
    elif set(grouped) != {CANDIDATE, reference}:
        raise ValueError("comparison requires exactly candidate and reference")
    if (
        tuple(sorted(grouped[CANDIDATE])) != TRAINING_SEEDS
        or tuple(sorted(grouped[reference])) != TRAINING_SEEDS
    ):
        raise ValueError("comparison does not contain the five preregistered seeds")
    all_runs = list(grouped[CANDIDATE].values()) + list(grouped[reference].values())
    if (
        len({run["protocol_sha256"] for run in all_runs}) != 1
        or len({run["coordinate_sha256"] for run in all_runs}) != 1
    ):
        raise ValueError("comparison mixes protocols or coordinates")
    for run in all_runs[1:]:
        if run["source_provenance"] != all_runs[0]["source_provenance"] or any(
            run["test"][field] != all_runs[0]["test"][field]
            for field in (
                "sample_coordinates",
                "per_sample_persistence_weighted_smooth_l1",
            )
        ):
            raise ValueError(
                "comparison mixes source or actual paired test observations"
            )

    overall = _paired(
        grouped[CANDIDATE],
        grouped[reference],
        resamples=resamples,
        bootstrap_seed=seed,
    )
    coordinates = grouped[CANDIDATE][TRAINING_SEEDS[0]]["test"]["sample_coordinates"]
    bucket_counts = {
        bucket: sum(
            coordinate.get("long_gap_bucket") == bucket for coordinate in coordinates
        )
        for bucket in LONG_GAP_BUCKETS
    }
    gap_results = {}
    for offset, bucket in enumerate(LONG_GAP_BUCKETS, start=1):
        if bucket_counts[bucket] >= 50:
            gap_results[bucket] = _paired(
                grouped[CANDIDATE],
                grouped[reference],
                resamples=resamples,
                bootstrap_seed=seed + offset,
                bucket=bucket,
            )
    result = {
        "schema": MULTISEED_SCHEMA,
        "study_role": "confirmatory" if resamples == 10000 else "posthoc_comparison",
        "publication_evidence": resamples == 10000,
        "preregistration_sha256": V4_PREREGISTRATION_SHA256,
        "protocol_sha256": all_runs[0]["protocol_sha256"],
        "coordinate_sha256": all_runs[0]["coordinate_sha256"],
        "candidate": CANDIDATE,
        "reference": reference,
        "training_seeds": list(TRAINING_SEEDS),
        "hierarchical_bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "bootstrap_contract": "seeds_then_flow_generations_then_coordinates",
        "overall": overall,
        "primary_bucket_counts": bucket_counts,
        "sufficiently_populated_long_gap_buckets": gap_results,
        "gates": {
            "candidate_better_overall": overall["paired_weighted_forecast_loss"][
                "mean_baseline_minus_candidate"
            ]
            > 0,
            "paired_interval_strictly_positive": overall[
                "paired_weighted_forecast_loss"
            ]["bootstrap_95pct_low"]
            > 0,
            "relative_improvement_at_least_five_percent": overall[
                "mean_relative_improvement"
            ]
            >= 0.05,
            "all_populated_long_gap_means_positive": bool(gap_results)
            and all(
                value["paired_weighted_forecast_loss"]["mean_baseline_minus_candidate"]
                > 0
                for value in gap_results.values()
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
        choices=REFERENCES,
        required=True,
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260904)
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

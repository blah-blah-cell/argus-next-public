"""Hierarchical paired bootstrap across flow-v2 training seeds."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from .flow_v2_compare import _load, _load_runtime
from .flow_v2_protocol import FEATURE_SCHEMA, PREREGISTRATION_SHA256, write_manifest

MULTISEED_SCHEMA = "argus.flow-v2-multiseed-comparison/v1"
MINIMUM_PRIMARY_RELATIVE_IMPROVEMENT = 0.05


def hierarchical_bootstrap(
    differences_by_seed: list[list[float]],
    *,
    cluster_ids_by_seed: list[list[int]] | None = None,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if not differences_by_seed or any(not values for values in differences_by_seed):
        raise ValueError("hierarchical bootstrap needs non-empty seed samples")
    if resamples < 100:
        raise ValueError("hierarchical bootstrap requires at least 100 resamples")
    if cluster_ids_by_seed is None:
        cluster_ids_by_seed = [
            list(range(len(values))) for values in differences_by_seed
        ]
    if len(cluster_ids_by_seed) != len(differences_by_seed) or any(
        len(cluster_ids) != len(values)
        for cluster_ids, values in zip(
            cluster_ids_by_seed, differences_by_seed, strict=True
        )
    ):
        raise ValueError("hierarchical bootstrap cluster IDs must match observations")
    clustered_by_seed: list[list[list[float]]] = []
    for values, cluster_ids in zip(
        differences_by_seed, cluster_ids_by_seed, strict=True
    ):
        grouped: dict[int, list[float]] = defaultdict(list)
        for cluster_id, value in zip(cluster_ids, values, strict=True):
            grouped[cluster_id].append(value)
        clustered_by_seed.append(list(grouped.values()))
    rng = random.Random(seed)
    seed_count = len(differences_by_seed)
    sampled_means = []
    for _ in range(resamples):
        selected_seeds = [rng.randrange(seed_count) for _ in range(seed_count)]
        seed_means = []
        for seed_index in selected_seeds:
            clusters = clustered_by_seed[seed_index]
            sampled_values = []
            for _ in range(len(clusters)):
                cluster = clusters[rng.randrange(len(clusters))]
                sampled_values.extend(
                    cluster[rng.randrange(len(cluster))] for _ in range(len(cluster))
                )
            seed_means.append(fmean(sampled_values))
        sampled_means.append(fmean(seed_means))
    sampled_means.sort()
    seed_means = [fmean(values) for values in differences_by_seed]
    return {
        "mean_baseline_minus_candidate": fmean(seed_means),
        "seed_means": seed_means,
        "flow_cluster_counts_by_seed": [
            len(clusters) for clusters in clustered_by_seed
        ],
        "bootstrap_95pct_low": sampled_means[int(0.025 * resamples)],
        "bootstrap_95pct_high": sampled_means[
            min(resamples - 1, int(0.975 * resamples))
        ],
        "candidate_per_sample_win_rate": sum(
            value > 0.0 for values in differences_by_seed for value in values
        )
        / sum(len(values) for values in differences_by_seed),
    }


def bootstrap_mean(
    values: list[float], *, resamples: int, seed: int
) -> dict[str, float]:
    if not values:
        raise ValueError("bootstrap mean needs observations")
    if resamples < 100:
        raise ValueError("bootstrap mean requires at least 100 resamples")
    rng = random.Random(seed)
    size = len(values)
    sampled = sorted(
        fmean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(resamples)
    )
    return {
        "mean": fmean(values),
        "bootstrap_95pct_low": sampled[int(0.025 * resamples)],
        "bootstrap_95pct_high": sampled[min(resamples - 1, int(0.975 * resamples))],
    }


def compare_multiseed(
    metric_paths: list[Path],
    *,
    runtime_paths: list[Path] | None = None,
    reference: str = "lstm",
    resamples: int = 10_000,
    seed: int = 20260824,
) -> dict[str, Any]:
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for path in metric_paths:
        run = _load(path)
        model = run["model"]["name"]
        training_seed = int(run["training"]["seed"])
        if training_seed in grouped[model]:
            raise ValueError(f"duplicate {model!r} run for seed {training_seed}")
        grouped[model][training_seed] = run
    if reference not in grouped:
        raise ValueError(f"reference model {reference!r} is absent")
    if len(grouped) < 2:
        raise ValueError("multi-seed comparison requires at least two models")
    expected_seeds = set(grouped[reference])
    if not expected_seeds:
        raise ValueError("reference model has no training seeds")
    if any(set(runs) != expected_seeds for runs in grouped.values()):
        raise ValueError("every model must have the exact same training seeds")

    expected_runtime_keys = {
        (model, training_seed) for model in grouped for training_seed in expected_seeds
    }

    all_runs = [run for runs in grouped.values() for run in runs.values()]
    study_roles = {run["study_role"] for run in all_runs}
    if len(study_roles) != 1:
        raise ValueError("multi-seed runs mix confirmatory and development evidence")
    study_role = next(iter(study_roles))
    protocol_hashes = {run["protocol_sha256"] for run in all_runs}
    coordinate_hashes = {run["coordinate_sha256"] for run in all_runs}
    if len(protocol_hashes) != 1 or len(coordinate_hashes) != 1:
        raise ValueError("multi-seed runs do not share one frozen protocol")
    contexts = {int(run["protocol"]["config"]["context_packets"]) for run in all_runs}
    if len(contexts) != 1:
        raise ValueError("multi-seed runs do not share one context length")
    protocol_configs = {
        json.dumps(run["protocol"]["config"], sort_keys=True) for run in all_runs
    }
    if len(protocol_configs) != 1:
        raise ValueError("multi-seed runs do not share one protocol configuration")
    protocol_splits = {
        json.dumps(run["protocol"]["split"], sort_keys=True) for run in all_runs
    }
    if len(protocol_splits) != 1:
        raise ValueError("multi-seed runs do not share one source split")
    preregistration_hashes = {
        run["protocol"].get("preregistration_sha256") for run in all_runs
    }
    if preregistration_hashes != {PREREGISTRATION_SHA256}:
        raise ValueError("multi-seed runs do not match the pre-registration")
    training_fields = (
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "patience",
        "checkpoint_every",
        "seed",
    )
    training_contracts: dict[int, dict[str, Any]] = {}
    for training_seed in expected_seeds:
        contracts = {
            json.dumps(
                {field: run["training"][field] for field in training_fields},
                sort_keys=True,
            )
            for runs in grouped.values()
            for seed_value, run in runs.items()
            if seed_value == training_seed
        }
        if len(contracts) != 1:
            raise ValueError("models use different training configurations")
        training_contracts[training_seed] = json.loads(next(iter(contracts)))
    protocol_sha256 = next(iter(protocol_hashes))
    coordinate_sha256 = next(iter(coordinate_hashes))
    source_provenance = {
        json.dumps(run.get("source_provenance"), sort_keys=True) for run in all_runs
    }
    if len(source_provenance) != 1:
        raise ValueError("multi-seed runs do not share one source revision and state")

    runtimes: dict[tuple[str, int], dict[str, Any]] = {}
    for path in runtime_paths or []:
        runtime = _load_runtime(path)
        key = (runtime["model"], int(runtime["training_seed"]))
        if key in runtimes:
            raise ValueError(f"duplicate runtime benchmark for {key}")
        if runtime["protocol_sha256"] != protocol_sha256:
            raise ValueError("runtime benchmark uses another protocol")
        if runtime["coordinate_sha256"] != coordinate_sha256:
            raise ValueError("runtime benchmark uses other test coordinates")
        if key not in expected_runtime_keys:
            raise ValueError(f"runtime benchmark has no matching model run: {key}")
        if runtime.get("training_source_provenance") != grouped[key[0]][key[1]].get(
            "source_provenance"
        ):
            raise ValueError("runtime benchmark belongs to another training source")
        runtimes[key] = runtime

    comparisons = {}
    ordered_seeds = sorted(expected_seeds)
    for model, runs in sorted(grouped.items()):
        if model == reference:
            continue
        differences_by_seed = []
        cluster_ids_by_seed = []
        latency_ratios = []
        candidate_seed_means = []
        baseline_seed_means = []
        quality_by_seed = []
        memory_effect_by_seed = []
        for training_seed in ordered_seeds:
            candidate = runs[training_seed]
            baseline = grouped[reference][training_seed]
            candidate_losses = candidate["test"]["per_sample_weighted_smooth_l1"]
            baseline_losses = baseline["test"]["per_sample_weighted_smooth_l1"]
            if len(candidate_losses) != len(baseline_losses):
                raise ValueError("paired test losses changed length across models")
            candidate_clusters = candidate["test"]["sample_stream_indices"]
            baseline_clusters = baseline["test"]["sample_stream_indices"]
            if candidate_clusters != baseline_clusters:
                raise ValueError("paired test flow clusters changed across models")
            cluster_ids_by_seed.append(candidate_clusters)
            differences_by_seed.append(
                [
                    baseline_value - candidate_value
                    for candidate_value, baseline_value in zip(
                        candidate_losses, baseline_losses, strict=True
                    )
                ]
            )
            candidate_seed_means.append(fmean(candidate_losses))
            baseline_seed_means.append(fmean(baseline_losses))
            quality_by_seed.append(
                {"seed": training_seed, **candidate["test"]["quality"]}
            )
            memory_effect_by_seed.append(
                {
                    "seed": training_seed,
                    **candidate["test"].get("memory_effect", {"available": False}),
                }
            )
            candidate_runtime = runtimes.get((model, training_seed))
            baseline_runtime = runtimes.get((reference, training_seed))
            if candidate_runtime is not None and baseline_runtime is not None:
                latency_ratios.append(
                    baseline_runtime["latency"]["warm_p95_ms"]
                    / candidate_runtime["latency"]["warm_p95_ms"]
                )
        paired = hierarchical_bootstrap(
            differences_by_seed,
            cluster_ids_by_seed=cluster_ids_by_seed,
            resamples=resamples,
            seed=seed,
        )
        baseline_mean = fmean(baseline_seed_means)
        relative_improvement = paired["mean_baseline_minus_candidate"] / max(
            baseline_mean, 1e-12
        )
        latency_interval = (
            bootstrap_mean(latency_ratios, resamples=resamples, seed=seed + 1)
            if latency_ratios
            else None
        )
        comparisons[model] = {
            "paired_weighted_forecast_loss": paired,
            "mean_candidate_weighted_forecast_loss": fmean(candidate_seed_means),
            "mean_reference_weighted_forecast_loss": baseline_mean,
            "relative_forecast_loss_improvement": relative_improvement,
            "quality_by_seed": quality_by_seed,
            "memory_effect_by_seed": memory_effect_by_seed,
            "cpu_p95_speedup_by_seed": latency_ratios,
            "mean_cpu_p95_speedup": (fmean(latency_ratios) if latency_ratios else None),
            "cpu_p95_speedup_bootstrap": latency_interval,
            "gates": {
                "at_least_three_seeds": len(ordered_seeds) >= 3,
                "paired_hierarchical_interval_strictly_positive": paired[
                    "bootstrap_95pct_low"
                ]
                > 0.0,
                "at_least_five_percent_relative_improvement": relative_improvement
                >= MINIMUM_PRIMARY_RELATIVE_IMPROVEMENT,
                "fresh_process_runtime_complete": len(latency_ratios)
                == len(ordered_seeds),
                "cpu_p95_speedup_interval_no_slower": (
                    latency_interval["bootstrap_95pct_low"] >= 1.0
                    if latency_interval is not None
                    else None
                ),
            },
        }

    blockers = [
        "requires untouched source-day replication",
        "retrieval needs immutable causal evidence-pair labels",
        "forecast quality is not intrusion-detection efficacy",
    ]
    if len(ordered_seeds) < 3:
        blockers.append("fewer than three training seeds")
    if set(runtimes) != expected_runtime_keys:
        blockers.append("fresh-process runtime benchmark matrix is incomplete")
    frozen_source = all_runs[0].get("source_provenance") or {}
    if frozen_source.get("available") is not True:
        blockers.append("training source revision is unavailable")
    elif frozen_source.get("git_dirty") is not False:
        blockers.append("training source tree was dirty")
    if any(
        run.get("protocol", {}).get("journal", {}).get("provenance_complete")
        is not True
        or run.get("protocol", {}).get("journal", {}).get("datasets")
        != ["CSE-CIC-IDS2018"]
        for run in all_runs
    ):
        blockers.append("primary journal CSE-CIC-IDS2018 provenance is incomplete")
    if study_role != "confirmatory":
        blockers.append("runs are development-only rather than confirmatory")
    if any(
        runtime.get("benchmark_source_provenance", {}).get("available") is not True
        or runtime.get("benchmark_source_provenance", {}).get("git_dirty") is not False
        for runtime in runtimes.values()
    ):
        blockers.append("runtime benchmark source was unavailable or dirty")
    return {
        "schema": MULTISEED_SCHEMA,
        "task": "causal_per_flow_next_packet_forecast_only",
        "not_intrusion_detection_efficacy": True,
        "not_retrieval_accuracy": True,
        "feature_schema": FEATURE_SCHEMA,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "protocol_sha256": protocol_sha256,
        "coordinate_sha256": coordinate_sha256,
        "context_packets": next(iter(contexts)),
        "protocol_config": json.loads(next(iter(protocol_configs))),
        "reference_model": reference,
        "study_role": study_role,
        "protocol_split": json.loads(next(iter(protocol_splits))),
        "models": sorted(grouped),
        "training_seeds": ordered_seeds,
        "training_contracts": [
            training_contracts[training_seed] for training_seed in ordered_seeds
        ],
        "seed_count_per_model": len(ordered_seeds),
        "hierarchical_bootstrap_resamples": resamples,
        "bootstrap_contract": (
            "paired_resample_training_seeds_then_flow_streams_then_coordinates"
        ),
        "minimum_primary_relative_improvement": MINIMUM_PRIMARY_RELATIVE_IMPROVEMENT,
        "publication_evidence": False,
        "publication_blockers": blockers,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, action="append", required=True)
    parser.add_argument("--runtime", type=Path, action="append")
    parser.add_argument("--reference", default="lstm")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_multiseed(
        args.metrics,
        runtime_paths=args.runtime,
        reference=args.reference,
        resamples=args.resamples,
        seed=args.seed,
    )
    write_manifest(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

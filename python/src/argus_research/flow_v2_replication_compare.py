"""Aggregate untouched flow-v2 replication runs across training seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from .flow_v2_multiseed import (
    MINIMUM_PRIMARY_RELATIVE_IMPROVEMENT,
    hierarchical_bootstrap,
)
from .flow_v2_protocol import FEATURE_SCHEMA, PREREGISTRATION_SHA256, write_manifest
from .flow_v2_replication import REPLICATION_SCHEMA
from .flow_v2_replication_protocol import REPLICATION_PROTOCOL_SCHEMA
from .flow_v2_train import _canonical_sha256

REPLICATION_COMPARISON_SCHEMA = "argus.flow-v2-replication-comparison/v1"


def compare_replication(
    paths: list[Path],
    *,
    reference: str = "lstm",
    resamples: int = 10_000,
    seed: int = 20260828,
) -> dict[str, Any]:
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for path in paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("schema") != REPLICATION_SCHEMA:
            raise ValueError(f"{path} is not an untouched replication run")
        if run.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError(f"{path} uses another feature schema")
        if run.get("study_role") != "confirmatory":
            raise ValueError(f"{path} is not confirmatory replication evidence")
        losses = run.get("test", {}).get("per_sample_weighted_smooth_l1")
        if not isinstance(losses, list) or not losses:
            raise ValueError(f"{path} lacks per-sample replication evidence")
        losses_sha256 = hashlib.sha256(
            json.dumps(losses, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if run["test"].get("per_sample_sha256") != losses_sha256:
            raise ValueError(f"{path} has invalid per-sample replication evidence")
        stream_indices = run["test"].get("sample_stream_indices")
        stream_sha256 = hashlib.sha256(
            json.dumps(stream_indices, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            not isinstance(stream_indices, list)
            or len(stream_indices) != len(losses)
            or any(not isinstance(item, int) or item < 0 for item in stream_indices)
            or run["test"].get("sample_stream_indices_sha256") != stream_sha256
        ):
            raise ValueError(f"{path} lacks valid replication flow clusters")
        protocol = run.get("replication_protocol", {})
        if (
            protocol.get("schema") != REPLICATION_PROTOCOL_SCHEMA
            or protocol.get("preregistration_sha256") != PREREGISTRATION_SHA256
            or _canonical_sha256(protocol) != run.get("replication_protocol_sha256")
            or protocol.get("primary_protocol_sha256")
            != run.get("primary_protocol_sha256")
            or protocol.get("coordinate_sha256")
            != run.get("replication_coordinate_sha256")
            or protocol.get("journal", {}).get("provenance_complete") is not True
            or protocol.get("journal", {}).get("datasets") != ["CIC-IDS2017"]
        ):
            raise ValueError(f"{path} has invalid frozen replication evidence")
        model = run["model"]["name"]
        training_seed = int(run["training"]["seed"])
        if training_seed in grouped[model]:
            raise ValueError(
                f"duplicate {model!r} replication for seed {training_seed}"
            )
        grouped[model][training_seed] = run
    if reference not in grouped or len(grouped) < 2:
        raise ValueError("replication comparison needs a reference and candidate")
    expected_seeds = set(grouped[reference])
    if any(set(runs) != expected_seeds for runs in grouped.values()):
        raise ValueError("every replication model must use the same training seeds")
    all_runs = [run for runs in grouped.values() for run in runs.values()]
    for field in (
        "primary_protocol_sha256",
        "primary_coordinate_sha256",
        "replication_protocol_sha256",
        "replication_coordinate_sha256",
    ):
        if len({run[field] for run in all_runs}) != 1:
            raise ValueError(f"replication runs disagree on {field}")
    source_states = {
        json.dumps(run.get("source_provenance"), sort_keys=True) for run in all_runs
    }
    if len(source_states) != 1:
        raise ValueError("replication runs do not share one source revision and state")
    evaluation_states = {
        json.dumps(run.get("evaluation_source_provenance"), sort_keys=True)
        for run in all_runs
    }
    if len(evaluation_states) != 1:
        raise ValueError("replication runs use different evaluator source states")
    if evaluation_states != source_states:
        raise ValueError("replication evaluator source differs from training")
    contexts = {int(run["replication_protocol"]["context_packets"]) for run in all_runs}
    if len(contexts) != 1:
        raise ValueError("replication runs do not share one context length")

    comparisons = {}
    ordered_seeds = sorted(expected_seeds)
    for model, runs in sorted(grouped.items()):
        if model == reference:
            continue
        differences_by_seed = []
        cluster_ids_by_seed = []
        candidate_means = []
        baseline_means = []
        for training_seed in ordered_seeds:
            candidate = runs[training_seed]["test"]["per_sample_weighted_smooth_l1"]
            baseline = grouped[reference][training_seed]["test"][
                "per_sample_weighted_smooth_l1"
            ]
            if len(candidate) != len(baseline):
                raise ValueError("replication losses are not paired")
            candidate_clusters = runs[training_seed]["test"]["sample_stream_indices"]
            baseline_clusters = grouped[reference][training_seed]["test"][
                "sample_stream_indices"
            ]
            if candidate_clusters != baseline_clusters:
                raise ValueError("replication flow clusters are not paired")
            cluster_ids_by_seed.append(candidate_clusters)
            differences_by_seed.append(
                [
                    baseline_value - candidate_value
                    for candidate_value, baseline_value in zip(
                        candidate, baseline, strict=True
                    )
                ]
            )
            candidate_means.append(fmean(candidate))
            baseline_means.append(fmean(baseline))
        paired = hierarchical_bootstrap(
            differences_by_seed,
            cluster_ids_by_seed=cluster_ids_by_seed,
            resamples=resamples,
            seed=seed,
        )
        baseline_mean = fmean(baseline_means)
        relative = paired["mean_baseline_minus_candidate"] / max(baseline_mean, 1e-12)
        comparisons[model] = {
            "paired_weighted_forecast_loss": paired,
            "mean_candidate_weighted_forecast_loss": fmean(candidate_means),
            "mean_reference_weighted_forecast_loss": baseline_mean,
            "relative_forecast_loss_improvement": relative,
            "gates": {
                "at_least_three_seeds": len(ordered_seeds) >= 3,
                "paired_hierarchical_interval_strictly_positive": paired[
                    "bootstrap_95pct_low"
                ]
                > 0.0,
                "at_least_five_percent_relative_improvement": relative
                >= MINIMUM_PRIMARY_RELATIVE_IMPROVEMENT,
            },
        }

    source = all_runs[0].get("source_provenance") or {}
    blockers = []
    if len(ordered_seeds) < 3:
        blockers.append("fewer than three training seeds")
    if source.get("available") is not True:
        blockers.append("training source revision is unavailable")
    elif source.get("git_dirty") is not False:
        blockers.append("training source tree was dirty")
    evaluation_source = all_runs[0].get("evaluation_source_provenance") or {}
    if evaluation_source.get("available") is not True:
        blockers.append("replication evaluator source revision is unavailable")
    elif evaluation_source.get("git_dirty") is not False:
        blockers.append("replication evaluator source tree was dirty")
    return {
        "schema": REPLICATION_COMPARISON_SCHEMA,
        "task": "untouched_causal_per_flow_next_packet_forecast_only",
        "not_intrusion_detection_efficacy": True,
        "feature_schema": FEATURE_SCHEMA,
        "study_role": "confirmatory",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "primary_protocol_sha256": all_runs[0]["primary_protocol_sha256"],
        "replication_protocol_sha256": all_runs[0]["replication_protocol_sha256"],
        "replication_coordinate_sha256": all_runs[0]["replication_coordinate_sha256"],
        "context_packets": next(iter(contexts)),
        "reference_model": reference,
        "models": sorted(grouped),
        "training_seeds": ordered_seeds,
        "hierarchical_bootstrap_resamples": resamples,
        "bootstrap_contract": (
            "paired_resample_training_seeds_then_flow_streams_then_coordinates"
        ),
        "publication_evidence": not blockers,
        "publication_blockers": blockers,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, action="append", required=True)
    parser.add_argument("--reference", default="lstm")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_replication(
        args.metrics,
        reference=args.reference,
        resamples=args.resamples,
        seed=args.seed,
    )
    write_manifest(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

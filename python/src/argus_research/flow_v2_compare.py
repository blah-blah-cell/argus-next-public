"""Compare flow-v2 model runs on identical frozen test coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import fmean
from typing import Any

from .flow_v2_protocol import FEATURE_SCHEMA, write_manifest
from .flow_v2_runtime_benchmark import BENCHMARK_SCHEMA
from .flow_v2_train import RUN_SCHEMA

COMPARISON_SCHEMA = "argus.flow-v2-paired-comparison/v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != RUN_SCHEMA:
        raise ValueError(f"{path} is not a flow-v2 model run")
    if value.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError(f"{path} uses an incompatible feature schema")
    if value.get("study_role") not in {
        "confirmatory",
        "exploratory",
        "diagnostic",
    }:
        raise ValueError(f"{path} lacks an explicit study role")
    losses = value.get("test", {}).get("per_sample_weighted_smooth_l1")
    if not isinstance(losses, list) or not losses:
        raise ValueError(f"{path} lacks per-sample confirmatory test losses")
    losses_sha256 = hashlib.sha256(
        json.dumps(losses, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if value["test"].get("per_sample_sha256") != losses_sha256:
        raise ValueError(f"{path} has invalid per-sample test evidence")
    stream_indices = value["test"].get("sample_stream_indices")
    if (
        not isinstance(stream_indices, list)
        or len(stream_indices) != len(losses)
        or any(not isinstance(value, int) or value < 0 for value in stream_indices)
        or value["test"].get("sample_stream_indices_sha256")
        != hashlib.sha256(
            json.dumps(stream_indices, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        raise ValueError(f"{path} lacks valid per-flow cluster evidence")
    access = value.get("test_access", {})
    if (
        access.get("schema") != "argus.flow-v2-test-access/v1"
        or access.get("status") != "completed"
        or access.get("attempt") != 1
        or access.get("test_per_sample_sha256") != losses_sha256
        or access.get("test_sample_stream_indices_sha256")
        != value["test"].get("sample_stream_indices_sha256")
    ):
        raise ValueError(f"{path} lacks a valid one-attempt test-access ledger")
    return value


def _load_runtime(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != BENCHMARK_SCHEMA:
        raise ValueError(f"{path} is not a fresh-process runtime benchmark")
    return value


def paired_bootstrap(
    candidate: list[float],
    baseline: list[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("paired losses must have equal non-zero length")
    if resamples < 100:
        raise ValueError("paired bootstrap requires at least 100 resamples")
    differences = [
        baseline_value - candidate_value
        for candidate_value, baseline_value in zip(candidate, baseline, strict=True)
    ]
    rng = random.Random(seed)
    size = len(differences)
    sampled = sorted(
        fmean(differences[rng.randrange(size)] for _ in range(size))
        for _ in range(resamples)
    )
    low = sampled[int(0.025 * resamples)]
    high = sampled[min(resamples - 1, int(0.975 * resamples))]
    return {
        "mean_baseline_minus_candidate": fmean(differences),
        "bootstrap_95pct_low": low,
        "bootstrap_95pct_high": high,
        "candidate_per_sample_win_rate": sum(value > 0 for value in differences) / size,
    }


def compare(
    paths: list[Path],
    *,
    runtime_paths: list[Path] | None = None,
    reference: str = "lstm",
    resamples: int = 10_000,
    seed: int = 20260823,
) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("paired comparison requires at least two model runs")
    runs: dict[str, dict[str, Any]] = {}
    for path in paths:
        run = _load(path)
        name = run["model"]["name"]
        if name in runs:
            raise ValueError(f"duplicate model run {name!r}")
        runs[name] = run
    if reference not in runs:
        raise ValueError(f"reference model {reference!r} is absent")

    runtimes: dict[str, dict[str, Any]] = {}
    for path in runtime_paths or []:
        runtime = _load_runtime(path)
        name = runtime["model"]
        if name in runtimes:
            raise ValueError(f"duplicate runtime benchmark {name!r}")
        runtimes[name] = runtime

    protocol_hashes = {run["protocol_sha256"] for run in runs.values()}
    coordinate_hashes = {run["coordinate_sha256"] for run in runs.values()}
    if len(protocol_hashes) != 1 or len(coordinate_hashes) != 1:
        raise ValueError("model runs do not share one frozen comparison protocol")
    protocol_sha256 = next(iter(protocol_hashes))
    coordinate_sha256 = next(iter(coordinate_hashes))
    source_provenance = {
        json.dumps(run.get("source_provenance"), sort_keys=True)
        for run in runs.values()
    }
    if len(source_provenance) != 1:
        raise ValueError("model runs do not share one source revision and state")
    study_roles = {run["study_role"] for run in runs.values()}
    if len(study_roles) != 1:
        raise ValueError("model runs mix confirmatory and development evidence")
    for name, runtime in runtimes.items():
        if name not in runs:
            raise ValueError(f"runtime benchmark has no matching model run: {name}")
        if runtime.get("protocol_sha256") != protocol_sha256:
            raise ValueError("runtime benchmark uses another protocol")
        if runtime.get("coordinate_sha256") != coordinate_sha256:
            raise ValueError("runtime benchmark uses other test coordinates")
        if int(runtime.get("training_seed", -1)) != int(runs[name]["training"]["seed"]):
            raise ValueError("runtime benchmark uses another training seed")
        if runtime.get("training_source_provenance") != runs[name].get(
            "source_provenance"
        ):
            raise ValueError("runtime benchmark belongs to another training source")
    baseline = runs[reference]
    baseline_losses = baseline["test"]["per_sample_weighted_smooth_l1"]
    comparisons = {}
    for name, run in runs.items():
        if name == reference:
            continue
        candidate_losses = run["test"]["per_sample_weighted_smooth_l1"]
        if (
            run["test"]["sample_stream_indices"]
            != baseline["test"]["sample_stream_indices"]
        ):
            raise ValueError("paired test flow clusters changed across models")
        paired = paired_bootstrap(
            candidate_losses,
            baseline_losses,
            resamples=resamples,
            seed=seed,
        )
        candidate_runtime = runtimes.get(name)
        baseline_runtime = runtimes.get(reference)
        candidate_p95 = (
            float(candidate_runtime["latency"]["warm_p95_ms"])
            if candidate_runtime is not None
            else None
        )
        baseline_p95 = (
            float(baseline_runtime["latency"]["warm_p95_ms"])
            if baseline_runtime is not None
            else None
        )
        speedup = (
            baseline_p95 / candidate_p95
            if candidate_p95 is not None and baseline_p95 is not None
            else None
        )
        comparisons[name] = {
            "paired_weighted_forecast_loss": paired,
            "quality": run["test"]["quality"],
            "reference_quality": baseline["test"]["quality"],
            "cpu_p95_latency_ms": candidate_p95,
            "reference_cpu_p95_latency_ms": baseline_p95,
            "cpu_p95_speedup_vs_reference": speedup,
            "runtime_benchmark_scope": (
                candidate_runtime["latency"]["scope"]
                if candidate_runtime is not None
                else None
            ),
            "memory_effect": run["test"].get("memory_effect", {"available": False}),
            "gates": {
                "paired_loss_interval_strictly_positive": paired["bootstrap_95pct_low"]
                > 0.0,
                "cpu_p95_no_slower_than_reference": (
                    candidate_p95 <= baseline_p95
                    if candidate_p95 is not None and baseline_p95 is not None
                    else None
                ),
            },
        }
    blockers = [
        "single training seed per model",
        "requires untouched source-day replication",
        "retrieval needs immutable causal evidence-pair labels",
        "forecast quality is not intrusion-detection efficacy",
    ]
    if set(runtimes) != set(runs):
        blockers.append(
            "fresh-process runtime benchmark missing for one or more models"
        )
    frozen_source = next(iter(runs.values())).get("source_provenance") or {}
    if frozen_source.get("available") is not True:
        blockers.append("training source revision is unavailable")
    elif frozen_source.get("git_dirty") is not False:
        blockers.append("training source tree was dirty")
    if any(
        runtime.get("benchmark_source_provenance", {}).get("available") is not True
        or runtime.get("benchmark_source_provenance", {}).get("git_dirty") is not False
        for runtime in runtimes.values()
    ):
        blockers.append("runtime benchmark source was unavailable or dirty")
    return {
        "schema": COMPARISON_SCHEMA,
        "task": "causal_per_flow_next_packet_forecast_only",
        "not_intrusion_detection_efficacy": True,
        "not_retrieval_accuracy": True,
        "feature_schema": FEATURE_SCHEMA,
        "protocol_sha256": protocol_sha256,
        "coordinate_sha256": coordinate_sha256,
        "reference_model": reference,
        "study_role": next(iter(study_roles)),
        "models": sorted(runs),
        "seed_count_per_model": 1,
        "publication_evidence": False,
        "publication_blockers": blockers,
        "bootstrap_resamples": resamples,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, action="append", required=True)
    parser.add_argument("--runtime", type=Path, action="append")
    parser.add_argument("--reference", default="lstm")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
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

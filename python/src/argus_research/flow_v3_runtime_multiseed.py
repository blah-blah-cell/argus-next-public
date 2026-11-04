"""Aggregate confirmatory flow-v3 runtime ratios across training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .flow_v2_multiseed import bootstrap_mean
from .flow_v2_protocol import write_manifest
from .flow_v3_runtime_benchmark import BENCHMARK_SCHEMA
from .flow_v3_protocol import PREREGISTRATION_SHA256

RUNTIME_MULTISEED_SCHEMA = "argus.flow-v3-runtime-multiseed/v1"


def aggregate(
    paths: list[Path],
    *,
    context: int = 30_000,
    resamples: int = 10_000,
    seed: int = 20260829,
) -> dict[str, Any]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(reports) != 5:
        raise ValueError("runtime aggregation requires exactly five seed reports")
    if any(report.get("schema") != BENCHMARK_SCHEMA for report in reports):
        raise ValueError("runtime aggregation received another benchmark schema")
    if any(
        report.get("preregistration_sha256") != PREREGISTRATION_SHA256
        for report in reports
    ):
        raise ValueError("runtime reports belong to another preregistration")
    if any(
        report.get("study_role") != "confirmatory"
        or report.get("publication_evidence") is not True
        for report in reports
    ):
        raise ValueError("runtime aggregation requires confirmatory reports")
    training_seeds = [int(report["training_seed"]) for report in reports]
    if len(set(training_seeds)) != 5:
        raise ValueError("runtime aggregation contains duplicate training seeds")
    if len({report["protocol_sha256"] for report in reports}) != 1:
        raise ValueError("runtime reports use different protocols")
    ratios = []
    for report in reports:
        measurement = report.get("measurements", {}).get(str(context))
        if measurement is None:
            raise ValueError(f"runtime report lacks the {context}-packet endpoint")
        error = float(measurement["cache_native_equivalence_max_abs_error"])
        if error > 1e-6:
            raise ValueError("cache-native equivalence tolerance failed")
        ratios.append(
            float(measurement["lstm_full_context"]["p95_ms"])
            / float(measurement["ahem_cache_native"]["p95_ms"])
        )
    interval = bootstrap_mean(ratios, resamples=resamples, seed=seed)
    return {
        "schema": RUNTIME_MULTISEED_SCHEMA,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "study_role": "confirmatory",
        "protocol_sha256": reports[0]["protocol_sha256"],
        "training_seeds": sorted(training_seeds),
        "context_packets": context,
        "metric": "lstm_p95_divided_by_cache_native_ahem_p95",
        "ratios_by_report_order": ratios,
        "bootstrap_resamples": resamples,
        "bootstrap": interval,
        "gate": {"lower_bound_at_least_one": interval["bootstrap_95pct_low"] >= 1.0},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, nargs="+", required=True)
    parser.add_argument("--context", type=int, default=30_000)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_manifest(
        args.output,
        aggregate(
            args.runtime,
            context=args.context,
            resamples=args.resamples,
            seed=args.seed,
        ),
    )


if __name__ == "__main__":
    main()

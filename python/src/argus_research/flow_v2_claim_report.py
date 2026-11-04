"""Render a fail-closed verdict for the pre-registered AHEM forecast claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .flow_v2_multiseed import MULTISEED_SCHEMA
from .flow_v2_protocol import PREREGISTRATION_SHA256, write_manifest
from .flow_v2_protocol import (
    PRIMARY_TEST_SOURCE,
    PRIMARY_TRAIN_SOURCES,
    PRIMARY_VALIDATION_SOURCE,
)
from .flow_v2_replication_compare import REPLICATION_COMPARISON_SCHEMA

CLAIM_SCHEMA = "argus.flow-v2-preregistered-claim-report/v1"
CANDIDATE = "ahem_v3.2"
FROZEN_TRAINING_SEEDS = [20260822, 20260823, 20260824]
FROZEN_MODELS = {
    "ahem_v3.2",
    "ahem_v3.2_local",
    "ahem_v3.2_recency",
    "lstm",
    "gru",
    "causal_cnn",
    "long_tcn",
    "s4d",
}
FROZEN_PROTOCOL_CONFIG = {
    "context_packets": 10_000,
    "train_samples": 2_048,
    "validation_samples": 512,
    "test_samples": 512,
    "seed": 20260821,
}
FROZEN_BOOTSTRAP_CONTRACT = (
    "paired_resample_training_seeds_then_flow_streams_then_coordinates"
)


def _read(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema:
        raise ValueError(f"{path} has an incompatible claim-evidence schema")
    return value


def _quality_gates(report: dict[str, Any]) -> dict[str, bool]:
    comparison = report["comparisons"].get(CANDIDATE)
    if comparison is None:
        raise ValueError(f"comparison does not contain {CANDIDATE}")
    gates = comparison["gates"]
    return {
        "at_least_three_seeds": gates["at_least_three_seeds"] is True,
        "interval_strictly_positive": gates[
            "paired_hierarchical_interval_strictly_positive"
        ]
        is True,
        "minimum_relative_effect": gates["at_least_five_percent_relative_improvement"]
        is True,
    }


def build_claim_report(
    primary_lstm: dict[str, Any],
    primary_local: dict[str, Any],
    primary_recency: dict[str, Any],
    replication_lstm: dict[str, Any],
) -> dict[str, Any]:
    primary_reports = (primary_lstm, primary_local, primary_recency)
    if any(report.get("schema") != MULTISEED_SCHEMA for report in primary_reports):
        raise ValueError("primary claim evidence is not multi-seed flow-v2 output")
    if replication_lstm.get("schema") != REPLICATION_COMPARISON_SCHEMA:
        raise ValueError("claim lacks untouched replication evidence")
    if any(
        report.get("study_role") != "confirmatory"
        for report in (*primary_reports, replication_lstm)
    ):
        raise ValueError("claim rejects development-only evidence")
    expected_split = {
        "method": "journal_source_file_disjoint",
        "validation_source": PRIMARY_VALIDATION_SOURCE,
        "test_source": PRIMARY_TEST_SOURCE,
        "train_sources": sorted(PRIMARY_TRAIN_SOURCES),
    }
    if any(
        report.get("protocol_split") != expected_split for report in primary_reports
    ):
        raise ValueError("claim evidence uses another primary source split")
    if any(
        report.get("preregistration_sha256") != PREREGISTRATION_SHA256
        for report in (*primary_reports, replication_lstm)
    ):
        raise ValueError("claim evidence differs from the pre-registration")
    if any(
        report.get("hierarchical_bootstrap_resamples") != 10_000
        or report.get("bootstrap_contract") != FROZEN_BOOTSTRAP_CONTRACT
        for report in (*primary_reports, replication_lstm)
    ):
        raise ValueError("claim evidence uses another bootstrap contract")
    protocol_hashes = {report["protocol_sha256"] for report in primary_reports}
    if len(protocol_hashes) != 1:
        raise ValueError("primary claim reports use different protocols")
    protocol_sha256 = next(iter(protocol_hashes))
    if replication_lstm["primary_protocol_sha256"] != protocol_sha256:
        raise ValueError("replication evidence belongs to another primary protocol")
    if any(report.get("context_packets") != 10_000 for report in primary_reports):
        raise ValueError("pre-registered primary claim requires 10k-packet context")
    if replication_lstm.get("context_packets") != 10_000:
        raise ValueError("pre-registered replication requires 10k-packet context")
    if replication_lstm.get("training_seeds") != FROZEN_TRAINING_SEEDS:
        raise ValueError("replication uses non-pre-registered training seeds")
    for report in primary_reports:
        if set(report.get("models", [])) != FROZEN_MODELS:
            raise ValueError("primary evidence omits a pre-registered model")
        if report.get("protocol_config") != FROZEN_PROTOCOL_CONFIG:
            raise ValueError(
                "primary evidence differs from the pre-registered protocol"
            )
        if report.get("training_seeds") != FROZEN_TRAINING_SEEDS:
            raise ValueError("primary evidence uses non-pre-registered training seeds")
        contracts = report.get("training_contracts")
        if not isinstance(contracts, list) or len(contracts) != 3:
            raise ValueError("primary evidence lacks frozen training contracts")
        for contract, training_seed in zip(
            contracts, FROZEN_TRAINING_SEEDS, strict=True
        ):
            expected = {
                "batch_size": 4,
                "checkpoint_every": 5,
                "epochs": 50,
                "learning_rate": 0.0002,
                "patience": 8,
                "seed": training_seed,
                "weight_decay": 0.0001,
            }
            if contract != expected:
                raise ValueError("training contract differs from the pre-registration")
    expected_references = (
        (primary_lstm, "lstm"),
        (primary_local, "ahem_v3.2_local"),
        (primary_recency, "ahem_v3.2_recency"),
        (replication_lstm, "lstm"),
    )
    for report, reference in expected_references:
        if report["reference_model"] != reference:
            raise ValueError(f"claim report expected reference {reference!r}")

    gates: dict[str, bool] = {}
    for prefix, report in (
        ("primary_vs_lstm", primary_lstm),
        ("primary_vs_local", primary_local),
        ("primary_vs_recency", primary_recency),
        ("replication_vs_lstm", replication_lstm),
    ):
        for name, passed in _quality_gates(report).items():
            gates[f"{prefix}.{name}"] = passed
    primary_latency = primary_lstm["comparisons"][CANDIDATE]["gates"]
    gates["primary_vs_lstm.runtime_complete"] = (
        primary_latency["fresh_process_runtime_complete"] is True
    )
    gates["primary_vs_lstm.cpu_p95_interval_no_slower"] = (
        primary_latency["cpu_p95_speedup_interval_no_slower"] is True
    )
    gates["replication_artifact_complete"] = (
        replication_lstm.get("publication_evidence") is True
    )
    allowed_primary_blockers = {
        "requires untouched source-day replication",
        "retrieval needs immutable causal evidence-pair labels",
        "forecast quality is not intrusion-detection efficacy",
    }
    for name, report in (
        ("primary_vs_lstm", primary_lstm),
        ("primary_vs_local", primary_local),
        ("primary_vs_recency", primary_recency),
    ):
        unexpected = set(report.get("publication_blockers", [])) - (
            allowed_primary_blockers
        )
        gates[f"{name}.artifact_contract_complete"] = not unexpected
    failed = sorted(name for name, passed in gates.items() if not passed)
    return {
        "schema": CLAIM_SCHEMA,
        "claim": (
            "AHEM-v3.2 improves causal 10k-packet per-flow next-packet forecast "
            "loss over a parameter-scale LSTM and its local/recency ablations, "
            "without slower fresh-process single-thread CPU p95 model-forward "
            "latency, and replicates the forecast improvement on an untouched source"
        ),
        "scope": "forecasting_only",
        "not_retrieval_accuracy": True,
        "not_intrusion_detection_efficacy": True,
        "candidate": CANDIDATE,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "primary_protocol_sha256": protocol_sha256,
        "replication_protocol_sha256": replication_lstm["replication_protocol_sha256"],
        "gates": gates,
        "verdict": "supported" if not failed else "rejected",
        "failed_gates": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-lstm", type=Path, required=True)
    parser.add_argument("--primary-local", type=Path, required=True)
    parser.add_argument("--primary-recency", type=Path, required=True)
    parser.add_argument("--replication-lstm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_claim_report(
        _read(args.primary_lstm, MULTISEED_SCHEMA),
        _read(args.primary_local, MULTISEED_SCHEMA),
        _read(args.primary_recency, MULTISEED_SCHEMA),
        _read(args.replication_lstm, REPLICATION_COMPARISON_SCHEMA),
    )
    write_manifest(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

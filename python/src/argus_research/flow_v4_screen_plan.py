"""Compile the short, development-only AHEM-v5 falsification screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .flow_v2_protocol import write_manifest
from .flow_v2_train import _canonical_sha256, _source_provenance
from .flow_v3_protocol import ProtocolConfig, compile_source_disjoint
from .flow_v3_train import (
    PRIMARY_TEST_SOURCE,
    PRIMARY_VALIDATION_SOURCE,
    V4_PREREGISTRATION_SHA256,
)
from .flow_v4_preflight import source_identity_bindings
from .flow_v4_screen_report import SCREEN_MODELS, SCREEN_SEEDS

SCREEN_PLAN_SCHEMA = "argus.ahem-v5-development-screen-plan/v1"
SCREEN_VALIDATION_SOURCE = "Wednesday-21-02-2018"
SCREEN_TEST_SOURCE = "Wednesday-28-02-2018"
SCREEN_CONTEXTS = (257, 10_000, 30_000)


def build_plan(
    journal: Path,
    run_root: Path,
    *,
    source_provenance: dict[str, Any],
    availability_recovery: bool = False,
) -> dict[str, Any]:
    if (
        source_provenance.get("available") is not True
        or source_provenance.get("git_dirty") is not False
    ):
        raise ValueError("development screen requires a clean Git revision")
    resolved_journal = journal.resolve(strict=True)
    if not resolved_journal.is_dir():
        raise ValueError("development journal must be a directory")
    resolved_root = run_root.resolve()
    config = ProtocolConfig(
        context_packets=10_000,
        train_samples=1_024,
        validation_samples=224 if availability_recovery else 256,
        test_samples=256,
        seed=20260831,
        require_memory_match=True,
        sampling_policy="natural" if availability_recovery else "balanced",
    )
    protocol = compile_source_disjoint(
        resolved_journal,
        SCREEN_VALIDATION_SOURCE,
        SCREEN_TEST_SOURCE,
        config,
    )
    split = protocol.manifest["split"]
    accessed_sources = set(split["train_sources"]) | {
        split["validation_source"],
        split["test_source"],
    }
    if {PRIMARY_VALIDATION_SOURCE, PRIMARY_TEST_SOURCE} & accessed_sources:
        raise ValueError("development screen journal exposes a primary holdout")
    journal_manifest = protocol.manifest["journal"]
    if journal_manifest.get("provenance_complete") is not True or journal_manifest.get(
        "datasets"
    ) != ["CSE-CIC-IDS2018"]:
        raise ValueError("development screen lacks audited CSE-CIC-IDS2018 provenance")
    source_bindings = source_identity_bindings(resolved_journal, accessed_sources)
    if source_bindings["complete"] is not True:
        raise ValueError("development screen source-day identity is not bound")
    protocol.manifest["preregistration_sha256"] = V4_PREREGISTRATION_SHA256
    protocol_sha256 = _canonical_sha256(protocol.manifest)

    training_tasks = []
    for seed in SCREEN_SEEDS:
        for model in SCREEN_MODELS:
            destination = resolved_root / f"seed-{seed}" / model
            training_tasks.append(
                {
                    "id": f"screen-train:{seed}:{model}",
                    "output": str(destination),
                    "argv": [
                        "python",
                        "-m",
                        "argus_research.flow_v4_train",
                        "--journal",
                        str(resolved_journal),
                        "--output",
                        str(destination),
                        "--model",
                        model,
                        "--validation-source",
                        SCREEN_VALIDATION_SOURCE,
                        "--test-source",
                        SCREEN_TEST_SOURCE,
                        "--context",
                        "10000",
                        "--train-samples",
                        "1024",
                        "--validation-samples",
                        str(config.validation_samples),
                        "--test-samples",
                        "256",
                        "--protocol-seed",
                        "20260831",
                        "--epochs",
                        "30",
                        "--patience",
                        "8",
                        "--checkpoint-every",
                        "5",
                        "--seed",
                        str(seed),
                        "--device",
                        "cuda",
                        "--allow-dirty-source",
                        "--sampling-policy",
                        config.sampling_policy,
                    ],
                    "expected_artifacts": [
                        "run.json",
                        "best.pt",
                        "latest.pt",
                        "metrics.json",
                        "test-access.json",
                    ],
                }
            )

    mechanism_tasks = []
    for seed in SCREEN_SEEDS:
        seed_root = resolved_root / f"seed-{seed}"
        output = seed_root / "mechanism.json"
        mechanism_tasks.append(
            {
                "id": f"screen-mechanism:{seed}",
                "output": str(output),
                "argv": [
                    "python",
                    "-m",
                    "argus_research.flow_v4_mechanism_eval",
                    "--candidate-metrics",
                    str(seed_root / SCREEN_MODELS[0] / "metrics.json"),
                    "--candidate-checkpoint",
                    str(seed_root / SCREEN_MODELS[0] / "best.pt"),
                    "--local-metrics",
                    str(seed_root / SCREEN_MODELS[1] / "metrics.json"),
                    "--recency-metrics",
                    str(seed_root / SCREEN_MODELS[2] / "metrics.json"),
                    "--journal",
                    str(resolved_journal),
                    "--output",
                    str(output),
                    "--device",
                    "cuda",
                ],
                "expected_artifacts": [str(output)],
            }
        )

    selected_seed_root = resolved_root / f"seed-{SCREEN_SEEDS[0]}"
    runtime_tasks = []
    for context in SCREEN_CONTEXTS:
        output = resolved_root / "runtime" / f"context-{context}.json"
        runtime_tasks.append(
            {
                "id": f"screen-runtime:{context}",
                "output": str(output),
                "argv": [
                    "python",
                    "-m",
                    "argus_research.flow_v4_runtime_benchmark",
                    "--ahem-checkpoint",
                    str(selected_seed_root / SCREEN_MODELS[0] / "best.pt"),
                    "--lstm-checkpoint",
                    str(selected_seed_root / "lstm" / "best.pt"),
                    "--output",
                    str(output),
                    "--lengths",
                    str(context),
                    "--warmup",
                    "20",
                    "--iterations",
                    "100",
                    "--threads",
                    "1",
                    "--device",
                    "cpu",
                ],
                "expected_artifacts": [str(output)],
            }
        )

    metrics = [
        resolved_root / f"seed-{seed}" / model / "metrics.json"
        for seed in SCREEN_SEEDS
        for model in SCREEN_MODELS
    ]
    report_output = resolved_root / "screen-report.json"
    report_task = {
        "id": "screen-report",
        "output": str(report_output),
        "argv": [
            "python",
            "-m",
            "argus_research.flow_v4_screen_report",
            "--metrics",
            *[str(path) for path in metrics],
            "--mechanisms",
            *[task["output"] for task in mechanism_tasks],
            "--runtimes",
            *[task["output"] for task in runtime_tasks],
            "--output",
            str(report_output),
        ],
        "expected_artifacts": [str(report_output)],
    }
    unsigned = {
        "schema": SCREEN_PLAN_SCHEMA,
        "availability_recovery": availability_recovery,
        "original_frozen_screen": not availability_recovery,
        "preregistration_sha256": V4_PREREGISTRATION_SHA256,
        "source_provenance": source_provenance,
        "journal": str(resolved_journal),
        "run_root": str(resolved_root),
        "protocol": protocol.manifest,
        "source_identity_bindings": source_bindings,
        "protocol_sha256": protocol_sha256,
        "coordinate_sha256": protocol.manifest["coordinate_sha256"],
        "training_tasks": training_tasks,
        "mechanism_tasks": mechanism_tasks,
        "runtime_tasks": runtime_tasks,
        "report_task": report_task,
        "task_counts": {
            "training": len(training_tasks),
            "mechanism": len(mechanism_tasks),
            "runtime": len(runtime_tasks),
            "report": 1,
        },
    }
    return {**unsigned, "plan_sha256": _canonical_sha256(unsigned)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--availability-recovery", action="store_true")
    args = parser.parse_args()
    plan = build_plan(
        args.journal,
        args.run_root,
        source_provenance=_source_provenance(),
        availability_recovery=args.availability_recovery,
    )
    write_manifest(args.output, plan)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()

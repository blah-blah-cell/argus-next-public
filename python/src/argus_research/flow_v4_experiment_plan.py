"""Compile the frozen AHEM-v5 study into explicit executable tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .flow_v2_protocol import _journal_manifest, write_manifest
from .flow_v2_train import _canonical_sha256, _source_provenance
from .flow_v3_replication import REPLICATION_DATASET, REPLICATION_SOURCE
from .flow_v3_train import V4_PREREGISTRATION_SHA256
from .flow_v4_multiseed import CANDIDATE, PERSISTENCE, REFERENCES, TRAINING_SEEDS
from .flow_v4_preflight import (
    FROZEN_MODELS,
    PREFLIGHT_SCHEMA,
    source_identity_bindings,
)

PLAN_SCHEMA = "argus.ahem-v5-experiment-plan/v1"
CONTEXT_LENGTHS = (257, 10_000, 30_000)
RUNTIME_PROCESSES = 5


def build_plan(
    preflight: dict[str, Any],
    *,
    journal: Path,
    replication_journal: Path,
    run_root: Path,
    source_provenance: dict[str, Any],
) -> dict[str, Any]:
    if preflight.get("schema") != PREFLIGHT_SCHEMA:
        raise ValueError("experiment plan requires a v4 preflight")
    if preflight.get("preregistration_sha256") != V4_PREREGISTRATION_SHA256:
        raise ValueError("preflight belongs to another preregistration")
    if preflight.get("confirmatory_ready") is not True:
        raise ValueError("preflight is not confirmatory-ready")
    if (
        source_provenance.get("available") is not True
        or source_provenance.get("git_dirty") is not False
        or source_provenance.get("git_revision")
        != preflight.get("source_provenance", {}).get("git_revision")
    ):
        raise ValueError("experiment plan source differs from clean preflight source")
    resolved_journal = journal.resolve(strict=True)
    resolved_replication = replication_journal.resolve(strict=True)
    if not resolved_journal.is_dir() or not resolved_replication.is_dir():
        raise ValueError("experiment journals must be directories")
    primary_manifest = preflight.get("protocol", {}).get("journal", {})
    replication_manifest = _journal_manifest(resolved_replication, {REPLICATION_SOURCE})
    replication_bindings = source_identity_bindings(
        resolved_replication, {REPLICATION_SOURCE}
    )
    if (
        replication_manifest.get("provenance_complete") is not True
        or replication_manifest.get("datasets") != [REPLICATION_DATASET]
        or replication_bindings["complete"] is not True
    ):
        raise ValueError("replication journal lacks audited CIC-IDS2017 provenance")
    primary_hashes = {
        member.get("sha256") for member in primary_manifest.get("members", [])
    }
    replication_hashes = {
        member.get("sha256") for member in replication_manifest.get("members", [])
    }
    if (
        not primary_hashes
        or not replication_hashes
        or primary_hashes & replication_hashes
    ):
        raise ValueError("primary and replication journal bytes are not disjoint")
    resolved_root = run_root.resolve()

    training_tasks = []
    for training_seed in TRAINING_SEEDS:
        for model in FROZEN_MODELS:
            destination = resolved_root / f"seed-{training_seed}" / model
            training_tasks.append(
                {
                    "id": f"train:{training_seed}:{model}",
                    "model": model,
                    "training_seed": training_seed,
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
                        "--seed",
                        str(training_seed),
                        "--device",
                        "cuda",
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
    for training_seed in TRAINING_SEEDS:
        seed_root = resolved_root / f"seed-{training_seed}"
        output = seed_root / "mechanism.json"
        mechanism_tasks.append(
            {
                "id": f"mechanism:{training_seed}",
                "output": str(output),
                "depends_on": [
                    f"train:{training_seed}:ahem_v5_source_aligned",
                    f"train:{training_seed}:ahem_v5_local_only",
                    f"train:{training_seed}:ahem_v5_recency",
                ],
                "argv": [
                    "python",
                    "-m",
                    "argus_research.flow_v4_mechanism_eval",
                    "--candidate-metrics",
                    str(seed_root / CANDIDATE / "metrics.json"),
                    "--candidate-checkpoint",
                    str(seed_root / CANDIDATE / "best.pt"),
                    "--local-metrics",
                    str(seed_root / "ahem_v5_local_only" / "metrics.json"),
                    "--recency-metrics",
                    str(seed_root / "ahem_v5_recency" / "metrics.json"),
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

    runtime_tasks = []
    runtime_aggregate_tasks = []
    for training_seed in TRAINING_SEEDS:
        seed_root = resolved_root / f"seed-{training_seed}"
        for context in CONTEXT_LENGTHS:
            report_paths = []
            for process_index in range(1, RUNTIME_PROCESSES + 1):
                output = (
                    seed_root / "runtime" / f"{context}-process-{process_index}.json"
                )
                report_paths.append(output)
                runtime_tasks.append(
                    {
                        "id": f"runtime:{training_seed}:{context}:{process_index}",
                        "output": str(output),
                        "depends_on": [
                            f"train:{training_seed}:{CANDIDATE}",
                            f"train:{training_seed}:lstm",
                        ],
                        "argv": [
                            "python",
                            "-m",
                            "argus_research.flow_v4_runtime_benchmark",
                            "--ahem-checkpoint",
                            str(seed_root / CANDIDATE / "best.pt"),
                            "--lstm-checkpoint",
                            str(seed_root / "lstm" / "best.pt"),
                            "--output",
                            str(output),
                            "--lengths",
                            str(context),
                            "--warmup",
                            "50",
                            "--iterations",
                            "200",
                            "--threads",
                            "1",
                            "--device",
                            "cpu",
                        ],
                        "expected_artifacts": [str(output)],
                    }
                )
            aggregate_output = seed_root / "runtime" / f"{context}-aggregate.json"
            runtime_aggregate_tasks.append(
                {
                    "id": f"runtime-aggregate:{training_seed}:{context}",
                    "output": str(aggregate_output),
                    "depends_on": [
                        f"runtime:{training_seed}:{context}:{index}"
                        for index in range(1, RUNTIME_PROCESSES + 1)
                    ],
                    "argv": [
                        "python",
                        "-m",
                        "argus_research.flow_v4_runtime_replicates",
                        "--reports",
                        *[str(path) for path in report_paths],
                        "--output",
                        str(aggregate_output),
                    ],
                    "expected_artifacts": [str(aggregate_output)],
                }
            )

    comparison_tasks = []
    for reference in REFERENCES:
        output = resolved_root / f"multiseed-vs-{reference}.json"
        metrics = []
        for training_seed in TRAINING_SEEDS:
            seed_root = resolved_root / f"seed-{training_seed}"
            metrics.append(seed_root / CANDIDATE / "metrics.json")
            if reference != PERSISTENCE:
                metrics.append(seed_root / reference / "metrics.json")
        comparison_tasks.append(
            {
                "id": f"comparison:{reference}",
                "output": str(output),
                "argv": [
                    "python",
                    "-m",
                    "argus_research.flow_v4_multiseed",
                    "--metrics",
                    *[str(path) for path in metrics],
                    "--reference",
                    reference,
                    "--output",
                    str(output),
                ],
                "expected_artifacts": [str(output)],
            }
        )

    ahem_runs = [
        resolved_root / f"seed-{seed}" / CANDIDATE / "run.json"
        for seed in TRAINING_SEEDS
    ]
    ahem_checkpoints = [
        resolved_root / f"seed-{seed}" / CANDIDATE / "best.pt"
        for seed in TRAINING_SEEDS
    ]
    lstm_runs = [
        resolved_root / f"seed-{seed}" / "lstm" / "run.json" for seed in TRAINING_SEEDS
    ]
    lstm_checkpoints = [
        resolved_root / f"seed-{seed}" / "lstm" / "best.pt" for seed in TRAINING_SEEDS
    ]
    replication_output = resolved_root / "replication-vs-lstm.json"
    replication_task = {
        "id": "replication",
        "output": str(replication_output),
        "argv": [
            "python",
            "-m",
            "argus_research.flow_v4_replication",
            "--journal",
            str(resolved_replication),
            "--output",
            str(replication_output),
            "--ahem-run",
            *[str(path) for path in ahem_runs],
            "--ahem-checkpoint",
            *[str(path) for path in ahem_checkpoints],
            "--lstm-run",
            *[str(path) for path in lstm_runs],
            "--lstm-checkpoint",
            *[str(path) for path in lstm_checkpoints],
            "--device",
            "cuda",
        ],
        "expected_artifacts": [
            str(replication_output),
            str(replication_output.with_suffix(".json.access.json")),
        ],
    }

    comparison_outputs = {
        task["id"].split(":", 1)[1]: task["output"] for task in comparison_tasks
    }
    claim_output = resolved_root / "claim-report.json"
    claim_task = {
        "id": "claim",
        "output": str(claim_output),
        "argv": [
            "python",
            "-m",
            "argus_research.flow_v4_claim_report",
            "--lstm",
            comparison_outputs["lstm"],
            "--local",
            comparison_outputs["ahem_v5_local_only"],
            "--recency",
            comparison_outputs["ahem_v5_recency"],
            "--gru",
            comparison_outputs["gru"],
            "--tcn",
            comparison_outputs["long_tcn"],
            "--s4d",
            comparison_outputs["s4d"],
            "--persistence",
            comparison_outputs[PERSISTENCE],
            "--mechanism",
            *[task["output"] for task in mechanism_tasks],
            "--runtime",
            *[task["output"] for task in runtime_aggregate_tasks],
            "--replication",
            str(replication_output),
            "--output",
            str(claim_output),
        ],
        "expected_artifacts": [str(claim_output)],
    }

    unsigned = {
        "schema": PLAN_SCHEMA,
        "preregistration_sha256": V4_PREREGISTRATION_SHA256,
        "preflight_sha256": _canonical_sha256(preflight),
        "protocol_sha256": preflight["protocol_sha256"],
        "coordinate_sha256": preflight["coordinate_sha256"],
        "source_provenance": source_provenance,
        "journal": str(resolved_journal),
        "replication_journal": str(resolved_replication),
        "replication_journal_manifest": replication_manifest,
        "replication_source_identity_bindings": replication_bindings,
        "run_root": str(resolved_root),
        "training_tasks": training_tasks,
        "mechanism_tasks": mechanism_tasks,
        "runtime_tasks": runtime_tasks,
        "runtime_aggregate_tasks": runtime_aggregate_tasks,
        "comparison_tasks": comparison_tasks,
        "replication_task": replication_task,
        "claim_task": claim_task,
        "task_counts": {
            "training": len(training_tasks),
            "mechanism": len(mechanism_tasks),
            "runtime_process": len(runtime_tasks),
            "runtime_aggregate": len(runtime_aggregate_tasks),
            "comparison": len(comparison_tasks),
            "replication": 1,
            "claim": 1,
        },
    }
    return {**unsigned, "plan_sha256": _canonical_sha256(unsigned)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--replication-journal", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    plan = build_plan(
        preflight,
        journal=args.journal,
        replication_journal=args.replication_journal,
        run_root=args.run_root,
        source_provenance=_source_provenance(),
    )
    write_manifest(args.output, plan)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()

"""Compile the pre-registered flow-v2 experiment into explicit process tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .flow_v2_claim_report import (
    FROZEN_MODELS,
    FROZEN_PROTOCOL_CONFIG,
    FROZEN_TRAINING_SEEDS,
)
from .flow_v2_protocol import (
    PRIMARY_TEST_SOURCE,
    PRIMARY_TRAIN_SOURCES,
    PRIMARY_VALIDATION_SOURCE,
    PREREGISTRATION_SHA256,
    ProtocolConfig,
    compile_protocol,
    write_manifest,
)
from .flow_v2_replication_protocol import (
    REPLICATION_PROTOCOL_SCHEMA,
    compile_replication_protocol,
)
from .flow_v2_train import _canonical_sha256, _source_provenance

PLAN_SCHEMA = "argus.flow-v2-experiment-plan/v1"


def build_plan(
    protocol: dict[str, Any],
    *,
    journal: Path,
    replication_journal: Path,
    replication_protocol: dict[str, Any],
    run_root: Path,
    source_provenance: dict[str, Any],
) -> dict[str, Any]:
    if protocol.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("experiment protocol is not bound to the pre-registration")
    if protocol.get("config") != FROZEN_PROTOCOL_CONFIG:
        raise ValueError("experiment protocol differs from the frozen configuration")
    expected_split = {
        "method": "journal_source_file_disjoint",
        "validation_source": PRIMARY_VALIDATION_SOURCE,
        "test_source": PRIMARY_TEST_SOURCE,
        "train_sources": sorted(PRIMARY_TRAIN_SOURCES),
    }
    if protocol.get("split") != expected_split:
        raise ValueError("experiment protocol differs from the frozen source split")
    if set(protocol.get("models", {})) != FROZEN_MODELS:
        raise ValueError("experiment protocol omits a pre-registered model")
    if protocol.get("journal", {}).get(
        "provenance_complete"
    ) is not True or protocol.get("journal", {}).get("datasets") != ["CSE-CIC-IDS2018"]:
        raise ValueError("experiment primary journal lacks CSE-CIC-IDS2018 provenance")
    if (
        source_provenance.get("available") is not True
        or source_provenance.get("git_dirty") is not False
    ):
        raise ValueError("experiment plan requires an available clean Git source")
    protocol_sha256 = _canonical_sha256(protocol)
    if replication_protocol.get("schema") != REPLICATION_PROTOCOL_SCHEMA:
        raise ValueError("experiment lacks a frozen replication protocol")
    if replication_protocol.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("replication protocol is not bound to the pre-registration")
    if replication_protocol.get("primary_protocol_sha256") != protocol_sha256:
        raise ValueError("replication protocol belongs to another primary protocol")
    if replication_protocol.get("journal", {}).get(
        "provenance_complete"
    ) is not True or replication_protocol.get("journal", {}).get("datasets") != [
        "CIC-IDS2017"
    ]:
        raise ValueError("experiment replication journal lacks CIC-IDS2017 provenance")
    frozen_replication_fields = {
        "source": "Friday-WorkingHours",
        "context_packets": 10_000,
        "samples": 512,
        "seed": 20260827,
    }
    if any(
        replication_protocol.get(field) != value
        for field, value in frozen_replication_fields.items()
    ):
        raise ValueError("replication protocol differs from the pre-registration")
    resolved_journal = journal.resolve(strict=True)
    if not resolved_journal.is_dir():
        raise ValueError("primary journal is not a directory")
    resolved_replication_journal = replication_journal.resolve(strict=True)
    if not resolved_replication_journal.is_dir():
        raise ValueError("replication journal is not a directory")
    resolved_root = run_root.resolve()
    if not resolved_root.is_absolute():
        raise ValueError("experiment run root must be absolute")

    training_tasks = []
    runtime_tasks = []
    for training_seed in FROZEN_TRAINING_SEEDS:
        seed_root = resolved_root / f"seed-{training_seed}"
        for model in sorted(FROZEN_MODELS):
            output = seed_root / model
            training_tasks.append(
                {
                    "id": f"train:{training_seed}:{model}",
                    "model": model,
                    "training_seed": training_seed,
                    "output": str(output),
                    "argv": [
                        "python",
                        "-m",
                        "argus_research.flow_v2_train",
                        "--journal",
                        str(resolved_journal),
                        "--validation-source",
                        protocol["split"]["validation_source"],
                        "--test-source",
                        protocol["split"]["test_source"],
                        "--model",
                        model,
                        "--output",
                        str(output),
                        "--context",
                        "10000",
                        "--train-samples",
                        "2048",
                        "--validation-samples",
                        "512",
                        "--test-samples",
                        "512",
                        "--protocol-seed",
                        "20260821",
                        "--epochs",
                        "50",
                        "--batch-size",
                        "4",
                        "--learning-rate",
                        "0.0002",
                        "--weight-decay",
                        "0.0001",
                        "--patience",
                        "8",
                        "--checkpoint-every",
                        "5",
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
            runtime_tasks.append(
                {
                    "id": f"runtime:{training_seed}:{model}",
                    "depends_on": f"train:{training_seed}:{model}",
                    "model": model,
                    "training_seed": training_seed,
                    "argv": [
                        "python",
                        "-m",
                        "argus_research.flow_v2_runtime_benchmark",
                        "--run",
                        str(output / "run.json"),
                        "--checkpoint",
                        str(output / "best.pt"),
                        "--journal",
                        str(resolved_journal),
                        "--output",
                        str(output / "runtime.json"),
                        "--windows",
                        "64",
                        "--rounds",
                        "3",
                        "--threads",
                        "1",
                    ],
                    "expected_artifacts": ["runtime.json"],
                }
            )

    comparisons = []
    for reference in ("lstm", "ahem_v3.2_local", "ahem_v3.2_recency"):
        output = resolved_root / f"multiseed-vs-{reference}.json"
        argv = ["python", "-m", "argus_research.flow_v2_multiseed"]
        for task in training_tasks:
            argv.extend(("--metrics", str(Path(task["output"]) / "metrics.json")))
        for task in runtime_tasks:
            argv.extend(("--runtime", task["argv"][task["argv"].index("--output") + 1]))
        argv.extend(("--reference", reference, "--output", str(output)))
        comparisons.append(
            {
                "id": f"multiseed:{reference}",
                "reference": reference,
                "output": str(output),
                "argv": argv,
                "required_training_tasks": [task["id"] for task in training_tasks],
                "required_runtime_tasks": [task["id"] for task in runtime_tasks],
            }
        )

    replication_protocol_path = resolved_root / "replication-protocol.json"
    replication_tasks = []
    for training_seed in FROZEN_TRAINING_SEEDS:
        for model in ("ahem_v3.2", "lstm"):
            trained_output = resolved_root / f"seed-{training_seed}" / model
            output = (
                resolved_root / "replication" / f"seed-{training_seed}-{model}.json"
            )
            replication_tasks.append(
                {
                    "id": f"replication:{training_seed}:{model}",
                    "depends_on": f"train:{training_seed}:{model}",
                    "model": model,
                    "training_seed": training_seed,
                    "output": str(output),
                    "argv": [
                        "python",
                        "-m",
                        "argus_research.flow_v2_replication",
                        "--metrics",
                        str(trained_output / "metrics.json"),
                        "--checkpoint",
                        str(trained_output / "best.pt"),
                        "--journal",
                        str(resolved_replication_journal),
                        "--replication-protocol",
                        str(replication_protocol_path),
                        "--output",
                        str(output),
                        "--batch-size",
                        "4",
                        "--device",
                        "cuda",
                    ],
                    "expected_artifacts": [str(output)],
                }
            )
    replication_comparison_output = (
        resolved_root / "replication" / "multiseed-vs-lstm.json"
    )
    replication_argv = [
        "python",
        "-m",
        "argus_research.flow_v2_replication_compare",
    ]
    for task in replication_tasks:
        replication_argv.extend(("--metrics", task["output"]))
    replication_argv.extend(
        ("--reference", "lstm", "--output", str(replication_comparison_output))
    )
    comparison_outputs = {task["reference"]: task["output"] for task in comparisons}
    claim_output = resolved_root / "claim-report.json"
    claim_task = {
        "id": "claim-report",
        "output": str(claim_output),
        "argv": [
            "python",
            "-m",
            "argus_research.flow_v2_claim_report",
            "--primary-lstm",
            comparison_outputs["lstm"],
            "--primary-local",
            comparison_outputs["ahem_v3.2_local"],
            "--primary-recency",
            comparison_outputs["ahem_v3.2_recency"],
            "--replication-lstm",
            str(replication_comparison_output),
            "--output",
            str(claim_output),
        ],
    }
    unsigned = {
        "schema": PLAN_SCHEMA,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "protocol_sha256": protocol_sha256,
        "coordinate_sha256": protocol["coordinate_sha256"],
        "source_provenance": source_provenance,
        "journal": str(resolved_journal),
        "replication_journal": str(resolved_replication_journal),
        "replication_protocol_sha256": _canonical_sha256(replication_protocol),
        "run_root": str(resolved_root),
        "training_task_count": len(training_tasks),
        "runtime_task_count": len(runtime_tasks),
        "replication_task_count": len(replication_tasks),
        "training_tasks": training_tasks,
        "runtime_tasks": runtime_tasks,
        "comparison_tasks": comparisons,
        "replication_tasks": replication_tasks,
        "replication_comparison_task": {
            "id": "replication-multiseed:lstm",
            "output": str(replication_comparison_output),
            "argv": replication_argv,
        },
        "claim_task": claim_task,
        "execution_contract": {
            "training_parallelism": (
                "scheduler-controlled; it may change wall-clock time but not task inputs"
            ),
            "one_model_per_process": True,
            "runtime_tasks_parallel": False,
            "runtime_reason": "fresh-process single-thread measurements must not contend",
            "shell_interpolation_required": False,
            "argv_is_authoritative": True,
        },
        "remaining_after_plan": ["SHA-256 evidence bundle"],
    }
    return {**unsigned, "plan_sha256": _canonical_sha256(unsigned)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--replication-journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-source", default=PRIMARY_VALIDATION_SOURCE)
    parser.add_argument("--test-source", default=PRIMARY_TEST_SOURCE)
    parser.add_argument("--allow-dirty-source", action="store_true")
    args = parser.parse_args()
    source = _source_provenance()
    if not args.allow_dirty_source and (
        source.get("available") is not True or source.get("git_dirty") is not False
    ):
        parser.error("publication plan requires an available clean Git source")
    config = ProtocolConfig(**FROZEN_PROTOCOL_CONFIG)
    compiled = compile_protocol(
        args.journal, args.validation_source, args.test_source, config
    )
    frozen_replication = compile_replication_protocol(
        compiled.manifest,
        args.replication_journal,
        "Friday-WorkingHours",
        context_packets=10_000,
        samples=512,
        seed=20260827,
        source_provenance=source,
    )
    protocol_path = args.run_root / "protocol.json"
    replication_protocol_path = args.run_root / "replication-protocol.json"
    write_manifest(protocol_path, compiled.manifest)
    write_manifest(replication_protocol_path, frozen_replication.manifest)
    plan = build_plan(
        compiled.manifest,
        journal=args.journal,
        replication_journal=args.replication_journal,
        replication_protocol=frozen_replication.manifest,
        run_root=args.run_root,
        source_provenance=source,
    )
    write_manifest(args.output, plan)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()

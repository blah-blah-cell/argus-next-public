"""Compile reusable development-only AHEM pilot cycles without holdout access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .flow_v2_protocol import (
    PRIMARY_TRAIN_SOURCES,
    PREREGISTRATION_SHA256,
    ProtocolConfig,
    compile_protocol,
    write_manifest,
)
from .flow_v2_train import _canonical_sha256, _source_provenance

PILOT_PLAN_SCHEMA = "argus.flow-v2-pilot-plan/v1"
PILOT_REPORT_SCHEMA = "argus.flow-v2-pilot-promotion/v1"
MAX_PILOT_CYCLES = 5
PILOT_MODELS = (
    "ahem_v3.2",
    "ahem_v3.2_local",
    "ahem_v3.2_recency",
    "lstm",
    "s4d",
)
PILOT_VALIDATION_SOURCE = "Wednesday-21-02-2018"
PILOT_CHALLENGE_SOURCE = "Friday-16-02-2018"
PILOT_TRAIN_SOURCES = tuple(
    source
    for source in PRIMARY_TRAIN_SOURCES
    if source not in {PILOT_VALIDATION_SOURCE, PILOT_CHALLENGE_SOURCE}
)
PILOT_TIERS = {
    "screen": {
        "train_samples": 256,
        "validation_samples": 128,
        "test_samples": 128,
        "epochs": 5,
        "patience": 3,
        "seeds": [20260830],
    },
    "gate": {
        "train_samples": 1_024,
        "validation_samples": 256,
        "test_samples": 256,
        "epochs": 15,
        "patience": 5,
        "seeds": [20260830, 20260831, 20260832],
    },
}


def build_pilot_plan(
    protocol: dict[str, Any],
    *,
    journal: Path,
    run_root: Path,
    cycle: int,
    tier: str,
    source_provenance: dict[str, Any],
    screen_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if cycle <= 0 or cycle > MAX_PILOT_CYCLES:
        raise ValueError(f"pilot cycle must be between 1 and {MAX_PILOT_CYCLES}")
    if tier not in PILOT_TIERS:
        raise ValueError(f"unknown pilot tier {tier!r}")
    expected_split = {
        "method": "journal_source_file_disjoint",
        "validation_source": PILOT_VALIDATION_SOURCE,
        "test_source": PILOT_CHALLENGE_SOURCE,
        "train_sources": sorted(PILOT_TRAIN_SOURCES),
    }
    if protocol.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("pilot protocol differs from the current methods freeze")
    if protocol.get("split") != expected_split:
        raise ValueError(
            "pilot journal must contain only the eight development sources"
        )
    if protocol.get("journal", {}).get(
        "provenance_complete"
    ) is not True or protocol.get("journal", {}).get("datasets") != ["CSE-CIC-IDS2018"]:
        raise ValueError("pilot requires provenance-complete CSE-CIC-IDS2018 data")
    if (
        source_provenance.get("available") is not True
        or source_provenance.get("git_dirty") is not False
    ):
        raise ValueError("pilot requires a clean Git source revision")
    if tier == "gate" and (
        screen_report is None
        or screen_report.get("schema") != PILOT_REPORT_SCHEMA
        or screen_report.get("tier") != "screen"
        or screen_report.get("cycle") != cycle
        or screen_report.get("protocol_sha256") != _canonical_sha256(protocol)
        or screen_report.get("passed") is not True
        or screen_report.get("decision") != "advance_to_gate"
    ):
        raise ValueError("gate requires the passing screen report from this cycle")

    resolved_journal = journal.resolve(strict=True)
    resolved_root = run_root.resolve() / f"pilot-{tier}-cycle-{cycle:03d}"
    settings = PILOT_TIERS[tier]
    tasks = []
    for seed in settings["seeds"]:
        for model in PILOT_MODELS:
            output = resolved_root / f"seed-{seed}" / model
            tasks.append(
                {
                    "id": f"pilot-train:{tier}:{cycle}:{seed}:{model}",
                    "model": model,
                    "training_seed": seed,
                    "output": str(output),
                    "argv": [
                        "python",
                        "-m",
                        "argus_research.flow_v2_train",
                        "--journal",
                        str(resolved_journal),
                        "--validation-source",
                        PILOT_VALIDATION_SOURCE,
                        "--test-source",
                        PILOT_CHALLENGE_SOURCE,
                        "--model",
                        model,
                        "--output",
                        str(output),
                        "--context",
                        "10000",
                        "--train-samples",
                        str(settings["train_samples"]),
                        "--validation-samples",
                        str(settings["validation_samples"]),
                        "--test-samples",
                        str(settings["test_samples"]),
                        "--protocol-seed",
                        "20260829",
                        "--epochs",
                        str(settings["epochs"]),
                        "--batch-size",
                        "4",
                        "--learning-rate",
                        "0.0002",
                        "--weight-decay",
                        "0.0001",
                        "--patience",
                        str(settings["patience"]),
                        "--checkpoint-every",
                        "5",
                        "--seed",
                        str(seed),
                        "--device",
                        "cuda",
                        "--exploratory-run",
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

    comparisons = []
    for reference in ("lstm", "ahem_v3.2_local", "ahem_v3.2_recency"):
        output = resolved_root / f"vs-{reference}.json"
        argv = ["python", "-m", "argus_research.flow_v2_multiseed"]
        for task in tasks:
            argv.extend(("--metrics", str(Path(task["output"]) / "metrics.json")))
        argv.extend(("--reference", reference, "--output", str(output)))
        comparisons.append(
            {
                "id": f"pilot-compare:{tier}:{cycle}:{reference}",
                "reference": reference,
                "output": str(output),
                "argv": argv,
                "depends_on": [task["id"] for task in tasks],
            }
        )
    report_output = resolved_root / "promotion-report.json"
    comparison_outputs = {value["reference"]: value["output"] for value in comparisons}
    report_task = {
        "id": f"pilot-report:{tier}:{cycle}",
        "output": str(report_output),
        "argv": [
            "python",
            "-m",
            "argus_research.flow_v2_pilot_report",
            "--tier",
            tier,
            "--cycle",
            str(cycle),
            "--lstm",
            comparison_outputs["lstm"],
            "--local",
            comparison_outputs["ahem_v3.2_local"],
            "--recency",
            comparison_outputs["ahem_v3.2_recency"],
            "--output",
            str(report_output),
        ],
        "depends_on": [value["id"] for value in comparisons],
    }
    unsigned = {
        "schema": PILOT_PLAN_SCHEMA,
        "study_role": "exploratory",
        "publication_evidence": False,
        "cycle": cycle,
        "maximum_cycles": MAX_PILOT_CYCLES,
        "tier": tier,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "protocol_sha256": _canonical_sha256(protocol),
        "coordinate_sha256": protocol["coordinate_sha256"],
        "source_provenance": source_provenance,
        "journal": str(resolved_journal),
        "run_root": str(resolved_root),
        "holdouts_forbidden": [
            "Thursday-22-02-2018",
            "Friday-23-02-2018",
            "CIC-IDS2017 Friday-WorkingHours",
        ],
        "training_tasks": tasks,
        "comparison_tasks": comparisons,
        "report_task": report_task,
        "next_if_passed": (
            "run gate tier"
            if tier == "screen"
            else "freeze code before confirmatory run"
        ),
        "next_if_failed": "revise architecture and create a new numbered pilot cycle",
    }
    return {**unsigned, "plan_sha256": _canonical_sha256(unsigned)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--tier", choices=tuple(PILOT_TIERS), default="screen")
    parser.add_argument("--screen-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = PILOT_TIERS[args.tier]
    source = _source_provenance()
    config = ProtocolConfig(
        context_packets=10_000,
        train_samples=settings["train_samples"],
        validation_samples=settings["validation_samples"],
        test_samples=settings["test_samples"],
        seed=20260829,
    )
    compiled = compile_protocol(
        args.journal,
        PILOT_VALIDATION_SOURCE,
        PILOT_CHALLENGE_SOURCE,
        config,
    )
    screen_report = None
    if args.screen_report is not None:
        screen_report = json.loads(args.screen_report.read_text(encoding="utf-8"))
    plan = build_pilot_plan(
        compiled.manifest,
        journal=args.journal,
        run_root=args.run_root,
        cycle=args.cycle,
        tier=args.tier,
        source_provenance=source,
        screen_report=screen_report,
    )
    write_manifest(args.output, plan)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()

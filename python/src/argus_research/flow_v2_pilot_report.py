"""Fail-closed promotion decision for development-only AHEM pilot cycles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from .flow_v2_multiseed import MULTISEED_SCHEMA
from .flow_v2_protocol import write_manifest

PILOT_REPORT_SCHEMA = "argus.flow-v2-pilot-promotion/v1"
CANDIDATE = "ahem_v3.2"


def _candidate(report: dict[str, Any], reference: str) -> dict[str, Any]:
    if report.get("schema") != MULTISEED_SCHEMA:
        raise ValueError("pilot report requires flow-v2 multi-seed evidence")
    if report.get("study_role") != "exploratory":
        raise ValueError("pilot report rejects confirmatory or diagnostic evidence")
    if report.get("reference_model") != reference:
        raise ValueError(f"pilot expected reference {reference!r}")
    try:
        return report["comparisons"][CANDIDATE]
    except KeyError as error:
        raise ValueError("pilot comparison lacks AHEM-v3.2") from error


def build_pilot_report(
    lstm: dict[str, Any],
    local: dict[str, Any],
    recency: dict[str, Any],
    *,
    tier: str,
    cycle: int,
) -> dict[str, Any]:
    if tier not in {"screen", "gate"}:
        raise ValueError(f"unknown pilot tier {tier!r}")
    if cycle <= 0:
        raise ValueError("pilot cycle must be positive")
    reports = (lstm, local, recency)
    if len({report.get("protocol_sha256") for report in reports}) != 1:
        raise ValueError("pilot comparisons use different protocols")
    comparisons = {
        "lstm": _candidate(lstm, "lstm"),
        "local": _candidate(local, "ahem_v3.2_local"),
        "recency": _candidate(recency, "ahem_v3.2_recency"),
    }
    gates: dict[str, bool] = {}
    for name, comparison in comparisons.items():
        paired = comparison["paired_weighted_forecast_loss"]
        gates[f"lower_mean_loss_vs_{name}"] = (
            paired["mean_baseline_minus_candidate"] > 0.0
        )
        if tier == "gate":
            gates[f"positive_cluster_interval_vs_{name}"] = (
                paired["bootstrap_95pct_low"] > 0.0
            )
            gates[f"two_percent_effect_vs_{name}"] = (
                comparison["relative_forecast_loss_improvement"] >= 0.02
            )
    memory_rows = comparisons["lstm"].get("memory_effect_by_seed", [])
    memory_reductions = [
        float(row["mean_weighted_loss_reduction"])
        for row in memory_rows
        if row.get("available") is True
        and row.get("mean_weighted_loss_reduction") is not None
    ]
    gates["transition_memory_has_positive_mean_utility"] = bool(memory_reductions) and (
        fmean(memory_reductions) > 0.0
    )
    seed_count = int(lstm.get("seed_count_per_model", 0))
    gates["tier_seed_count"] = seed_count >= (3 if tier == "gate" else 1)
    failed = sorted(name for name, passed in gates.items() if not passed)
    passed = not failed
    decision = (
        "advance_to_gate"
        if passed and tier == "screen"
        else "freeze_candidate_before_confirmatory"
        if passed
        else "revise_candidate_in_new_pilot_cycle"
    )
    return {
        "schema": PILOT_REPORT_SCHEMA,
        "study_role": "exploratory",
        "publication_evidence": False,
        "tier": tier,
        "cycle": cycle,
        "candidate": CANDIDATE,
        "protocol_sha256": lstm["protocol_sha256"],
        "seed_count": seed_count,
        "gates": gates,
        "failed_gates": failed,
        "passed": passed,
        "decision": decision,
        "claim_boundary": (
            "Passing selects an architecture for a future frozen experiment; "
            "it does not establish superiority or permit confirmatory retesting."
        ),
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("screen", "gate"), required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--lstm", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--recency", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_pilot_report(
        _read(args.lstm),
        _read(args.local),
        _read(args.recency),
        tier=args.tier,
        cycle=args.cycle,
    )
    write_manifest(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

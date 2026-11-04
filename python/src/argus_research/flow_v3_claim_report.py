"""Assemble the fail-closed AHEM-v3.4 confirmatory verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .flow_v2_protocol import write_manifest
from .flow_v3_multiseed import MULTISEED_SCHEMA
from .flow_v3_protocol import PREREGISTRATION_SHA256
from .flow_v3_runtime_multiseed import RUNTIME_MULTISEED_SCHEMA

CLAIM_SCHEMA = "argus.flow-v3-claim-report/v1"


def _load(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema:
        raise ValueError(f"{path.name} has an incompatible evidence schema")
    return value


def assemble(
    *,
    lstm: Path,
    local_only: Path,
    recency: Path,
    runtime: Path,
    replication: Path,
) -> dict[str, Any]:
    comparisons = {
        "lstm": _load(lstm, MULTISEED_SCHEMA),
        "local_only": _load(local_only, MULTISEED_SCHEMA),
        "recency": _load(recency, MULTISEED_SCHEMA),
    }
    expected_references = {
        "lstm": "lstm",
        "local_only": "ahem_v3.4_local_only",
        "recency": "ahem_v3.4_recency",
    }
    for name, comparison in comparisons.items():
        if comparison.get("preregistration_sha256") != PREREGISTRATION_SHA256:
            raise ValueError(f"{name} comparison belongs to another preregistration")
        if comparison.get("reference") != expected_references[name]:
            raise ValueError(f"{name} comparison uses another reference")
    if len({value["protocol_sha256"] for value in comparisons.values()}) != 1:
        raise ValueError("primary comparisons do not share one protocol")
    runtime_value = _load(runtime, RUNTIME_MULTISEED_SCHEMA)
    if runtime_value.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("runtime belongs to another preregistration")
    if runtime_value.get("protocol_sha256") != comparisons["lstm"]["protocol_sha256"]:
        raise ValueError("runtime and quality protocols differ")
    replication_value = _load(replication, MULTISEED_SCHEMA)
    if replication_value.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("replication belongs to another preregistration")
    if replication_value.get("reference") != "lstm":
        raise ValueError("replication must compare AHEM with LSTM")
    if (
        replication_value.get("protocol_sha256")
        == comparisons["lstm"]["protocol_sha256"]
    ):
        raise ValueError("replication reused the primary protocol")

    gates = {
        "h1_lstm_quality": comparisons["lstm"]["gates"]["comparison_passed"],
        "h2_local_mechanism": comparisons["local_only"]["gates"]["comparison_passed"],
        "h2_recency_mechanism": comparisons["recency"]["gates"]["comparison_passed"],
        "h3_runtime": runtime_value["gate"]["lower_bound_at_least_one"],
        "h4_replication": replication_value["gates"]["comparison_passed"],
    }
    return {
        "schema": CLAIM_SCHEMA,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "claim": "bounded exact flow-indexed memory improves next-dynamic-event forecasting and 30k cache-native runtime over the preregistered LSTM reference",
        "not_intrusion_detection_efficacy": True,
        "not_industry_replacement_evidence": True,
        "gates": gates,
        "compound_claim_supported": all(gates.values()),
        "primary_protocol_sha256": comparisons["lstm"]["protocol_sha256"],
        "replication_protocol_sha256": replication_value["protocol_sha256"],
        "evidence": {
            "lstm": comparisons["lstm"],
            "local_only": comparisons["local_only"],
            "recency": comparisons["recency"],
            "runtime": runtime_value,
            "replication": replication_value,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm", type=Path, required=True)
    parser.add_argument("--local-only", type=Path, required=True)
    parser.add_argument("--recency", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--replication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_manifest(
        args.output,
        assemble(
            lstm=args.lstm,
            local_only=args.local_only,
            recency=args.recency,
            runtime=args.runtime,
            replication=args.replication,
        ),
    )


if __name__ == "__main__":
    main()

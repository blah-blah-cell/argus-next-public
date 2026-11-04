"""Frozen causal-memory corruption evaluation for selected AHEM-v5 weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .ahem_v4 import (
    AhemV5QueryRotationNextPacket,
    AhemV5ValuePermutationNextPacket,
)
from .flow_v2_protocol import write_manifest
from .flow_v2_train import _canonical_sha256
from .flow_v3_protocol import ProtocolConfig, compile_source_disjoint
from .flow_v3_smoke import _evaluate
from .flow_v3_train import (
    V4_CHECKPOINT_SCHEMA,
    V4_PREREGISTRATION_SHA256,
    V4_RUN_SCHEMA,
)

MECHANISM_SCHEMA = "argus.ahem-v5-mechanism-evaluation/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(path: Path, expected_model: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != V4_RUN_SCHEMA:
        raise ValueError(f"{path.name} is not a v5 run result")
    if value.get("preregistration_sha256") != V4_PREREGISTRATION_SHA256:
        raise ValueError(f"{path.name} belongs to another preregistration")
    if value.get("model", {}).get("name") != expected_model:
        raise ValueError(f"{path.name} is not the expected {expected_model} result")
    if value.get("test_access", {}).get("status") != "completed":
        raise ValueError(f"{path.name} has no completed test-access ledger")
    return value


def summarize(
    candidate: float,
    local: float,
    recency: float,
    query_rotation: float,
    value_permutation: float,
) -> dict[str, Any]:
    gain = local - candidate
    query_penalty = query_rotation - candidate
    value_penalty = value_permutation - candidate
    query_fraction = query_penalty / gain if gain > 0 else float("-inf")
    value_fraction = value_penalty / gain if gain > 0 else float("-inf")
    return {
        "weighted_smooth_l1": {
            "candidate": candidate,
            "local_only": local,
            "recency": recency,
            "query_rotation": query_rotation,
            "value_permutation": value_permutation,
        },
        "candidate_gain_over_local": gain,
        "query_rotation_penalty": query_penalty,
        "value_permutation_penalty": value_penalty,
        "query_rotation_gain_removed_fraction": query_fraction,
        "value_permutation_gain_removed_fraction": value_fraction,
        "gates": {
            "candidate_better_than_local": gain > 0,
            "candidate_better_than_recency": candidate < recency,
            "query_rotation_removes_at_least_25_percent": query_fraction >= 0.25,
            "value_permutation_removes_at_least_25_percent": value_fraction >= 0.25,
        },
    }


def run(
    candidate_metrics: Path,
    candidate_checkpoint: Path,
    local_metrics: Path,
    recency_metrics: Path,
    journal: Path,
    output: Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    candidate = _metrics(candidate_metrics, "ahem_v5_source_aligned")
    local = _metrics(local_metrics, "ahem_v5_local_only")
    recency = _metrics(recency_metrics, "ahem_v5_recency")
    immutable = ("protocol_sha256", "coordinate_sha256")
    if any(
        value.get(field) != candidate.get(field)
        for value in (local, recency)
        for field in immutable
    ):
        raise ValueError("mechanism evaluation mixes protocols or coordinates")
    seeds = {
        value.get("training", {}).get("seed") for value in (candidate, local, recency)
    }
    if len(seeds) != 1:
        raise ValueError("mechanism evaluation mixes training seeds")
    checkpoint = torch.load(candidate_checkpoint, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("schema") != V4_CHECKPOINT_SCHEMA
        or checkpoint.get("model") != "ahem_v5_source_aligned"
        or checkpoint.get("preregistration_sha256") != V4_PREREGISTRATION_SHA256
    ):
        raise ValueError("candidate checkpoint is not a frozen v5 artifact")
    if candidate["test_access"].get("best_checkpoint_sha256") != _sha256(
        candidate_checkpoint
    ):
        raise ValueError("candidate metrics do not bind the supplied checkpoint")

    config = ProtocolConfig(**candidate["protocol"]["config"])
    protocol = compile_source_disjoint(
        journal,
        candidate["protocol"]["split"]["validation_source"],
        candidate["protocol"]["split"]["test_source"],
        config,
    )
    protocol.manifest["preregistration_sha256"] = V4_PREREGISTRATION_SHA256
    if (
        protocol.manifest["coordinate_sha256"] != candidate["coordinate_sha256"]
        or _canonical_sha256(protocol.manifest) != candidate["protocol_sha256"]
    ):
        raise ValueError("recompiled mechanism protocol differs from training")
    batch_size = int(candidate["training"]["batch_size"])
    if batch_size < 2 or len(protocol.coordinates["test"]) % batch_size != 0:
        raise ValueError("query rotation requires complete evaluation batches")
    query = AhemV5QueryRotationNextPacket().to(device).eval()
    permutation = AhemV5ValuePermutationNextPacket().to(device).eval()
    query.load_state_dict(checkpoint["state_dict"], strict=True)
    permutation.load_state_dict(checkpoint["state_dict"], strict=True)
    query_result = _evaluate(
        query, protocol, "test", config.context_packets, batch_size, device
    )
    permutation_result = _evaluate(
        permutation, protocol, "test", config.context_packets, batch_size, device
    )
    summary = summarize(
        float(candidate["test"]["weighted_smooth_l1"]),
        float(local["test"]["weighted_smooth_l1"]),
        float(recency["test"]["weighted_smooth_l1"]),
        float(query_result["weighted_smooth_l1"]),
        float(permutation_result["weighted_smooth_l1"]),
    )
    result = {
        "schema": MECHANISM_SCHEMA,
        "preregistration_sha256": V4_PREREGISTRATION_SHA256,
        "study_role": candidate["study_role"],
        "publication_evidence": bool(
            candidate.get("publication_eligible")
            and local.get("publication_eligible")
            and recency.get("publication_eligible")
        ),
        "protocol_sha256": candidate["protocol_sha256"],
        "coordinate_sha256": candidate["coordinate_sha256"],
        "training_seed": next(iter(seeds)),
        "candidate_checkpoint_sha256": _sha256(candidate_checkpoint),
        "summary": summary,
        "query_rotation": query_result,
        "value_permutation": permutation_result,
        "all_mechanism_gates_pass": all(summary["gates"].values()),
    }
    write_manifest(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--local-metrics", type=Path, required=True)
    parser.add_argument("--recency-metrics", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.candidate_metrics,
                args.candidate_checkpoint,
                args.local_metrics,
                args.recency_metrics,
                args.journal,
                args.output,
                device=torch.device(args.device),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

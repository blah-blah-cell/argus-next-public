"""Evaluate one frozen flow-v2 checkpoint on an untouched ARGUSLR2 source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .flow_v2_compare import _load
from .flow_v2_protocol import FEATURE_SCHEMA, write_manifest
from .flow_v2_replication_protocol import (
    REPLICATION_PROTOCOL_SCHEMA,
    compile_replication_protocol,
)
from .flow_v2_train import (
    CHECKPOINT_SCHEMA,
    _canonical_sha256,
    _evaluate,
    _file_sha256,
    _source_provenance,
    model_spec,
)

REPLICATION_SCHEMA = "argus.flow-v2-untouched-replication-run/v1"


def run(args: argparse.Namespace) -> dict[str, Any]:
    trained = _load(args.metrics)
    if trained.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("replication metrics use another feature schema")
    if trained.get("study_role") != "confirmatory" and not getattr(
        args, "allow_nonconfirmatory", False
    ):
        raise ValueError("untouched replication rejects development-only checkpoints")
    frozen_protocol = json.loads(args.replication_protocol.read_text(encoding="utf-8"))
    if frozen_protocol.get("schema") != REPLICATION_PROTOCOL_SCHEMA:
        raise ValueError("replication requires a frozen pre-training protocol")
    evaluation_source = _source_provenance()
    training_source = trained.get("source_provenance", {})
    if (
        training_source.get("available") is True
        and evaluation_source.get("available") is True
        and training_source.get("git_revision") != evaluation_source.get("git_revision")
    ):
        raise ValueError("replication evaluator source differs from training")
    context_packets = int(frozen_protocol["context_packets"])
    if trained["protocol"]["config"]["context_packets"] != context_packets:
        raise ValueError("replication context differs from the primary protocol")
    protocol = compile_replication_protocol(
        trained["protocol"],
        args.journal,
        frozen_protocol["source"],
        context_packets=context_packets,
        samples=int(frozen_protocol["samples"]),
        seed=int(frozen_protocol["seed"]),
        source_provenance=frozen_protocol.get("source_provenance"),
    )
    if protocol.manifest != frozen_protocol:
        raise ValueError("replication journal or coordinates changed after preflight")
    replication_sha256 = _canonical_sha256(frozen_protocol)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("replication checkpoint has an incompatible schema")
    if checkpoint.get("protocol_sha256") != trained["protocol_sha256"]:
        raise ValueError("replication checkpoint belongs to another primary protocol")
    if checkpoint.get("coordinate_sha256") != trained["coordinate_sha256"]:
        raise ValueError("replication checkpoint uses other primary coordinates")
    if checkpoint.get("training_config_sha256") != trained["training"].get(
        "config_sha256"
    ):
        raise ValueError("replication checkpoint uses another training configuration")
    if _file_sha256(args.checkpoint) != trained.get("test_access", {}).get(
        "best_checkpoint_sha256"
    ):
        raise ValueError(
            "replication checkpoint bytes differ from confirmatory evidence"
        )
    model_name = trained["model"]["name"]
    if checkpoint.get("model") != model_name:
        raise ValueError("replication checkpoint belongs to another model")

    spec = model_spec(model_name, context_packets)
    model = spec.constructor()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    model.to(device)
    test = _evaluate(
        model,
        spec,
        protocol,
        "replication",
        context_packets,
        args.batch_size,
        device,
    )
    result = {
        "schema": REPLICATION_SCHEMA,
        "task": "untouched_causal_per_flow_next_packet_forecast_only",
        "not_intrusion_detection_efficacy": True,
        "feature_schema": FEATURE_SCHEMA,
        "study_role": trained["study_role"],
        "primary_protocol_sha256": trained["protocol_sha256"],
        "primary_coordinate_sha256": trained["coordinate_sha256"],
        "replication_protocol_sha256": replication_sha256,
        "replication_coordinate_sha256": frozen_protocol["coordinate_sha256"],
        "replication_protocol": frozen_protocol,
        "model": trained["model"],
        "training": trained["training"],
        "source_provenance": trained.get("source_provenance"),
        "evaluation_source_provenance": evaluation_source,
        "checkpoint": {
            "epoch": checkpoint["epoch"],
            "best_epoch": checkpoint["best_epoch"],
        },
        "test": test,
    }
    write_manifest(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--replication-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--allow-nonconfirmatory",
        action="store_true",
        help="diagnostic-only override; output remains publication-ineligible",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("replication batch size must be positive")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

"""Run a non-evidentiary v6.3 contract check on real local ARGUSLR2 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

import torch

from .ahem_v6 import AhemV6StateTransitionNextPacket
from .ahem_v63 import AhemV63Config, AhemV63FrozenBackbone, ahem_v63_loss
from .flow_v3_protocol import ProtocolConfig, compile_local_development
from .flow_v3_protocol import materialize_batch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_backbone(path: Path) -> tuple[AhemV6StateTransitionNextPacket, str]:
    if not path.is_file():
        raise ValueError("v6.1 runtime artifact is absent")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema": "argus.ahem-v6-runtime-artifact/v1",
        "model_id": "argus.ahem-v6.1.state-transition-forecaster",
        "security_detector": False,
        "feature_schema": "packet-causal-flow-v2",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"v6.1 runtime artifact {key} differs")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("v6.1 runtime artifact has no state_dict")
    model = AhemV6StateTransitionNextPacket(
        packet_dim=int(payload["packet_dim"]),
        output_dim=int(payload["output_dim"]),
        width=int(payload["width"]),
        key_dim=int(payload["key_dim"]),
        local_packets=int(payload["local_packets"]),
        max_transitions=int(payload["max_transitions"]),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, _sha256(path)


def run_smoke(
    journal: Path,
    backbone_path: Path,
    *,
    context_packets: int = 512,
    samples: int = 16,
) -> dict[str, object]:
    if context_packets < 257:
        raise ValueError("real-data smoke context must be at least 257 packets")
    if samples <= 0:
        raise ValueError("real-data smoke samples must be positive")
    backbone, artifact_hash = _load_backbone(backbone_path)
    protocol = compile_local_development(
        journal,
        ProtocolConfig(
            context_packets=context_packets,
            local_packets=128,
            train_samples=samples,
            validation_samples=samples,
            test_samples=samples,
            seed=20260913,
            require_memory_match=True,
            sampling_policy="natural",
            minimum_completed_transitions=2,
        ),
    )
    contexts, target = materialize_batch(
        protocol.partitions["train"],
        protocol.coordinates["train"],
        context_packets,
    )
    config = AhemV63Config(horizon=context_packets)
    candidate = AhemV63FrozenBackbone(backbone, config).train()
    started = perf_counter()
    output = candidate.forward_diagnostics(contexts)
    forward_ms = (perf_counter() - started) * 1000
    initial_deviation = float(
        (output.prediction - output.backbone_prediction).abs().max()
    )
    loss, parts = ahem_v63_loss(output, target, config)
    loss.backward()
    reader_gradients = [
        parameter.grad
        for name, parameter in candidate.named_parameters()
        if not name.startswith("backbone.") and parameter.grad is not None
    ]
    backbone_gradients = [
        parameter.grad
        for parameter in candidate.backbone.parameters()
        if parameter.grad is not None
    ]
    finite_gradients = all(torch.isfinite(value).all() for value in reader_gradients)
    nonzero_gradient = any(float(value.abs().sum()) > 0 for value in reader_gradients)
    if (
        initial_deviation != 0
        or backbone_gradients
        or not reader_gradients
        or not finite_gradients
        or not nonzero_gradient
        or not math.isfinite(float(loss))
    ):
        raise RuntimeError("v6.3 real-data smoke contract failed")
    return {
        "schema": "argus.ahem-v6.3-real-data-smoke/v1",
        "study_role": "engineering_smoke_only",
        "publication_evidence": False,
        "promotion_eligible": False,
        "reason": "local reused traffic and an untrained v6.3 reader",
        "backbone_artifact": {
            "path": str(backbone_path.resolve()),
            "sha256": artifact_hash,
            "frozen": True,
        },
        "data": {
            "real_arguslr2": True,
            "split": protocol.manifest["split"],
            "coordinate_sha256": protocol.manifest["coordinate_sha256"],
            "context_packets": context_packets,
            "samples": len(contexts),
        },
        "parameters": {
            "backbone": sum(value.numel() for value in candidate.backbone.parameters()),
            "reader_trainable": sum(
                value.numel() for value in candidate.parameters() if value.requires_grad
            ),
            "total": sum(value.numel() for value in candidate.parameters()),
        },
        "checks": {
            "initial_prediction_exactly_backbone": initial_deviation == 0,
            "maximum_initial_deviation": initial_deviation,
            "backbone_gradients_absent": not backbone_gradients,
            "reader_gradients_finite": finite_gradients,
            "reader_gradient_nonzero": nonzero_gradient,
        },
        "objective": {name: float(value) for name, value in parts.items()},
        "forward_batch_ms": forward_ms,
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--journal",
        type=Path,
        required=True,
        help="directory containing at least five complete ARGUSLR2 records",
    )
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--context-packets", type=int, default=512)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    result = run_smoke(
        args.journal,
        args.backbone,
        context_packets=args.context_packets,
        samples=args.samples,
    )
    if args.output:
        _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

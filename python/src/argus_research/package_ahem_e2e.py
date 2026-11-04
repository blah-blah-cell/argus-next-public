"""Export the recovered research checkpoint as path-free inference weights.

This is not a resumable training artifact or evidence of detector efficacy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

SOURCE_SHA256 = "d206c62997c283bd050d4a29e09988a1aa431e73b4fbd1729bb0f2c94ce3549f"
SCHEMA = "argus.ahem-e2e-inference-weights/v1"
CONFIG = {
    "width": 256, "key_dim": 64, "phase_width": 128,
    "local_packets": 128, "slots": 16, "horizon": 10_000,
}


def package(source: Path, destination: Path) -> dict[str, object]:
    """Whitelist metadata and verify strict, finite FP32 weights before export."""
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("source checkpoint hash differs from recovered step 3808")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("schema") != "argus.full-ahem-training/v1"
        or checkpoint.get("config") != CONFIG
        or checkpoint.get("step") != 3808
        or checkpoint.get("binding", {}).get("seed") != 20261018
    ):
        raise ValueError("source checkpoint identity differs")
    from .ahem_endtoend_fast_20261103 import FastEndToEndAhem

    state = checkpoint.get("model")
    if not isinstance(state, dict) or not state or any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or not torch.isfinite(value).all()
        for value in state.values()
    ):
        raise ValueError("source weights must be finite FP32 tensors")
    model = FastEndToEndAhem(**CONFIG)
    model.load_state_dict(state, strict=True)
    payload = {
        "schema": SCHEMA,
        "config": CONFIG,
        "step": 3808,
        "seed": 20261018,
        "source_checkpoint_sha256": SOURCE_SHA256,
        "security_detector": False,
        "publication_evidence": False,
        "promotion_eligible": False,
        "evaluation_status": "unvalidated_recovered_incomplete_run",
        "state_dict": {key: value.detach().cpu().clone() for key, value in state.items()},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    manifest = {
        key: value for key, value in payload.items() if key != "state_dict"
    }
    manifest.update(
        artifact_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        parameters=sum(value.numel() for value in model.parameters()),
        state_elements=sum(value.numel() for value in state.values()),
        inference_only=True,
        includes_optimizer_or_rng_state=False,
    )
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()

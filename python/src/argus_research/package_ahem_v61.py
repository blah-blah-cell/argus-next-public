"""Package the frozen AHEM-v6.1 checkpoint for forecast-only desktop use."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from . import ahem_v3 as ahem_v3_module
from . import ahem_v4 as ahem_v4_module
from . import ahem_v6 as ahem_v6_module
from .ahem_v6 import AhemV6StateTransitionNextPacket
from .flow_v2_features import FEATURE_CONTRACT_SHA256

SOURCE_CHECKPOINT_SHA256 = (
    "0edff7f3be0045d82f6ae3e682393965a8942501102ed4fd2d84cdad5e56ad89"
)
FINAL_RESULT_SHA256 = (
    "fbb468ad92c54a98a6a94754458ff44cd21b944a146604959f3b8402e7e00a9a"
)
RUNTIME_SCHEMA = "argus.ahem-v6-runtime-artifact/v1"
MANIFEST_SCHEMA = "argus.ahem-v6-runtime-manifest/v1"
MODEL_ID = "argus.ahem-v6.1.state-transition-forecaster"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_sha256() -> str:
    """Hash normalized model sources so line-ending conversion is irrelevant."""
    digest = hashlib.sha256()
    for module in (ahem_v3_module, ahem_v4_module, ahem_v6_module):
        source_path = Path(module.__file__)
        normalized = source_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        digest.update(source_path.name.encode())
        digest.update(b"\0")
        digest.update(normalized.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def package(checkpoint: Path, final_result: Path, destination: Path) -> dict[str, object]:
    source_hash = sha256(checkpoint)
    if source_hash != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("source checkpoint SHA-256 is not the frozen seed-20260906 artifact")
    result_hash = sha256(final_result)
    if result_hash != FINAL_RESULT_SHA256:
        raise ValueError("final result SHA-256 differs from the frozen evaluation")
    result = json.loads(final_result.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("final result is not a JSON object")
    required_result = {
        "schema": "argus.ahem-v6-final-evaluation-run/v2",
        "decision": "supported",
        "confirmatory_claim_eligible": True,
        "not_intrusion_detection_efficacy": True,
    }
    for key, expected in required_result.items():
        if result.get(key) != expected:
            raise ValueError(f"final result {key} differs from the frozen contract")

    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    required_checkpoint = {
        "schema": "argus.ahem-v6-training-development/v2",
        "model": "ahem_v6_state_transition",
        "seed": 20260906,
        "best_epoch": 48,
    }
    for key, expected in required_checkpoint.items():
        if source.get(key) != expected:
            raise ValueError(f"source checkpoint {key} differs from the selected run")
    state = source.get("best_state_dict")
    if not isinstance(state, dict):
        raise ValueError("source checkpoint has no best_state_dict")
    model = AhemV6StateTransitionNextPacket()
    model.load_state_dict(state, strict=True)
    model.eval()
    implementation_hash = implementation_sha256()

    payload = {
        "schema": RUNTIME_SCHEMA,
        "model_id": MODEL_ID,
        "architecture": "ahem-v6.1-bounded-causal-state-transition-memory",
        "feature_schema": "packet-causal-flow-v2",
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "runtime_adapter": "argus.causal-flow/v1",
        "supervision": "next_same_flow_dynamic_event_forecast_only",
        "security_detector": False,
        "calibrated_security_verdict": False,
        "state_dict": model.state_dict(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "packet_dim": model.packet_dim,
        "output_dim": model.output_dim,
        "width": model.width,
        "key_dim": model.key_dim,
        "local_packets": model.local_packets,
        "max_transitions": model.max_transitions,
        "maximum_context_packets": 10_000,
        "minimum_completed_transitions": 2,
        "loss": "(smooth_l1(log_length)+4*smooth_l1(log_per_flow_interarrival))/5",
        "selected_seed": 20260906,
        "selected_best_epoch": 48,
        "selection": "lowest development validation loss before final holdout access",
        "source_checkpoint_sha256": source_hash,
        "final_result_sha256": result_hash,
        "model_implementation_sha256": implementation_hash,
        "model_implementation_files": ["ahem_v3.py", "ahem_v4.py", "ahem_v6.py"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "model_id": MODEL_ID,
        "artifact_sha256": sha256(destination),
        "source_checkpoint_sha256": source_hash,
        "final_result_sha256": result_hash,
        "model_implementation_sha256": implementation_hash,
        "model_implementation_files": ["ahem_v3.py", "ahem_v4.py", "ahem_v6.py"],
        "security_detector": False,
        "feature_schema": "packet-causal-flow-v2",
        "cache_scope_requirement": "stable opaque flow identity within one capture",
    }
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(manifest_tmp, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--final-result", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            package(args.checkpoint, args.final_result, args.destination),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

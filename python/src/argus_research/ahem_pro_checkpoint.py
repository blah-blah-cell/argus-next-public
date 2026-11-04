"""Atomic, weights-only Pro development artifacts; not desktop admissions.

Optimizer/RNG state is deliberately absent: loading initializes a new training
run, not an exact resume. Caller records and verifies external transport hashes.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from .ahem_pro import AhemProNextPacket, ProConfig, objective_contract

SCHEMA = "argus.ahem-pro-development-weights/v1"


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for name in (
        "ahem_pro.py",
        "ahem_pro_stream.py",
        "ahem_pro_checkpoint.py",
        "ahem_v3.py",
        "ahem_v4.py",
        "ahem_v6.py",
        "flow_v6_objective.py",
        "flow_v2_features.py",
    ):
        digest.update(name.encode() + b"\0")
        digest.update(
            Path(__file__).with_name(name).read_text(encoding="utf-8").encode() + b"\0"
        )
    return digest.hexdigest()


def _validate_provenance(provenance: object) -> dict[str, object]:
    if not isinstance(provenance, dict):
        raise ValueError("Pro provenance must be an object")
    run_id = provenance.get("run_id")
    steps = provenance.get("completed_optimizer_steps")
    fingerprint = provenance.get("training_data_sha256")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Pro provenance needs a run identity")
    if type(steps) is not int or steps < 0:
        raise ValueError("Pro provenance needs nonnegative completed optimizer steps")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(c not in "0123456789abcdef" for c in fingerprint)
    ):
        raise ValueError(
            "Pro provenance needs a lowercase SHA-256 of its training-data manifest"
        )
    if (
        provenance.get("evaluation_status") != "unvalidated"
        or provenance.get("security_detector") is not False
    ):
        raise ValueError("Pro development weights cannot claim validated detection")
    return provenance


def save_pro_weights(
    model: AhemProNextPacket,
    destination: Path,
    *,
    run_id: str,
    training_data_sha256: str,
    completed_optimizer_steps: int,
) -> str:
    """Replace a specifically named development checkpoint atomically; return SHA-256."""
    provenance = _validate_provenance(
        {
            "run_id": run_id,
            "training_data_sha256": training_data_sha256,
            "completed_optimizer_steps": completed_optimizer_steps,
            "evaluation_status": "unvalidated",
            "security_detector": False,
        }
    )
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    if any(
        t.dtype != torch.float32 or not torch.isfinite(t).all() for t in state.values()
    ):
        raise ValueError("Pro development checkpoints require finite FP32 weights")
    payload = {
        "schema": SCHEMA,
        "config": asdict(model.config),
        "architecture_contract": model.config.contract(),
        "architecture_sha256": model.config.contract_sha256(),
        "implementation_sha256": implementation_sha256(),
        "objective_contract": objective_contract(),
        "provenance": provenance,
        "state_dict": state,
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=destination.name + ".",
            suffix=".part",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Fully decode and validate before publishing the final name.
        load_pro_weights(temporary, expected_config=model.config)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, destination)
        return digest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_pro_weights(
    path: Path, *, expected_config: ProConfig
) -> tuple[AhemProNextPacket, dict[str, object]]:
    """Reject other families, objective drift, source drift and corrupt weights."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("not a Pro development weights checkpoint")
    expected = {
        "config": asdict(expected_config),
        "architecture_contract": expected_config.contract(),
        "architecture_sha256": expected_config.contract_sha256(),
        "implementation_sha256": implementation_sha256(),
        "objective_contract": objective_contract(),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Pro checkpoint {key} does not match the requested implementation"
            )
    provenance = _validate_provenance(payload.get("provenance"))
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Pro checkpoint has no state dictionary")
    if any(
        not isinstance(t, torch.Tensor)
        or t.dtype != torch.float32
        or not torch.isfinite(t).all()
        for t in state.values()
    ):
        raise ValueError("Pro checkpoint weights must be finite FP32 tensors")
    # Loading must not consume the caller's training RNG sequence.
    with torch.random.fork_rng(devices=[]):
        model = AhemProNextPacket(expected_config)
    expected_state = model.state_dict()
    if set(state) != set(expected_state) or any(
        state[k].shape != expected_state[k].shape for k in expected_state
    ):
        raise ValueError("Pro checkpoint state names or shapes do not match")
    model.load_state_dict(state, strict=True)
    return model.eval(), provenance

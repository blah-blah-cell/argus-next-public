"""New-run epoch recovery; immutable commits, safe loads, explicit RNG restore.

No existing trainer or frozen artifact is changed. Call only at completed optimizer
step / epoch boundaries with no pending gradient accumulation. Local mirroring is
not proof of durable cloud storage: the caller must verify its export destination.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch

SCHEMA = "argus.epoch-recovery/v1"
BINDINGS = {"protocol_sha256", "source_manifest_sha256", "implementation_sha256"}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _binding(value):
    if not isinstance(value, dict) or set(value) != BINDINGS:
        raise ValueError("exact protocol/source/implementation bindings required")
    if any(not isinstance(h, str) or len(h) != 64 or
           any(c not in "0123456789abcdef" for c in h) for h in value.values()):
        raise ValueError("bindings must be lowercase SHA-256")
    return dict(value)


def _cpu(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite checkpoint tensor")
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(v) for v in value)
    return copy.deepcopy(value)


def _scope(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    groups, seen = [], set()
    for group in optimizer.param_groups:
        row = []
        for p in group["params"]:
            if id(p) not in names or id(p) in seen:
                raise ValueError("optimizer scope is outside model or duplicated")
            seen.add(id(p))
            row.append(names[id(p)])
        groups.append(row)
    return groups


def _rng():
    state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state().clone(),
            "numpy": (state[0], torch.from_numpy(state[1].astype(np.int64)),
                      state[2], state[3], state[4]),
            "cuda": [s.cpu().clone() for s in torch.cuda.get_rng_state_all()]
                    if torch.cuda.is_initialized() else []}


def _write(path, writer):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_epoch(root, model, optimizer, *, epoch, optimizer_steps, binding, metrics):
    """Publish trained weights + resume state; receipt is the final commit marker.

    Epoch identities never overwrite. Incomplete directories remain evidence and
    cannot be resumed. The returned receipt hash must be retained externally.
    """
    if type(epoch) is not int or epoch < 1 or type(optimizer_steps) is not int or optimizer_steps < 1:
        raise ValueError("completed trained epoch and positive optimizer steps required")
    binding = _binding(binding)
    # Validate metrics before creating anything; no NaN/Infinity JSON.
    metrics = json.loads(json.dumps(metrics, allow_nan=False))
    if any(p.grad is not None and torch.count_nonzero(p.grad).item() for p in model.parameters()):
        raise ValueError("clear gradients before committing an epoch boundary")
    payload = {"schema": SCHEMA, "binding": binding, "epoch": epoch,
               "optimizer_steps": optimizer_steps, "model_state": _cpu(model.state_dict()),
               "optimizer_state": _cpu(optimizer.state_dict()),
               "optimizer_class": type(optimizer).__module__ + "." + type(optimizer).__qualname__,
               "optimizer_scope": _scope(model, optimizer), "rng": _rng(),
               "model_training": model.training, "metrics": metrics}
    directory = Path(root) / f"epoch-{epoch:06d}"
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()  # Exclusive reservation; an incomplete epoch is not overwritten.
    path = directory / "checkpoint.pt"
    _write(path, lambda stream: torch.save(payload, stream))
    decoded = torch.load(path, map_location="cpu", weights_only=True)
    if decoded["schema"] != SCHEMA or decoded["binding"] != binding:
        raise ValueError("checkpoint verification failed before commit")
    receipt = {"schema": SCHEMA, "binding": binding, "epoch": epoch,
               "optimizer_steps": optimizer_steps, "checkpoint_sha256": sha256(path),
               "bytes": path.stat().st_size, "complete": True}
    marker = directory / "COMMITTED.json"
    _write(marker, lambda stream: stream.write(json.dumps(receipt, allow_nan=False,
                                                        sort_keys=True).encode()))
    return directory, sha256(marker)


def _verified(directory, receipt_sha256, binding):
    directory = Path(directory)
    marker = directory / "COMMITTED.json"
    if sha256(marker) != receipt_sha256:
        raise ValueError("external receipt hash mismatch")
    receipt = json.loads(marker.read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("complete") is not True or receipt.get("binding") != _binding(binding):
        raise ValueError("uncommitted epoch or wrong experiment binding")
    path = directory / "checkpoint.pt"
    if path.stat().st_size != receipt["bytes"] or sha256(path) != receipt["checkpoint_sha256"]:
        raise ValueError("checkpoint hash/size mismatch")
    return receipt, path


def mirror_epoch(directory, destination_root, *, receipt_sha256, binding):
    """Copy completed artifacts and verify bytes before committing the mirror."""
    receipt, source = _verified(directory, receipt_sha256, binding)
    destination = Path(destination_root) / Path(directory).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    def copy_to(stream):
        with source.open("rb") as incoming:
            shutil.copyfileobj(incoming, stream)
    _write(destination / "checkpoint.pt", copy_to)
    if sha256(destination / "checkpoint.pt") != receipt["checkpoint_sha256"]:
        raise ValueError("mirror hash mismatch; no commit marker published")
    marker_bytes = (Path(directory) / "COMMITTED.json").read_bytes()
    _write(destination / "COMMITTED.json", lambda stream: stream.write(marker_bytes))
    _verified(destination, receipt_sha256, binding)
    return destination


def resume_epoch(directory, model, optimizer, *, receipt_sha256, binding):
    """Strict restore, including Python/NumPy/Torch/CUDA RNG; same CUDA topology.

    Scheduler, GradScaler, sampler position and custom generators are NOT covered.
    This contract therefore applies to fixed-LR FP32, epoch-boundary trainers using
    these global RNGs. A future AMP/scheduler trainer needs an extended schema.
    """
    receipt, path = _verified(directory, receipt_sha256, binding)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("schema") != SCHEMA or payload.get("binding") != binding or
        payload.get("epoch") != receipt["epoch"] or payload.get("optimizer_steps") != receipt["optimizer_steps"]):
        raise ValueError("checkpoint/receipt inconsistency")
    if payload["optimizer_scope"] != _scope(model, optimizer) or payload["optimizer_class"] != type(optimizer).__module__ + "." + type(optimizer).__qualname__:
        raise ValueError("optimizer type or parameter ordering changed")
    _cpu(payload["model_state"])
    _cpu(payload["optimizer_state"])
    cuda = payload["rng"]["cuda"]
    if cuda and (not torch.cuda.is_available() or len(cuda) != torch.cuda.device_count()):
        raise ValueError("CUDA RNG topology changed; exact resume unavailable")
    # Validate loads on jointly deep-copied model/optimizer before touching caller.
    probe_model, probe_optimizer = copy.deepcopy((model, optimizer))
    probe_model.load_state_dict(payload["model_state"], strict=True)
    probe_optimizer.load_state_dict(payload["optimizer_state"])
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    model.train(payload["model_training"])
    random.setstate(payload["rng"]["python"])
    state = payload["rng"]["numpy"]
    np.random.set_state((state[0], state[1].numpy().astype(np.uint32), *state[2:]))
    torch.set_rng_state(payload["rng"]["torch"])
    if cuda:
        torch.cuda.set_rng_state_all(cuda)
    return {"epoch": payload["epoch"], "optimizer_steps": payload["optimizer_steps"],
            "metrics": payload["metrics"]}

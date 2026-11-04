"""Research-only October 12 bounded pilot and fail-closed training primitives.

Does not alter the retained parent, retrieval, deployed models or test access.
Projection caches are an audited preparation boundary, not proof of causality.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import torch
from torch import nn
from .flow_v6_objective import dynamic_loss

METRICS = ("weighted_smooth_l1", "byte_mae", "delay_mae_ms")
BOUNDS = (math.log1p(65535), math.log1p(60))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_write(path, writer):
    """Unique same-directory temporary; fsync then atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return digest(path)


def atomic_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    return atomic_write(path, lambda stream: stream.write(raw))


def validate_cache(cache, context_width):
    keys = {"features", "context", "current", "raw", "available", "target", "ordinal"}
    if set(cache) != keys or not all(isinstance(v, torch.Tensor) for v in cache.values()):
        raise ValueError("exact projected cache schema required")
    n = len(cache["target"])
    shapes = {"features": (n, 10), "context": (n, context_width),
              "current": (n, 2), "raw": (n, 2), "target": (n, 2),
              "available": (n, 1), "ordinal": (n,)}
    if not n or any(cache[k].shape != shape for k, shape in shapes.items()):
        raise ValueError("empty or unaligned projected cache")
    if cache["available"].dtype != torch.bool or cache["ordinal"].dtype not in (torch.int32, torch.int64):
        raise ValueError("boolean memory mask and integer ordinals required")
    if any(cache[k].dtype != torch.float32 for k in ("features", "context", "current", "raw", "target")):
        raise ValueError("float32 projected fields required for this pilot")
    if (cache["ordinal"] < 0).any() or any(not torch.isfinite(v).all() for v in cache.values()):
        raise ValueError("nonfinite cache or negative ordinals")
    for key in ("current", "raw", "target"):
        bound = cache[key].new_tensor(BOUNDS)
        if not cache[key].is_floating_point() or ((cache[key] < 0) | (cache[key] > bound + 1e-6)).any():
            raise ValueError(f"{key} violates the declared packet-size/60-second delay contract")


class BoundedGateAdapter(nn.Module):
    """Small zero-logit correction; targets never enter the inference API."""
    def __init__(self, head, startup_scale, local_packets, width=32):
        super().__init__()
        if not 1 <= width <= 256 or local_packets < 1:
            raise ValueError("bounded small-head pilot required")
        self.parent_head = copy.deepcopy(head).requires_grad_(False)
        scale = torch.as_tensor(startup_scale).detach().clone()
        if scale.shape != (2,) or not torch.isfinite(scale).all() or ((scale < 0) | (scale > 1)).any():
            raise ValueError("invalid frozen startup scale")
        self.register_buffer("startup_scale", scale)
        self.local_packets = local_packets
        self.delta = nn.Sequential(nn.Linear(10 + len(head.context_mean), width),
                                   nn.GELU(), nn.Linear(width, 2))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, features, context, current, raw, available, ordinal):
        if available.dtype != torch.bool or available.shape != (len(current), 1):
            raise ValueError("aligned boolean availability required")
        if ordinal.shape != (len(current),) or ordinal.dtype not in (torch.int32, torch.int64):
            raise ValueError("aligned integer ordinal required")
        head = self.parent_head
        z = ((context - head.context_mean) / head.context_scale).clamp(-10, 10)
        logits = head.network(features)
        if head.context_enabled:
            logits = logits + head.context_delta(z)
        coefficient = (logits + self.delta(torch.cat((features, z), 1))).sigmoid()
        coefficient = coefficient * available.to(coefficient.dtype)
        result = current + coefficient * (raw - current)
        corrected = result + (1 - self.startup_scale) * (current - result)
        return torch.where((ordinal < self.local_packets - 1)[:, None], corrected, result)


def predict(model, cache):
    return model(*(cache[k] for k in ("features", "context", "current", "raw", "available", "ordinal")))


@torch.no_grad()
def measure(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 2 or not len(target):
        raise ValueError("aligned nonempty forecast/target required")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("nonfinite prediction or target")
    bounds = target.new_tensor(BOUNDS)
    if ((target < 0) | (target > bounds + 1e-6)).any():
        raise ValueError("invalid physical target")
    # Unclipped physical errors: invalid predictions are not silently repaired.
    error = (prediction.double().expm1() - target.double().expm1()).abs().mean(0)
    if not torch.isfinite(error).all():
        raise ValueError("physical inverse overflow")
    return {"queries": len(target), "weighted_smooth_l1": float(dynamic_loss(prediction.float(), target.float())),
            "byte_mae": float(error[0]), "delay_mae_ms": float(error[1] * 1000),
            "invalid_outputs": int(((prediction < -1e-6) | (prediction > bounds + 1e-6)).any(1).sum())}


def acceptance(candidate, baseline, *, relative_cap=0.05, absolute_tolerances=None):
    if not 0 <= relative_cap <= 0.05:
        raise ValueError("cap cannot exceed agreed 5%")
    tolerances = absolute_tolerances or {k: 0.0 for k in METRICS}
    if set(tolerances) != set(METRICS) or any(not math.isfinite(v) or v < 0 for v in tolerances.values()):
        raise ValueError("explicit finite nonnegative metric tolerances required")
    failures = []
    if not baseline or set(candidate) != set(baseline):
        return {"accepted": False, "failures": ["source population changed"]}
    for source, parent in baseline.items():
        row = candidate[source]
        if row.get("queries") != parent["queries"]:
            failures.append(f"{source}: coverage changed")
        for metric in METRICS:
            value, reference = row.get(metric), parent[metric]
            if value is None or not math.isfinite(value) or value < 0 or value > reference * (1 + relative_cap) + tolerances[metric]:
                failures.append(f"{source}: {metric}")
        if row.get("invalid_outputs") != 0:
            failures.append(f"{source}: invalid outputs")
    return {"accepted": not failures, "failures": failures}


def memory_admission(free_bytes, required_bytes, *, reserve_bytes=2 * 2**30):
    if min(free_bytes, required_bytes, reserve_bytes) < 0:
        raise ValueError("negative memory budget")
    return required_bytes + reserve_bytes <= free_bytes


def snapshot(model, optimizer, scheduler, sampler, epoch, binding, metrics, best_score, stale):
    return {"schema": "argus.oct12-bounded-pilot/v1", "epoch": epoch,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "sampler_rng": sampler.get_state(),
            "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "binding": binding, "metrics": metrics, "best_score": best_score, "stale": stale}


def restore(payload, model, optimizer, scheduler, sampler, binding):
    if payload.get("schema") != "argus.oct12-bounded-pilot/v1" or payload["binding"] != binding:
        raise ValueError("resume binding changed")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    sampler.set_state(payload["sampler_rng"])
    torch.set_rng_state(payload["torch_rng"])
    random.setstate(payload["python_rng"])
    if payload["cuda_rng"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA resume requires CUDA")
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["epoch"], payload["best_score"], payload["stale"]


class Checkpoints:
    def __init__(self, folder, save_every):
        self.folder = Path(folder)
        if save_every < 1:
            raise ValueError("positive snapshot cadence required")
        self.save_every = save_every

    def commit(self, payload, *, improved):
        receipts = {}
        names = ["latest.pt"]
        if improved:
            names.append("best.pt")
        if payload["epoch"] % self.save_every == 0 or payload["epoch"] == 0:
            names.append(f"epoch-{payload['epoch']:04d}.pt")
        for name in names:
            path = self.folder / name
            if name.startswith("epoch-") and path.exists():
                raise FileExistsError(path)
            receipts[name] = atomic_write(path, lambda stream: torch.save(payload, stream))
            atomic_json(path.with_suffix(".sha256.json"), {"sha256": receipts[name], "epoch": payload["epoch"]})
        return receipts

    def load(self, name="latest.pt"):
        if name not in ("latest.pt", "best.pt"):
            raise ValueError("only committed latest/best resume names allowed")
        path = self.folder / name
        receipt = json.loads(path.with_suffix(".sha256.json").read_text())
        if digest(path) != receipt["sha256"]:
            raise ValueError("resume checkpoint hash mismatch")
        return torch.load(path, map_location="cpu", weights_only=True)

    def load_latest(self):
        return self.load()

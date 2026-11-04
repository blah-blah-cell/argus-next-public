"""Grouped complete-forecast learnability diagnostic, never a runtime policy."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def one_slot_forecasts(state: dict, data: dict) -> torch.Tensor:
    """Each completed slot read alone, with original count evidence and frozen gate."""
    raw = data["successors"]
    n, k, _ = raw.shape
    current = data["current"][:, None].expand_as(raw)
    features = torch.cat(
        (
            current,
            raw,
            raw - current,
            torch.zeros_like(raw),
            raw.new_zeros(n, k, 1),
            (data["valid"].sum(1).to(raw.dtype) / 16)[:, None, None].expand(n, k, 1),
        ),
        2,
    )
    z = (
        (data["context"] - state["readout.head.context_mean"])
        / state["readout.head.context_scale"]
    ).clamp(-10, 10)

    def linear(x, prefix):
        return F.linear(x, state[prefix + ".weight"], state[prefix + ".bias"])

    coefficient = (
        linear(
            F.gelu(linear(features, "readout.head.network.0")), "readout.head.network.2"
        )
        + linear(z, "readout.head.context_delta")[:, None]
    ).sigmoid()
    return current + coefficient * (raw - current)


def huber_cost(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    error = np.abs(
        np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    )
    fields = np.where(error < 1, error * error / 2, error - 0.5)
    return (fields[..., 0] + 4 * fields[..., 1]) / 5


def group_query(forecasts, descriptors):
    """Exact complete-forecast groups, with permutation-invariant causal features."""
    forecasts, descriptors = np.asarray(forecasts), np.asarray(descriptors)
    if (
        forecasts.ndim != 2
        or forecasts.shape[1] != 2
        or descriptors.ndim != 2
        or len(descriptors) != len(forecasts)
        or not len(forecasts)
        or not np.isfinite(forecasts).all()
        or not np.isfinite(descriptors).all()
    ):
        raise ValueError("nonempty aligned finite forecasts/descriptors required")
    unique, inverse = np.unique(forecasts, axis=0, return_inverse=True)
    # Sum in float64, so permutation cannot materially change a group descriptor.
    sums = np.zeros((len(unique), descriptors.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, descriptors)
    grouped = sums / np.bincount(inverse)[:, None]
    return unique, grouped.astype(np.float32), inverse


def build_choices(
    data: dict, slot_predictions: torch.Tensor, retained: torch.Tensor, state: dict
) -> dict[str, np.ndarray]:
    """No future target is read, passed or required here."""
    n = len(retained)
    valid = data["valid"].numpy()
    z = (
        (
            (data["context"] - state["readout.head.context_mean"])
            / state["readout.head.context_scale"]
        )
        .clamp(-10, 10)
        .numpy()
    )
    slots, anchor = slot_predictions.numpy(), retained.numpy()
    current, feature = data["current"].numpy(), data["features"].numpy()
    protected = (data["ordinal"] < 127).numpy() | ~valid.any(1)
    predictions, descriptors, offsets, parent, recent, scored = [], [], [0], [], [], []
    for i in range(n):
        indices = np.flatnonzero(valid[i]) if not protected[i] else np.array([], int)
        options = np.concatenate(
            (slots[i, indices], current[i : i + 1], anchor[i : i + 1])
        )
        if protected[i]:
            options = anchor[i : i + 1]
        x = np.zeros((len(options), 165), dtype=np.float32)
        if len(indices):
            x[: len(indices), :18] = feature[i, indices]
            x[: len(indices), 22] = 1  # Historical slot action.
        if not protected[i]:
            x[-2, 23] = 1  # Persistence action.
        x[-1, 24] = 1  # Retained action always present.
        x[:, 18:20], x[:, 20:22], x[:, 25:] = options, options - anchor[i], z[i]
        unique, grouped, inverse = group_query(options, x)
        start = offsets[-1]
        predictions.append(unique)
        descriptors.append(grouped)
        parent.append(start + int(inverse[-1]))
        if len(indices):
            recent_local = int(np.argmin(feature[i, indices, 1]))
            scored_local = int(np.argmax(data["scores"][i, indices].numpy()))
            recent.append(start + int(inverse[recent_local]))
            scored.append(start + int(inverse[scored_local]))
        else:
            recent.append(parent[-1])
            scored.append(parent[-1])
        offsets.append(start + len(unique))
    return {
        "x": np.concatenate(descriptors),
        "forecasts": np.concatenate(predictions),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "parent": np.asarray(parent),
        "recent": np.asarray(recent),
        "scored": np.asarray(scored),
        "protected": protected,
        "current": current,
        "retained": anchor,
    }


def select_rows(pack: dict, queries: np.ndarray) -> dict:
    queries = np.asarray(queries, dtype=np.int64)
    lengths = np.diff(pack["offsets"])[queries]
    offsets = np.r_[0, np.cumsum(lengths)]
    rows = np.concatenate(
        [np.arange(pack["offsets"][q], pack["offsets"][q + 1]) for q in queries]
    )
    result = {k: pack[k][rows] for k in ("x", "forecasts")}
    result["offsets"] = offsets
    for k in ("parent", "recent", "scored"):
        result[k] = offsets[:-1] + pack[k][queries] - pack["offsets"][queries]
    for k in ("protected", "current", "retained", "target", "coordinates"):
        result[k] = pack[k][queries]
    return result


def training_arrays(pack: dict, *, shuffle_seed: int | None = None):
    target = pack["target"]
    permutation = np.arange(len(target))
    if shuffle_seed is not None:
        permutation = np.random.default_rng(shuffle_seed).permutation(len(target))
    lengths = np.diff(pack["offsets"])
    advantage = huber_cost(pack["retained"], target[permutation]).repeat(
        lengths
    ) - huber_cost(pack["forecasts"], target[permutation].repeat(lengths, 0))
    weights = np.repeat(1.0 / lengths, lengths)
    return pack["x"], advantage, weights, permutation


def select(pack: dict, scores: np.ndarray) -> np.ndarray:
    if scores.shape != (len(pack["forecasts"]),) or not np.isfinite(scores).all():
        raise ValueError("finite aligned predicted advantages required")
    selected = pack["parent"].copy()
    for i, (a, b) in enumerate(zip(pack["offsets"][:-1], pack["offsets"][1:])):
        best = a + int(np.argmax(scores[a:b]))
        if scores[best] > 0 and not pack["protected"][i]:
            selected[i] = best
    return selected


def assess(pack: dict, prediction: np.ndarray) -> dict:
    """Future-aware labels/strata used ONLY after a fixed prediction exists."""
    target = pack["target"]
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError("nonempty finite aligned forecasts required")
    costs = huber_cost(pack["forecasts"], target.repeat(np.diff(pack["offsets"]), 0))
    base = huber_cost(pack["retained"], target)
    chosen = huber_cost(prediction, target)
    oracle = np.minimum.reduceat(costs, pack["offsets"][:-1])
    uniform = np.add.reduceat(costs, pack["offsets"][:-1]) / np.diff(pack["offsets"])
    ties = np.abs(costs - oracle.repeat(np.diff(pack["offsets"]))) <= 1e-10
    chance = np.add.reduceat(ties.astype(float), pack["offsets"][:-1]) / np.diff(
        pack["offsets"]
    )
    consequence = base - oracle >= 0.01
    physical = np.abs(
        np.expm1(prediction.astype(float)) - np.expm1(target.astype(float))
    )
    bounds = np.log1p([65535, 60])
    return {
        "queries": len(target),
        "weighted_smooth_l1": float(chosen.mean()),
        "byte_mae": float(physical[:, 0].mean()),
        "delay_mae_ms": float(physical[:, 1].mean() * 1000),
        "invalid_outputs": int(
            ((prediction < -1e-6) | (prediction > bounds + 1e-6)).any(1).sum()
        ),
        "coverage": 1.0,
        "oracle_loss": float(oracle.mean()),
        "mean_regret": float((chosen - oracle).mean()),
        "best_group_accuracy": float((np.abs(chosen - oracle) <= 1e-10).mean()),
        "uniform_expected_loss": float(uniform.mean()),
        "uniform_expected_best_group_accuracy": float(chance.mean()),
        "loss_reduction_percent": float(100 * (1 - chosen.mean() / base.mean())),
        "consequential_queries": int(consequence.sum()),
        "consequential_loss": float(chosen[consequence].mean())
        if consequence.any()
        else None,
        "consequential_retained_loss": float(base[consequence].mean())
        if consequence.any()
        else None,
        "consequential_regret": float((chosen - oracle)[consequence].mean())
        if consequence.any()
        else None,
        "consequential_best_group_accuracy": float(
            (np.abs(chosen - oracle)[consequence] <= 1e-10).mean()
        )
        if consequence.any()
        else None,
        "consequential_uniform_accuracy": float(chance[consequence].mean())
        if consequence.any()
        else None,
        "changed_forecasts": int(np.any(prediction != pack["retained"], axis=1).sum()),
        "protected_queries": int(pack["protected"].sum()),
    }

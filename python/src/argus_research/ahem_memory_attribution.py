"""Retrospective memory attribution; no model fitting or online policy.

Targets are used only to evaluate historical candidate opportunity. Candidate
enumeration itself accepts no target. A best candidate is not a bound on an
arbitrary mixture/persistence forecast.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def field_loss(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.shape[-1:] != (2,):
        raise ValueError("aligned two-field arrays required")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("nonfinite evaluation data")
    error = np.abs(prediction - target)
    return np.where(error < 1, error**2 / 2, error - 0.5)


def loss(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return (field_loss(prediction, target) * np.array([1.0, 4.0])).sum(-1) / 5


def phase_footprints(
    previous: np.ndarray, *, local_packets: int, phase_packets: int, horizon: int
) -> np.ndarray:
    previous = np.asarray(previous)
    if (
        previous.ndim != 1
        or not np.issubdtype(previous.dtype, np.integer)
        or min(local_packets, phase_packets, horizon) <= 0
        or np.any((previous < -1) | (previous >= np.arange(len(previous))))
    ):
        raise ValueError("causal recurrence links and positive dimensions required")
    earliest = np.zeros(len(previous), dtype=np.int64)
    for index in range(len(previous)):
        oldest, cursor = max(0, index - horizon + 1), index
        for _ in range(phase_packets - 1):
            before = int(previous[cursor])
            if before < oldest:
                break
            cursor = before
        earliest[index] = min(cursor, max(0, index - local_packets + 1))
    return earliest


def candidate_bank(
    previous: np.ndarray,
    earliest: np.ndarray,
    anchor: int,
    *,
    capacity: int = 16,
    horizon: int = 10_000,
) -> dict[str, np.ndarray]:
    """Newest-first completed same-flow transitions; FIFO occurs before phase mask."""
    previous, earliest = np.asarray(previous), np.asarray(earliest)
    if (
        previous.ndim != 1
        or earliest.shape != previous.shape
        or not np.issubdtype(previous.dtype, np.integer)
        or not np.issubdtype(earliest.dtype, np.integer)
        or not 0 <= anchor < len(previous)
        or min(capacity, horizon) <= 0
        or np.any((previous < -1) | (previous >= np.arange(len(previous))))
        or np.any((earliest < 0) | (earliest > np.arange(len(previous))))
    ):
        raise ValueError("invalid causal bank inputs")
    oldest = max(0, anchor - horizon + 1)
    chain, cursor = [anchor], anchor
    while int(previous[cursor]) >= oldest:
        cursor = int(previous[cursor])
        chain.append(cursor)
    source = np.asarray(chain[1:-1], dtype=np.int64)
    successor = np.asarray(chain[:-2], dtype=np.int64)
    phase_valid = earliest[source] >= oldest
    retained = np.arange(len(source)) < capacity
    return {
        "source": source,
        "successor": successor,
        "phase_valid": phase_valid,
        "retained": retained,
        "bank_valid": phase_valid & retained,
    }


def attribute_query(
    *,
    target: np.ndarray,
    current: np.ndarray,
    successors: np.ndarray,
    full_valid: np.ndarray,
    bank_valid: np.ndarray,
    weights: np.ndarray,
    raw_read: np.ndarray,
    prediction: np.ndarray,
    startup_scale: np.ndarray,
    tie_tolerance: float = 1e-8,
) -> dict[str, Any]:
    target, current, successors, weights, raw_read, prediction, scale = (
        np.asarray(value, dtype=np.float64)
        for value in (
            target,
            current,
            successors,
            weights,
            raw_read,
            prediction,
            startup_scale,
        )
    )
    full_valid, bank_valid = np.asarray(full_valid), np.asarray(bank_valid)
    n = len(successors)
    if (
        successors.shape != (n, 2)
        or any(x.shape != (2,) for x in (target, current, raw_read, prediction, scale))
        or any(x.shape != (n,) for x in (full_valid, bank_valid, weights))
        or full_valid.dtype != bool
        or bank_valid.dtype != bool
        or np.any(bank_valid & ~full_valid)
        or np.any(weights < 0)
        or np.any(weights[~bank_valid] != 0)
        or any(
            not np.isfinite(x).all()
            for x in (target, current, successors, weights, raw_read, prediction, scale)
        )
        or np.any(target < 0)
        or np.any((scale < 0) | (scale > 1))
        or not np.isfinite(tie_tolerance)
        or tie_tolerance < 0
    ):
        raise ValueError("invalid aligned attribution data")
    count, full_count = int(bank_valid.sum()), int(full_valid.sum())
    if not np.isclose(weights.sum(), float(count > 0), atol=1e-6):
        raise ValueError("weights must sum to one with memory, zero without")
    expected_raw = (weights[:, None] * successors).sum(0) if count else current
    if not np.allclose(raw_read, expected_raw, rtol=1e-6, atol=2e-6):
        raise ValueError("raw read does not match actual weights")
    if not count and (
        not np.array_equal(raw_read, current) or not np.array_equal(prediction, current)
    ):
        raise ValueError("no-memory outputs must equal persistence")
    losses = loss(successors, np.broadcast_to(target, successors.shape))
    persistence = float(loss(current, target))
    raw_loss, forecast = float(loss(raw_read, target)), float(loss(prediction, target))
    full_best = (
        min(persistence, float(losses[full_valid].min())) if full_count else persistence
    )
    bank_best = (
        min(persistence, float(losses[bank_valid].min())) if count else persistence
    )
    end = current + scale * (raw_read - current)
    low, high = np.minimum(current, end), np.maximum(current, end)
    if np.any(prediction < low - 2e-6) or np.any(prediction > high + 2e-6):
        raise ValueError("prediction outside calibrated per-field blend interval")
    oracle_gate = np.clip(target, low, high)
    gate_best = float(loss(oracle_gate, target))
    result: dict[str, Any] = {
        "bank_candidates": count,
        "history_candidates": full_count,
        "persistence_loss": persistence,
        "forecast_loss": forecast,
        "weighted_raw_read_loss": raw_loss,
        "full_candidate_or_persistence_oracle_loss": full_best,
        "bank_candidate_or_persistence_oracle_loss": bank_best,
        "storage_candidate_opportunity_gap": max(0.0, bank_best - full_best),
        "calibrated_per_field_gate_oracle_loss": gate_best,
        "fixed_read_gate_headroom": max(0.0, forecast - gate_best),
        "forecast_minus_actual_raw_loss": forecast - raw_loss,
        "forecast_minus_persistence_loss": forecast - persistence,
        "raw_read": raw_read.tolist(),
        "prediction": prediction.tolist(),
        "gate_oracle_prediction": oracle_gate.tolist(),
        "target": target.tolist(),
        "current": current.tolist(),
    }
    if count:
        selected = int(np.where(bank_valid, weights, -1).argmax())
        best = float(losses[bank_valid].min())
        worst = float(losses[bank_valid].max())
        tied = bank_valid & (np.abs(losses - best) <= tie_tolerance)
        recent = int(np.flatnonzero(bank_valid)[0])
        result.update(
            {
                "ranking_regret": float(losses[selected] - best),
                "selected_tie_correct": bool(tied[selected]),
                "random_tie_accuracy": float(tied.sum() / count),
                "optimal_ties": int(tied.sum()),
                "bank_spread": worst - best,
                "recency_consequential": bool(
                    count > 1 and losses[recent] - best > 0.01
                ),
                "spread_consequential": bool(count > 1 and worst - best > 0.01),
            }
        )
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("nonempty query rows required")
    from .ahem_forecast_audit import forecast_quality

    numeric = [
        "forecast_loss",
        "persistence_loss",
        "weighted_raw_read_loss",
        "storage_candidate_opportunity_gap",
        "fixed_read_gate_headroom",
        "full_candidate_or_persistence_oracle_loss",
        "bank_candidate_or_persistence_oracle_loss",
        "calibrated_per_field_gate_oracle_loss",
        "forecast_minus_actual_raw_loss",
        "forecast_minus_persistence_loss",
    ]
    choices = [row for row in rows if row["bank_candidates"] > 1]
    target = np.asarray([row["target"] for row in rows])
    output = {
        "queries": len(rows),
        "no_memory": sum(row["bank_candidates"] == 0 for row in rows),
        "multi_candidate_queries": len(choices),
        "storage_opportunity_queries": sum(
            row["storage_candidate_opportunity_gap"] > 1e-8 for row in rows
        ),
        "gate_headroom_queries": sum(
            row["fixed_read_gate_headroom"] > 1e-8 for row in rows
        ),
        "forecast_better_than_raw_queries": sum(
            row["forecast_minus_actual_raw_loss"] < -1e-8 for row in rows
        ),
        "forecast_worse_than_raw_queries": sum(
            row["forecast_minus_actual_raw_loss"] > 1e-8 for row in rows
        ),
        "forecast_better_than_persistence_queries": sum(
            row["forecast_minus_persistence_loss"] < -1e-8 for row in rows
        ),
        "forecast_worse_than_persistence_queries": sum(
            row["forecast_minus_persistence_loss"] > 1e-8 for row in rows
        ),
        "means": {key: float(np.mean([row[key] for row in rows])) for key in numeric},
        "quality": {
            key: forecast_quality(np.asarray([row[key] for row in rows]), target)
            for key in ["prediction", "raw_read", "current", "gate_oracle_prediction"]
        },
    }
    output["retrieval"] = {
        "queries": len(choices),
        "top1_tie_accuracy": float(
            np.mean([row["selected_tie_correct"] for row in choices])
        )
        if choices
        else None,
        "random_tie_accuracy": float(
            np.mean([row["random_tie_accuracy"] for row in choices])
        )
        if choices
        else None,
        "mean_regret": float(np.mean([row["ranking_regret"] for row in choices]))
        if choices
        else None,
    }
    for label in ["recency_consequential", "spread_consequential"]:
        subset = [row for row in choices if row[label]]
        output["retrieval"][label] = {
            "queries": len(subset),
            "top1_tie_accuracy": float(
                np.mean([row["selected_tie_correct"] for row in subset])
            )
            if subset
            else None,
            "random_tie_accuracy": float(
                np.mean([row["random_tie_accuracy"] for row in subset])
            )
            if subset
            else None,
            "mean_regret": float(np.mean([row["ranking_regret"] for row in subset]))
            if subset
            else None,
        }
    return output

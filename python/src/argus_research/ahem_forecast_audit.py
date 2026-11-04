"""Transparent causal controls and physical-unit forecast diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _aligned(features: np.ndarray, previous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features, previous = np.asarray(features), np.asarray(previous)
    if features.ndim != 2 or features.shape[1] != 12 or previous.shape != (len(features),):
        raise ValueError("expected twelve features and aligned previous links")
    if not np.isfinite(features).all() or not np.issubdtype(previous.dtype, np.integer):
        raise ValueError("features must be finite and recurrence links integer")
    if np.any((previous < -1) | (previous >= np.arange(len(features)))):
        raise ValueError("previous links must point to causal history")
    return features, previous


def causal_ewma(features: np.ndarray, previous: np.ndarray, *, alpha: float = 0.2) -> np.ndarray:
    """Predict from current and prior same-flow observations, including cold starts."""
    features, previous = _aligned(features, previous)
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must lie in (0, 1]")
    result = features[:, :2].astype(np.float64).copy()
    for index, prior in enumerate(previous):
        if prior >= 0:
            result[index] = alpha * features[index, :2] + (1 - alpha) * result[prior]
    return result


@dataclass(frozen=True)
class ProtocolMedian:
    """Fit on completed training transitions; protocol is observed at the anchor."""
    global_median: np.ndarray
    medians: dict[tuple[bool, bool], np.ndarray]

    @classmethod
    def fit(cls, anchors: np.ndarray, targets: np.ndarray) -> "ProtocolMedian":
        anchors, targets = np.asarray(anchors), np.asarray(targets)
        if anchors.ndim != 2 or anchors.shape[1] != 12 or targets.shape != (len(anchors), 2) or not len(anchors):
            raise ValueError("aligned training anchors and two-field targets are required")
        if not np.isfinite(anchors).all() or not np.isfinite(targets).all():
            raise ValueError("training data must be finite")
        groups = [(bool(row[4] > 0.5), bool(row[5] > 0.5)) for row in anchors]
        medians = {
            key: np.median(targets[[group == key for group in groups]], axis=0)
            for key in set(groups)
        }
        return cls(np.median(targets, axis=0), medians)

    def predict(self, anchors: np.ndarray) -> np.ndarray:
        anchors = np.asarray(anchors)
        if anchors.ndim != 2 or anchors.shape[1] != 12 or not np.isfinite(anchors).all():
            raise ValueError("finite twelve-field anchors required")
        return np.asarray([
            self.medians.get((bool(row[4] > 0.5), bool(row[5] > 0.5)), self.global_median)
            for row in anchors
        ], dtype=np.float64).reshape(-1, 2)


def forecast_quality(prediction: np.ndarray, target: np.ndarray) -> dict:
    """Report exact log objective and unrounded physical-unit errors.

    Delay targets in the frozen feature contract are clipped to 60 seconds;
    these metrics cannot describe unobserved delays beyond that cap. Negative
    predicted lengths are floored at zero only for the physical-unit view.
    """
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if prediction.ndim != 2 or prediction.shape[1] != 2 or target.shape != prediction.shape or not len(target):
        raise ValueError("nonempty aligned two-field predictions and targets required")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("finite forecasts and nonnegative observed log targets required")
    error = np.abs(prediction - target)
    huber = np.where(error < 1, 0.5 * error ** 2, error - 0.5)
    with np.errstate(over="raise", invalid="raise"):
        physical_prediction = np.expm1(np.maximum(prediction, 0))
        physical_target = np.expm1(target)
    physical_error = np.abs(physical_prediction - physical_target)
    physical_error[:, 1] *= 1000
    weighted = (huber[:, 0] + 4 * huber[:, 1]) / 5
    return {
        "queries": len(target), "weighted_smooth_l1": float(weighted.mean()),
        "size_smooth_l1": float(huber[:, 0].mean()),
        "delay_smooth_l1": float(huber[:, 1].mean()),
        "size_weighted_contribution": float(huber[:, 0].mean() / 5),
        "delay_weighted_contribution": float(4 * huber[:, 1].mean() / 5),
        "size_mae_bytes": float(physical_error[:, 0].mean()),
        "delay_mae_ms": float(physical_error[:, 1].mean()),
        "size_error_median_bytes": float(np.median(physical_error[:, 0])),
        "delay_error_median_ms": float(np.median(physical_error[:, 1])),
        "size_error_p95_bytes": float(np.quantile(physical_error[:, 0], 0.95)),
        "delay_error_p95_ms": float(np.quantile(physical_error[:, 1], 0.95)),
        "negative_length_predictions": int((prediction[:, 0] < 0).sum()),
        "target_delay_cap_seconds": 60,
    }

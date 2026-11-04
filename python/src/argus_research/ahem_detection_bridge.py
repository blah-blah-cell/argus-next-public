"""Causal metrics and feature contract for the frozen AHEM detection bridge.

This module does not label traffic and does not train AHEM.  It turns a
forecast that was made before a same-flow packet arrived into residual evidence
at that packet's arrival time, and evaluates already-scored packet events.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

RESIDUAL_FEATURES = (
    "signed_log_frame_length_error",
    "signed_log_interarrival_error",
    "absolute_log_frame_length_error",
    "absolute_log_interarrival_error",
    "smooth_l1_log_frame_length",
    "smooth_l1_log_interarrival",
    "weighted_smooth_l1",
)
PLAN_SCHEMA = "argus.ahem-v61-detection-bridge/v1"
EXPECTED_PLAN_SHA256 = (
    "49fe6eb8b5e1ad498569a8c4b277aa3b8a042285c124b5c075578b0e15ea2c58"
)


class ForecastResidualTracker:
    """Bind a forecast to the next packet of the same flow generation."""

    def __init__(self, capture_id: str) -> None:
        self.reset(capture_id)

    def reset(self, capture_id: str) -> None:
        if not isinstance(capture_id, str) or not capture_id:
            raise ValueError("a non-empty capture identity is required")
        self.capture_id = capture_id
        self._pending: dict[object, np.ndarray] = {}

    def observe(
        self,
        flow_generation: object,
        observed: np.ndarray,
        next_prediction: np.ndarray | None,
        *,
        capture_id: str,
    ) -> np.ndarray | None:
        if capture_id != self.capture_id:
            raise ValueError("capture changed; explicitly reset residual state")
        hash(flow_generation)
        observed = np.asarray(observed, dtype=np.float64)
        if observed.shape != (2,) or not np.isfinite(observed).all():
            raise ValueError("observed target must be a finite two-field vector")
        if next_prediction is not None:
            next_prediction = np.asarray(next_prediction, dtype=np.float64)
            if next_prediction.shape != (2,) or not np.isfinite(
                next_prediction
            ).all():
                raise ValueError("next prediction must be a finite two-field vector")
        previous = self._pending.pop(flow_generation, None)
        if next_prediction is not None:
            self._pending[flow_generation] = next_prediction.copy()
        if previous is None:
            return None
        return forecast_residual_features(previous[None], observed[None])[0]

    @property
    def pending_flows(self) -> int:
        return len(self._pending)


def load_frozen_plan(path: Path) -> dict[str, Any]:
    """Load only the exact reviewed detection-bridge plan bytes."""
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != EXPECTED_PLAN_SHA256:
        raise ValueError("detection-bridge plan bytes differ from the frozen plan")
    plan = json.loads(payload)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("detection-bridge plan has an incompatible schema")
    if plan.get("architecture_changes_permitted") is not False:
        raise ValueError("detection bridge must keep the forecast models frozen")
    if plan.get("status") != "frozen_before_cic-ids2017_packet_access":
        raise ValueError("detection-bridge plan was not frozen before test access")
    models = plan.get("forecast_models", {})
    if models.get("seeds") != [20260906, 20260907, 20260908]:
        raise ValueError("detection bridge requires all three frozen seeds")
    if models.get("training_or_finetuning_permitted") is not False:
        raise ValueError("detection bridge cannot train the frozen forecasters")
    endpoints = plan.get("primary_endpoints", {})
    if endpoints.get("bootstrap", {}).get("resamples") != 10_000:
        raise ValueError("detection bridge requires the frozen bootstrap count")
    return plan


def _matrix(name: str, values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite [events, 2] matrix")
    return result


def forecast_residual_features(
    prediction: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    """Return seven residual fields available when the target packet arrives."""
    prediction = _matrix("prediction", prediction)
    observed = _matrix("observed", observed)
    if prediction.shape != observed.shape:
        raise ValueError("prediction and observed matrices must have the same shape")
    signed = observed - prediction
    absolute = np.abs(signed)
    smooth_l1 = np.where(absolute < 1.0, 0.5 * signed**2, absolute - 0.5)
    weighted = (smooth_l1[:, :1] + 4.0 * smooth_l1[:, 1:2]) / 5.0
    return np.concatenate((signed, absolute, smooth_l1, weighted), axis=1)


def causal_base_features(
    packet_features: np.ndarray,
    flow_packet_counts: np.ndarray,
    flow_age_seconds: np.ndarray,
) -> np.ndarray:
    """Build the frozen 14-field detector base at target-packet arrival."""
    packet_features = np.asarray(packet_features, dtype=np.float64)
    counts = np.asarray(flow_packet_counts)
    ages = np.asarray(flow_age_seconds, dtype=np.float64)
    events = len(packet_features)
    if (
        packet_features.shape != (events, 12)
        or not np.isfinite(packet_features).all()
        or counts.shape != (events,)
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts < 1)
        or ages.shape != (events,)
        or not np.isfinite(ages).all()
        or np.any(ages < 0)
    ):
        raise ValueError("causal base inputs violate the 12+count+age contract")
    return np.concatenate(
        (
            packet_features,
            np.log1p(counts.astype(np.float64))[:, None],
            np.log1p(ages)[:, None],
        ),
        axis=1,
    )


def _event_vectors(
    probabilities: np.ndarray,
    labels: np.ndarray,
    capture_ids: Sequence[str],
    flow_ids: Sequence[str],
    timestamps_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    timestamps_ns = np.asarray(timestamps_ns)
    count = len(probabilities)
    if count == 0 or probabilities.ndim != 1 or not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be a finite vector")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    if labels.shape != (count,) or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be a binary vector")
    if timestamps_ns.shape != (count,) or not np.issubdtype(
        timestamps_ns.dtype, np.integer
    ):
        raise ValueError("timestamps_ns must be an integer vector")
    if len(capture_ids) != count or len(flow_ids) != count:
        raise ValueError("capture and flow identifiers must align with events")
    prior: dict[str, int] = {}
    for capture_id, timestamp in zip(capture_ids, timestamps_ns, strict=True):
        if not capture_id:
            raise ValueError("capture identifiers must be non-empty")
        timestamp = int(timestamp)
        if timestamp < prior.get(capture_id, -1):
            raise ValueError("timestamps must be monotonic inside each capture")
        prior[capture_id] = timestamp
    if any(not flow_id for flow_id in flow_ids):
        raise ValueError("flow identifiers must be non-empty")
    return probabilities, labels.astype(np.int8), timestamps_ns.astype(np.int64)


def alert_indices(
    probabilities: np.ndarray,
    threshold: float,
    capture_ids: Sequence[str],
    flow_ids: Sequence[str],
    timestamps_ns: np.ndarray,
    *,
    refractory_ns: int,
) -> np.ndarray:
    """Collapse repeated above-threshold packets into per-flow alert episodes."""
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    if refractory_ns <= 0:
        raise ValueError("refractory_ns must be positive")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    timestamps_ns = np.asarray(timestamps_ns)
    if probabilities.ndim != 1 or timestamps_ns.shape != probabilities.shape:
        raise ValueError("probabilities and timestamps must be aligned vectors")
    if len(capture_ids) != len(probabilities) or len(flow_ids) != len(probabilities):
        raise ValueError("event identifiers must align with probabilities")
    last_alert: dict[tuple[str, str], int] = {}
    emitted: list[int] = []
    for index, (probability, capture, flow, timestamp) in enumerate(
        zip(probabilities, capture_ids, flow_ids, timestamps_ns, strict=True)
    ):
        if not np.isfinite(probability):
            raise ValueError("probabilities must be finite")
        if probability < threshold:
            continue
        key = (capture, flow)
        timestamp = int(timestamp)
        previous = last_alert.get(key)
        if previous is None or timestamp - previous >= refractory_ns:
            emitted.append(index)
            last_alert[key] = timestamp
    return np.asarray(emitted, dtype=np.int64)


def false_alerts_per_hour(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    capture_ids: Sequence[str],
    flow_ids: Sequence[str],
    timestamps_ns: np.ndarray,
    *,
    monitored_seconds: float,
    refractory_ns: int,
) -> tuple[float, int]:
    if not np.isfinite(monitored_seconds) or monitored_seconds <= 0:
        raise ValueError("monitored_seconds must be finite and positive")
    probabilities, labels, timestamps_ns = _event_vectors(
        probabilities, labels, capture_ids, flow_ids, timestamps_ns
    )
    alerts = alert_indices(
        probabilities,
        threshold,
        capture_ids,
        flow_ids,
        timestamps_ns,
        refractory_ns=refractory_ns,
    )
    false_alerts = int((labels[alerts] == 0).sum())
    return false_alerts / (monitored_seconds / 3600.0), false_alerts


def select_threshold_for_false_alert_budget(
    probabilities: np.ndarray,
    capture_ids: Sequence[str],
    flow_ids: Sequence[str],
    timestamps_ns: np.ndarray,
    *,
    monitored_seconds: float,
    maximum_false_alerts_per_hour: float,
    refractory_ns: int,
) -> float:
    """Choose the least-conservative empirical benign-only threshold in budget."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.zeros(len(probabilities), dtype=np.int8)
    if maximum_false_alerts_per_hour < 0 or not np.isfinite(
        maximum_false_alerts_per_hour
    ):
        raise ValueError("false-alert budget must be finite and non-negative")
    _event_vectors(probabilities, labels, capture_ids, flow_ids, timestamps_ns)
    candidates = np.unique(np.concatenate((probabilities, np.asarray([1.0]))))
    passing: list[float] = []
    for threshold in candidates:
        rate, _ = false_alerts_per_hour(
            probabilities,
            labels,
            float(threshold),
            capture_ids,
            flow_ids,
            timestamps_ns,
            monitored_seconds=monitored_seconds,
            refractory_ns=refractory_ns,
        )
        if rate <= maximum_false_alerts_per_hour:
            passing.append(float(threshold))
    if passing:
        return min(passing)
    # A score of exactly one can otherwise make a zero-alert budget impossible.
    return float(np.nextafter(1.0, np.inf))


def packets_to_alert(
    probabilities: np.ndarray,
    labels: np.ndarray,
    attack_episode_ids: Sequence[str | None],
    threshold: float,
) -> dict[str, object]:
    """Measure attack-packet count to first alert; missed episodes stay censored."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    if probabilities.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("probabilities and labels must be aligned vectors")
    if len(attack_episode_ids) != len(probabilities):
        raise ValueError("attack episodes must align with events")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary")
    episode_rows: dict[str, list[int]] = defaultdict(list)
    for index, (label, episode) in enumerate(
        zip(labels, attack_episode_ids, strict=True)
    ):
        if label == 1:
            if not episode:
                raise ValueError("every attack event needs an attack episode id")
            episode_rows[episode].append(index)
        elif episode is not None:
            raise ValueError("benign events cannot have an attack episode id")
    detected: dict[str, int] = {}
    missed: list[str] = []
    for episode, rows in sorted(episode_rows.items()):
        first = next(
            (
                offset
                for offset, row in enumerate(rows)
                if probabilities[row] >= threshold
            ),
            None,
        )
        if first is None:
            missed.append(episode)
        else:
            detected[episode] = first + 1
    values = np.asarray(list(detected.values()), dtype=np.float64)
    return {
        "episodes": len(episode_rows),
        "detected_episodes": len(detected),
        "missed_episodes": missed,
        "detection_rate": len(detected) / max(len(episode_rows), 1),
        "median_packets_to_alert_detected": (
            float(np.median(values)) if len(values) else None
        ),
        "p95_packets_to_alert_detected": (
            float(np.quantile(values, 0.95)) if len(values) else None
        ),
        "per_episode": detected,
    }


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, *, bins: int = 15
) -> float:
    if bins <= 1:
        raise ValueError("bins must exceed one")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if probabilities.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("probabilities and labels must be aligned vectors")
    total = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if selected.any():
            total += selected.mean() * abs(
                probabilities[selected].mean() - labels[selected].mean()
            )
    return float(total)


def bridge_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    capture_ids: Sequence[str],
    flow_ids: Sequence[str],
    timestamps_ns: np.ndarray,
    attack_episode_ids: Sequence[str | None],
    *,
    threshold: float,
    monitored_seconds: float,
    refractory_ns: int = 60_000_000_000,
) -> dict[str, object]:
    probabilities, labels, timestamps_ns = _event_vectors(
        probabilities, labels, capture_ids, flow_ids, timestamps_ns
    )
    if labels.min() == labels.max():
        raise ValueError("evaluation requires both benign and attack events")
    false_rate, false_count = false_alerts_per_hour(
        probabilities,
        labels,
        threshold,
        capture_ids,
        flow_ids,
        timestamps_ns,
        monitored_seconds=monitored_seconds,
        refractory_ns=refractory_ns,
    )
    predictions = probabilities >= threshold
    return {
        "events": len(labels),
        "attack_prevalence": float(labels.mean()),
        "attack_pr_auc": float(average_precision_score(labels, probabilities)),
        "attack_auroc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece_15_bin": expected_calibration_error(probabilities, labels),
        "threshold": float(threshold),
        "false_alerts": false_count,
        "false_alerts_per_hour": false_rate,
        "packet_recall_at_threshold": float(predictions[labels == 1].mean()),
        "packets_to_alert": packets_to_alert(
            probabilities, labels, attack_episode_ids, threshold
        ),
    }


def paired_hierarchical_pr_auc_bootstrap(
    base_probabilities: np.ndarray,
    ahem_probabilities: np.ndarray,
    lstm_probabilities: np.ndarray,
    labels: np.ndarray,
    capture_ids: Sequence[str],
    timestamps_ns: np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = 20260914,
    block_ns: int = 300_000_000_000,
) -> dict[str, object]:
    """Paired day-then-time-block bootstrap for the two superiority effects."""
    if resamples <= 0 or block_ns <= 0:
        raise ValueError("bootstrap counts and block width must be positive")
    vectors = [
        np.asarray(values, dtype=np.float64)
        for values in (base_probabilities, ahem_probabilities, lstm_probabilities)
    ]
    labels = np.asarray(labels)
    timestamps_ns = np.asarray(timestamps_ns)
    count = len(labels)
    if (
        any(values.shape != (count,) or not np.isfinite(values).all() for values in vectors)
        or timestamps_ns.shape != (count,)
        or len(capture_ids) != count
    ):
        raise ValueError("bootstrap event vectors must be finite and aligned")
    if count == 0 or not np.isin(labels, (0, 1)).all() or labels.min() == labels.max():
        raise ValueError("bootstrap requires both benign and attack events")
    captures = sorted(set(capture_ids))
    if len(captures) < 2:
        raise ValueError("hierarchical bootstrap requires at least two capture days")
    grouped: dict[str, list[np.ndarray]] = {}
    for capture in captures:
        rows = np.flatnonzero(np.asarray(capture_ids) == capture)
        if not len(rows):
            raise AssertionError("capture grouping lost an event")
        ordered = rows[np.argsort(timestamps_ns[rows], kind="stable")]
        first = int(timestamps_ns[ordered[0]])
        block_ids = (timestamps_ns[ordered] - first) // block_ns
        grouped[capture] = [ordered[block_ids == block] for block in np.unique(block_ids)]
    base, ahem, lstm = vectors

    def effects(rows: np.ndarray) -> tuple[float, float]:
        target = labels[rows]
        ahem_ap = average_precision_score(target, ahem[rows])
        return (
            float(ahem_ap - average_precision_score(target, base[rows])),
            float(ahem_ap - average_precision_score(target, lstm[rows])),
        )

    observed = effects(np.arange(count))
    rng = np.random.default_rng(seed)
    draws: list[tuple[float, float]] = []
    attempts = 0
    maximum_attempts = max(resamples * 20, 100)
    while len(draws) < resamples and attempts < maximum_attempts:
        attempts += 1
        sampled_rows: list[np.ndarray] = []
        for capture in rng.choice(captures, size=len(captures), replace=True):
            blocks = grouped[str(capture)]
            for block_index in rng.integers(0, len(blocks), size=len(blocks)):
                sampled_rows.append(blocks[int(block_index)])
        rows = np.concatenate(sampled_rows)
        if labels[rows].min() == labels[rows].max():
            continue
        draws.append(effects(rows))
    if len(draws) != resamples:
        raise RuntimeError("too many one-class bootstrap draws")
    distribution = np.asarray(draws, dtype=np.float64)
    # Two one-sided superiority tests share alpha=.05; Bonferroni therefore
    # uses the 2.5th percentile as each lower confidence bound.
    lower = np.quantile(distribution, 0.025, axis=0)
    return {
        "resamples": resamples,
        "attempts": attempts,
        "discarded_one_class_draws": attempts - resamples,
        "seed": seed,
        "block_ns": block_ns,
        "observed": {
            "ahem_minus_base_pr_auc": observed[0],
            "ahem_minus_lstm_pr_auc": observed[1],
        },
        "bonferroni_one_sided_95pct_lower": {
            "ahem_minus_base_pr_auc": float(lower[0]),
            "ahem_minus_lstm_pr_auc": float(lower[1]),
        },
        "both_supported": bool(
            observed[0] > 0
            and observed[1] > 0
            and lower[0] > 0
            and lower[1] > 0
        ),
    }

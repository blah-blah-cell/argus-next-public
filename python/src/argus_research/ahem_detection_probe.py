"""Frozen-capacity downstream probes for the AHEM detection bridge."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .ahem_detection_bridge import RESIDUAL_FEATURES


@dataclass(frozen=True)
class BridgeEvents:
    """Aligned causal evidence for all three frozen forecast seeds."""

    base: np.ndarray
    ahem_residuals: np.ndarray
    lstm_residuals: np.ndarray
    labels: np.ndarray

    def validate(self, *, require_both_classes: bool = True) -> None:
        base = np.asarray(self.base)
        ahem = np.asarray(self.ahem_residuals)
        lstm = np.asarray(self.lstm_residuals)
        labels = np.asarray(self.labels)
        events = len(labels)
        if events == 0 or labels.shape != (events,) or not np.isin(labels, (0, 1)).all():
            raise ValueError("bridge labels must be a non-empty binary vector")
        if require_both_classes and labels.min() == labels.max():
            raise ValueError("bridge probe fitting requires both classes")
        if base.ndim != 2 or base.shape[0] != events or base.shape[1] != 14:
            raise ValueError("bridge base evidence must be [events, 14]")
        expected = (events, 3, len(RESIDUAL_FEATURES))
        if ahem.shape != expected or lstm.shape != expected:
            raise ValueError("bridge residual evidence must be [events, 3, 7]")
        if not all(np.isfinite(values).all() for values in (base, ahem, lstm)):
            raise ValueError("bridge probe evidence must be finite")

    @property
    def event_count(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class Preprocessor:
    median: np.ndarray
    low: np.ndarray
    high: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.median):
            raise ValueError("probe matrix has the wrong feature width")
        values = np.where(np.isfinite(values), values, self.median)
        values = np.clip(values, self.low, self.high)
        return ((values - self.mean) / self.std).astype(np.float32)


@dataclass(frozen=True)
class FittedProbe:
    variant: str
    seed_index: int
    preprocessor: Preprocessor
    coefficient: np.ndarray
    intercept: float
    temperature: float

    def logits(self, values: np.ndarray) -> np.ndarray:
        transformed = self.preprocessor.transform(values)
        return transformed @ self.coefficient + self.intercept

    def probabilities(self, values: np.ndarray) -> np.ndarray:
        logits = self.logits(values) / self.temperature
        positive = logits >= 0
        output = np.empty_like(logits, dtype=np.float64)
        output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_values = np.exp(logits[~positive])
        output[~positive] = exp_values / (1.0 + exp_values)
        return output


VARIANTS = (
    "base",
    "base_plus_ahem",
    "base_plus_lstm",
    "ahem_residual_only",
    "lstm_residual_only",
)


def variant_matrix(events: BridgeEvents, variant: str, seed_index: int) -> np.ndarray:
    events.validate(require_both_classes=False)
    if variant not in VARIANTS:
        raise ValueError(f"unknown detection-bridge probe {variant!r}")
    if type(seed_index) is not int or not 0 <= seed_index < 3:
        raise ValueError("forecast seed index must be 0, 1, or 2")
    base = np.asarray(events.base, dtype=np.float64)
    ahem = np.asarray(events.ahem_residuals[:, seed_index], dtype=np.float64)
    lstm = np.asarray(events.lstm_residuals[:, seed_index], dtype=np.float64)
    if variant == "base":
        return base
    if variant == "base_plus_ahem":
        return np.concatenate((base, ahem), axis=1)
    if variant == "base_plus_lstm":
        return np.concatenate((base, lstm), axis=1)
    return ahem if variant == "ahem_residual_only" else lstm


def fit_preprocessor(values: np.ndarray) -> Preprocessor:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("probe fitting requires a non-empty matrix")
    finite = np.where(np.isfinite(values), values, np.nan)
    median = np.nanmedian(finite, axis=0)
    median[~np.isfinite(median)] = 0.0
    filled = np.where(np.isfinite(values), values, median)
    low = np.quantile(filled, 0.005, axis=0)
    high = np.quantile(filled, 0.995, axis=0)
    clipped = np.clip(filled, low, high)
    mean = clipped.mean(axis=0)
    std = clipped.std(axis=0)
    std[std < 1e-12] = 1.0
    return Preprocessor(median, low, high, mean, std)


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    values = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    targets = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            values / log_temperature.exp(), targets
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().detach().clamp(0.05, 20.0))


def fit_probe(
    train: BridgeEvents,
    calibration: BridgeEvents,
    variant: str,
    seed_index: int,
) -> FittedProbe:
    """Fit weights on training days and temperature on a separate day."""
    train.validate()
    calibration.validate()
    train_values = variant_matrix(train, variant, seed_index)
    calibration_values = variant_matrix(calibration, variant, seed_index)
    preprocessor = fit_preprocessor(train_values)
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=20260913,
    )
    classifier.fit(preprocessor.transform(train_values), train.labels)
    coefficient = classifier.coef_[0].astype(np.float64, copy=True)
    intercept = float(classifier.intercept_[0])
    calibration_logits = (
        preprocessor.transform(calibration_values) @ coefficient + intercept
    )
    temperature = _fit_temperature(calibration_logits, calibration.labels)
    return FittedProbe(
        variant=variant,
        seed_index=seed_index,
        preprocessor=preprocessor,
        coefficient=coefficient,
        intercept=intercept,
        temperature=temperature,
    )


def fit_probe_ensemble(
    train: BridgeEvents, calibration: BridgeEvents, variant: str
) -> tuple[FittedProbe, FittedProbe, FittedProbe]:
    return tuple(
        fit_probe(train, calibration, variant, seed_index)
        for seed_index in range(3)
    )


def ensemble_probabilities(
    probes: tuple[FittedProbe, FittedProbe, FittedProbe],
    events: BridgeEvents,
) -> np.ndarray:
    if len(probes) != 3 or [probe.seed_index for probe in probes] != [0, 1, 2]:
        raise ValueError("probe ensemble must contain the three ordered frozen seeds")
    if len({probe.variant for probe in probes}) != 1:
        raise ValueError("probe ensemble cannot mix evidence variants")
    predictions = [
        probe.probabilities(variant_matrix(events, probe.variant, probe.seed_index))
        for probe in probes
    ]
    return np.mean(predictions, axis=0)

"""Day-held-out calibrated CSE-CIC-IDS2018 completed-flow detector.

This detector is deliberately retrospective. CICFlowMeter features include
completed-flow aggregates and must never be presented as packet-time evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import Tensor, nn

from .real_flow_baseline import IDENTITY_OR_TIME, FlowMlp

CHUNK_ROWS = 100_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [column.strip() for column in frame.columns]
    return frame


def feature_names(path: Path) -> list[str]:
    header = _normalize_columns(pd.read_csv(path, nrows=0))
    label = next(
        (column for column in header.columns if column.lower() == "label"), None
    )
    if label is None:
        raise ValueError(f"{path} has no Label column")
    return [
        column
        for column in header.columns
        if column != label and column.lower() not in IDENTITY_OR_TIME
    ]


def _chunk_values(
    chunk: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunk = _normalize_columns(chunk)
    label_column = next(column for column in chunk.columns if column.lower() == "label")
    labels = chunk[label_column].astype(str).str.strip()
    valid = labels.str.lower() != "label"
    labels = labels[valid]
    numeric = chunk.loc[valid, features].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    target = (labels.str.lower() != "benign").to_numpy(dtype=np.float32)
    return values, target, labels.to_numpy(dtype=str)


def training_sample(
    paths: list[Path], features: list[str], stride: int
) -> tuple[np.ndarray, np.ndarray]:
    sampled_x: list[np.ndarray] = []
    sampled_y: list[np.ndarray] = []
    ordinal = 0
    for path in paths:
        for chunk in pd.read_csv(path, chunksize=CHUNK_ROWS, low_memory=False):
            values, target, _ = _chunk_values(chunk, features)
            select = (np.arange(len(values)) + ordinal) % stride == 0
            sampled_x.append(values[select])
            sampled_y.append(target[select])
            ordinal += len(values)
    return np.concatenate(sampled_x), np.concatenate(sampled_y)


def fit_preprocessor(values: np.ndarray, features: list[str]) -> dict[str, object]:
    values = np.array(values, dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    median = np.nanmedian(values, axis=0)
    median[~np.isfinite(median)] = 0.0
    filled = np.where(np.isnan(values), median, values)
    low = np.quantile(filled, 0.005, axis=0)
    high = np.quantile(filled, 0.995, axis=0)
    clipped = np.clip(filled, low, high)
    mean = clipped.mean(axis=0)
    std = clipped.std(axis=0)
    std[std < 1e-12] = 1.0
    return {
        "feature_names": features,
        "median": median.tolist(),
        "clip_low": low.tolist(),
        "clip_high": high.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "fit_scope": "training_days_deterministic_stride_sample_only",
    }


def transform(values: np.ndarray, preprocessor: dict[str, object]) -> np.ndarray:
    median = np.asarray(preprocessor["median"])
    values = np.array(values, dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    values = np.where(np.isnan(values), median, values)
    values = np.clip(values, preprocessor["clip_low"], preprocessor["clip_high"])
    return ((values - preprocessor["mean"]) / preprocessor["std"]).astype(np.float32)


def train_model(
    values: np.ndarray, target: np.ndarray, epochs: int, device: torch.device
) -> FlowMlp:
    x = torch.from_numpy(values).to(device)
    y = torch.from_numpy(target).to(device)
    model = FlowMlp(x.size(1)).to(device)
    positive_weight = (len(target) - float(target.sum())) / max(
        float(target.sum()), 1.0
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(20260810)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x), generator=generator, device=device)
        for indices in order.split(8192):
            loss = criterion(model(x[indices]), y[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite detector training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def predict_paths(
    model: FlowMlp,
    paths: list[Path],
    features: list[str],
    preprocessor: dict[str, object],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    observed_labels: set[str] = set()
    model.eval()
    with torch.inference_mode():
        for path in paths:
            for chunk in pd.read_csv(path, chunksize=CHUNK_ROWS, low_memory=False):
                values, target, labels = _chunk_values(chunk, features)
                batch = torch.from_numpy(transform(values, preprocessor)).to(device)
                logits.append(model(batch).cpu().numpy())
                targets.append(target)
                observed_labels.update(labels.tolist())
    return np.concatenate(logits), np.concatenate(targets), sorted(observed_labels)


def fit_temperature(logits: np.ndarray, target: np.ndarray) -> float:
    values = torch.from_numpy(logits).float()
    labels = torch.from_numpy(target).float()
    log_temperature = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            values / log_temperature.exp(), labels
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().detach().clamp(0.05, 20.0))


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    output = np.empty_like(values, dtype=np.float64)
    output[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1 + exp_values)
    return output


def expected_calibration_error(
    probabilities: np.ndarray, target: np.ndarray, bins: int = 15
) -> float:
    total = 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        upper = lower + 1 / bins
        selected = (probabilities >= lower) & (
            probabilities < upper if upper < 1 else probabilities <= 1
        )
        if selected.any():
            total += selected.mean() * abs(
                probabilities[selected].mean() - target[selected].mean()
            )
    return float(total)


def metrics(
    probabilities: np.ndarray, target: np.ndarray, threshold: float
) -> dict[str, object]:
    prediction = probabilities >= threshold
    benign = target == 0
    attack = target == 1
    tn = int((~prediction & benign).sum())
    fp = int((prediction & benign).sum())
    fn = int((~prediction & attack).sum())
    tp = int((prediction & attack).sum())
    return {
        "rows": len(target),
        "attack_prevalence": float(target.mean()),
        "attack_average_precision": float(
            average_precision_score(target, probabilities)
        ),
        "auroc": float(roc_auc_score(target, probabilities)),
        "brier": float(brier_score_loss(target, probabilities)),
        "ece_15_bin": expected_calibration_error(probabilities, target),
        "false_positive_rate": fp / max(fp + tn, 1),
        "recall": tp / max(tp + fn, 1),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def run(
    train_paths: list[Path],
    calibration_paths: list[Path],
    test_paths: list[Path],
    output: Path,
    epochs: int,
    sample_stride: int,
    target_fpr: float,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(20260810)
    features = feature_names(train_paths[0])
    if any(
        feature_names(path) != features
        for path in train_paths + calibration_paths + test_paths
    ):
        raise ValueError("dataset days do not share one exact ordered feature schema")
    sampled_x, sampled_y = training_sample(train_paths, features, sample_stride)
    preprocessor = fit_preprocessor(sampled_x, features)
    model = train_model(transform(sampled_x, preprocessor), sampled_y, epochs, device)
    calibration_logits, calibration_y, calibration_labels = predict_paths(
        model, calibration_paths, features, preprocessor, device
    )
    temperature = fit_temperature(calibration_logits, calibration_y)
    calibration_probabilities = sigmoid(calibration_logits / temperature)
    benign_probabilities = calibration_probabilities[calibration_y == 0]
    if not len(benign_probabilities):
        raise ValueError("calibration days contain no benign examples")
    threshold = float(
        np.quantile(benign_probabilities, 1 - target_fpr, method="higher")
    )
    test_logits, test_y, test_labels = predict_paths(
        model, test_paths, features, preprocessor, device
    )
    test_probabilities = sigmoid(test_logits / temperature)
    report = {
        "artifact_type": "calibrated_retrospective_flow_detector",
        "admitted_for_live_packet_detection": False,
        "reason_not_live": "CICFlowMeter completed-flow aggregates are unavailable at packet decision time",
        "dataset": "CSE-CIC-IDS2018 official processed flow CSV",
        "split_policy": "whole_day_disjoint",
        "train_files": [path.name for path in train_paths],
        "calibration_files": [path.name for path in calibration_paths],
        "test_files": [path.name for path in test_paths],
        "source_sha256": {
            path.name: _sha256(path)
            for path in train_paths + calibration_paths + test_paths
        },
        "train_sample_rows": len(sampled_y),
        "train_sample_attack_prevalence": float(sampled_y.mean()),
        "epochs": epochs,
        "device": str(device),
        "temperature": temperature,
        "target_calibration_fpr": target_fpr,
        "threshold": threshold,
        "preprocessor": preprocessor,
        "calibration_labels": calibration_labels,
        "test_labels": test_labels,
        "calibration_metrics": metrics(
            calibration_probabilities, calibration_y, threshold
        ),
        "test_metrics": metrics(test_probabilities, test_y, threshold),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "manifest": report}, output)
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.train,
                args.calibration,
                args.test,
                args.output,
                args.epochs,
                args.sample_stride,
                args.target_fpr,
                torch.device(args.device),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

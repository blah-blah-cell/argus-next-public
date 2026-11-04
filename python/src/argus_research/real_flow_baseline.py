"""Day-held-out, real-data retrospective flow baseline.

This runner is intentionally separate from AHEM packet-memory experiments:
CICFlowMeter rows include completed-flow statistics and therefore cannot justify
an online packet-time claim.  It is a real-data detection/open-family baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import Tensor, nn

IDENTITY_OR_TIME = {
    "timestamp",
    "flow id",
    "src ip",
    "dst ip",
    "source ip",
    "destination ip",
}


class FlowMlp(nn.Module):
    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values).squeeze(-1)


def _load(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [column.strip() for column in frame.columns]
    label_column = next(
        (column for column in frame.columns if column.lower() == "label"), None
    )
    if label_column is None:
        raise ValueError(f"{path} has no Label column")
    labels = frame.pop(label_column).astype(str).str.strip()
    # CICFlowMeter CSVs may contain an embedded repeated header row.
    valid_rows = labels.str.lower() != label_column.lower()
    frame = frame.loc[valid_rows].reset_index(drop=True)
    labels = labels.loc[valid_rows].reset_index(drop=True)
    target = (labels.str.lower() != "benign").to_numpy(dtype=np.float32)
    numeric = []
    for column in frame.columns:
        if column.strip().lower() in IDENTITY_OR_TIME:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().any():
            numeric.append(column)
            frame[column] = converted
    if not numeric:
        raise ValueError("no numeric non-identity features")
    return frame[numeric], target, sorted(labels.unique())


def _fit_transform(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    shared = [column for column in train.columns if column in test.columns]
    train_values = train[shared].replace([np.inf, -np.inf], np.nan)
    test_values = test[shared].replace([np.inf, -np.inf], np.nan)
    median = train_values.median(axis=0).fillna(0.0)
    train_values = train_values.fillna(median)
    test_values = test_values.fillna(median)
    low = train_values.quantile(0.005)
    high = train_values.quantile(0.995)
    train_values = train_values.clip(low, high, axis=1)
    test_values = test_values.clip(low, high, axis=1)
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0).replace(0.0, 1.0)
    metadata = {
        "feature_count": len(shared),
        "clip_quantiles": [0.005, 0.995],
        "fit_scope": "train_day_only",
    }
    return (
        ((train_values - mean) / std).to_numpy(dtype=np.float32),
        ((test_values - mean) / std).to_numpy(dtype=np.float32),
        metadata,
    )


def _predict(model: FlowMlp, values: Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in values.split(batch_size):
            chunks.append(torch.sigmoid(model(batch)).cpu())
    return torch.cat(chunks).numpy()


def run(
    train_path: Path, test_path: Path, output: Path, epochs: int, device: torch.device
) -> dict[str, object]:
    torch.manual_seed(20260809)
    train_frame, train_y, train_labels = _load(train_path)
    test_frame, test_y, test_labels = _load(test_path)
    train_x, test_x, prep = _fit_transform(train_frame, test_frame)
    x_train = torch.from_numpy(train_x).to(device)
    y_train = torch.from_numpy(train_y).to(device)
    x_test = torch.from_numpy(test_x).to(device)
    model = FlowMlp(x_train.size(1)).to(device)
    positive_weight = (len(train_y) - float(train_y.sum())) / max(
        float(train_y.sum()), 1.0
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batch_size = 8192
    for _ in range(epochs):
        model.train()
        order = torch.randperm(x_train.size(0), device=device)
        for indices in order.split(batch_size):
            logits = model(x_train[indices])
            loss = criterion(logits, y_train[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite real-data training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    probabilities = _predict(model, x_test, batch_size)
    threshold = 0.5
    prediction = probabilities >= threshold
    tn = int(np.logical_and(~prediction, test_y == 0).sum())
    fp = int(np.logical_and(prediction, test_y == 0).sum())
    fn = int(np.logical_and(~prediction, test_y == 1).sum())
    tp = int(np.logical_and(prediction, test_y == 1).sum())
    result = {
        "benchmark": "cse_cic_ids2018_day_held_out_flow_baseline",
        "decision_time": "retrospective_completed_flow",
        "not_packet_memory_efficacy": True,
        "train_source": str(train_path),
        "test_source": str(test_path),
        "train_labels": train_labels,
        "test_labels": test_labels,
        "train_rows": int(len(train_y)),
        "test_rows": int(len(test_y)),
        "epochs": epochs,
        "device": str(device),
        "feature_contract": prep,
        "threshold": threshold,
        "test_auroc": float(roc_auc_score(test_y, probabilities)),
        "test_attack_average_precision": float(
            average_precision_score(test_y, probabilities)
        ),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_count": x_train.size(1),
            "real_data": True,
        },
        output / "flow_mlp.pt",
    )
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    result = run(
        args.train, args.test, args.output, args.epochs, torch.device(args.device)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

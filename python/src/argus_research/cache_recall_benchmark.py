"""Real-trace keyed-cache recall benchmark versus an unkeyed long LSTM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import Tensor, nn


class KeyedCacheReader(nn.Module):
    """A learned AHEM cache reader; opaque flow IDs choose entries outside the model."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.query = nn.Sequential(nn.Linear(feature_dim, 48), nn.GELU())
        self.value = nn.Sequential(nn.Linear(feature_dim, 48), nn.GELU())
        self.classifier = nn.Sequential(nn.Linear(97, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, current: Tensor, cached: Tensor, age: Tensor) -> Tensor:
        return self.classifier(
            torch.cat((self.query(current), self.value(cached), age), dim=1)
        ).squeeze(1)


class GlobalLstmReader(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, 96, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, context: Tensor) -> Tensor:
        state, _ = self.lstm(context)
        return self.classifier(state[:, -1]).squeeze(1)


def _lstm_batch(
    streams: list[np.ndarray], source_ids: np.ndarray, indices: np.ndarray, context: int
) -> np.ndarray:
    values = []
    for source_id, index in zip(source_ids, indices, strict=True):
        stream = streams[int(source_id)]
        start = max(0, int(index) - context + 1)
        window = stream[start : int(index) + 1]
        if len(window) < context:
            window = np.pad(window, ((context - len(window), 0), (0, 0)))
        values.append(window)
    return np.stack(values)


def _train_cache(
    model: KeyedCacheReader,
    data: dict[str, np.ndarray],
    train: np.ndarray,
    device: torch.device,
) -> None:
    rng = np.random.default_rng(20260809)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    model.train()
    for _ in range(100):
        for _ in range(50):
            selected = rng.choice(train, size=128, replace=True)
            current = torch.from_numpy(data["current"][selected]).to(device)
            cached = torch.from_numpy(data["cached"][selected]).to(device)
            age = torch.from_numpy(data["age"][selected, None]).to(device)
            target = torch.from_numpy(data["label"][selected].astype(np.float32)).to(
                device
            )
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(current, cached, age), target
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def _train_lstm(
    model: GlobalLstmReader,
    data: dict[str, np.ndarray],
    streams: list[np.ndarray],
    train: np.ndarray,
    context: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    device: torch.device,
) -> None:
    rng = np.random.default_rng(20260809)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            selected = rng.choice(train, size=batch_size, replace=True)
            values = _lstm_batch(
                streams,
                data["source_id"][selected],
                data["query_index"][selected],
                context,
            )
            target = torch.from_numpy(data["label"][selected].astype(np.float32)).to(
                device
            )
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(torch.from_numpy(values).to(device)), target
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite LSTM cache-recall loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def _score_cache(
    model: KeyedCacheReader,
    data: dict[str, np.ndarray],
    test: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        logits = model(
            torch.from_numpy(data["current"][test]).to(device),
            torch.from_numpy(data["cached"][test]).to(device),
            torch.from_numpy(data["age"][test, None]).to(device),
        )
    return torch.sigmoid(logits).cpu().numpy()


def _score_lstm(
    model: GlobalLstmReader,
    data: dict[str, np.ndarray],
    streams: list[np.ndarray],
    test: np.ndarray,
    context: int,
    device: torch.device,
) -> np.ndarray:
    scores = []
    with torch.no_grad():
        for selected in np.array_split(test, max(1, len(test) // 2)):
            values = _lstm_batch(
                streams,
                data["source_id"][selected],
                data["query_index"][selected],
                context,
            )
            scores.append(
                torch.sigmoid(model(torch.from_numpy(values).to(device))).cpu().numpy()
            )
    return np.concatenate(scores)


def run(
    probe: Path,
    inputs: list[Path],
    output: Path,
    model_choice: str,
    lstm_epochs: int,
    lstm_steps_per_epoch: int,
    lstm_batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    archive = np.load(probe)
    data = {name: archive[name] for name in archive.files}
    streams = [np.load(path)["features"].astype(np.float32) for path in inputs]
    train = np.flatnonzero(~data["is_test"])
    test = np.flatnonzero(data["is_test"])
    result_models: dict[str, object] = {}
    target = data["label"][test]
    if model_choice in ("cache", "both"):
        cache = KeyedCacheReader(data["current"].shape[1]).to(device)
        _train_cache(cache, data, train, device)
        cache_score = _score_cache(cache, data, test, device)
        result_models["keyed_cache_reader"] = {
            "parameters": sum(p.numel() for p in cache.parameters()),
            "auroc": float(roc_auc_score(target, cache_score)),
            "accuracy": float(((cache_score >= 0.5) == target).mean()),
        }
    if model_choice in ("lstm", "both"):
        lstm = GlobalLstmReader(data["current"].shape[1]).to(device)
        _train_lstm(
            lstm,
            data,
            streams,
            train,
            30_000,
            lstm_epochs,
            lstm_steps_per_epoch,
            lstm_batch_size,
            device,
        )
        lstm_score = _score_lstm(lstm, data, streams, test, 30_000, device)
        result_models["global_lstm"] = {
            "parameters": sum(p.numel() for p in lstm.parameters()),
            "auroc": float(roc_auc_score(target, lstm_score)),
            "accuracy": float(((lstm_score >= 0.5) == target).mean()),
            "epochs": lstm_epochs,
            "steps_per_epoch": lstm_steps_per_epoch,
            "batch_size": lstm_batch_size,
        }
    result = {
        "benchmark": "real_trace_keyed_cache_recall_vs_global_lstm",
        "not_intrusion_detection_efficacy": True,
        "global_context_packets": 30_000,
        "keyed_cache_entries_per_flow": 256,
        "test_pairs": int(len(test)),
        "models": result_models,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("cache", "lstm", "both"), default="both")
    parser.add_argument("--lstm-epochs", type=int, default=100)
    parser.add_argument("--lstm-steps-per-epoch", type=int, default=10)
    parser.add_argument("--lstm-batch-size", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.probe,
                args.input,
                args.output,
                args.model,
                args.lstm_epochs,
                args.lstm_steps_per_epoch,
                args.lstm_batch_size,
                torch.device(args.device),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Real-PCAP self-supervised pretraining for the bounded AHEM encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .ahem_v2 import AhemV2NextPacket
from .memory_models import AhemFixedClassifier


class AhemNextPacket(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.encoder = AhemFixedClassifier(feature_dim)
        # Reuse the bounded encoder then predict the next causally unseen packet.
        self.encoder.head = nn.Identity()
        self.predict = nn.Sequential(
            nn.Linear(128, 96), nn.GELU(), nn.Linear(96, feature_dim)
        )

    def forward(self, context: Tensor) -> Tensor:
        return self.predict(self.encoder(context))


class LstmNextPacket(nn.Module):
    """Standard recurrent baseline under the exact same input/target protocol."""

    def __init__(self, feature_dim: int, output_dim: int | None = None) -> None:
        super().__init__()
        output_dim = feature_dim if output_dim is None else output_dim
        self.encoder = nn.LSTM(feature_dim, 96, batch_first=True)
        self.predict = nn.Sequential(
            nn.Linear(96, 96), nn.GELU(), nn.Linear(96, output_dim)
        )

    def forward(self, context: Tensor) -> Tensor:
        encoded, _ = self.encoder(context)
        return self.predict(encoded[:, -1])

    def forward_step(
        self,
        packet: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Process exactly one new packet while carrying recurrent state.

        Runtime comparisons must use this method. Re-running ``forward`` over
        the complete history at every arrival is not representative of a
        streaming LSTM and can create a false long-context speedup.
        """
        if packet.ndim == 2:
            packet = packet[:, None, :]
        if packet.ndim != 3 or packet.size(1) != 1:
            raise ValueError("an LSTM streaming step expects one packet per batch")
        encoded, next_state = self.encoder(packet, state)
        return self.predict(encoded[:, -1]), next_state


class GruNextPacket(nn.Module):
    """Parameter-matched recurrent baseline with a single gated state."""

    def __init__(
        self, feature_dim: int, hidden_dim: int = 104, output_dim: int | None = None
    ) -> None:
        super().__init__()
        output_dim = feature_dim if output_dim is None else output_dim
        self.encoder = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.predict = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, context: Tensor) -> Tensor:
        encoded, _ = self.encoder(context)
        return self.predict(encoded[:, -1])


class CausalCnnNextPacket(nn.Module):
    """Small local-context causal CNN specialist (31-packet receptive field)."""

    def __init__(
        self, feature_dim: int, channels: int = 96, output_dim: int | None = None
    ) -> None:
        super().__init__()
        output_dim = feature_dim if output_dim is None else output_dim
        self.input = nn.Conv1d(feature_dim, channels, 1)
        self.blocks = nn.ModuleList(nn.Conv1d(channels, channels, 3) for _ in range(4))
        self.predict = nn.Sequential(
            nn.Linear(channels, 96), nn.GELU(), nn.Linear(96, output_dim)
        )

    def forward(self, context: Tensor) -> Tensor:
        state = self.input(context.transpose(1, 2))
        for block in self.blocks:
            state = F.gelu(block(F.pad(state, (2, 0)))) + state
        return self.predict(state[:, :, -1])


class DilatedCnnNextPacket(nn.Module):
    """A 31-packet causal CNN matching the source-day training artifact."""

    def __init__(
        self, feature_dim: int, channels: int = 96, output_dim: int | None = None
    ) -> None:
        super().__init__()
        output_dim = feature_dim if output_dim is None else output_dim
        self.input = nn.Conv1d(feature_dim, channels, 1)
        self.dilations = (1, 2, 4, 8)
        self.blocks = nn.ModuleList(
            nn.Conv1d(channels, channels, 3, dilation=value) for value in self.dilations
        )
        self.predict = nn.Sequential(
            nn.Linear(channels, 96), nn.GELU(), nn.Linear(96, output_dim)
        )

    def forward(self, context: Tensor) -> Tensor:
        state = self.input(context.transpose(1, 2))
        for dilation, block in zip(self.dilations, self.blocks, strict=True):
            state = F.gelu(block(F.pad(state, (2 * dilation, 0)))) + state
        return self.predict(state[:, :, -1])


class LongTcnNextPacket(nn.Module):
    """Dilated causal CNN whose receptive field exceeds 30,000 packets."""

    def __init__(
        self, feature_dim: int, channels: int = 48, output_dim: int | None = None
    ) -> None:
        super().__init__()
        output_dim = feature_dim if output_dim is None else output_dim
        self.input = nn.Conv1d(feature_dim, channels, 1)
        self.dilations = tuple(2**index for index in range(15))
        self.blocks = nn.ModuleList(
            nn.Conv1d(channels, channels, 3, dilation=d) for d in self.dilations
        )
        self.predict = nn.Sequential(
            nn.Linear(channels, 96), nn.GELU(), nn.Linear(96, output_dim)
        )

    def forward(self, context: Tensor) -> Tensor:
        state = self.input(context.transpose(1, 2))
        for dilation, block in zip(self.dilations, self.blocks, strict=True):
            state = F.gelu(block(F.pad(state, (2 * dilation, 0)))) + state
        return self.predict(state[:, :, -1])


def _sample_batch(
    streams: list[np.ndarray],
    starts: list[tuple[int, int]],
    context: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    sample_x = []
    sample_y = []
    for _ in range(batch_size):
        source = int(rng.integers(len(streams)))
        stream = streams[source]
        low, high = starts[source]
        start = int(rng.integers(low, high))
        sample_x.append(stream[start : start + context])
        sample_y.append(stream[start + context])
    return np.stack(sample_x), np.stack(sample_y)


def _evaluate(
    model: nn.Module,
    streams: list[np.ndarray],
    starts: list[tuple[int, int]],
    context: int,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    rng = np.random.default_rng(20260810)
    losses = []
    with torch.no_grad():
        for _ in range(32):
            x, y = _sample_batch(streams, starts, context, batch_size, rng)
            loss = nn.functional.smooth_l1_loss(
                model(torch.from_numpy(x).to(device)), torch.from_numpy(y).to(device)
            )
            losses.append(float(loss.cpu()))
    return float(np.mean(losses))


def _train_one(
    name: str,
    model: nn.Module,
    streams: list[np.ndarray],
    train_starts: list[tuple[int, int]],
    test_starts: list[tuple[int, int]],
    context: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, object], nn.Module]:
    rng = np.random.default_rng(20260809)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    losses = []
    model.train()
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            x, y = _sample_batch(streams, train_starts, context, batch_size, rng)
            prediction = model(torch.from_numpy(x).to(device))
            loss = nn.functional.smooth_l1_loss(
                prediction, torch.from_numpy(y).to(device)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {name} self-supervised loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return (
        {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "train_loss_first": losses[0],
            "train_loss_last": losses[-1],
            "chronological_holdout_loss": _evaluate(
                model, streams, test_starts, context, batch_size, device
            ),
        },
        model,
    )


def run(
    inputs: list[Path],
    output: Path,
    epochs: int,
    steps_per_epoch: int,
    context: int,
    batch_size: int,
    model_name: str,
    device: torch.device,
) -> dict[str, object]:
    arrays = [np.load(path)["features"].astype(np.float32) for path in inputs]
    if len({array.shape[1] for array in arrays}) != 1:
        raise ValueError("all sources must share one causal feature schema")
    feature_dim = arrays[0].shape[1]
    if any(len(array) <= context * 3 for array in arrays):
        raise ValueError("a source is too short for the declared context")
    torch.manual_seed(20260809)
    splits = [int(len(array) * 0.8) for array in arrays]
    train_starts = [(0, split - context) for split in splits]
    # The target packet is in the future chronological suffix; the preceding
    # context remains observable evidence, never a future feature.
    test_starts = [
        (split - context, len(array) - context - 1)
        for array, split in zip(arrays, splits, strict=True)
    ]
    available_models = {
        "ahem_fixed": AhemNextPacket(feature_dim).to(device),
        "ahem_v2": AhemV2NextPacket(feature_dim).to(device),
        "lstm": LstmNextPacket(feature_dim).to(device),
        "cnn_local": CausalCnnNextPacket(feature_dim).to(device),
        "tcn_long": LongTcnNextPacket(feature_dim).to(device),
    }
    models = (
        available_models
        if model_name == "all"
        else {model_name: available_models[model_name]}
    )
    comparison = {}
    for name, model in models.items():
        metrics, trained = _train_one(
            name,
            model,
            arrays,
            train_starts,
            test_starts,
            context,
            epochs,
            steps_per_epoch,
            batch_size,
            device,
        )
        comparison[name] = metrics
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": trained.state_dict(),
                "feature_schema": "packet-causal-v1",
                "supervision": "next_packet_only",
            },
            output / f"{name}_self_supervised.pt",
        )
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "benchmark": "real_pcap_temporal_models_self_supervised",
        "supervision": "causal_next_packet_prediction_only",
        "not_intrusion_detection_efficacy": True,
        "feature_schema": "packet-causal-v1",
        "context_packets": context,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size,
        "device": str(device),
        "source_files": [str(path) for path in inputs],
        "comparison": comparison,
    }
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--model",
        choices=("ahem_fixed", "ahem_v2", "lstm", "cnn_local", "tcn_long", "all"),
        default="all",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.input,
                args.output,
                args.epochs,
                args.steps_per_epoch,
                args.context,
                args.batch_size,
                args.model,
                torch.device(args.device),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

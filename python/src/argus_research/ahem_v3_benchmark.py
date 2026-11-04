"""Strict real-journal AHEM-v3 versus LSTM feasibility benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor, nn

from .ahem_pretrain import LstmNextPacket
from .ahem_v3 import AhemV3NextPacket, AhemV3Output
from .continual_learning import iter_segments


@dataclass(frozen=True)
class Quality:
    smooth_l1: float
    log_length_mae: float
    log_interarrival_mae: float
    binary_feature_accuracy: float


def _eligible(streams: list[Tensor], context: int) -> list[Tensor]:
    return [stream for stream in streams if len(stream) > context]


def _batch(
    streams: list[Tensor], context: int, batch_size: int, rng: random.Random
) -> tuple[Tensor, Tensor]:
    choices = _eligible(streams, context)
    if not choices:
        raise ValueError(f"no whole segment exceeds the {context}-packet context")
    windows = []
    targets = []
    for _ in range(batch_size):
        stream = rng.choice(choices)
        target = rng.randrange(context, len(stream))
        windows.append(stream[target - context : target])
        targets.append(stream[target])
    return torch.stack(windows), torch.stack(targets)


def _quality(prediction: Tensor, target: Tensor) -> Quality:
    return Quality(
        smooth_l1=float(nn.functional.smooth_l1_loss(prediction, target)),
        log_length_mae=float((prediction[:, 0] - target[:, 0]).abs().mean()),
        log_interarrival_mae=float((prediction[:, 1] - target[:, 1]).abs().mean()),
        binary_feature_accuracy=float(
            ((prediction[:, 2:] >= 0.5) == (target[:, 2:] >= 0.5)).float().mean()
        ),
    )


def _evaluate(
    model: nn.Module,
    streams: list[Tensor],
    context: int,
    samples: int,
    seed: int,
) -> Quality:
    model.eval()
    rng = random.Random(seed)
    predictions = []
    targets = []
    with torch.inference_mode():
        for _ in range(math.ceil(samples / 4)):
            x, y = _batch(streams, context, min(4, samples), rng)
            predictions.append(model(x))
            targets.append(y)
    return _quality(torch.cat(predictions)[:samples], torch.cat(targets)[:samples])


def _ahem_loss(output: AhemV3Output, target: Tensor) -> Tensor:
    prediction_loss = nn.functional.smooth_l1_loss(output.prediction, target)
    local_per_sample = nn.functional.smooth_l1_loss(
        output.local_prediction, target, reduction="none"
    ).mean(dim=1)
    memory_per_sample = nn.functional.smooth_l1_loss(
        output.memory_candidate, target, reduction="none"
    ).mean(dim=1)
    utility_target = (memory_per_sample.detach() < local_per_sample.detach()).float()
    utility_loss = nn.functional.binary_cross_entropy(
        output.memory_gate.squeeze(1), utility_target
    )
    return prediction_loss + 0.10 * local_per_sample.mean() + 0.05 * utility_loss


def _train(
    model: nn.Module,
    streams: list[Tensor],
    validation: list[Tensor],
    context: int,
    epochs: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    ahem: bool,
) -> tuple[nn.Module, list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    rng = random.Random(20260814)
    best = deepcopy(model.state_dict())
    best_validation = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for _ in range(steps):
            x, y = _batch(streams, context, batch_size, rng)
            if ahem:
                output = model.forward_diagnostics(x)
                loss = _ahem_loss(output, y)
            else:
                loss = nn.functional.smooth_l1_loss(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite temporal training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_quality = _evaluate(
            model, validation, context, samples=32, seed=20260815
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": sum(losses) / len(losses),
                "validation_smooth_l1": validation_quality.smooth_l1,
            }
        )
        if validation_quality.smooth_l1 < best_validation:
            best_validation = validation_quality.smooth_l1
            best = deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model.eval(), history


def _latency(model: nn.Module, sample: Tensor, iterations: int) -> dict[str, float]:
    values = []
    model.eval()
    with torch.inference_mode():
        for _ in range(3):
            model(sample)
        for _ in range(iterations):
            started = time.perf_counter_ns()
            model(sample)
            values.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(values)

    def percentile(value: float) -> float:
        index = min(len(ordered) - 1, math.ceil(value * len(ordered)) - 1)
        return ordered[index]

    return {"p50_ms": percentile(0.50), "p95_ms": percentile(0.95)}


def _ahem_diagnostics(
    model: AhemV3NextPacket, streams: list[Tensor], context: int
) -> dict[str, float | int]:
    x, y = _batch(streams, context, 8, random.Random(20260817))
    with torch.inference_mode():
        output = model.forward_diagnostics(x)
    full = nn.functional.smooth_l1_loss(output.prediction, y)
    local = nn.functional.smooth_l1_loss(output.local_prediction, y)
    return {
        "full_smooth_l1": float(full),
        "local_only_smooth_l1": float(local),
        "memory_gain_fraction": float((local - full) / local.clamp_min(1e-8)),
        "mean_memory_gate": float(output.memory_gate.mean()),
        "retrieved_chunks": output.selected_chunks.size(1),
        "available_chunks": (context - model.local_packets) // model.chunk_size,
        "cached_bytes_per_flow": (
            (context - model.local_packets) // model.chunk_size * model.width * 4
        ),
    }


def _paired_quality_gain(
    ahem: nn.Module,
    lstm: nn.Module,
    streams: list[Tensor],
    context: int,
    samples: int = 128,
) -> dict[str, float]:
    rng = random.Random(20260819)
    differences = []
    ahem.eval()
    lstm.eval()
    with torch.inference_mode():
        for _ in range(math.ceil(samples / 4)):
            x, y = _batch(streams, context, min(4, samples), rng)
            ahem_loss = nn.functional.smooth_l1_loss(ahem(x), y, reduction="none").mean(
                dim=1
            )
            lstm_loss = nn.functional.smooth_l1_loss(lstm(x), y, reduction="none").mean(
                dim=1
            )
            differences.extend((lstm_loss - ahem_loss).tolist())
    differences = differences[:samples]
    bootstrap = random.Random(20260820)
    means = sorted(
        sum(bootstrap.choice(differences) for _ in differences) / len(differences)
        for _ in range(1_000)
    )
    return {
        "mean_lstm_minus_ahem_smooth_l1": sum(differences) / len(differences),
        "bootstrap_95pct_low": means[24],
        "bootstrap_95pct_high": means[974],
        "ahem_per_sample_win_rate": sum(value > 0 for value in differences)
        / len(differences),
    }


def run(
    journal: Path,
    output: Path,
    context: int,
    epochs: int,
    steps: int,
    batch_size: int,
    latency_iterations: int,
) -> dict[str, object]:
    torch.manual_seed(20260814)
    random.seed(20260814)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    records = list(iter_segments(journal))
    if len(records) < 12:
        raise ValueError("at least twelve real chronological segments are required")
    streams = [values for _, values in records]
    train_end = int(len(streams) * 0.70)
    validation_end = int(len(streams) * 0.85)
    train_streams = streams[:train_end]
    validation_streams = streams[train_end:validation_end]
    test_streams = streams[validation_end:]
    for name, split in (
        ("train", train_streams),
        ("validation", validation_streams),
        ("test", test_streams),
    ):
        if not _eligible(split, context):
            raise ValueError(f"{name} split has no segment longer than the context")
    constructors: dict[str, tuple[Callable[[], nn.Module], float, bool]] = {
        "ahem_v3": (lambda: AhemV3NextPacket(12), 7e-4, True),
        "lstm": (lambda: LstmNextPacket(12), 7e-4, False),
    }
    trained = {}
    histories = {}
    for name, (constructor, rate, is_ahem) in constructors.items():
        torch.manual_seed(20260814)
        model, history = _train(
            constructor(),
            train_streams,
            validation_streams,
            context,
            epochs,
            steps,
            batch_size,
            rate,
            is_ahem,
        )
        trained[name] = model
        histories[name] = history
    test_quality = {
        name: asdict(_evaluate(model, test_streams, context, samples=64, seed=20260816))
        for name, model in trained.items()
    }
    latency_sample, _ = _batch(test_streams, context, 1, random.Random(20260818))
    latency = {
        name: _latency(model, latency_sample, latency_iterations)
        for name, model in trained.items()
    }
    parameters = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in trained.items()
    }
    quality_checks = {
        "smooth_l1": test_quality["ahem_v3"]["smooth_l1"]
        < test_quality["lstm"]["smooth_l1"],
        "log_length_mae": test_quality["ahem_v3"]["log_length_mae"]
        < test_quality["lstm"]["log_length_mae"],
        "log_interarrival_mae": test_quality["ahem_v3"]["log_interarrival_mae"]
        < test_quality["lstm"]["log_interarrival_mae"],
        "binary_feature_accuracy": test_quality["ahem_v3"]["binary_feature_accuracy"]
        > test_quality["lstm"]["binary_feature_accuracy"],
    }
    paired = _paired_quality_gain(
        trained["ahem_v3"], trained["lstm"], test_streams, context
    )
    quality_win = all(quality_checks.values()) and paired["bootstrap_95pct_low"] > 0
    latency_win = latency["ahem_v3"]["p95_ms"] < latency["lstm"]["p95_ms"]
    result = {
        "benchmark": "ahem_v3_strict_real_journal_vs_lstm",
        "created_utc": datetime.now(UTC).isoformat(),
        "task": "causal_next_packet_forecast_only",
        "not_intrusion_detection_efficacy": True,
        "real_data": True,
        "split": "whole_segment_chronological_70_15_15",
        "context_packets": context,
        "train_segments": train_end,
        "validation_segments": validation_end - train_end,
        "test_segments": len(streams) - validation_end,
        "epochs": epochs,
        "steps_per_epoch": steps,
        "batch_size": batch_size,
        "parameters": parameters,
        "test_quality": test_quality,
        "quality_checks": quality_checks,
        "paired_quality": paired,
        "latency": latency,
        "ahem_diagnostics": _ahem_diagnostics(
            trained["ahem_v3"], test_streams, context
        ),
        "strict_pareto_win_over_lstm": quality_win and latency_win,
        "quality_win": quality_win,
        "latency_win": latency_win,
        "training_history": histories,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, model in trained.items():
        temporary = output / f"{name}.tmp"
        destination = output / f"{name}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "feature_schema": "packet-causal-v1",
                "supervision": "next_packet_only",
                "benchmark": "ahem_v3_strict_real_journal_vs_lstm",
            },
            temporary,
        )
        os.replace(temporary, destination)
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--latency-iterations", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.journal,
                args.output,
                args.context,
                args.epochs,
                args.steps,
                args.batch_size,
                args.latency_iterations,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

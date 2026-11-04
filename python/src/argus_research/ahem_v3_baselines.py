"""Extended parameter-matched real-journal baselines for frozen AHEM-v3."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

from .ahem_pretrain import GruNextPacket, LongTcnNextPacket, LstmNextPacket
from .ahem_v3 import AhemV3NextPacket
from .ahem_v3_benchmark import _batch, _evaluate, _latency, _train
from .continual_learning import iter_segments


def _load(model: nn.Module, path: Path) -> nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def run(
    journal: Path,
    frozen_directory: Path,
    output: Path,
    context: int,
    epochs: int,
    steps: int,
) -> dict[str, object]:
    torch.manual_seed(20260814)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    streams = [values for _, values in iter_segments(journal)]
    train_end = int(len(streams) * 0.70)
    validation_end = int(len(streams) * 0.85)
    train = streams[:train_end]
    validation = streams[train_end:validation_end]
    test = streams[validation_end:]
    models: dict[str, nn.Module] = {
        "ahem_v3": _load(AhemV3NextPacket(12), frozen_directory / "ahem_v3.pt"),
        "lstm": _load(LstmNextPacket(12), frozen_directory / "lstm.pt"),
    }
    histories = {}
    for name, model in (
        ("gru", GruNextPacket(12, 104)),
        ("tcn", LongTcnNextPacket(12, 32)),
    ):
        torch.manual_seed(20260814)
        trained, history = _train(
            model,
            train,
            validation,
            context,
            epochs,
            steps,
            2,
            7e-4,
            False,
        )
        models[name] = trained
        histories[name] = history
    quality = {
        name: asdict(_evaluate(model, test, context, 64, 20260816))
        for name, model in models.items()
    }
    sample, _ = _batch(test, context, 1, random.Random(20260823))
    latency = {name: _latency(model, sample, 20) for name, model in models.items()}
    parameters = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }
    ahem_loss = quality["ahem_v3"]["smooth_l1"]
    ahem_latency = latency["ahem_v3"]["p95_ms"]
    checks = {
        name: {
            "quality": ahem_loss < quality[name]["smooth_l1"],
            "latency": ahem_latency < latency[name]["p95_ms"],
            "parameters": parameters["ahem_v3"] <= parameters[name] * 1.05,
        }
        for name in ("lstm", "gru", "tcn")
    }
    result = {
        "benchmark": "ahem_v3_parameter_matched_extended_baselines",
        "created_utc": datetime.now(UTC).isoformat(),
        "real_data": True,
        "not_intrusion_detection_efficacy": True,
        "split": "whole_segment_chronological_70_15_15",
        "context_packets": context,
        "epochs": epochs,
        "steps_per_epoch": steps,
        "parameters": parameters,
        "quality": quality,
        "latency": latency,
        "ahem_wins": checks,
        "strict_all_baseline_win": all(
            all(values.values()) for values in checks.values()
        ),
        "training_history": histories,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name in ("gru", "tcn"):
        torch.save(
            {
                "state_dict": models[name].state_dict(),
                "feature_schema": "packet-causal-v1",
                "supervision": "next_packet_only",
            },
            output / f"{name}.pt",
        )
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.journal,
                args.frozen,
                args.output,
                args.context,
                args.epochs,
                args.steps,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

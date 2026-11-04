"""Controlled real-journal ablations for the frozen AHEM-v3 candidate."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor, nn

from .ahem_v3 import AhemV3NextPacket, AhemV3Output
from .ahem_v3_benchmark import _batch, _evaluate, _latency
from .continual_learning import iter_segments


class AblationModel(AhemV3NextPacket):
    """A named structural intervention with an appropriate training loss."""

    def __init__(self, feature_dim: int, variant: str) -> None:
        options = {
            "local_only": {"local_only": True},
            "flat_retrieval": {"hierarchical_retrieval": False},
            "no_age_prior": {"use_age_prior": False},
            "no_utility_gate": {"use_utility_gate": False},
            "adaptive_no_utility": {
                "use_utility_gate": False,
                "flat_retrieval_until_chunks": 512,
            },
        }
        if variant not in options:
            raise ValueError(f"unknown AHEM-v3 ablation: {variant}")
        super().__init__(feature_dim, **options[variant])
        self.variant = variant


def _ablation_loss(output: AhemV3Output, target: Tensor, variant: str) -> Tensor:
    prediction = nn.functional.smooth_l1_loss(output.prediction, target)
    if variant == "local_only":
        return prediction
    local_per_sample = nn.functional.smooth_l1_loss(
        output.local_prediction, target, reduction="none"
    ).mean(dim=1)
    if variant == "no_utility_gate":
        return prediction + 0.10 * local_per_sample.mean()
    memory_per_sample = nn.functional.smooth_l1_loss(
        output.memory_candidate, target, reduction="none"
    ).mean(dim=1)
    utility_target = (memory_per_sample.detach() < local_per_sample.detach()).float()
    utility = nn.functional.binary_cross_entropy(
        output.memory_gate.squeeze(1), utility_target
    )
    return prediction + 0.10 * local_per_sample.mean() + 0.05 * utility


def _train_ablation(
    model: AblationModel,
    train: list[Tensor],
    validation: list[Tensor],
    context: int,
    epochs: int,
    steps: int,
) -> tuple[nn.Module, list[dict[str, float]]]:
    # Keep the primary benchmark's batching, validation, clipping, and
    # checkpoint-selection semantics while swapping only the objective.
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    rng = random.Random(20260814)
    best = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_validation = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for _ in range(steps):
            x, y = _batch(train, context, 2, rng)
            loss = _ablation_loss(model.forward_diagnostics(x), y, model.variant)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite AHEM-v3 ablation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        quality = _evaluate(model, validation, context, 32, 20260815)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": sum(losses) / len(losses),
                "validation_smooth_l1": quality.smooth_l1,
            }
        )
        if quality.smooth_l1 < best_validation:
            best_validation = quality.smooth_l1
            best = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
    model.load_state_dict(best)
    return model.eval(), history


def _load_full(path: Path) -> AhemV3NextPacket:
    model = AhemV3NextPacket(12)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def run(
    journal: Path,
    frozen_full: Path,
    output: Path,
    context: int,
    epochs: int,
    steps: int,
    variants: tuple[str, ...] = (
        "local_only",
        "flat_retrieval",
        "no_age_prior",
        "no_utility_gate",
        "adaptive_no_utility",
    ),
) -> dict[str, object]:
    torch.manual_seed(20260814)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    streams = [values for _, values in iter_segments(journal)]
    train_end = int(len(streams) * 0.70)
    validation_end = int(len(streams) * 0.85)
    train = streams[:train_end]
    validation = streams[train_end:validation_end]
    test = streams[validation_end:]
    models: dict[str, nn.Module] = {"full": _load_full(frozen_full)}
    histories = {}
    for variant in variants:
        torch.manual_seed(20260814)
        model, history = _train_ablation(
            AblationModel(12, variant),
            train,
            validation,
            context,
            epochs,
            steps,
        )
        models[variant] = model
        histories[variant] = history
    quality = {
        name: asdict(_evaluate(model, test, context, 64, 20260816))
        for name, model in models.items()
    }
    sample, _ = _batch(test, context, 1, random.Random(20260824))
    latency = {name: _latency(model, sample, 20) for name, model in models.items()}
    full_loss = quality["full"]["smooth_l1"]
    result = {
        "benchmark": "ahem_v3_controlled_component_ablations",
        "created_utc": datetime.now(UTC).isoformat(),
        "real_data": True,
        "not_intrusion_detection_efficacy": True,
        "split": "whole_segment_chronological_70_15_15",
        "context_packets": context,
        "epochs": epochs,
        "steps_per_epoch": steps,
        "quality": quality,
        "latency": latency,
        "full_has_lower_loss": {
            name: full_loss < quality[name]["smooth_l1"]
            for name in models
            if name != "full"
        },
        "training_history": histories,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        if name == "full":
            continue
        torch.save(
            {
                "state_dict": model.state_dict(),
                "variant": name,
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
    parser.add_argument("--frozen-full", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--variant",
        action="append",
        choices=(
            "local_only",
            "flat_retrieval",
            "no_age_prior",
            "no_utility_gate",
            "adaptive_no_utility",
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.journal,
                args.frozen_full,
                args.output,
                args.context,
                args.epochs,
                args.steps,
                tuple(args.variant)
                if args.variant
                else (
                    "local_only",
                    "flat_retrieval",
                    "no_age_prior",
                    "no_utility_gate",
                    "adaptive_no_utility",
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

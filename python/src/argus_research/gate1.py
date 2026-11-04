"""Reproducible Gate-1 training: baselines first, router only on a held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn

from .memory_models import (
    AhemFixedClassifier,
    CausalRouter,
    CausalTcnClassifier,
    GruClassifier,
    causal_router_features,
)
from .synthetic_memory import (
    SyntheticMemoryConfig,
    make_mixed_distance_streams,
    make_streams,
)


def _train(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    model.train()
    for _ in range(epochs):
        order = torch.randperm(x.size(0), device=x.device)
        for indices in order.split(batch_size):
            logits = model(x[indices])
            if not torch.isfinite(logits).all():
                raise FloatingPointError(
                    f"{type(model).__name__} emitted non-finite logits"
                )
            loss = nn.functional.cross_entropy(logits, y[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{type(model).__name__} emitted non-finite loss"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


@torch.no_grad()
def _predict(model: nn.Module, x: Tensor, batch_size: int) -> Tensor:
    model.eval()
    logits = torch.cat([model(indices) for indices in x.split(batch_size)], dim=0)
    if not torch.isfinite(logits).all():
        raise FloatingPointError(
            f"{type(model).__name__} emitted non-finite evaluation logits"
        )
    return logits


@torch.no_grad()
def _accuracy(model: nn.Module, x: Tensor, y: Tensor, batch_size: int) -> float:
    logits = _predict(model, x, batch_size)
    return (logits.argmax(dim=1) == y).float().mean().item()


def run(
    output: Path,
    sequence_length: int,
    distance: int,
    device: torch.device,
    mixed_router_regimes: bool = False,
    epochs: int = 10,
    train_samples: int = 2048,
    evaluation_samples: int = 1024,
    batch_size: int = 64,
) -> dict[str, object]:
    torch.manual_seed(17)
    config = SyntheticMemoryConfig(
        sequence_length=sequence_length, evidence_distance=distance
    )
    # These are disjoint deterministic sets: experts / router target construction / final test.
    maker = make_streams
    if mixed_router_regimes:
        short_distance = 16

        def maker(count, _config, seed, target):
            return make_mixed_distance_streams(
                count, _config, (short_distance, distance), seed, target
            )

    train_x, train_y = maker(train_samples, config, 101, device)
    route_x, route_y = maker(evaluation_samples, config, 202, device)
    test_x, test_y = maker(evaluation_samples, config, 303, device)
    constructors: dict[str, Callable[[], nn.Module]] = {
        "gru": lambda: GruClassifier(config.feature_dim).to(device),
        "tcn": lambda: CausalTcnClassifier(config.feature_dim).to(device),
        "ahem_fixed": lambda: AhemFixedClassifier(config.feature_dim).to(device),
    }
    experts = {name: build() for name, build in constructors.items()}
    learning_rates = {"gru": 5e-4, "tcn": 2e-3, "ahem_fixed": 2e-3}
    for name, model in experts.items():
        _train(model, train_x, train_y, epochs, batch_size, learning_rates[name])
    standalone = {
        name: _accuracy(model, test_x, test_y, batch_size)
        for name, model in experts.items()
    }

    # Router target = lowest frozen expert CE loss on a split unseen by all experts.
    with torch.no_grad():
        losses = torch.stack(
            [
                nn.functional.cross_entropy(
                    _predict(model, route_x, batch_size), route_y, reduction="none"
                )
                for model in experts.values()
            ],
            dim=1,
        )
        # The declared utility price makes routing meaningful: an accurate
        # short-context expert is preferable to long-memory retrieval when its
        # loss is comparable.  It is fixed before this run, never fit on test.
        declared_cost = torch.tensor((0.03, 0.02, 0.06), device=device)
        route_target = (losses + declared_cost).argmin(dim=1)
    router = CausalRouter(config.feature_dim * 2, len(experts)).to(device)
    _train(
        router, causal_router_features(route_x), route_target, epochs, batch_size, 2e-3
    )
    with torch.no_grad():
        ordered = list(experts.values())
        choice = router(causal_router_features(test_x)).argmax(dim=1)
        all_logits = torch.stack(
            [_predict(model, test_x, batch_size) for model in ordered], dim=1
        )
        routed = all_logits[torch.arange(test_x.size(0), device=device), choice]
        router_accuracy = (routed.argmax(dim=1) == test_y).float().mean().item()
        oracle_choice = (
            nn.functional.cross_entropy(
                all_logits.flatten(0, 1),
                test_y.repeat_interleave(len(ordered)),
                reduction="none",
            )
            .reshape(test_x.size(0), len(ordered))
            .argmin(dim=1)
        )
        oracle_logits = all_logits[
            torch.arange(test_x.size(0), device=device), oracle_choice
        ]
        oracle_accuracy = (oracle_logits.argmax(dim=1) == test_y).float().mean().item()
    output.mkdir(parents=True, exist_ok=True)
    for name, model in experts.items():
        torch.save(
            {"state_dict": model.state_dict(), "synthetic_only": True},
            output / f"{name}.pt",
        )
    torch.save(
        {"state_dict": router.state_dict(), "synthetic_only": True},
        output / "router.pt",
    )
    target_counts = torch.bincount(route_target, minlength=len(experts)).tolist()
    return {
        "benchmark": "synthetic_long_range_memory_gate1",
        "not_nids_efficacy": True,
        "sequence_length": sequence_length,
        "evidence_distance": distance,
        "mixed_router_regimes": mixed_router_regimes,
        "device": str(device),
        "epochs": epochs,
        "train_samples": train_samples,
        "evaluation_samples": evaluation_samples,
        "batch_size": batch_size,
        "expert_learning_rates": learning_rates,
        "experts_train_seed": 101,
        "router_target_seed": 202,
        "final_test_seed": 303,
        "standalone_test_accuracy": standalone,
        "router_test_accuracy": router_accuracy,
        "oracle_per_example_expert_test_accuracy": oracle_accuracy,
        "router_inputs": "last_packet_plus_causal_prefix_mean; no_expert_scores",
        "router_target_counts": dict(zip(experts, target_counts, strict=True)),
        "router_utility_cost": dict(zip(experts, declared_cost.tolist(), strict=True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--distance", type=int, default=768)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--mixed-router-regimes", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--evaluation-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    result = run(
        args.output,
        args.sequence_length,
        args.distance,
        torch.device(args.device),
        args.mixed_router_regimes,
        args.epochs,
        args.train_samples,
        args.evaluation_samples,
        args.batch_size,
    )
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Train the matched compact-router control without opening the locked test."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .laya_router_vm_train import _load_manifest, _sha256
from .memory_models import CausalRouter
from .train_temporal_router import fit_confidence_threshold, fit_temperature


def _counterfactuals(path: Path, names: list[str]) -> tuple[Tensor, Tensor, Tensor]:
    features = []
    forecast_losses = []
    utilities = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
                row_features = [float(value) for value in row["compact_router_features"]]
                row_forecast = [float(row["forecast_loss"][name]) for name in names]
                row_utility = [float(row["utility"][name]) for name in names]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid counterfactual row {line_number}") from error
            if len(row_features) != 24:
                raise ValueError(f"compact router feature width differs at row {line_number}")
            features.append(row_features)
            forecast_losses.append(row_forecast)
            utilities.append(row_utility)
    if not features:
        raise ValueError("counterfactual dataset is empty")
    tensors = (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(forecast_losses, dtype=torch.float32),
        torch.tensor(utilities, dtype=torch.float32),
    )
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("counterfactual dataset contains non-finite values")
    return tensors


def _evaluate(
    router: CausalRouter,
    features: Tensor,
    forecast: Tensor,
    utility: Tensor,
    names: list[str],
    temperature: float,
    confidence: float,
    fallback: int,
) -> dict[str, Any]:
    with torch.inference_mode():
        probabilities = (router(features) / temperature).softmax(1)
        selected = probabilities.argmax(1)
        selected[probabilities.max(1).values < confidence] = fallback
        selected_forecast = forecast.gather(1, selected[:, None]).squeeze(1)
        selected_utility = utility.gather(1, selected[:, None]).squeeze(1)
        oracle = utility.min(1).values
        best_fixed = int(utility.mean(0).argmin())
    return {
        "samples": len(features),
        "mean_selected_forecast_loss": float(selected_forecast.mean()),
        "mean_selected_utility": float(selected_utility.mean()),
        "mean_oracle_utility": float(oracle.mean()),
        "mean_route_regret": float((selected_utility - oracle).mean()),
        "best_fixed_expert": names[best_fixed],
        "best_fixed_utility": float(utility[:, best_fixed].mean()),
        "router_gain_over_best_fixed": float(utility[:, best_fixed].mean() - selected_utility.mean()),
        "selection_rate": {
            name: float((selected == index).float().mean()) for index, name in enumerate(names)
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("baseline output already exists")
    manifest = _load_manifest(args.prepared)
    names = [expert["expert_id"] for expert in manifest["experts"]]
    for partition in ("train", "calibration"):
        name = f"{partition}.counterfactual.jsonl"
        path = args.prepared / name
        if not path.is_file() or _sha256(path) != manifest["files"].get(name):
            raise ValueError(f"prepared {name} SHA-256 differs")
    train = _counterfactuals(args.prepared / "train.counterfactual.jsonl", names)
    calibration = _counterfactuals(
        args.prepared / "calibration.counterfactual.jsonl", names
    )
    torch.manual_seed(args.seed)
    router = CausalRouter(24, len(names))
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    features, _, utility = train
    generator = torch.Generator().manual_seed(args.seed)
    for _ in range(args.epochs):
        order = torch.randperm(len(features), generator=generator)
        for indices in order.split(args.batch_size):
            logits = router(features[indices])
            probabilities = logits.softmax(1)
            expected_utility = (probabilities * utility[indices]).sum(1).mean()
            supervised = nn.functional.cross_entropy(logits, utility[indices].argmin(1))
            balance = (probabilities.mean(0) - 1 / len(names)).square().mean()
            loss = expected_utility + 0.02 * supervised + 0.01 * balance
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    router.eval()
    calibration_features, _, calibration_utility = calibration
    with torch.no_grad():
        calibration_logits = router(calibration_features).detach().clone()
    temperature = fit_temperature(calibration_logits, calibration_utility.argmin(1))
    fallback = int(train[2].mean(0).argmin())
    confidence = fit_confidence_threshold(
        router,
        calibration_features,
        calibration_utility,
        temperature,
        fallback,
    )
    metrics = {
        "train": _evaluate(router, *train, names, temperature, confidence, fallback),
        "calibration": _evaluate(
            router, *calibration, names, temperature, confidence, fallback
        ),
    }
    result = {
        "schema": "argus.laya-matched-compact-router/v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "security_detector": False,
        "locked_test_opened": False,
        "preparation_manifest_sha256": _sha256(args.prepared / "manifest.json"),
        "experts": names,
        "expert_sha256": {
            expert["expert_id"]: expert["checkpoint_sha256"]
            for expert in manifest["experts"]
        },
        "epochs": args.epochs,
        "seed": args.seed,
        "temperature": temperature,
        "confidence_threshold": confidence,
        "fallback_expert": names[fallback],
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": router.state_dict(), "manifest": result}, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) <= 0 or args.learning_rate <= 0:
        parser.error("training coordinates must be positive")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

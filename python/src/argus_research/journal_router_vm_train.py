"""Train a causal, compute-aware router from frozen real-journal experts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .ahem_profiles import get_profile
from .ahem_v3_benchmark import _batch
from .ahem_v32_benchmark import _weighted_loss
from .ahem_vm_train import _journal_manifest, _split_streams
from .memory_models import CausalRouter, causal_router_features
from .temporal_expert_vm_train import EXPERTS
from .train_temporal_router import fit_confidence_threshold, fit_temperature


@dataclass(frozen=True)
class FrozenExpert:
    name: str
    input_context: int
    checkpoint: Path
    sha256: str
    model: nn.Module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_experts(
    ahem_checkpoint: Path,
    expert_root: Path,
    journal_sha256: str,
    device: torch.device,
) -> list[FrozenExpert]:
    loaded: list[FrozenExpert] = []
    ahem_payload = torch.load(ahem_checkpoint, map_location="cpu", weights_only=True)
    profile = get_profile("flash")
    if ahem_payload.get("profile") != profile.manifest():
        raise ValueError("AHEM router expert is not the Flash profile")
    if ahem_payload.get("journal_sha256") != journal_sha256:
        raise ValueError("AHEM router expert used a different journal")
    ahem = profile.build()
    ahem.load_state_dict(ahem_payload["state_dict"], strict=True)
    loaded.append(
        FrozenExpert(
            "ahem_flash",
            30_000,
            ahem_checkpoint,
            _sha256(ahem_checkpoint),
            ahem.to(device).eval(),
        )
    )
    for name, spec in EXPERTS.items():
        checkpoint = expert_root / name / "best.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        checks = {
            "expert": name,
            "journal_sha256": journal_sha256,
            "feature_schema": "packet-causal-v1",
            "supervision": "next_packet_only",
            "security_detector": False,
        }
        for key, expected in checks.items():
            if payload.get(key) != expected:
                raise ValueError(f"{name} checkpoint {key} differs")
        model = spec.constructor()
        model.load_state_dict(payload["state_dict"], strict=True)
        loaded.append(
            FrozenExpert(
                name,
                spec.input_context,
                checkpoint,
                _sha256(checkpoint),
                model.to(device).eval(),
            )
        )
    return loaded


def _profile_latency(
    experts: list[FrozenExpert], sample: Tensor, device: torch.device
) -> dict[str, float]:
    values: dict[str, float] = {}
    with torch.inference_mode():
        for expert in experts:
            context = sample[:, -expert.input_context :].to(device)
            for _ in range(5):
                expert.model(context)
            torch.cuda.synchronize()
            timings = []
            for _ in range(20):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                expert.model(context)
                end.record()
                torch.cuda.synchronize()
                timings.append(start.elapsed_time(end))
            timings.sort()
            values[expert.name] = float(timings[len(timings) // 2])
    return values


def _collect_counterfactuals(
    experts: list[FrozenExpert],
    streams: list[Tensor],
    context: int,
    samples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    features: list[Tensor] = []
    losses: list[Tensor] = []
    rng = random.Random(seed)
    with torch.inference_mode():
        for _ in range(math.ceil(samples / batch_size)):
            packets, target = _batch(streams, context, batch_size, rng)
            features.append(causal_router_features(packets))
            target = target.to(device)
            expert_losses = []
            for expert in experts:
                prediction = expert.model(
                    packets[:, -expert.input_context :].to(device)
                )
                expert_losses.append(
                    _weighted_loss(prediction, target, reduction="none").float()
                )
            losses.append(torch.stack(expert_losses, dim=1).cpu())
    return torch.cat(features)[:samples], torch.cat(losses)[:samples]


def _route(
    router: CausalRouter,
    features: Tensor,
    temperature: float,
    confidence_threshold: float,
    fallback_expert: int,
) -> tuple[Tensor, Tensor]:
    with torch.inference_mode():
        probabilities = (router(features) / temperature).softmax(1)
        selected = probabilities.argmax(1)
        selected[probabilities.max(1).values < confidence_threshold] = fallback_expert
    return selected, probabilities


def _evaluate(
    router: CausalRouter,
    features: Tensor,
    forecast_losses: Tensor,
    utilities: Tensor,
    names: list[str],
    temperature: float,
    confidence_threshold: float,
    fallback_expert: int,
) -> dict[str, Any]:
    selected, probabilities = _route(
        router,
        features,
        temperature,
        confidence_threshold,
        fallback_expert,
    )
    selected_forecast = forecast_losses.gather(1, selected[:, None]).squeeze(1)
    selected_utility = utilities.gather(1, selected[:, None]).squeeze(1)
    oracle = utilities.argmin(1)
    best_fixed = int(utilities.mean(0).argmin())
    return {
        "samples": len(features),
        "cost_aware_route_accuracy": float((selected == oracle).float().mean()),
        "mean_selected_forecast_loss": float(selected_forecast.mean()),
        "mean_selected_utility": float(selected_utility.mean()),
        "mean_oracle_utility": float(utilities.min(1).values.mean()),
        "mean_route_regret": float((selected_utility - utilities.min(1).values).mean()),
        "best_fixed_expert": names[best_fixed],
        "best_fixed_utility": float(utilities[:, best_fixed].mean()),
        "router_gain_over_best_fixed": float(
            utilities[:, best_fixed].mean() - selected_utility.mean()
        ),
        "selection_rate": {
            name: float((selected == index).float().mean())
            for index, name in enumerate(names)
        },
        "mean_entropy": float(
            (-(probabilities * probabilities.clamp_min(1e-8).log()).sum(1)).mean()
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("journal router training requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    journal = _journal_manifest(args.journal)
    train, validation, test, split = _split_streams(
        args.journal,
        args.context,
        args.validation_source,
        args.test_source,
    )
    experts = _load_experts(
        args.ahem_checkpoint, args.expert_root, journal["sha256"], device
    )
    names = [expert.name for expert in experts]
    sample, _ = _batch(train, args.context, 1, random.Random(args.seed))
    latency_ms = _profile_latency(experts, sample, device)
    max_latency = max(latency_ms.values())
    normalized_cost = torch.tensor(
        [latency_ms[name] / max_latency for name in names], dtype=torch.float32
    )
    partitions = []
    for index, (streams, samples) in enumerate(
        (
            (train, args.train_samples),
            (validation, args.validation_samples),
            (test, args.test_samples),
        )
    ):
        features, forecast = _collect_counterfactuals(
            experts,
            streams,
            args.context,
            samples,
            args.counterfactual_batch_size,
            args.seed + index + 1,
            device,
        )
        utility = forecast + args.compute_weight * normalized_cost[None]
        partitions.append((features, forecast, utility))
    router = CausalRouter(24, len(experts)).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    train_features = partitions[0][0].to(device)
    train_utility = partitions[0][2].to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for _ in range(args.epochs):
        order = torch.randperm(len(train_features), generator=generator, device=device)
        for indices in order.split(args.router_batch_size):
            logits = router(train_features[indices])
            probabilities = logits.softmax(1)
            utility = (probabilities * train_utility[indices]).sum(1).mean()
            labels = train_utility[indices].argmin(1)
            supervised = nn.functional.cross_entropy(logits, labels)
            balance = (probabilities.mean(0) - 1 / len(experts)).square().mean()
            loss = utility + 0.02 * supervised + 0.01 * balance
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    router.cpu().eval()
    calibration_features, _, calibration_utility = partitions[1]
    with torch.inference_mode():
        calibration_logits = router(calibration_features)
    temperature = fit_temperature(calibration_logits, calibration_utility.argmin(1))
    fallback = int(partitions[0][2].mean(0).argmin())
    confidence = fit_confidence_threshold(
        router,
        calibration_features,
        calibration_utility,
        temperature,
        fallback,
    )
    metrics = {
        label: _evaluate(
            router,
            features,
            forecast,
            utility,
            names,
            temperature,
            confidence,
            fallback,
        )
        for label, (features, forecast, utility) in zip(
            ("train", "validation", "test"), partitions, strict=True
        )
    }
    manifest = {
        "schema": "argus.compute-aware-temporal-router/v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "artifact_type": "causal_temporal_forecaster_router",
        "security_detector": False,
        "real_data": True,
        "feature_schema": "packet-causal-v1",
        "router_inputs": "last_packet_plus_streaming_prefix_mean",
        "expert_outputs_are_router_inputs": False,
        "counterfactual_target": "forecast_loss_plus_normalized_latency_cost",
        "context_packets": args.context,
        "compute_weight": args.compute_weight,
        "profiled_isolated_median_latency_ms": latency_ms,
        "experts": names,
        "expert_sha256": {expert.name: expert.sha256 for expert in experts},
        "journal": journal,
        "split": split,
        "epochs": args.epochs,
        "temperature": temperature,
        "confidence_threshold": confidence,
        "fallback_expert": names[fallback],
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": router.state_dict(), "manifest": manifest}, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--ahem-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-source", required=True)
    parser.add_argument("--test-source", required=True)
    parser.add_argument("--context", type=int, default=30_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--counterfactual-batch-size", type=int, default=4)
    parser.add_argument("--router-batch-size", type=int, default=256)
    parser.add_argument("--compute-weight", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    positive = (
        args.context,
        args.epochs,
        args.train_samples,
        args.validation_samples,
        args.test_samples,
        args.counterfactual_batch_size,
        args.router_batch_size,
    )
    if min(positive) <= 0 or args.compute_weight < 0:
        parser.error("counts must be positive and compute weight non-negative")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

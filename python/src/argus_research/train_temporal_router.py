"""Train a causal compute router from frozen-expert counterfactual losses.

This is a forecaster router, not an intrusion detector. It uses only real
packet observations available before expert execution and never consumes a
security label or an expert score as an input feature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor, nn

from .ahem_pretrain import (
    AhemNextPacket,
    CausalCnnNextPacket,
    LongTcnNextPacket,
    LstmNextPacket,
)
from .experimental_cnn import adapt_runtime_rows
from .memory_models import CausalRouter, causal_router_features

EXPERTS = (
    ("ahem_fixed", "ahem_fixed_real_pcap_30k.pt", AhemNextPacket),
    ("cnn_local", "cnn_local_real_pcap_30k.pt", CausalCnnNextPacket),
    ("lstm", "lstm_real_pcap_30k.pt", LstmNextPacket),
    ("tcn_long", "tcn_long_real_pcap_30k.pt", LongTcnNextPacket),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_experts(
    model_dir: Path, device: torch.device
) -> tuple[list[nn.Module], dict[str, str]]:
    models: list[nn.Module] = []
    hashes: dict[str, str] = {}
    for name, file_name, model_type in EXPERTS:
        checkpoint = model_dir / file_name
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            payload.get("feature_schema") != "packet-causal-v1"
            or payload.get("supervision") != "next_packet_only"
        ):
            raise ValueError(f"{name} has an incompatible causal contract")
        model = model_type(12).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        models.append(model)
        hashes[name] = _sha256(checkpoint)
    return models, hashes


def load_partitions(feature_dir: Path) -> tuple[list[Path], list[Path], list[Path]]:
    paths = sorted(feature_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime_ns)
    if len(paths) < 10:
        raise ValueError(
            "at least ten chronological real-capture feature files are required"
        )
    train_end = max(1, int(len(paths) * 0.6))
    calibration_end = max(train_end + 1, int(len(paths) * 0.8))
    return paths[:train_end], paths[train_end:calibration_end], paths[calibration_end:]


def make_windows(paths: list[Path], context: int, cap: int) -> tuple[Tensor, Tensor]:
    contexts: list[Tensor] = []
    targets: list[Tensor] = []
    per_file = max(1, cap // len(paths))
    for path in paths:
        values, _ = adapt_runtime_rows(path)
        if len(values) <= context:
            continue
        available = len(values) - context
        stride = max(1, available // per_file)
        for index in range(context, len(values), stride):
            contexts.append(values[index - context : index])
            targets.append(values[index])
            if len(contexts) >= cap:
                break
        if len(contexts) >= cap:
            break
    if not contexts:
        raise ValueError(
            "partition has no sequence long enough for the declared context"
        )
    return torch.stack(contexts), torch.stack(targets)


def counterfactual_losses(
    models: list[nn.Module],
    contexts: Tensor,
    targets: Tensor,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    all_losses: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(contexts), batch_size):
            batch = contexts[start : start + batch_size].to(device)
            target = targets[start : start + batch_size].to(device)
            losses = [
                torch.nn.functional.smooth_l1_loss(
                    model(batch), target, reduction="none"
                ).mean(1)
                for model in models
            ]
            all_losses.append(torch.stack(losses, dim=1).cpu())
    return torch.cat(all_losses)


def fit_temperature(logits: Tensor, labels: Tensor) -> float:
    log_temperature = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().detach().clamp(0.05, 20.0))


def route_indices(
    router: CausalRouter,
    features: Tensor,
    temperature: float,
    confidence_threshold: float,
    fallback_expert: int,
) -> tuple[Tensor, Tensor]:
    router.eval()
    with torch.inference_mode():
        probabilities = (router(features) / temperature).softmax(1)
        selected = probabilities.argmax(1)
        selected[probabilities.max(1).values < confidence_threshold] = fallback_expert
    return selected, probabilities


def fit_confidence_threshold(
    router: CausalRouter,
    features: Tensor,
    losses: Tensor,
    temperature: float,
    fallback_expert: int,
) -> float:
    best = (float("inf"), 1.0)
    for step in range(101):
        threshold = step / 100
        selected, _ = route_indices(
            router, features, temperature, threshold, fallback_expert
        )
        mean_loss = float(losses.gather(1, selected[:, None]).mean())
        if mean_loss < best[0]:
            best = (mean_loss, threshold)
    return best[1]


def evaluate(
    router: CausalRouter,
    features: Tensor,
    losses: Tensor,
    temperature: float,
    confidence_threshold: float,
    fallback_expert: int,
) -> dict[str, object]:
    with torch.inference_mode():
        selected, probabilities = route_indices(
            router, features, temperature, confidence_threshold, fallback_expert
        )
        oracle = losses.argmin(1)
        selected_loss = losses.gather(1, selected[:, None]).squeeze(1)
        oracle_loss = losses.min(1).values
        fixed_losses = losses.mean(0)
        best_fixed_loss = fixed_losses.min()
    return {
        "samples": float(len(features)),
        "route_accuracy": float((selected == oracle).float().mean()),
        "mean_selected_forecast_loss": float(selected_loss.mean()),
        "mean_oracle_forecast_loss": float(oracle_loss.mean()),
        "mean_route_regret": float((selected_loss - oracle_loss).mean()),
        "fixed_expert_mean_losses": {
            name: float(fixed_losses[index])
            for index, (name, _, _) in enumerate(EXPERTS)
        },
        "best_fixed_expert_loss": float(best_fixed_loss),
        "router_gain_over_best_fixed": float(best_fixed_loss - selected_loss.mean()),
        "mean_entropy": float(
            (-(probabilities * probabilities.clamp_min(1e-8).log()).sum(1)).mean()
        ),
    }


def run(
    feature_dir: Path,
    model_dir: Path,
    output: Path,
    context: int,
    samples: int,
    epochs: int,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(20260810)
    models, expert_hashes = load_experts(model_dir, device)
    partitions = load_partitions(feature_dir)
    prepared = [make_windows(paths, context, samples) for paths in partitions]
    losses = [counterfactual_losses(models, x, y, device, 16) for x, y in prepared]
    router_features = [causal_router_features(x) for x, _ in prepared]
    router = CausalRouter(24, len(models))
    optimizer = torch.optim.AdamW(router.parameters(), lr=2e-3, weight_decay=1e-4)
    labels = losses[0].argmin(1)
    for _ in range(epochs):
        router.train()
        logits = router(router_features[0])
        probabilities = logits.softmax(1)
        utility = (probabilities * losses[0]).sum(1).mean()
        loss = utility + 0.02 * torch.nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    router.eval()
    with torch.no_grad():
        calibration_logits = router(router_features[1])
    temperature = fit_temperature(calibration_logits, losses[1].argmin(1))
    fallback_expert = int(losses[0].mean(0).argmin())
    confidence_threshold = fit_confidence_threshold(
        router,
        router_features[1],
        losses[1],
        temperature,
        fallback_expert,
    )
    metrics = {
        "train": evaluate(
            router,
            router_features[0],
            losses[0],
            temperature,
            confidence_threshold,
            fallback_expert,
        ),
        "calibration": evaluate(
            router,
            router_features[1],
            losses[1],
            temperature,
            confidence_threshold,
            fallback_expert,
        ),
        "test": evaluate(
            router,
            router_features[2],
            losses[2],
            temperature,
            confidence_threshold,
            fallback_expert,
        ),
    }
    manifest = {
        "artifact_type": "causal_temporal_forecaster_router",
        "security_detector": False,
        "real_data": True,
        "feature_schema": "packet-causal-v1",
        "router_inputs": "last_observed_packet_plus_causal_prefix_mean",
        "expert_outputs_are_router_inputs": False,
        "target": "minimum_frozen_expert_next_packet_loss",
        "context_packets": context,
        "temperature": temperature,
        "confidence_threshold": confidence_threshold,
        "fallback_expert": EXPERTS[fallback_expert][0],
        "experts": [name for name, _, _ in EXPERTS],
        "expert_sha256": expert_hashes,
        "split": {
            name: [path.name for path in paths]
            for name, paths in zip(
                ("train", "calibration", "test"), partitions, strict=True
            )
        },
        "metrics": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": router.state_dict(), "manifest": manifest}, output)
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.features,
                args.models,
                args.output,
                args.context,
                args.samples,
                args.epochs,
                torch.device(args.device),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""GPU training for causal temporal experts on source-day-held-out journals."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ahem_pretrain import (
    DilatedCnnNextPacket,
    GruNextPacket,
    LongTcnNextPacket,
    LstmNextPacket,
)
from .ahem_v3_benchmark import Quality, _eligible, _quality
from .ahem_v32_benchmark import _weighted_loss
from .ahem_vm_train import (
    _append_progress,
    _atomic_checkpoint,
    _cpu_state,
    _journal_manifest,
    _optimizer_to_device,
    _sha256,
    _split_streams,
)


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    input_context: int
    constructor: Callable[[], nn.Module]


EXPERTS = {
    "cnn": ExpertSpec("cnn", 31, lambda: DilatedCnnNextPacket(12)),
    "lstm": ExpertSpec("lstm", 30_000, lambda: LstmNextPacket(12)),
    "gru": ExpertSpec("gru", 30_000, lambda: GruNextPacket(12, 104)),
    "tcn": ExpertSpec("tcn", 30_000, lambda: LongTcnNextPacket(12, 48)),
}


def get_expert(name: str) -> ExpertSpec:
    try:
        return EXPERTS[name]
    except KeyError as error:
        raise ValueError(f"unknown temporal expert {name!r}") from error


def _model_input(packets: Tensor, input_context: int) -> Tensor:
    if packets.size(1) < input_context:
        raise ValueError("sampled context is shorter than the expert contract")
    return packets[:, -input_context:]


def _expert_batch(
    streams: list[Tensor],
    sampling_context: int,
    input_context: int,
    batch_size: int,
    rng: random.Random,
) -> tuple[Tensor, Tensor]:
    """Use common target positions without materializing unused old packets."""

    choices = _eligible(streams, sampling_context)
    if not choices:
        raise ValueError(
            f"no whole segment exceeds the {sampling_context}-packet context"
        )
    windows: list[Tensor] = []
    targets: list[Tensor] = []
    for _ in range(batch_size):
        stream = rng.choice(choices)
        target = rng.randrange(sampling_context, len(stream))
        windows.append(stream[target - input_context : target])
        targets.append(stream[target])
    return torch.stack(windows), torch.stack(targets)


def _evaluate_expert(
    model: nn.Module,
    streams: list[Tensor],
    sampling_context: int,
    input_context: int,
    samples: int,
    seed: int,
    device: torch.device,
) -> Quality:
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    rng = random.Random(seed)
    model.eval()
    with torch.inference_mode():
        for _ in range(math.ceil(samples / 4)):
            packets, target = _expert_batch(
                streams,
                sampling_context,
                input_context,
                min(4, samples),
                rng,
            )
            packets = packets.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            predictions.append(model(packets).float().cpu())
            targets.append(target.float().cpu())
    return _quality(torch.cat(predictions)[:samples], torch.cat(targets)[:samples])


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("temporal expert VM training requires CUDA")
    spec = get_expert(args.expert)
    if args.sampling_context < spec.input_context:
        raise ValueError("sampling context is shorter than the expert input")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    journal_manifest = _journal_manifest(args.journal)
    train, validation, test, split_manifest = _split_streams(
        args.journal,
        args.sampling_context,
        args.validation_source,
        args.test_source,
    )
    model = spec.constructor().to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    rng = random.Random(args.seed)
    best_validation = float("inf")
    best_state = deepcopy(_cpu_state(model))
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    resumed_from = None
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        if payload.get("expert") != spec.name:
            raise ValueError("resume checkpoint belongs to another expert")
        if payload.get("journal_sha256") != journal_manifest["sha256"]:
            raise ValueError("resume checkpoint journal fingerprint changed")
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(device)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _optimizer_to_device(optimizer, device)
        scaler.load_state_dict(payload.get("scaler_state_dict", {}))
        history = list(payload.get("history", []))
        best_validation = float(payload["best_validation_smooth_l1"])
        best_epoch = int(payload["best_epoch"])
        best_state = payload.get("best_state_dict", payload["state_dict"])
        stale_epochs = int(payload.get("stale_epochs", 0))
        start_epoch = int(payload["epoch"]) + 1
        if "python_rng_state" in payload:
            rng.setstate(payload["python_rng_state"])
        resumed_from = {
            "path": str(args.resume),
            "sha256": _sha256(args.resume),
            "epoch": start_epoch - 1,
        }
    run_manifest = {
        "schema": "argus.temporal-expert-run/v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "task": "causal_next_packet_forecast",
        "not_intrusion_detection_accuracy": True,
        "real_data_required": True,
        "expert": spec.name,
        "parameters": parameters,
        "sampling_context_packets": args.sampling_context,
        "input_context_packets": spec.input_context,
        "epochs_requested": args.epochs,
        "steps_per_epoch": args.steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "precision": "bfloat16" if use_bf16 else "float16",
        "device": torch.cuda.get_device_name(0),
        "journal": journal_manifest,
        "split": split_manifest,
        "resumed_from": resumed_from,
    }
    (args.output / "run.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    progress_path = args.output / "progress.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        for step in range(args.steps):
            packets, target = _expert_batch(
                train,
                args.sampling_context,
                spec.input_context,
                args.batch_size,
                rng,
            )
            packets = packets.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                prediction = model(packets)
                forecast = _weighted_loss(prediction, target)
                time_loss = F.smooth_l1_loss(prediction[:, 1], target[:, 1])
                loss = forecast + 0.5 * time_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite temporal expert loss")
            scaler.scale(loss / args.gradient_accumulation).backward()
            if (step + 1) % args.gradient_accumulation == 0 or step + 1 == args.steps:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
        validation_quality = _evaluate_expert(
            model,
            validation,
            args.sampling_context,
            spec.input_context,
            args.validation_samples,
            args.seed + 1,
            device,
        )
        record = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "validation": asdict(validation_quality),
            "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
            "created_utc": datetime.now(UTC).isoformat(),
        }
        torch.cuda.reset_peak_memory_stats()
        history.append(record)
        _append_progress(progress_path, record)
        improved = validation_quality.smooth_l1 < best_validation
        if improved:
            best_validation = validation_quality.smooth_l1
            best_epoch = epoch
            best_state = deepcopy(_cpu_state(model))
            stale_epochs = 0
        else:
            stale_epochs += 1
        checkpoint = {
            "schema": "argus.temporal-expert-checkpoint/v1",
            "expert": spec.name,
            "feature_schema": "packet-causal-v1",
            "supervision": "next_packet_only",
            "security_detector": False,
            "sampling_context_packets": args.sampling_context,
            "input_context_packets": spec.input_context,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_smooth_l1": best_validation,
            "journal_sha256": journal_manifest["sha256"],
            "state_dict": _cpu_state(model),
            "best_state_dict": best_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "stale_epochs": stale_epochs,
            "python_rng_state": rng.getstate(),
            "history": history,
        }
        _atomic_checkpoint(args.output / "latest.pt", checkpoint)
        if improved:
            _atomic_checkpoint(
                args.output / "best.pt", {**checkpoint, "state_dict": best_state}
            )
        if epoch % args.checkpoint_every == 0:
            _atomic_checkpoint(args.output / f"epoch-{epoch:04d}.pt", checkpoint)
        print(json.dumps(record), flush=True)
        if stale_epochs >= args.patience:
            break
    model.load_state_dict(best_state, strict=True)
    model.to(device).eval()
    test_quality = _evaluate_expert(
        model,
        test,
        args.sampling_context,
        spec.input_context,
        args.test_samples,
        args.seed + 2,
        device,
    )
    result = {
        **run_manifest,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "test_quality": asdict(test_quality),
        "history": history,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert", choices=tuple(EXPERTS), required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--sampling-context", type=int, default=30_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--validation-source")
    parser.add_argument("--test-source")
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    positive = (
        args.sampling_context,
        args.epochs,
        args.steps,
        args.batch_size,
        args.gradient_accumulation,
        args.patience,
        args.checkpoint_every,
    )
    if min(positive) <= 0:
        parser.error("context and training counts must be positive")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

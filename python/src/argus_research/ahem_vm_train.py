"""GPU-only AHEM tier training with real-journal provenance and checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .ahem_profiles import get_profile, warm_start_from_smaller
from .ahem_v3_benchmark import Quality, _batch, _quality
from .ahem_v32_benchmark import _retrieval_quality, _training_loss
from .continual_learning import iter_segments_with_source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _journal_manifest(directory: Path) -> dict[str, Any]:
    files = sorted(directory.glob("*.arguslr"))
    if not files:
        raise ValueError("journal directory contains no .arguslr files")
    members = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in files
    ]
    combined = hashlib.sha256(
        json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "argus.training-journal/v1",
        "members": members,
        "sha256": combined,
    }


def _split_streams(
    journal: Path,
    context: int,
    validation_source: str | None = None,
    test_source: str | None = None,
) -> tuple[list[Tensor], list[Tensor], list[Tensor], dict[str, Any]]:
    records = list(iter_segments_with_source(journal))
    if len(records) < 12:
        raise ValueError("at least twelve chronological real segments are required")
    if (validation_source is None) != (test_source is None):
        raise ValueError("validation_source and test_source must be specified together")
    if validation_source is not None and test_source is not None:
        if validation_source == test_source:
            raise ValueError("validation and test source files must differ")
        available = {source for source, _, _ in records}
        missing = {validation_source, test_source} - available
        if missing:
            raise ValueError(f"requested holdout sources are absent: {sorted(missing)}")
        train_records = [
            record
            for record in records
            if record[0] not in {validation_source, test_source}
        ]
        validation_records = [
            record for record in records if record[0] == validation_source
        ]
        test_records = [record for record in records if record[0] == test_source]
        split_records = (train_records, validation_records, test_records)
        method = "journal_source_file_disjoint"
    else:
        train_end = int(len(records) * 0.70)
        validation_end = int(len(records) * 0.85)
        split_records = (
            records[:train_end],
            records[train_end:validation_end],
            records[validation_end:],
        )
        method = "whole_segment_chronological_70_15_15"
    splits = tuple([values for _, _, values in split] for split in split_records)
    for name, values in zip(("train", "validation", "test"), splits, strict=True):
        if not any(len(stream) > context for stream in values):
            raise ValueError(f"{name} has no whole segment longer than {context}")
    manifest = {
        "method": method,
        "sources": {
            name: sorted({source for source, _, _ in records_for_split})
            for name, records_for_split in zip(
                ("train", "validation", "test"), split_records, strict=True
            )
        },
        "segment_ids_sha256": hashlib.sha256(
            "\n".join(
                f"{source}:{identifier}" for source, identifier, _ in records
            ).encode()
        ).hexdigest(),
        "segments": {
            "train": len(splits[0]),
            "validation": len(splits[1]),
            "test": len(splits[2]),
        },
        "packets": {
            "train": sum(map(len, splits[0])),
            "validation": sum(map(len, splits[1])),
            "test": sum(map(len, splits[2])),
        },
    }
    return splits[0], splits[1], splits[2], manifest


def _evaluate_device(
    model: nn.Module,
    streams: list[Tensor],
    context: int,
    samples: int,
    seed: int,
    device: torch.device,
) -> Quality:
    predictions = []
    targets = []
    rng = random.Random(seed)
    model.eval()
    with torch.inference_mode():
        for _ in range(math.ceil(samples / 4)):
            packets, target = _batch(streams, context, min(4, samples), rng)
            packets = packets.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            predictions.append(model(packets).float().cpu())
            targets.append(target.float().cpu())
    return _quality(torch.cat(predictions)[:samples], torch.cat(targets)[:samples])


def _cpu_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _append_progress(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, Tensor):
                state[name] = value.to(device)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cuda":
        raise ValueError("VM training is CUDA-only; CPU runs use the feasibility tools")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    profile = get_profile(args.profile)
    journal_manifest = _journal_manifest(args.journal)
    train, validation, test, split_manifest = _split_streams(
        args.journal,
        args.context,
        args.validation_source,
        args.test_source,
    )
    model = profile.build().to(device)
    transfer = None
    if args.warm_start is not None:
        transfer = warm_start_from_smaller(model, args.warm_start).manifest()
        model.to(device)
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
    args.output.mkdir(parents=True, exist_ok=True)
    progress_path = args.output / "progress.jsonl"
    resumed_from = None
    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        if payload.get("profile") != profile.manifest():
            raise ValueError(
                "resume checkpoint profile does not match the requested tier"
            )
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
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"])
        if "cuda_rng_state_all" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        resumed_from = {
            "path": str(args.resume),
            "sha256": _sha256(args.resume),
            "epoch": start_epoch - 1,
        }
    run_manifest = {
        "schema": "argus.ahem-vm-run/v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "task": "causal_next_packet_forecast_and_retrieval",
        "not_intrusion_detection_accuracy": True,
        "real_data_required": True,
        "profile": profile.manifest(),
        "parameters": parameters,
        "context_packets": args.context,
        "epochs_requested": args.epochs,
        "steps_per_epoch": args.steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "precision": "bfloat16" if use_bf16 else "float16",
        "device": torch.cuda.get_device_name(0),
        "journal": journal_manifest,
        "split": split_manifest,
        "warm_start": transfer,
        "resumed_from": resumed_from,
    }
    (args.output / "run.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        rankings = []
        for step in range(args.steps):
            packets, target = _batch(train, args.context, args.batch_size, rng)
            packets = packets.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                loss, components = _training_loss(model, packets, target)
                scaled_loss = loss / args.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite AHEM VM training loss")
            scaler.scale(scaled_loss).backward()
            if (step + 1) % args.gradient_accumulation == 0 or step + 1 == args.steps:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            rankings.append(components["ranking"])
        validation_quality = _evaluate_device(
            model,
            validation,
            args.context,
            args.validation_samples,
            args.seed + 1,
            device,
        )
        record = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "ranking_loss": sum(rankings) / len(rankings),
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
            "schema": "argus.ahem-training-checkpoint/v1",
            "profile": profile.manifest(),
            "feature_schema": "packet-causal-v1",
            "supervision": "next_packet_only",
            "security_detector": False,
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
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
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
    test_quality = _evaluate_device(
        model, test, args.context, args.test_samples, args.seed + 2, device
    )
    # Retrieval diagnostics operate on CPU to keep their oracle computation
    # independent of mixed precision and deployment GPU choice.
    model.cpu()
    retrieval = _retrieval_quality(
        model, test, args.context, args.retrieval_samples, args.seed + 3
    )
    result = {
        **run_manifest,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "test_quality": asdict(test_quality),
        "retrieval": retrieval,
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
    parser.add_argument("--profile", choices=("flash", "pro", "max"), required=True)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--context", type=int, default=30_000)
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
    parser.add_argument("--retrieval-samples", type=int, default=256)
    parser.add_argument("--validation-source")
    parser.add_argument("--test-source")
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    positive = (
        args.context,
        args.epochs,
        args.steps,
        args.batch_size,
        args.gradient_accumulation,
        args.patience,
        args.checkpoint_every,
    )
    if min(positive) <= 0:
        parser.error("context and training counts must be positive")
    if args.warm_start is not None and args.resume is not None:
        parser.error("choose either --warm-start or --resume")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

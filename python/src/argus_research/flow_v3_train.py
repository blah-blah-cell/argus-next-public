"""Fail-closed, one-model confirmatory runner for AHEM-v3.4."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .ahem_pretrain import GruNextPacket, LongTcnNextPacket, LstmNextPacket
from .ahem_v3 import (
    AhemV34FlowIndexedNextPacket,
    AhemV34LocalOnlyNextPacket,
    AhemV34RecencyNextPacket,
)
from .ahem_v4 import (
    AhemV41LocalOnlyNextPacket,
    AhemV41ProjectedStreamNextPacket,
    AhemV41RecencyNextPacket,
    AhemV4LocalOnlyNextPacket,
    AhemV4ProjectedStreamNextPacket,
    AhemV4RecencyNextPacket,
    AhemV5LocalOnlyNextPacket,
    AhemV5RecencyNextPacket,
    AhemV5SourceAlignedStreamNextPacket,
)
from .flow_v2_protocol import write_manifest
from .flow_v2_train import _canonical_sha256, _file_sha256, _source_provenance
from .flow_v3_protocol import (
    PREREGISTRATION_SHA256,
    PROTOCOL_SCHEMA,
    ProtocolConfig,
    compile_source_disjoint,
    materialize_batch,
)
from .flow_v3_smoke import _batches, _evaluate, _loss
from .s4d_baseline import S4DNextPacket

RUN_SCHEMA = "argus.flow-v3-model-run/v1"
CHECKPOINT_SCHEMA = "argus.flow-v3-model-checkpoint/v1"
TEST_ACCESS_SCHEMA = "argus.flow-v3-test-access/v1"
V4_RUN_SCHEMA = "argus.ahem-v5-model-run/v1"
V4_CHECKPOINT_SCHEMA = "argus.ahem-v5-model-checkpoint/v1"
V4_TEST_ACCESS_SCHEMA = "argus.ahem-v5-test-access/v1"
V4_PREREGISTRATION_SHA256 = (
    "b1a1de81fa1bd154a5e6863d7fcee71dcda239b347be2f979365949097b3a7c5"
)
V4_MODEL_NAMES = (
    "ahem_v5_source_aligned",
    "ahem_v5_local_only",
    "ahem_v5_recency",
    "lstm",
    "gru",
    "long_tcn",
    "s4d",
)
PRIMARY_VALIDATION_SOURCE = "Thursday-22-02-2018"
PRIMARY_TEST_SOURCE = "Friday-23-02-2018"
PRIMARY_TRAIN_SOURCES = {
    "Wednesday-14-02-2018",
    "Thursday-15-02-2018",
    "Friday-16-02-2018",
    "Tuesday-20-02-2018",
    "Wednesday-21-02-2018",
    "Wednesday-28-02-2018",
    "Thursday-01-03-2018",
    "Friday-02-03-2018",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    constructor: Callable[[], nn.Module]


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    patience: int = 8
    checkpoint_every: int = 5
    seed: int = 20260829

    def validate(self) -> None:
        if min(self.epochs, self.batch_size, self.patience, self.checkpoint_every) <= 0:
            raise ValueError("training counts must be positive")
        if (
            not math.isfinite(self.learning_rate)
            or not math.isfinite(self.weight_decay)
            or self.learning_rate <= 0
            or self.weight_decay < 0
            or self.seed < 0
        ):
            raise ValueError("training scalar configuration is invalid")


def model_spec(name: str) -> ModelSpec:
    specs = {
        "ahem_v3.4_flow_indexed": ModelSpec(
            "ahem_v3.4_flow_indexed",
            "bounded_exact_flow_transition_memory",
            AhemV34FlowIndexedNextPacket,
        ),
        "ahem_v3.4_local_only": ModelSpec(
            "ahem_v3.4_local_only",
            "exact_local_ablation",
            AhemV34LocalOnlyNextPacket,
        ),
        "ahem_v3.4_recency": ModelSpec(
            "ahem_v3.4_recency",
            "fixed_recency_retrieval_ablation",
            AhemV34RecencyNextPacket,
        ),
        "ahem_v4_projected_stream": ModelSpec(
            "ahem_v4_projected_stream",
            "online_projected_flow_transition_memory",
            AhemV4ProjectedStreamNextPacket,
        ),
        "ahem_v4_local_only": ModelSpec(
            "ahem_v4_local_only",
            "exact_v4_local_ablation",
            AhemV4LocalOnlyNextPacket,
        ),
        "ahem_v4_recency": ModelSpec(
            "ahem_v4_recency",
            "fixed_recency_v4_retrieval_ablation",
            AhemV4RecencyNextPacket,
        ),
        "ahem_v4.1_projected_stream": ModelSpec(
            "ahem_v4.1_projected_stream",
            "online_projected_flow_transition_memory_bounded_semantic_scale",
            AhemV41ProjectedStreamNextPacket,
        ),
        "ahem_v4.1_local_only": ModelSpec(
            "ahem_v4.1_local_only",
            "exact_v4.1_local_ablation",
            AhemV41LocalOnlyNextPacket,
        ),
        "ahem_v4.1_recency": ModelSpec(
            "ahem_v4.1_recency",
            "fixed_recency_v4.1_retrieval_ablation",
            AhemV41RecencyNextPacket,
        ),
        "ahem_v5_source_aligned": ModelSpec(
            "ahem_v5_source_aligned",
            "online_source_aligned_flow_transition_memory",
            AhemV5SourceAlignedStreamNextPacket,
        ),
        "ahem_v5_local_only": ModelSpec(
            "ahem_v5_local_only",
            "exact_v5_local_ablation",
            AhemV5LocalOnlyNextPacket,
        ),
        "ahem_v5_recency": ModelSpec(
            "ahem_v5_recency",
            "fixed_recency_v5_retrieval_ablation",
            AhemV5RecencyNextPacket,
        ),
        "lstm": ModelSpec(
            "lstm", "recurrent", lambda: LstmNextPacket(13, output_dim=2)
        ),
        "gru": ModelSpec("gru", "recurrent", lambda: GruNextPacket(13, output_dim=2)),
        "long_tcn": ModelSpec(
            "long_tcn",
            "dilated_convolution",
            lambda: LongTcnNextPacket(13, output_dim=2),
        ),
        "s4d": ModelSpec(
            "s4d", "diagonal_state_space", lambda: S4DNextPacket(13, output_dim=2)
        ),
    }
    try:
        return specs[name]
    except KeyError as error:
        raise ValueError(f"unknown flow-v3 model: {name}") from error


def _cpu_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _reserve_test_access(
    output: Path,
    *,
    protocol_sha256: str,
    coordinate_sha256: str,
    model: str,
    seed: int,
    best_epoch: int,
    access_schema: str = TEST_ACCESS_SCHEMA,
) -> dict[str, Any]:
    checkpoint = output / "best.pt"
    if not checkpoint.is_file():
        raise ValueError("test access requires a frozen best checkpoint")
    value = {
        "schema": access_schema,
        "status": "reserved",
        "attempt": 1,
        "reserved_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol_sha256,
        "coordinate_sha256": coordinate_sha256,
        "model": model,
        "training_seed": seed,
        "best_epoch": best_epoch,
        "best_checkpoint_sha256": _file_sha256(checkpoint),
    }
    ledger = output / "test-access.json"
    try:
        with ledger.open("x", encoding="utf-8") as destination:
            destination.write(json.dumps(value, indent=2) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
    except FileExistsError as error:
        raise ValueError("confirmatory test access was already reserved") from error
    return value


def _checkpoint_payload(
    *,
    protocol_sha256: str,
    coordinate_sha256: str,
    training_sha256: str,
    model_name: str,
    training_seed: int,
    study_role: str,
    publication_eligible: bool,
    source_provenance: dict[str, Any],
    epoch: int,
    best_epoch: int,
    best_validation: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_state: dict[str, Tensor],
    stale_epochs: int,
    history: list[dict[str, Any]],
    preregistration_sha256: str = PREREGISTRATION_SHA256,
    checkpoint_schema: str = CHECKPOINT_SCHEMA,
) -> dict[str, Any]:
    return {
        "schema": checkpoint_schema,
        "preregistration_sha256": preregistration_sha256,
        "protocol_sha256": protocol_sha256,
        "coordinate_sha256": coordinate_sha256,
        "training_config_sha256": training_sha256,
        "model": model_name,
        "training_seed": training_seed,
        "study_role": study_role,
        "publication_eligible": publication_eligible,
        "source_provenance": source_provenance,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_weighted_smooth_l1": best_validation,
        "state_dict": _cpu_state(model),
        "best_state_dict": best_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "stale_epochs": stale_epochs,
        "history": history,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = _source_provenance()
    diagnostic = bool(getattr(args, "allow_dirty_source", False))
    if not diagnostic and (
        source.get("available") is not True or source.get("git_dirty") is not False
    ):
        raise RuntimeError(
            "confirmatory training requires a clean Git revision; dirty-source "
            "runs are diagnostics only"
        )
    training = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )
    training.validate()
    study = getattr(args, "study", "v3")
    if study not in {"v3", "v5"}:
        raise ValueError("unknown flow study")
    if study == "v5" and args.model not in V4_MODEL_NAMES:
        raise ValueError("v5 study requested a model outside its frozen inventory")
    if (
        study == "v5"
        and not diagnostic
        and training != TrainingConfig(seed=training.seed)
    ):
        raise ValueError("confirmatory v5 optimization differs from the frozen budget")
    preregistration_sha256 = (
        V4_PREREGISTRATION_SHA256 if study == "v5" else PREREGISTRATION_SHA256
    )
    run_schema = V4_RUN_SCHEMA if study == "v5" else RUN_SCHEMA
    checkpoint_schema = V4_CHECKPOINT_SCHEMA if study == "v5" else CHECKPOINT_SCHEMA
    test_access_schema = V4_TEST_ACCESS_SCHEMA if study == "v5" else TEST_ACCESS_SCHEMA
    allowed_seeds = (
        {20260904, 20260905, 20260906, 20260907, 20260908}
        if study == "v5"
        else {20260829, 20260830, 20260831, 20260901, 20260902}
    )
    if not diagnostic and training.seed not in allowed_seeds:
        raise ValueError("confirmatory run uses a seed outside the preregistration")
    sampling_policy = getattr(args, "sampling_policy", "balanced")
    if sampling_policy != "balanced" and (
        not diagnostic
        or {args.validation_source, args.test_source}
        & {PRIMARY_VALIDATION_SOURCE, PRIMARY_TEST_SOURCE}
    ):
        raise ValueError("natural sampling is diagnostic-only, not preregistered")
    protocol_config = ProtocolConfig(
        context_packets=args.context,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        seed=args.protocol_seed,
        require_memory_match=True,
        sampling_policy=sampling_policy,
    )
    protocol = compile_source_disjoint(
        args.journal,
        args.validation_source,
        args.test_source,
        protocol_config,
    )
    protocol.manifest["preregistration_sha256"] = preregistration_sha256
    if protocol.manifest["schema"] != PROTOCOL_SCHEMA:
        raise ValueError("unexpected flow-v3 protocol schema")
    expected_split = {
        "method": "journal_source_file_disjoint",
        "validation_source": PRIMARY_VALIDATION_SOURCE,
        "test_source": PRIMARY_TEST_SOURCE,
        "train_sources": sorted(PRIMARY_TRAIN_SOURCES),
    }
    if not diagnostic and protocol.manifest["split"] != expected_split:
        raise RuntimeError("confirmatory run requires the preregistered source split")
    journal = protocol.manifest.get("journal", {})
    if not diagnostic and (
        journal.get("provenance_complete") is not True
        or journal.get("datasets") != ["CSE-CIC-IDS2018"]
    ):
        raise RuntimeError(
            "confirmatory run requires provenance-complete CSE-CIC-IDS2018"
        )
    expected_protocol_seed = 20260830 if study == "v5" else 20260828
    if not diagnostic and (
        args.context != 10_000
        or args.protocol_seed != expected_protocol_seed
        or args.train_samples != 2_048
        or args.validation_samples != 512
        or args.test_samples != 512
    ):
        raise ValueError("confirmatory run differs from frozen sampling dimensions")

    spec = model_spec(args.model)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(training.seed)
    random.seed(training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training.seed)
    model = spec.constructor().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    protocol_sha256 = _canonical_sha256(protocol.manifest)
    training_sha256 = _canonical_sha256(asdict(training))
    run_manifest = {
        "schema": run_schema,
        "created_utc": datetime.now(UTC).isoformat(),
        "study_role": "diagnostic" if diagnostic else "confirmatory",
        "publication_eligible": not diagnostic,
        "task": "next_dynamic_event_forecast_not_intrusion_detection",
        "preregistration_sha256": preregistration_sha256,
        "protocol_sha256": protocol_sha256,
        "coordinate_sha256": protocol.manifest["coordinate_sha256"],
        "protocol": protocol.manifest,
        "source_provenance": source,
        "model": {
            "name": spec.name,
            "family": spec.family,
            "parameters": sum(value.numel() for value in model.parameters()),
            "fresh_random_initialization": True,
        },
        "training": {
            **asdict(training),
            "config_sha256": training_sha256,
            "coordinate_seed": args.protocol_seed,
            "examples_per_epoch": len(protocol.coordinates["train"]),
            "optimizer_steps_per_epoch": math.ceil(
                len(protocol.coordinates["train"]) / training.batch_size
            ),
            "device": str(device),
            "precision": "float32",
            "torch": torch.__version__,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    best_epoch = 0
    best_state = deepcopy(_cpu_state(model))
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1

    if (args.output / "test-access.json").exists():
        raise ValueError("output already contains a test-access ledger")
    if args.resume is None:
        occupied = [
            name
            for name in ("run.json", "latest.pt", "best.pt", "metrics.json")
            if (args.output / name).exists()
        ]
        if occupied:
            raise ValueError(
                "new run refuses occupied output; resume explicitly: "
                + ", ".join(occupied)
            )
        write_manifest(args.output / "run.json", run_manifest)
    else:
        original = json.loads((args.output / "run.json").read_text(encoding="utf-8"))
        for field in (
            "schema",
            "preregistration_sha256",
            "protocol_sha256",
            "coordinate_sha256",
            "source_provenance",
        ):
            if original.get(field) != run_manifest.get(field):
                raise ValueError(f"resume changed immutable run field: {field}")
        if original.get("model") != run_manifest.get("model") or original.get(
            "training"
        ) != run_manifest.get("training"):
            raise ValueError("resume changed model or training configuration")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        required = {
            "schema": checkpoint_schema,
            "preregistration_sha256": preregistration_sha256,
            "protocol_sha256": protocol_sha256,
            "coordinate_sha256": protocol.manifest["coordinate_sha256"],
            "training_config_sha256": training_sha256,
            "model": spec.name,
            "training_seed": training.seed,
            "study_role": run_manifest["study_role"],
            "publication_eligible": run_manifest["publication_eligible"],
            "source_provenance": source,
        }
        if any(checkpoint.get(key) != value for key, value in required.items()):
            raise ValueError("resume checkpoint fingerprint changed")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint["history"])
        best_validation = float(checkpoint["best_validation_weighted_smooth_l1"])
        best_epoch = int(checkpoint["best_epoch"])
        best_state = checkpoint["best_state_dict"]
        stale_epochs = int(checkpoint["stale_epochs"])
        start_epoch = int(checkpoint["epoch"]) + 1
        run_manifest = original

    for epoch in range(start_epoch, training.epochs + 1):
        if stale_epochs >= training.patience:
            break
        model.train()
        epoch_losses = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for batch in _batches(
            protocol.coordinates["train"],
            training.batch_size,
            training.seed + epoch * 1_000_003,
        ):
            contexts, targets = materialize_batch(
                protocol.partitions["train"], batch, args.context
            )
            contexts = contexts.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, contexts, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite flow-v3 training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        validation = _evaluate(
            model,
            protocol,
            "validation",
            args.context,
            training.batch_size,
            device,
        )
        value = float(validation["weighted_smooth_l1"])
        improved = value < best_validation
        if improved:
            best_validation = value
            best_epoch = epoch
            best_state = deepcopy(_cpu_state(model))
            stale_epochs = 0
        else:
            stale_epochs += 1
        record = {
            "epoch": epoch,
            "train_loss": sum(epoch_losses) / len(epoch_losses),
            "validation_weighted_smooth_l1": value,
            "improved": improved,
            "gpu_peak_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "created_utc": datetime.now(UTC).isoformat(),
        }
        history.append(record)
        checkpoint = _checkpoint_payload(
            protocol_sha256=protocol_sha256,
            coordinate_sha256=protocol.manifest["coordinate_sha256"],
            training_sha256=training_sha256,
            model_name=spec.name,
            training_seed=training.seed,
            study_role=run_manifest["study_role"],
            publication_eligible=run_manifest["publication_eligible"],
            source_provenance=source,
            epoch=epoch,
            best_epoch=best_epoch,
            best_validation=best_validation,
            model=model,
            optimizer=optimizer,
            best_state=best_state,
            stale_epochs=stale_epochs,
            history=history,
            preregistration_sha256=preregistration_sha256,
            checkpoint_schema=checkpoint_schema,
        )
        _atomic_checkpoint(args.output / "latest.pt", checkpoint)
        if improved:
            _atomic_checkpoint(
                args.output / "best.pt", {**checkpoint, "state_dict": best_state}
            )
        if epoch % training.checkpoint_every == 0:
            _atomic_checkpoint(args.output / f"epoch-{epoch:04d}.pt", checkpoint)
        print(json.dumps(record), flush=True)
        if stale_epochs >= training.patience:
            break

    ledger = _reserve_test_access(
        args.output,
        protocol_sha256=protocol_sha256,
        coordinate_sha256=protocol.manifest["coordinate_sha256"],
        model=spec.name,
        seed=training.seed,
        best_epoch=best_epoch,
        access_schema=test_access_schema,
    )
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    test = _evaluate(model, protocol, "test", args.context, training.batch_size, device)
    per_sample_sha256 = _canonical_sha256(test["per_sample_weighted_smooth_l1"])
    persistence_per_sample_sha256 = _canonical_sha256(
        test["per_sample_persistence_weighted_smooth_l1"]
    )
    coordinate_payload_sha256 = _canonical_sha256(test["sample_coordinates"])
    ledger = {
        **ledger,
        "status": "completed",
        "completed_utc": datetime.now(UTC).isoformat(),
        "test_samples": test["samples"],
        "per_sample_sha256": per_sample_sha256,
        "persistence_per_sample_sha256": persistence_per_sample_sha256,
        "sample_coordinates_sha256": coordinate_payload_sha256,
    }
    write_manifest(args.output / "test-access.json", ledger)
    run_manifest = {**run_manifest, "test_access": ledger}
    write_manifest(args.output / "run.json", run_manifest)
    result = {
        **run_manifest,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_weighted_smooth_l1": best_validation,
        "test": test,
        "history": history,
        "runtime_benchmark_required": True,
    }
    write_manifest(args.output / "metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=(
            "ahem_v3.4_flow_indexed",
            "ahem_v3.4_local_only",
            "ahem_v3.4_recency",
            "lstm",
            "gru",
            "long_tcn",
            "s4d",
        ),
        required=True,
    )
    parser.add_argument("--validation-source", default=PRIMARY_VALIDATION_SOURCE)
    parser.add_argument("--test-source", default=PRIMARY_TEST_SOURCE)
    parser.add_argument("--context", type=int, default=10_000)
    parser.add_argument("--train-samples", type=int, default=2_048)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--protocol-seed", type=int, default=20260828)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

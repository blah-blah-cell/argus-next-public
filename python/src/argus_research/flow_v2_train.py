"""Train one model under the frozen, source-disjoint ARGUSLR2 protocol.

The runner deliberately launches one model per process.  Every model receives
the exact same target coordinates and number of training examples, while its
architecture-specific input horizon and auxiliary losses remain explicit in
the run manifest.  Test coordinates are consumed once after model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ahem_pretrain import (
    CausalCnnNextPacket,
    GruNextPacket,
    LongTcnNextPacket,
    LstmNextPacket,
)
from .ahem_v3 import (
    AhemV32LocalOnlyNextPacket,
    AhemV32NextPacket,
    AhemV32RecencyNextPacket,
)
from .ahem_v3_benchmark import _quality
from .ahem_v32_benchmark import _training_loss, _weighted_loss
from .flow_v2_protocol import (
    FEATURE_SCHEMA,
    PRIMARY_TEST_SOURCE,
    PRIMARY_TRAIN_SOURCES,
    PRIMARY_VALIDATION_SOURCE,
    PROTOCOL_SCHEMA,
    CompiledProtocol,
    ProtocolConfig,
    WindowCoordinate,
    compile_protocol,
    materialize_batch,
    write_manifest,
)
from .s4d_baseline import S4DNextPacket

RUN_SCHEMA = "argus.flow-v2-model-run/v1"
CHECKPOINT_SCHEMA = "argus.flow-v2-model-checkpoint/v1"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    input_context: int
    constructor: Callable[[], nn.Module]
    auxiliary_objective: str


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    patience: int = 8
    checkpoint_every: int = 5
    seed: int = 20260822

    def validate(self) -> None:
        counts = (
            self.epochs,
            self.batch_size,
            self.patience,
            self.checkpoint_every,
        )
        if min(counts) <= 0:
            raise ValueError("training counts must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight decay must be non-negative")
        if self.seed < 0:
            raise ValueError("training seed must be non-negative")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_provenance() -> dict[str, Any]:
    """Record the exact Git state without making Git a runtime dependency."""
    repository = Path(__file__).resolve().parents[3]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"git_revision": revision, "git_dirty": dirty, "available": True}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"git_revision": None, "git_dirty": None, "available": False}


def model_spec(name: str, context_packets: int) -> ModelSpec:
    specs = {
        "ahem_v3.2": ModelSpec(
            "ahem_v3.2",
            "bounded_transition_memory",
            context_packets,
            lambda: AhemV32NextPacket(12),
            "weighted_forecast+time+local+counterfactual_ranking",
        ),
        "ahem_v3.2_local": ModelSpec(
            "ahem_v3.2_local",
            "local_ablation",
            context_packets,
            lambda: AhemV32LocalOnlyNextPacket(12),
            "weighted_forecast+time_without_transition_memory",
        ),
        "ahem_v3.2_recency": ModelSpec(
            "ahem_v3.2_recency",
            "recency_ablation",
            context_packets,
            lambda: AhemV32RecencyNextPacket(12),
            "weighted_forecast+time_with_fixed_recency_retrieval",
        ),
        "lstm": ModelSpec(
            "lstm",
            "recurrent",
            context_packets,
            lambda: LstmNextPacket(12),
            "weighted_forecast+time",
        ),
        "gru": ModelSpec(
            "gru",
            "recurrent",
            context_packets,
            lambda: GruNextPacket(12),
            "weighted_forecast+time",
        ),
        "causal_cnn": ModelSpec(
            "causal_cnn",
            "local_convolution",
            31,
            lambda: CausalCnnNextPacket(12),
            "weighted_forecast+time",
        ),
        "long_tcn": ModelSpec(
            "long_tcn",
            "dilated_convolution",
            context_packets,
            lambda: LongTcnNextPacket(12),
            "weighted_forecast+time",
        ),
        "s4d": ModelSpec(
            "s4d",
            "diagonal_state_space",
            context_packets,
            lambda: S4DNextPacket(12),
            "weighted_forecast+time",
        ),
    }
    try:
        return specs[name]
    except KeyError as error:
        raise ValueError(f"unknown flow-v2 model {name!r}") from error


def _coordinate_batches(
    coordinates: list[WindowCoordinate],
    batch_size: int,
    *,
    seed: int | None = None,
) -> list[list[WindowCoordinate]]:
    ordered = list(coordinates)
    if seed is not None:
        random.Random(seed).shuffle(ordered)
    return [
        ordered[index : index + batch_size]
        for index in range(0, len(ordered), batch_size)
    ]


def _model_input(contexts: Tensor, spec: ModelSpec) -> Tensor:
    if contexts.size(1) < spec.input_context:
        raise ValueError("protocol context is shorter than the model input contract")
    return contexts[:, -spec.input_context :]


def _loss(
    model: nn.Module, spec: ModelSpec, contexts: Tensor, targets: Tensor
) -> tuple[Tensor, dict[str, float]]:
    if spec.name == "ahem_v3.2":
        if not isinstance(model, AhemV32NextPacket):
            raise TypeError("AHEM model specification has an incompatible module")
        return _training_loss(model, contexts, targets)
    prediction = model(contexts)
    forecast = _weighted_loss(prediction, targets)
    time_loss = F.smooth_l1_loss(prediction[:, 1], targets[:, 1])
    total = forecast + 0.5 * time_loss
    return total, {
        "forecast": float(forecast.detach()),
        "ranking": 0.0,
        "time": float(time_loss.detach()),
    }


def _evaluate(
    model: nn.Module,
    spec: ModelSpec,
    protocol: CompiledProtocol,
    partition: str,
    context_packets: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    coordinates = protocol.coordinates[partition]
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    local_predictions: list[Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch in _coordinate_batches(coordinates, batch_size):
            contexts, target = materialize_batch(
                protocol.partitions[partition], batch, context_packets
            )
            contexts = _model_input(contexts, spec).to(device)
            target = target.to(device)
            if isinstance(model, AhemV32NextPacket):
                output = model.forward_diagnostics(contexts)
                prediction = output.prediction
                local_predictions.append(output.local_prediction.float().cpu())
            else:
                prediction = model(contexts)
            predictions.append(prediction.float().cpu())
            targets.append(target.float().cpu())
    joined_prediction = torch.cat(predictions)
    joined_target = torch.cat(targets)
    quality = asdict(_quality(joined_prediction, joined_target))
    per_sample = _weighted_loss(joined_prediction, joined_target, reduction="none")
    sample_stream_indices = [value.stream_index for value in coordinates]
    result: dict[str, Any] = {
        "quality": quality,
        "weighted_smooth_l1": float(per_sample.mean()),
        "per_sample_weighted_smooth_l1": per_sample.tolist(),
        "per_sample_sha256": _canonical_sha256(per_sample.tolist()),
        "sample_stream_indices": sample_stream_indices,
        "sample_stream_indices_sha256": _canonical_sha256(sample_stream_indices),
        "samples": len(per_sample),
    }
    if local_predictions:
        joined_local = torch.cat(local_predictions)
        local_loss = _weighted_loss(joined_local, joined_target, reduction="none")
        improvement = local_loss - per_sample
        result["memory_effect"] = {
            "available": True,
            "not_retrieval_accuracy": True,
            "help_rate_vs_local": float((improvement > 0).float().mean()),
            "mean_weighted_loss_reduction": float(improvement.mean()),
            "local_weighted_smooth_l1": float(local_loss.mean()),
            "full_weighted_smooth_l1": float(per_sample.mean()),
        }
    else:
        result["memory_effect"] = {"available": False}
    return result


def _cpu_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _reserve_test_access(
    output: Path,
    *,
    protocol_sha256: str,
    coordinate_sha256: str,
    model: str,
    training_seed: int,
    best_epoch: int,
) -> dict[str, Any]:
    """Durably reserve the only confirmatory test evaluation attempt."""
    checkpoint_path = output / "best.pt"
    if not checkpoint_path.is_file():
        raise ValueError("test access requires the frozen best checkpoint")
    ledger_path = output / "test-access.json"
    ledger = {
        "schema": "argus.flow-v2-test-access/v1",
        "status": "reserved",
        "attempt": 1,
        "reserved_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol_sha256,
        "coordinate_sha256": coordinate_sha256,
        "model": model,
        "training_seed": training_seed,
        "best_epoch": best_epoch,
        "best_checkpoint_sha256": _file_sha256(checkpoint_path),
    }
    output.mkdir(parents=True, exist_ok=True)
    try:
        with ledger_path.open("x", encoding="utf-8") as destination:
            destination.write(json.dumps(ledger, indent=2) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
    except FileExistsError as error:
        raise ValueError("confirmatory test access was already reserved") from error
    return ledger


def run(args: argparse.Namespace) -> dict[str, Any]:
    exploratory = bool(getattr(args, "exploratory_run", False))
    diagnostic = bool(getattr(args, "allow_dirty_source", False))
    study_role = (
        "exploratory" if exploratory else "diagnostic" if diagnostic else "confirmatory"
    )
    source_provenance = _source_provenance()
    if not getattr(args, "allow_dirty_source", False) and (
        source_provenance.get("available") is not True
        or source_provenance.get("git_dirty") is not False
    ):
        raise RuntimeError(
            "publication training requires an available clean Git source; "
            "use --allow-dirty-source only for non-publication diagnostics"
        )
    protocol_config = ProtocolConfig(
        context_packets=args.context,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        seed=args.protocol_seed,
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
    protocol = compile_protocol(
        args.journal,
        args.validation_source,
        args.test_source,
        protocol_config,
    )
    publication_data_eligible = protocol.manifest["journal"].get(
        "provenance_complete"
    ) is True and protocol.manifest["journal"].get("datasets") == ["CSE-CIC-IDS2018"]
    if not publication_data_eligible and not getattr(args, "allow_dirty_source", False):
        raise RuntimeError(
            "publication training requires provenance-complete CSE-CIC-IDS2018 journals"
        )
    if protocol.manifest["schema"] != PROTOCOL_SCHEMA:
        raise ValueError("unexpected comparison protocol schema")
    expected_confirmatory_split = {
        "method": "journal_source_file_disjoint",
        "validation_source": PRIMARY_VALIDATION_SOURCE,
        "test_source": PRIMARY_TEST_SOURCE,
        "train_sources": sorted(PRIMARY_TRAIN_SOURCES),
    }
    if (
        study_role == "confirmatory"
        and protocol.manifest["split"] != expected_confirmatory_split
    ):
        raise RuntimeError(
            "confirmatory training requires the exact frozen primary source split"
        )
    protocol_sha256 = _canonical_sha256(protocol.manifest)
    spec = model_spec(args.model, args.context)
    inventory = protocol.manifest["models"].get(spec.name)
    if inventory is None:
        raise ValueError("model is absent from the frozen protocol inventory")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    torch.manual_seed(training.seed)
    random.seed(training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    model = spec.constructor().to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != inventory["parameters"]:
        raise ValueError("constructed model differs from the protocol inventory")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    use_amp = device.type == "cuda"
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not use_bf16)
    precision = "bfloat16" if use_bf16 else "float16" if use_amp else "float32"
    training_sha256 = _canonical_sha256(
        {
            "optimization": asdict(training),
            "device": str(device),
            "precision": precision,
            "torch": torch.__version__,
        }
    )

    args.output.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema": RUN_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "task": "causal_per_flow_next_packet_forecast_only",
        "study_role": study_role,
        "publication_eligible": study_role == "confirmatory",
        "not_intrusion_detection_efficacy": True,
        "feature_schema": FEATURE_SCHEMA,
        "sequence_contract": protocol.manifest["sequence_contract"],
        "protocol_sha256": protocol_sha256,
        "coordinate_sha256": protocol.manifest["coordinate_sha256"],
        "protocol": protocol.manifest,
        "source_provenance": source_provenance,
        "model": {
            "name": spec.name,
            "family": spec.family,
            "parameters": parameters,
            "state_bytes_float32": inventory["state_bytes_float32"],
            "input_context_packets": spec.input_context,
            "random_initialization_only": True,
            "v1_weights_reused": False,
            "auxiliary_objective": spec.auxiliary_objective,
        },
        "training": {
            **asdict(training),
            "config_sha256": training_sha256,
            "examples_per_epoch": len(protocol.coordinates["train"]),
            "optimizer_steps_per_epoch": math.ceil(
                len(protocol.coordinates["train"]) / training.batch_size
            ),
            "equal_target_coordinates_across_models": True,
            "equal_examples_across_models": True,
            "equal_flops_claimed": False,
            "precision": precision,
            "torch": torch.__version__,
            "device": str(device),
        },
        "retrieval_evaluation": protocol.manifest["retrieval_evaluation"],
    }
    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    best_epoch = 0
    best_state = deepcopy(_cpu_state(model))
    stale_epochs = 0
    start_epoch = 1
    if args.resume is not None:
        manifest_path = args.output / "run.json"
        if not manifest_path.is_file():
            raise ValueError("resume output has no original run manifest")
        original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable_manifest_fields = (
            "schema",
            "protocol_sha256",
            "coordinate_sha256",
            "feature_schema",
        )
        if any(
            original_manifest.get(field) != run_manifest.get(field)
            for field in immutable_manifest_fields
        ):
            raise ValueError("resume run manifest changed")
        if original_manifest.get("model", {}).get("name") != spec.name:
            raise ValueError("resume run manifest belongs to another model")
        if (
            original_manifest.get("training", {}).get("config_sha256")
            != training_sha256
        ):
            raise ValueError("resume run manifest training configuration changed")
        if original_manifest.get("source_provenance") != run_manifest.get(
            "source_provenance"
        ):
            raise ValueError("resume source revision or tree state changed")
        run_manifest = original_manifest
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("resume file is not a flow-v2 checkpoint")
        if checkpoint.get("protocol_sha256") != protocol_sha256:
            raise ValueError("resume protocol fingerprint changed")
        if (
            checkpoint.get("coordinate_sha256")
            != protocol.manifest["coordinate_sha256"]
        ):
            raise ValueError("resume evaluation coordinates changed")
        if checkpoint.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("resume feature schema changed")
        if checkpoint.get("model") != spec.name:
            raise ValueError("resume checkpoint belongs to another model")
        if checkpoint.get("training_config_sha256") != training_sha256:
            raise ValueError("resume training configuration changed")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint["history"])
        best_validation = float(checkpoint["best_validation_weighted_smooth_l1"])
        best_epoch = int(checkpoint["best_epoch"])
        best_state = checkpoint["best_state_dict"]
        stale_epochs = int(checkpoint["stale_epochs"])
        start_epoch = int(checkpoint["epoch"]) + 1
    else:
        write_manifest(args.output / "run.json", run_manifest)

    for epoch in range(start_epoch, training.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        forecast_losses: list[float] = []
        ranking_losses: list[float] = []
        time_losses: list[float] = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        batches = _coordinate_batches(
            protocol.coordinates["train"],
            training.batch_size,
            seed=training.seed + epoch * 1_000_003,
        )
        for batch in batches:
            contexts, targets = materialize_batch(
                protocol.partitions["train"], batch, args.context
            )
            contexts = _model_input(contexts, spec).to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                loss, components = _loss(model, spec, contexts, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite flow-v2 training loss")
            scaler.scale(loss).backward()
            if use_amp:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach()))
            forecast_losses.append(components["forecast"])
            ranking_losses.append(components["ranking"])
            time_losses.append(components["time"])

        validation = _evaluate(
            model,
            spec,
            protocol,
            "validation",
            args.context,
            training.batch_size,
            device,
        )
        validation_loss = validation["weighted_smooth_l1"]
        improved = validation_loss < best_validation
        if improved:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = deepcopy(_cpu_state(model))
            stale_epochs = 0
        else:
            stale_epochs += 1
        record = {
            "epoch": epoch,
            "train_loss": sum(epoch_losses) / len(epoch_losses),
            "train_forecast_loss": sum(forecast_losses) / len(forecast_losses),
            "train_ranking_loss": sum(ranking_losses) / len(ranking_losses),
            "train_time_loss": sum(time_losses) / len(time_losses),
            "validation": validation["quality"],
            "improved": improved,
            "gpu_peak_bytes": (
                torch.cuda.max_memory_allocated() if device.type == "cuda" else None
            ),
            "created_utc": datetime.now(UTC).isoformat(),
        }
        history.append(record)
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "protocol_sha256": protocol_sha256,
            "coordinate_sha256": protocol.manifest["coordinate_sha256"],
            "feature_schema": FEATURE_SCHEMA,
            "training_config_sha256": training_sha256,
            "model": spec.name,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_weighted_smooth_l1": best_validation,
            "state_dict": _cpu_state(model),
            "best_state_dict": best_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "stale_epochs": stale_epochs,
            "history": history,
        }
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

    test_access = _reserve_test_access(
        args.output,
        protocol_sha256=protocol_sha256,
        coordinate_sha256=protocol.manifest["coordinate_sha256"],
        model=spec.name,
        training_seed=training.seed,
        best_epoch=best_epoch,
    )
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    test = _evaluate(
        model,
        spec,
        protocol,
        "test",
        args.context,
        training.batch_size,
        device,
    )
    test_access = {
        **test_access,
        "status": "completed",
        "completed_utc": datetime.now(UTC).isoformat(),
        "test_samples": test["samples"],
        "test_per_sample_sha256": test["per_sample_sha256"],
        "test_sample_stream_indices_sha256": test["sample_stream_indices_sha256"],
    }
    write_manifest(args.output / "test-access.json", test_access)
    run_manifest = {**run_manifest, "test_access": test_access}
    write_manifest(args.output / "run.json", run_manifest)
    result = {
        **run_manifest,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_weighted_smooth_l1": best_validation,
        "test": test,
        "runtime_benchmark_required": True,
        "history": history,
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
            "ahem_v3.2",
            "ahem_v3.2_local",
            "ahem_v3.2_recency",
            "lstm",
            "gru",
            "causal_cnn",
            "long_tcn",
            "s4d",
        ),
        required=True,
    )
    parser.add_argument("--validation-source", required=True)
    parser.add_argument("--test-source", required=True)
    parser.add_argument("--context", type=int, default=10_000)
    parser.add_argument("--train-samples", type=int, default=2_048)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--protocol-seed", type=int, default=20260821)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument(
        "--exploratory-run",
        action="store_true",
        help="mark a reusable development-only evaluation as publication-ineligible",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if (
        min(
            args.context,
            args.train_samples,
            args.validation_samples,
            args.test_samples,
        )
        <= 0
    ):
        parser.error("protocol counts must be positive")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

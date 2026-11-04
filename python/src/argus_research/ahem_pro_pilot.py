"""Exploratory source-disjoint real-PCAP pilot for AHEM Pro.

This runner is deliberately ineligible for publication claims. It never names
or opens the frozen v6.1 final holdouts and performs only one development seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor, nn

from .ahem_pro import (
    AhemProLocalOnlyNextPacket,
    AhemProNextPacket,
    AhemProRecencyNextPacket,
    ProConfig,
    objective_contract,
    pro_training_loss,
)
from .ahem_pro_checkpoint import implementation_sha256
from .ahem_v6 import FlowLaneLstmNextPacket
from .flow_v3_protocol import ProtocolConfig, compile_source_disjoint, materialize_batch
from .flow_v6_development import _retrieval_metrics
from .flow_v6_objective import dynamic_loss
from .pro_baselines import PersistenceResidualLstm

SCHEMA = "argus.ahem-pro-exploratory-pilot/v2.1"
SEED = 20260908
MODEL_NAMES = (
    "ahem_pro",
    "ahem_pro_recency",
    "ahem_pro_local_only",
    "flow_lane_lstm_h255",
    "persistence_residual_lstm_h255",
)
TRAIN_SOURCE = "Thursday-15-02-2018"
VALIDATION_SOURCE = "Wednesday-14-02-2018"
SCREEN_SOURCE = "Friday-16-02-2018"
ALLOWED_SOURCES = frozenset({TRAIN_SOURCE, VALIDATION_SOURCE, SCREEN_SOURCE})
DEVELOPMENT_SOURCE_ALLOWLIST = frozenset(
    {
        "Wednesday-14-02-2018",
        "Thursday-15-02-2018",
        "Friday-16-02-2018",
        "Tuesday-20-02-2018",
        "Wednesday-21-02-2018",
        "Wednesday-28-02-2018",
        "Thursday-01-03-2018",
        "Friday-02-03-2018",
    }
)
FORBIDDEN_FINAL_HOLDOUTS = frozenset(
    {"Thursday-22-02-2018", "Friday-23-02-2018", "CIC-IDS2017"}
)


@dataclass(frozen=True)
class PilotConfig:
    context_packets: int = 30_000
    train_samples: int = 256
    validation_samples: int = 64
    screen_samples: int = 64
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4

    def validate(self) -> None:
        if any(
            type(value) is not int or value <= 0
            for name, value in asdict(self).items()
            if name not in {"learning_rate", "weight_decay"}
        ):
            raise ValueError("pilot integer settings must be positive")
        if self.context_packets != ProConfig().horizon_packets:
            raise ValueError("pilot context must equal the Pro contract horizon")
        if (
            not math.isfinite(self.learning_rate)
            or not math.isfinite(self.weight_decay)
            or self.learning_rate <= 0
            or self.weight_decay < 0
        ):
            raise ValueError("pilot optimizer settings are invalid")


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(value, temporary)
    # Decode before the filename becomes visible as complete.
    torch.load(temporary, map_location="cpu", weights_only=True)
    os.replace(temporary, path)


def _inventory(
    journal: Path, allowed_sources: frozenset[str] = ALLOWED_SOURCES
) -> None:
    if (
        not ALLOWED_SOURCES <= allowed_sources
        or not allowed_sources <= DEVELOPMENT_SOURCE_ALLOWLIST
        or FORBIDDEN_FINAL_HOLDOUTS & allowed_sources
    ):
        raise ValueError("pilot source inventory is outside the development allowlist")
    journals = sorted(journal.glob("*.arguslr"))
    names = {path.stem for path in journals}
    if names != allowed_sources:
        raise ValueError(
            f"pilot requires exactly its declared sources, got {sorted(names)}"
        )
    for path in journals:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe pilot journal: {path}")
        if not path.with_suffix(path.suffix + ".manifest.json").is_file():
            raise ValueError(f"journal manifest is absent: {path.name}")
    if FORBIDDEN_FINAL_HOLDOUTS & names:
        raise ValueError("frozen final holdout entered the Pro pilot")


def _batches(count: int, size: int, *, seed: int | None = None) -> list[Tensor]:
    order = torch.arange(count)
    if seed is not None:
        order = order[
            torch.randperm(count, generator=torch.Generator().manual_seed(seed))
        ]
    return list(order.split(size))


def _materialize(protocol, config: PilotConfig) -> dict[str, tuple[Tensor, Tensor]]:
    result = {}
    for partition in ("train", "validation", "test"):
        contexts, targets = materialize_batch(
            protocol.partitions[partition],
            protocol.coordinates[partition],
            config.context_packets,
        )
        result[partition] = (contexts.contiguous(), targets.contiguous())
    return result


def _baseline_metrics(
    data: dict[str, tuple[Tensor, Tensor]],
) -> dict[str, dict[str, object]]:
    train_mean = data["train"][1].mean(0)
    result = {}
    for partition, (contexts, targets) in data.items():
        result[partition] = {
            "target_mean": targets.mean(0).tolist(),
            "target_std": targets.std(0, unbiased=False).tolist(),
            "flow_lane_events_mean": float(contexts[..., -1].sum(-1).float().mean()),
            "current_packet_persistence": float(
                dynamic_loss(contexts[:, -1, :2], targets)
            ),
            "zero": float(dynamic_loss(torch.zeros_like(targets), targets)),
            "training_target_mean": float(
                dynamic_loss(train_mean.expand_as(targets), targets)
            ),
        }
    return result


def _construct(name: str) -> nn.Module:
    constructors = {
        "ahem_pro": AhemProNextPacket,
        "ahem_pro_recency": AhemProRecencyNextPacket,
        "ahem_pro_local_only": AhemProLocalOnlyNextPacket,
        # 340,172 parameters: 0.264% above Pro's 339,275.
        "flow_lane_lstm_h255": lambda: FlowLaneLstmNextPacket(hidden_dim=255),
        "persistence_residual_lstm_h255": PersistenceResidualLstm,
    }
    try:
        return constructors[name]()
    except KeyError as error:
        raise ValueError(f"unknown Pro pilot model: {name}") from error


def _evaluate(
    model: nn.Module,
    values: tuple[Tensor, Tensor],
    config: PilotConfig,
    device: torch.device,
) -> dict[str, object]:
    losses, retrieval, branches = [], {}, {}
    model.eval()
    with torch.inference_mode():
        contexts, targets = values
        for indices in _batches(len(contexts), config.batch_size):
            batch = contexts[indices].to(device, non_blocking=True)
            target = targets[indices].to(device, non_blocking=True)
            if isinstance(model, AhemProNextPacket):
                output = model.forward_diagnostics(batch)
                prediction = output.prediction
                if model.uses_transition_memory:
                    for name, metric in _retrieval_metrics(output, target).items():
                        retrieval.setdefault(name, []).append(metric.cpu())
                    local_loss = dynamic_loss(
                        output.local_prediction, target, reduction="none"
                    )
                    memory_loss = dynamic_loss(
                        output.memory_candidate, target, reduction="none"
                    )
                    branch_values = {
                        "local_weighted_smooth_l1": local_loss,
                        "memory_weighted_smooth_l1": memory_loss,
                        "memory_minus_local_loss": memory_loss - local_loss,
                        "memory_gate": output.memory_gate[:, 0],
                    }
                    for name, metric in branch_values.items():
                        branches.setdefault(name, []).append(metric.cpu())
            else:
                prediction = model(batch)
            losses.append(dynamic_loss(prediction, target, reduction="none").cpu())
    joined = torch.cat(losses)
    if not torch.isfinite(joined).all():
        raise FloatingPointError("non-finite pilot evaluation loss")
    result: dict[str, object] = {
        "samples": len(joined),
        "weighted_smooth_l1": float(joined.mean()),
        "per_sample_weighted_smooth_l1": joined.tolist(),
    }
    if retrieval:
        result["retrieval"] = {
            name: float(torch.cat(metric).mean()) for name, metric in retrieval.items()
        }
    if branches:
        result["branch_diagnostics"] = {
            name: float(torch.cat(metric).mean()) for name, metric in branches.items()
        }
    return result


def _train(
    name: str,
    data: dict[str, tuple[Tensor, Tensor]],
    config: PilotConfig,
    device: torch.device,
    output: Path,
    *,
    run_binding: str,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = _construct(name).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state = {
        key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()
    }
    initial_validation = _evaluate(model, data["validation"], config, device)
    best_value = float(initial_validation["weighted_smooth_l1"])
    best_epoch = 0
    history = [
        {
            "epoch": 0,
            "train_objective": None,
            "validation_weighted_smooth_l1": best_value,
            "improved": True,
            "checkpoint_role": "untrained_initialization",
            "created_utc": datetime.now(UTC).isoformat(),
        }
    ]
    contexts, targets = data["train"]

    def publish_best() -> None:
        _atomic_checkpoint(
            output / "best.pt",
            {
                "schema": SCHEMA,
                "evaluation_status": "exploratory_unvalidated",
                "security_detector": False,
                "model": name,
                "seed": SEED,
                "run_binding": run_binding,
                "best_epoch": best_epoch,
                "selected_checkpoint_optimizer_steps": best_epoch
                * len(_batches(len(contexts), config.batch_size)),
                "training_improved_over_initialization": best_epoch > 0,
                "config": asdict(config),
                "pro_config": asdict(ProConfig()),
                "objective_contract": objective_contract(),
                "implementation_sha256": implementation_sha256(),
                "state_dict": best_state,
            },
        )

    publish_best()
    _atomic_json(output / "progress.json", {"status": "running", "history": history})
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []
        for indices in _batches(
            len(contexts), config.batch_size, seed=SEED + epoch * 1_000_003
        ):
            batch = contexts[indices].to(device, non_blocking=True)
            target = targets[indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, AhemProNextPacket) and model.uses_transition_memory:
                loss, _ = pro_training_loss(model.forward_diagnostics(batch), target)
            else:
                loss = dynamic_loss(model(batch), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite pilot loss for {name}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        validation = _evaluate(model, data["validation"], config, device)
        value = float(validation["weighted_smooth_l1"])
        improved = value < best_value
        if improved:
            best_value, best_epoch = value, epoch
            best_state = deepcopy(
                {
                    key: tensor.detach().cpu()
                    for key, tensor in model.state_dict().items()
                }
            )
        record = {
            "epoch": epoch,
            "train_objective": sum(epoch_losses) / len(epoch_losses),
            "validation_weighted_smooth_l1": value,
            "improved": improved,
            "created_utc": datetime.now(UTC).isoformat(),
        }
        history.append(record)
        if improved:
            publish_best()
        _atomic_json(
            output / "progress.json", {"status": "running", "history": history}
        )
        print(json.dumps({"model": name, **record}), flush=True)
    model.load_state_dict(best_state, strict=True)
    model.to(device).eval()
    metrics = {
        "model": name,
        "run_binding": run_binding,
        "seed": SEED,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "selected_checkpoint_optimizer_steps": best_epoch
        * len(_batches(len(contexts), config.batch_size)),
        "training_improved_over_initialization": best_epoch > 0,
        "best_validation_weighted_smooth_l1": best_value,
        "history": history,
        "train": _evaluate(model, data["train"], config, device),
        "validation": _evaluate(model, data["validation"], config, device),
        "screen": _evaluate(model, data["test"], config, device),
    }
    publish_best()
    _atomic_json(output / "metrics.json", metrics)
    _atomic_json(output / "progress.json", {"status": "complete", "history": history})
    return metrics


def _comparison_gates(results, baselines):
    pro = float(results["ahem_pro"]["screen"]["weighted_smooth_l1"])
    gates = {
        "pro_training_improves_over_initialization": results["ahem_pro"][
            "training_improved_over_initialization"
        ],
        "lstm_training_improves_over_initialization": results["flow_lane_lstm_h255"][
            "training_improved_over_initialization"
        ],
        "lstm_beats_zero_baseline": float(
            results["flow_lane_lstm_h255"]["screen"]["weighted_smooth_l1"]
        )
        < baselines["test"]["zero"],
        "pro_beats_current_packet_persistence": pro
        < baselines["test"]["current_packet_persistence"],
        "learned_retrieval_top1_beats_recency": results["ahem_pro"]["screen"][
            "retrieval"
        ]["top1_tie_aware"]
        > results["ahem_pro_recency"]["screen"]["retrieval"]["top1_tie_aware"],
    }
    for name in MODEL_NAMES[1:]:
        reference = float(results[name]["screen"]["weighted_smooth_l1"])
        if not math.isfinite(reference) or reference < 0:
            raise FloatingPointError(f"invalid comparison loss for {name}")
        gates[f"pro_beats_{name}"] = pro < reference
    if not math.isfinite(pro) or pro < 0:
        raise FloatingPointError("invalid Pro comparison loss")
    return gates


def run(
    journal: Path,
    output: Path,
    config: PilotConfig,
    allowed_sources: frozenset[str] = ALLOWED_SOURCES,
) -> dict[str, object]:
    config.validate()
    if output.exists():
        raise ValueError("pilot output already exists; use a fresh directory")
    _inventory(journal, allowed_sources)
    if not torch.cuda.is_available():
        raise RuntimeError("Pro real-data pilot requires CUDA")
    output.mkdir(parents=True)
    protocol = compile_source_disjoint(
        journal,
        VALIDATION_SOURCE,
        SCREEN_SOURCE,
        ProtocolConfig(
            context_packets=config.context_packets,
            local_packets=128,
            train_samples=config.train_samples,
            validation_samples=config.validation_samples,
            test_samples=config.screen_samples,
            seed=SEED,
            require_memory_match=True,
            sampling_policy=(
                "source_balanced_natural" if len(allowed_sources) > 3 else "natural"
            ),
            minimum_completed_transitions=2,
        ),
    )
    split = protocol.manifest["split"]
    expected_train_sources = allowed_sources - {VALIDATION_SOURCE, SCREEN_SOURCE}
    if set(split["train_sources"]) != expected_train_sources:
        raise RuntimeError("pilot training source changed")
    manifest = {
        "schema": SCHEMA,
        "study_role": "exploratory_architecture_pilot",
        "publication_eligible": False,
        "confirmatory_claim_eligible": False,
        "security_detector": False,
        "real_pcap_journals": True,
        "forbidden_final_holdouts_accessed": False,
        "declared_sources": sorted(allowed_sources),
        "training_sources": sorted(expected_train_sources),
        "config": asdict(config),
        "pro_config": asdict(ProConfig()),
        "objective_contract": objective_contract(),
        "implementation_sha256": implementation_sha256(),
        "architecture_contract": ProConfig().contract(),
        "models": list(MODEL_NAMES),
        "runtime": {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "protocol": protocol.manifest,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    digest = hashlib.sha256()
    for source_file in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source_file.name.encode() + b"\0")
        digest.update(source_file.read_text(encoding="utf-8").encode() + b"\0")
    manifest["research_source_sha256"] = digest.hexdigest()
    manifest["run_binding"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_json(output / "run.json", manifest)
    data = _materialize(protocol, config)
    baselines = _baseline_metrics(data)
    _atomic_json(output / "data_baselines.json", baselines)
    device = torch.device("cuda")
    results = {}
    for name in MODEL_NAMES:
        results[name] = _train(
            name,
            data,
            config,
            device,
            output / "models" / name,
            run_binding=manifest["run_binding"],
        )
    pro = float(results["ahem_pro"]["screen"]["weighted_smooth_l1"])
    comparisons = {}
    for name in MODEL_NAMES[1:]:
        baseline = float(results[name]["screen"]["weighted_smooth_l1"])
        comparisons[name] = {
            "relative_loss_reduction": (baseline - pro) / max(baseline, 1e-12)
        }
    gates = _comparison_gates(results, baselines)
    final = {
        "schema": SCHEMA,
        "run_binding": manifest["run_binding"],
        "decision": (
            "pilot_supports_larger_development_run"
            if all(gates.values())
            else "pilot_rejects_or_requires_repair"
        ),
        "models": results,
        "comparisons": comparisons,
        "data_baselines": baselines,
        "gates": gates,
        "forbidden_final_holdouts_accessed": False,
    }
    _atomic_json(output / "RESULT.json", final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--screen-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--additional-training-source",
        action="append",
        default=[],
        choices=sorted(DEVELOPMENT_SOURCE_ALLOWLIST - ALLOWED_SOURCES),
    )
    args = parser.parse_args()
    result = run(
        args.journal,
        args.output,
        PilotConfig(
            epochs=args.epochs,
            train_samples=args.train_samples,
            validation_samples=args.validation_samples,
            screen_samples=args.screen_samples,
            batch_size=args.batch_size,
        ),
        ALLOWED_SOURCES | frozenset(args.additional_training_source),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

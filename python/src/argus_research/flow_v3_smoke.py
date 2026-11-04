"""Exploratory AHEM flow-memory screens on provenance-bound real journals."""

from __future__ import annotations

import argparse
import json
import os
import random
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .ahem_pretrain import GruNextPacket, LongTcnNextPacket, LstmNextPacket
from .ahem_v3 import (
    AhemV32NextPacket,
    AhemV34FlowIndexedNextPacket,
    AhemV34LocalOnlyNextPacket,
    AhemV34RecencyNextPacket,
)
from .ahem_v4 import AhemV4ProjectedStreamNextPacket
from .flow_v2_protocol import write_manifest
from .flow_v3_protocol import (
    CompiledProtocol,
    ProtocolConfig,
    compile_local_development,
    compile_source_disjoint,
    materialize_batch,
)
from .s4d_baseline import S4DNextPacket

SMOKE_SCHEMA = "argus.flow-conditioned-global-smoke/v3"
CANDIDATE = "ahem_v3.4_flow_indexed"
LOCAL_ABLATION = "ahem_v3.4_local_only"
RECENCY_ABLATION = "ahem_v3.4_recency"
REDUCED_MODELS = frozenset({CANDIDATE, LOCAL_ABLATION, RECENCY_ABLATION, "lstm"})


def _batches(values: list[Any], batch_size: int, seed: int) -> list[list[Any]]:
    ordered = list(values)
    random.Random(seed).shuffle(ordered)
    return [
        ordered[index : index + batch_size]
        for index in range(0, len(ordered), batch_size)
    ]


def _loss(model: nn.Module, contexts: Tensor, targets: Tensor) -> Tensor:
    if (
        isinstance(
            model,
            (AhemV34FlowIndexedNextPacket, AhemV4ProjectedStreamNextPacket),
        )
        and model.uses_transition_memory
    ):
        output = model.forward_diagnostics(contexts)
        forecast = _target_loss(output.prediction, targets)
        local = _target_loss(output.local_prediction, targets, reduction="none")
        memory = _target_loss(output.memory_candidate, targets, reduction="none")
        utility_target = torch.sigmoid((local.detach() - memory.detach()) / 0.05)
        utility = nn.functional.binary_cross_entropy(
            output.memory_gate.squeeze(-1).clamp(1e-6, 1.0 - 1e-6),
            utility_target,
        )
        return forecast + 0.15 * local.mean() + 0.15 * memory.mean() + 0.10 * utility
    return _target_loss(model(contexts), targets)


def _target_loss(prediction: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    if prediction.shape != target.shape or prediction.size(-1) != 2:
        raise ValueError("dynamic-event loss expects length and interarrival")
    weights = prediction.new_tensor((1.0, 4.0))
    values = (
        nn.functional.smooth_l1_loss(prediction, target, reduction="none") * weights
    ).sum(dim=-1) / weights.sum()
    return values.mean() if reduction == "mean" else values


def _long_gap_metadata(
    flow_ids: Tensor, anchor_index: int, context_packets: int, retained_events: int = 17
) -> dict[str, int | str | None]:
    start = max(0, anchor_index - context_packets + 1)
    flow_generation = flow_ids[anchor_index]
    positions = (
        torch.nonzero(flow_ids[start : anchor_index + 1] == flow_generation)
        .flatten()
        .add(start)
    )
    retained = positions[-retained_events:]
    transition_count = max(0, retained.numel() - 1)
    if transition_count == 0:
        return {
            "oldest_retained_transition_age": None,
            "completed_transition_count": 0,
            "long_gap_bucket": "no_completed_transition",
        }
    age = anchor_index - int(retained[0])
    if age < 129:
        bucket = "below_129"
    elif age < 1_024:
        bucket = "129_to_1023"
    elif age < 4_096:
        bucket = "1024_to_4095"
    else:
        bucket = "4096_to_9999"
    return {
        "oldest_retained_transition_age": age,
        "completed_transition_count": transition_count,
        "long_gap_bucket": bucket,
    }


def _evaluate(
    model: nn.Module,
    protocol: CompiledProtocol,
    partition: str,
    context: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    losses: list[Tensor] = []
    persistence_losses: list[Tensor] = []
    local_losses: list[Tensor] = []
    changed_events: list[Tensor] = []
    memory_gates: list[Tensor] = []
    sample_coordinates: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch in _batches(protocol.coordinates[partition], batch_size, 0):
            for coordinate in batch:
                record = protocol.partitions[partition][coordinate.record_index]
                sample_coordinates.append(
                    {
                        "source_file": record.source_file,
                        "segment_id": record.segment_id,
                        "flow_generation": str(
                            int(record.flow_ids[coordinate.anchor_index])
                        ),
                        "anchor_index": coordinate.anchor_index,
                        "target_index": coordinate.target_index,
                        **_long_gap_metadata(
                            record.flow_ids,
                            coordinate.anchor_index,
                            context,
                        ),
                    }
                )
            contexts, targets = materialize_batch(
                protocol.partitions[partition], batch, context
            )
            contexts = contexts.to(device)
            targets = targets.to(device)
            changed_events.append(
                ((targets - contexts[:, -1, :2]).abs().amax(dim=-1) > 1e-6).cpu()
            )
            if isinstance(
                model,
                (
                    AhemV32NextPacket,
                    AhemV34FlowIndexedNextPacket,
                    AhemV4ProjectedStreamNextPacket,
                ),
            ):
                output = model.forward_diagnostics(contexts)
                predictions = output.prediction
                if model.uses_transition_memory:
                    memory_gates.append(output.memory_gate.float().cpu().flatten())
                    local_losses.append(
                        _target_loss(output.local_prediction, targets, reduction="none")
                        .float()
                        .cpu()
                    )
            else:
                predictions = model(contexts)
            losses.append(
                _target_loss(predictions, targets, reduction="none").float().cpu()
            )
            persistence_losses.append(
                _target_loss(contexts[:, -1, :2], targets, reduction="none")
                .float()
                .cpu()
            )
    values = torch.cat(losses)
    persistence_values = torch.cat(persistence_losses)
    changed = torch.cat(changed_events)
    result: dict[str, Any] = {
        "samples": len(values),
        "weighted_smooth_l1": float(values.mean()),
        "per_sample_weighted_smooth_l1": values.tolist(),
        "persistence_weighted_smooth_l1": float(persistence_values.mean()),
        "per_sample_persistence_weighted_smooth_l1": persistence_values.tolist(),
        "sample_coordinates": sample_coordinates,
        "changed_event": changed.tolist(),
        "strata": {
            name: {
                "samples": int(mask.sum()),
                "weighted_smooth_l1": float(values[mask].mean()),
            }
            for name, mask in (
                ("changed", changed),
                ("unchanged", ~changed),
            )
        },
    }
    if local_losses:
        local = torch.cat(local_losses)
        gates = torch.cat(memory_gates)
        improvement = local - values
        result["memory_effect"] = {
            "available": True,
            "help_rate_vs_local": float((improvement > 0).float().mean()),
            "mean_weighted_loss_reduction": float(improvement.mean()),
            "mean_gate": float(gates.mean()),
            "mean_gate_changed": float(gates[changed].mean()),
            "mean_gate_unchanged": float(gates[~changed].mean()),
        }
    return result


def _train_one(
    name: str,
    model: nn.Module,
    protocol: CompiledProtocol,
    config: ProtocolConfig,
    *,
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    best_state = deepcopy(model.state_dict())
    best_validation = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for batch in _batches(
            protocol.coordinates["train"], batch_size, seed + epoch * 1_000_003
        ):
            contexts, targets = materialize_batch(
                protocol.partitions["train"], batch, config.context_packets
            )
            contexts = contexts.to(device)
            targets = targets.to(device)
            value = _loss(model, contexts, targets)
            if not torch.isfinite(value):
                raise FloatingPointError(f"non-finite {name} smoke loss")
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(value.detach()))
        validation = _evaluate(
            model,
            protocol,
            "validation",
            config.context_packets,
            batch_size,
            device,
        )
        if validation["weighted_smooth_l1"] < best_validation:
            best_validation = validation["weighted_smooth_l1"]
            best_state = deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "train_loss": sum(losses) / len(losses),
                "validation_weighted_smooth_l1": validation["weighted_smooth_l1"],
                "gpu_peak_bytes": (
                    torch.cuda.max_memory_allocated() if device.type == "cuda" else None
                ),
            }
        )
    model.load_state_dict(best_state)
    result = {
        "model": name,
        "parameters": sum(value.numel() for value in model.parameters()),
        "best_validation_weighted_smooth_l1": best_validation,
        "history": history,
        "test": _evaluate(
            model,
            protocol,
            "test",
            config.context_packets,
            batch_size,
            device,
        ),
    }
    return model, result


def run(
    journal: Path,
    output: Path,
    *,
    context: int,
    epochs: int,
    batch_size: int,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    device: torch.device,
    seed: int = 20260828,
    reduced: bool = False,
    validation_source: str | None = None,
    test_source: str | None = None,
) -> dict[str, Any]:
    if min(epochs, batch_size) <= 0:
        raise ValueError("epochs and batch size must be positive")
    if bool(validation_source) != bool(test_source):
        raise ValueError("validation and test sources must be provided together")
    source_disjoint = validation_source is not None and test_source is not None
    config = ProtocolConfig(
        context_packets=context,
        train_samples=train_samples,
        validation_samples=validation_samples,
        test_samples=test_samples,
        seed=seed,
        require_memory_match=source_disjoint,
    )
    protocol = (
        compile_source_disjoint(journal, validation_source, test_source, config)
        if source_disjoint
        else compile_local_development(journal, config)
    )
    if (
        source_disjoint
        and protocol.manifest["journal"].get("provenance_complete") is not True
    ):
        raise ValueError(
            "source-disjoint training requires complete real-data provenance"
        )
    model_factories: dict[str, type[nn.Module] | Any] = {
        CANDIDATE: AhemV34FlowIndexedNextPacket,
        LOCAL_ABLATION: AhemV34LocalOnlyNextPacket,
        RECENCY_ABLATION: AhemV34RecencyNextPacket,
        "lstm": lambda: LstmNextPacket(13, output_dim=2),
        "gru": lambda: GruNextPacket(13, output_dim=2),
        "long_tcn": lambda: LongTcnNextPacket(13, output_dim=2),
        "s4d": lambda: S4DNextPacket(13, output_dim=2),
    }
    if reduced:
        model_factories = {
            name: factory
            for name, factory in model_factories.items()
            if name in REDUCED_MODELS
        }
    results = {}
    output.mkdir(parents=True, exist_ok=True)
    for name, factory in model_factories.items():
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = factory()
        trained, metrics = _train_one(
            name,
            model,
            protocol,
            config,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        results[name] = metrics
        temporary = output / f"{name}.pt.tmp"
        torch.save(
            {
                "schema": SMOKE_SCHEMA,
                "model": name,
                "state_dict": trained.state_dict(),
                "protocol": protocol.manifest,
            },
            temporary,
        )
        os.replace(temporary, output / f"{name}.pt")
    difference = (
        results["lstm"]["test"]["weighted_smooth_l1"]
        - results[CANDIDATE]["test"]["weighted_smooth_l1"]
    )
    memory_difference = (
        results[LOCAL_ABLATION]["test"]["weighted_smooth_l1"]
        - results[CANDIDATE]["test"]["weighted_smooth_l1"]
    )
    recency_difference = (
        results[RECENCY_ABLATION]["test"]["weighted_smooth_l1"]
        - results[CANDIDATE]["test"]["weighted_smooth_l1"]
    )
    persistence = {}
    for partition in ("validation", "test"):
        loss_batches = []
        changed_batches = []
        for batch in _batches(protocol.coordinates[partition], batch_size, 0):
            contexts, targets = materialize_batch(
                protocol.partitions[partition], batch, config.context_packets
            )
            loss_batches.append(
                _target_loss(contexts[:, -1, :2], targets, reduction="none")
            )
            changed_batches.append(
                (targets - contexts[:, -1, :2]).abs().amax(dim=-1) > 1e-6
            )
        values = torch.cat(loss_batches)
        changed = torch.cat(changed_batches)
        persistence[partition] = {
            "samples": len(values),
            "weighted_smooth_l1": float(values.mean()),
            "per_sample_weighted_smooth_l1": values.tolist(),
            "changed_event": changed.tolist(),
            "strata": {
                name: {
                    "samples": int(mask.sum()),
                    "weighted_smooth_l1": float(values[mask].mean()),
                }
                for name, mask in (
                    ("changed", changed),
                    ("unchanged", ~changed),
                )
            },
        }
    persistence_difference = (
        persistence["test"]["weighted_smooth_l1"]
        - results[CANDIDATE]["test"]["weighted_smooth_l1"]
    )
    full_neural_baselines = ("lstm", "gru", "long_tcn", "s4d")
    neural_advantages = {
        name: results[name]["test"]["weighted_smooth_l1"]
        - results[CANDIDATE]["test"]["weighted_smooth_l1"]
        for name in full_neural_baselines
        if name in results
    }
    baseline_set_complete = set(neural_advantages) == set(full_neural_baselines)
    report = {
        "schema": SMOKE_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "study_role": "exploratory",
        "publication_evidence": False,
        "real_arguslr2": True,
        "local_development": not source_disjoint,
        "device": str(device),
        "protocol": protocol.manifest,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "model_set": "reduced" if reduced else "full",
            "source_disjoint": source_disjoint,
            "config": asdict(config),
        },
        "models": results,
        "baselines": {"last_event_persistence": persistence},
        "ahem_advantage": {
            "lstm_minus_ahem_weighted_loss": difference,
            "ahem_lower_loss": difference > 0.0,
            "candidate": CANDIDATE,
        },
        "neural_baseline_gate": {
            "loss_reductions": neural_advantages,
            "evaluated_baselines": sorted(neural_advantages),
            "required_baselines": list(full_neural_baselines),
            "baseline_set_complete": baseline_set_complete,
            "ahem_lower_than_every_evaluated_baseline": all(
                value > 0.0 for value in neural_advantages.values()
            ),
            "ahem_lower_than_every_baseline": baseline_set_complete
            and all(value > 0.0 for value in neural_advantages.values()),
        },
        "memory_ablation": {
            "local_only_minus_full_weighted_loss": memory_difference,
            "learned_memory_lower_loss": memory_difference > 0.0,
            "candidate": CANDIDATE,
            "ablation": LOCAL_ABLATION,
        },
        "retrieval_ablation": {
            "recency_only_minus_learned_weighted_loss": recency_difference,
            "learned_retrieval_lower_loss": recency_difference > 0.0,
            "candidate": CANDIDATE,
            "ablation": RECENCY_ABLATION,
        },
        "persistence_gate": {
            "persistence_minus_ahem_weighted_loss": persistence_difference,
            "ahem_lower_loss": persistence_difference > 0.0,
        },
        "claim_boundary": "Architecture smoke only; not publication evidence.",
    }
    write_manifest(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=257)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--validation-samples", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--validation-source")
    parser.add_argument("--test-source")
    args = parser.parse_args()
    result = run(
        args.journal,
        args.output,
        context=args.context,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        device=torch.device(args.device),
        seed=args.seed,
        reduced=args.reduced,
        validation_source=args.validation_source,
        test_source=args.test_source,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

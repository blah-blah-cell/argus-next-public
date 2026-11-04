"""Fail-closed training-only development screen for AHEM-v6.1.

This runner never opens the frozen Thursday-22, Friday-23, or CIC-IDS2017
holdouts.  It first measures whether the bounded cache has non-recency oracle
headroom.  Expensive training starts only if that data-only gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .ahem_v6 import (
    AhemV62LocalOnlyNextPacket,
    AhemV62QueryRotationNextPacket,
    AhemV62RecencyNextPacket,
    AhemV62StateTransitionNextPacket,
    AhemV62ValuePermutationNextPacket,
    AhemV6LocalOnlyNextPacket,
    AhemV6Output,
    AhemV6QueryRotationNextPacket,
    AhemV6RecencyNextPacket,
    AhemV6StateTransitionNextPacket,
    AhemV6ValuePermutationNextPacket,
    FlowLaneLstmNextPacket,
    transition_slots,
)
from .flow_v2_train import _source_provenance
from .flow_v3_protocol import (
    CompiledProtocol,
    ProtocolConfig,
    compile_source_disjoint,
    materialize_batch,
)
from .flow_v6_objective import (
    dynamic_loss,
    frozen_utility_distribution,
    headroom,
    training_loss,
)

SCHEMA = "argus.ahem-v6-training-development/v2"
VALIDATION_SOURCE = "Thursday-01-03-2018"
SCREEN_SOURCE = "Friday-02-03-2018"
FORBIDDEN_SOURCES = frozenset(
    {"Thursday-22-02-2018", "Friday-23-02-2018", "CIC-IDS2017"}
)
DEVELOPMENT_SOURCES = frozenset(
    {
        "Wednesday-14-02-2018",
        "Thursday-15-02-2018",
        "Friday-16-02-2018",
        "Tuesday-20-02-2018",
        "Wednesday-21-02-2018",
        "Wednesday-28-02-2018",
        VALIDATION_SOURCE,
        SCREEN_SOURCE,
    }
)
SEEDS = (20260906, 20260907, 20260908)
MODELS = (
    "ahem_v6_state_transition",
    "ahem_v6_local_only",
    "ahem_v6_recency",
    "flow_lane_lstm",
)


@dataclass(frozen=True)
class RunConfig:
    context_packets: int = 10_000
    train_samples: int = 2_048
    validation_samples: int = 512
    screen_samples: int = 512
    protocol_seed: int = 20260906
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    patience: int = 8
    utility_temperature: float = 0.04
    minimum_completed_transitions: int = 2
    sampling_policy: str = "natural"
    retrieval_ranking_weight: float = 0.0
    retrieval_ranking_margin: float = 0.10


@dataclass(frozen=True)
class ExperimentSpec:
    schema: str
    candidate_name: str
    candidate_description: str
    protocol_revision: str
    models: tuple[str, ...]
    references: tuple[str, ...]
    minimum_validation_top1_gain: float | None = None
    maximum_train_to_screen_top1_drop: float | None = None


V61_SPEC = ExperimentSpec(
    schema=SCHEMA,
    candidate_name="ahem_v6_state_transition",
    candidate_description="causal_shared_two_event_state_transition_memory",
    protocol_revision="v6.1_minimum_two_causal_cache_alternatives",
    models=MODELS,
    references=("ahem_v6_local_only", "ahem_v6_recency", "flow_lane_lstm"),
)


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _batches(values: list[Any], size: int, seed: int | None = None) -> list[list[Any]]:
    ordered = list(values)
    if seed is not None:
        random.Random(seed).shuffle(ordered)
    return [ordered[index : index + size] for index in range(0, len(ordered), size)]


def _construct(name: str) -> nn.Module:
    constructors: dict[str, type[nn.Module]] = {
        "ahem_v6_state_transition": AhemV6StateTransitionNextPacket,
        "ahem_v6_local_only": AhemV6LocalOnlyNextPacket,
        "ahem_v6_recency": AhemV6RecencyNextPacket,
        "flow_lane_lstm": FlowLaneLstmNextPacket,
        "ahem_v62_state_transition": AhemV62StateTransitionNextPacket,
        "ahem_v62_local_only": AhemV62LocalOnlyNextPacket,
        "ahem_v62_recency": AhemV62RecencyNextPacket,
    }
    try:
        return constructors[name]()
    except KeyError as error:
        raise ValueError(f"unknown v6 development model: {name}") from error


def _check_development_inventory(journal: Path) -> None:
    """Reject foreign sources before any journal payload is read."""
    paths = sorted(journal.glob("*.arguslr"))
    if not paths:
        raise ValueError("no development journals found")
    for path in paths:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stem not in DEVELOPMENT_SOURCES
        ):
            raise ValueError(f"unapproved development journal: {path.name}")
    if {path.stem for path in paths} != DEVELOPMENT_SOURCES:
        raise ValueError("development journal inventory is incomplete")


def _rng_snapshot() -> dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
        "python": random.getstate(),
    }


def _restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if (
            not torch.cuda.is_available()
            or len(state["torch_cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError("resume CUDA random-state topology changed")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    random.setstate(state["python"])


def _headroom_report(
    protocol: CompiledProtocol, config: RunConfig, device: torch.device
) -> dict[str, Any]:
    successors: list[Tensor] = []
    targets: list[Tensor] = []
    masks: list[Tensor] = []
    for batch in _batches(protocol.coordinates["train"], config.batch_size):
        contexts, target = materialize_batch(
            protocol.partitions["train"], batch, config.context_packets
        )
        _, _, slots = transition_slots(contexts, max_transitions=16)
        successors.append(slots.successor[..., :2].to(device))
        targets.append(target.to(device))
        masks.append(slots.valid.to(device))
    successor = torch.cat(successors)
    target = torch.cat(targets)
    valid = torch.cat(masks)
    measured = asdict(
        headroom(
            successor,
            target,
            valid,
            utility_temperature=config.utility_temperature,
        )
    )
    gates = {
        "relative_oracle_gain_at_least_10pct": measured["relative_oracle_gain"] >= 0.10,
        "oracle_nonrecency_at_least_25pct": measured["oracle_not_recency_rate"] >= 0.25,
        "two_choice_examples_at_least_75pct": measured["informative_rate"] >= 0.75,
        "target_entropy_below_0_90": measured["normalized_target_entropy"] < 0.90,
    }
    return {
        "definition": "raw_observed_successor_dynamic_loss_independent_of_model",
        "partition": "training_sources_only",
        "metrics": measured,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _retrieval_metrics(output: AhemV6Output, target: Tensor) -> dict[str, Tensor]:
    if not output.valid.any(dim=-1).all():
        raise ValueError("retrieval metrics require at least one observed transition")
    relevance, losses = frozen_utility_distribution(
        output.successor_dynamic, target, output.valid
    )
    oracle = losses.argmin(dim=-1)
    weights = output.retrieval_weights.masked_fill(~output.valid, 0.0)
    if (
        not torch.isfinite(weights).all()
        or (weights < 0).any()
        or not torch.allclose(
            weights.sum(dim=-1), torch.ones_like(weights[:, 0]), atol=1e-5
        )
    ):
        raise ValueError("retrieval metrics require finite normalized attention")
    selected = weights.masked_fill(~output.valid, -1.0).argmax(dim=-1)
    ranks = losses.argsort(dim=-1).argsort(dim=-1)
    selected_rank = ranks.gather(1, selected[:, None]).squeeze(1)
    reciprocal_rank = 1.0 / (selected_rank.to(torch.float32) + 1.0)
    oracle_loss = losses.gather(1, oracle[:, None]).squeeze(1)
    selected_loss = losses.gather(1, selected[:, None]).squeeze(1)
    recency_index = output.valid.to(torch.int64).argmax(dim=-1)
    recency_loss = losses.gather(1, recency_index[:, None]).squeeze(1)
    oracle_ties = output.valid & ((losses - oracle_loss[:, None]).abs() <= 1e-8)
    # MRR ranks a relevant item in the predicted order, not the chosen item's
    # rank in oracle order. Stable ties in attention are broken newest first.
    predicted_order = weights.masked_fill(~output.valid, -1.0).argsort(
        dim=-1, descending=True, stable=True
    )
    predicted_ranks = predicted_order.argsort(dim=-1) + 1
    oracle_rank = predicted_ranks.masked_fill(~oracle_ties, weights.size(1) + 1).amin(
        -1
    )
    entropy = -(
        output.retrieval_weights * output.retrieval_weights.clamp_min(1e-8).log()
    ).sum(dim=-1)
    entropy = entropy / output.valid.sum(dim=-1).to(entropy.dtype).log().clamp_min(1.0)
    return {
        "top1": (selected == oracle).to(torch.float32),
        "top1_tie_aware": oracle_ties.gather(1, selected[:, None]).squeeze(1).float(),
        "selected_utility_reciprocal_rank": reciprocal_rank,
        "oracle_reciprocal_rank": 1.0 / oracle_rank.float(),
        "selected_transition_loss": selected_loss,
        "recency_transition_loss": recency_loss,
        "oracle_transition_loss": oracle_loss,
        "selected_oracle_regret": selected_loss - oracle_loss,
        "expected_transition_loss": (
            weights * losses.masked_fill(~output.valid, 0.0)
        ).sum(-1),
        "oracle_tie_count": oracle_ties.sum(-1).float(),
        "weight_entropy": entropy,
        "target_entropy": -(relevance * relevance.clamp_min(1e-8).log()).sum(dim=-1)
        / output.valid.sum(dim=-1).to(relevance.dtype).log().clamp_min(1.0),
        "random_top1": 1.0 / output.valid.sum(dim=-1).to(torch.float32),
    }


def _evaluate(
    model: nn.Module,
    protocol: CompiledProtocol,
    partition: str,
    config: RunConfig,
    device: torch.device,
) -> dict[str, Any]:
    losses: list[Tensor] = []
    retrieval: dict[str, list[Tensor]] = {}
    model.eval()
    with torch.inference_mode():
        for batch in _batches(protocol.coordinates[partition], config.batch_size):
            contexts, target = materialize_batch(
                protocol.partitions[partition], batch, config.context_packets
            )
            contexts = contexts.to(device)
            target = target.to(device)
            if isinstance(model, AhemV6StateTransitionNextPacket):
                output = model.forward_diagnostics(contexts)
                prediction = output.prediction
                if model.uses_transition_memory:
                    for name, values in _retrieval_metrics(output, target).items():
                        retrieval.setdefault(name, []).append(values.cpu())
            else:
                prediction = model(contexts)
            losses.append(dynamic_loss(prediction, target, reduction="none").cpu())
    joined = torch.cat(losses)
    if not torch.isfinite(joined).all():
        raise FloatingPointError("non-finite evaluation loss")
    result: dict[str, Any] = {
        "samples": len(joined),
        "weighted_smooth_l1": float(joined.mean()),
        "per_sample_weighted_smooth_l1": joined.tolist(),
    }
    if retrieval:
        result["retrieval_metric_schema"] = "argus.retrieval-diagnostics/v2"
        result["retrieval"] = {
            name: float(torch.cat(values).mean()) for name, values in retrieval.items()
        }
        result["per_sample_retrieval"] = {
            name: torch.cat(values).tolist() for name, values in retrieval.items()
        }
    return result


def _train_one(
    name: str,
    seed: int,
    protocol: CompiledProtocol,
    config: RunConfig,
    spec: ExperimentSpec,
    device: torch.device,
    output: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = _construct(name).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state = deepcopy(model.state_dict())
    best_validation = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    latest = output / "latest.pt"
    if not latest.is_file() and any(
        (output / filename).exists() for filename in ("best.pt", "metrics.json")
    ):
        raise ValueError(
            "existing artifacts lack a resumable latest.pt; use a fresh output"
        )
    binding = _canonical(
        {
            "config": asdict(config),
            "experiment": asdict(spec),
            "protocol": protocol.manifest,
        }
    )
    if latest.is_file():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
        if checkpoint.get("model") != name or checkpoint.get("seed") != seed:
            raise ValueError("resume checkpoint identity changed")
        if (
            checkpoint.get("training_binding") != binding
            or "rng_state" not in checkpoint
        ):
            raise ValueError(
                "resume checkpoint lacks matching protocol/config/RNG binding"
            )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_state = checkpoint["best_state_dict"]
        best_validation = float(checkpoint["best_validation"])
        best_epoch = int(checkpoint["best_epoch"])
        stale = int(checkpoint["stale_epochs"])
        history = list(checkpoint["history"])
        _restore_rng(checkpoint["rng_state"])
    for epoch in range(len(history) + 1, config.epochs + 1):
        if stale >= config.patience:
            break
        model.train()
        batch_losses = []
        for batch in _batches(
            protocol.coordinates["train"],
            config.batch_size,
            seed + epoch * 1_000_003,
        ):
            contexts, target = materialize_batch(
                protocol.partitions["train"], batch, config.context_packets
            )
            contexts = contexts.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if name == spec.candidate_name:
                assert isinstance(model, AhemV6StateTransitionNextPacket)
                loss, _ = training_loss(
                    model.forward_diagnostics(contexts),
                    target,
                    utility_temperature=config.utility_temperature,
                    ranking_weight=config.retrieval_ranking_weight,
                    ranking_margin=config.retrieval_ranking_margin,
                )
            else:
                loss = dynamic_loss(model(contexts), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite v6 development loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        validation = _evaluate(model, protocol, "validation", config, device)
        value = float(validation["weighted_smooth_l1"])
        improved = value < best_validation
        if improved:
            best_validation = value
            best_epoch = epoch
            best_state = deepcopy(
                {
                    key: tensor.detach().cpu()
                    for key, tensor in model.state_dict().items()
                }
            )
            stale = 0
        else:
            stale += 1
        record = {
            "epoch": epoch,
            "train_loss": sum(batch_losses) / len(batch_losses),
            "validation_weighted_smooth_l1": value,
            "improved": improved,
            "created_utc": datetime.now(UTC).isoformat(),
        }
        history.append(record)
        checkpoint = {
            "schema": spec.schema,
            "training_binding": binding,
            "rng_state": _rng_snapshot(),
            "model": name,
            "seed": seed,
            "epoch": epoch,
            "state_dict": {
                key: tensor.detach().cpu() for key, tensor in model.state_dict().items()
            },
            "best_state_dict": best_state,
            "best_validation": best_validation,
            "best_epoch": best_epoch,
            "stale_epochs": stale,
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        }
        _atomic_checkpoint(latest, checkpoint)
        if improved:
            _atomic_checkpoint(output / "best.pt", checkpoint)
        print(json.dumps({"model": name, "seed": seed, **record}), flush=True)
    model.load_state_dict(best_state, strict=True)
    model.to(device).eval()
    # latest.pt is the resumable optimizer state; publish best weights again
    # even if interruption occurred between writing latest.pt and best.pt.
    _atomic_checkpoint(
        output / "best.pt",
        {
            "schema": spec.schema,
            "model": name,
            "seed": seed,
            "epoch": best_epoch,
            "best_epoch": best_epoch,
            "state_dict": best_state,
            "best_state_dict": best_state,
            "best_validation": best_validation,
            "training_binding": binding,
        },
    )
    result = {
        "model": name,
        "seed": seed,
        "parameters": sum(value.numel() for value in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_weighted_smooth_l1": best_validation,
        "train": _evaluate(model, protocol, "train", config, device),
        "validation": _evaluate(model, protocol, "validation", config, device),
        "screen": _evaluate(model, protocol, "test", config, device),
    }
    _atomic_json(output / "metrics.json", result)
    return model, result


def _mechanism_controls(
    candidate: AhemV6StateTransitionNextPacket,
    protocol: CompiledProtocol,
    config: RunConfig,
    device: torch.device,
) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    constructors = (
        (
            ("query_rotation", AhemV62QueryRotationNextPacket),
            ("value_permutation", AhemV62ValuePermutationNextPacket),
        )
        if isinstance(candidate, AhemV62StateTransitionNextPacket)
        else (
            ("query_rotation", AhemV6QueryRotationNextPacket),
            ("value_permutation", AhemV6ValuePermutationNextPacket),
        )
    )
    for name, constructor in constructors:
        control = constructor().to(device)
        control.load_state_dict(candidate.state_dict(), strict=True)
        controls[name] = _evaluate(control, protocol, "test", config, device)
    return controls


def run_experiment(
    args: argparse.Namespace,
    *,
    config: RunConfig,
    spec: ExperimentSpec,
) -> dict[str, Any]:
    source = _source_provenance()
    if source.get("available") is not True or source.get("git_dirty") is not False:
        raise RuntimeError("v6 development requires a clean Git snapshot")
    if args.output.exists() and not (args.output / "run.json").is_file():
        allowed_supervisor_files = {"RUN_STATE.json", "JOURNALS.json", "supervisor.log"}
        unexpected = {
            path.name
            for path in args.output.iterdir()
            if path.name not in allowed_supervisor_files
        }
        if unexpected:
            raise ValueError("occupied v6 output lacks a compatible run manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen v6 development run requires CUDA")
    _check_development_inventory(args.journal)
    protocol = compile_source_disjoint(
        args.journal,
        VALIDATION_SOURCE,
        SCREEN_SOURCE,
        ProtocolConfig(
            context_packets=config.context_packets,
            train_samples=config.train_samples,
            validation_samples=config.validation_samples,
            test_samples=config.screen_samples,
            seed=config.protocol_seed,
            require_memory_match=True,
            sampling_policy=config.sampling_policy,
            minimum_completed_transitions=config.minimum_completed_transitions,
        ),
    )
    split = protocol.manifest.get("split", {})
    if FORBIDDEN_SOURCES & set(split.get("train_sources", [])):
        raise RuntimeError("v6 development exposed a forbidden holdout")
    if (
        split.get("validation_source") != VALIDATION_SOURCE
        or split.get("test_source") != SCREEN_SOURCE
    ):
        raise RuntimeError("v6 development split changed")
    run_manifest = {
        "schema": spec.schema,
        "created_utc": datetime.now(UTC).isoformat(),
        "study_role": "training_only_development",
        "publication_eligible": False,
        "task": "next_same_flow_dynamic_event_forecast",
        "candidate": spec.candidate_description,
        "protocol_revision": spec.protocol_revision,
        "implementation_audit": "2026-09-06-completion-audit-v1",
        "experiment_spec": asdict(spec),
        "config": asdict(config),
        "config_sha256": _canonical(asdict(config)),
        "source_provenance": source,
        "protocol": protocol.manifest,
    }
    run_path = args.output / "run.json"
    if run_path.is_file():
        existing = json.loads(run_path.read_text(encoding="utf-8"))
        for field in (
            "schema",
            "config_sha256",
            "source_provenance",
            "protocol",
            "experiment_spec",
        ):
            if existing.get(field) != run_manifest.get(field):
                raise ValueError(f"resume changed immutable run field: {field}")
        run_manifest = existing
    else:
        _atomic_json(run_path, run_manifest)
    preflight = _headroom_report(protocol, config, device)
    _atomic_json(args.output / "headroom.json", preflight)
    if not preflight["passed"]:
        result = {
            "schema": spec.schema,
            "decision": "reject_before_training_no_retrieval_headroom",
            "headroom": preflight,
        }
        _atomic_json(args.output / "RESULT.json", result)
        _atomic_json(args.output / "COMPLETED.json", {"complete": True, **result})
        return result

    all_results: dict[str, list[dict[str, Any]]] = {name: [] for name in spec.models}
    mechanism: list[dict[str, Any]] = []
    for seed in SEEDS:
        trained: dict[str, nn.Module] = {}
        for name in spec.models:
            model_output = args.output / "models" / str(seed) / name
            # Always validate checkpoint binding and regenerate metrics. A
            # metrics.json plus best.pt alone is not evidence of identity.
            model, metrics = _train_one(
                name, seed, protocol, config, spec, device, model_output
            )
            trained[name] = model
            all_results[name].append(metrics)
        candidate = trained[spec.candidate_name]
        assert isinstance(candidate, AhemV6StateTransitionNextPacket)
        mechanism.append(_mechanism_controls(candidate, protocol, config, device))

    candidate_losses = [
        value["screen"]["weighted_smooth_l1"]
        for value in all_results[spec.candidate_name]
    ]
    comparisons = {}
    gates = {}
    for reference in spec.references:
        reference_losses = [
            value["screen"]["weighted_smooth_l1"] for value in all_results[reference]
        ]
        improvements = [
            (baseline - candidate) / max(baseline, 1e-8)
            for candidate, baseline in zip(
                candidate_losses, reference_losses, strict=True
            )
        ]
        comparisons[reference] = {
            "per_seed_relative_loss_reduction": improvements,
            "mean_relative_loss_reduction": sum(improvements) / len(improvements),
        }
        gates[f"beats_{reference}_by_5pct_mean"] = (
            comparisons[reference]["mean_relative_loss_reduction"] >= 0.05
        )
    query_harm = [
        value["query_rotation"]["weighted_smooth_l1"] - candidate
        for value, candidate in zip(mechanism, candidate_losses, strict=True)
    ]
    gates["query_rotation_hurts_every_seed"] = all(value > 0.0 for value in query_harm)
    learned_top1 = [
        value["screen"]["retrieval"]["top1"]
        for value in all_results[spec.candidate_name]
    ]
    recency_top1 = [
        value["screen"]["retrieval"]["top1"]
        for value in all_results[spec.references[1]]
    ]
    gates["retrieval_top1_exceeds_recency_by_5pp"] = (
        sum(learned_top1) / len(learned_top1) - sum(recency_top1) / len(recency_top1)
        >= 0.05
    )
    if spec.minimum_validation_top1_gain is not None:
        learned_validation_top1 = [
            value["validation"]["retrieval"]["top1"]
            for value in all_results[spec.candidate_name]
        ]
        recency_validation_top1 = [
            value["validation"]["retrieval"]["top1"]
            for value in all_results[spec.references[1]]
        ]
        gates["retrieval_validation_top1_gain"] = (
            sum(learned_validation_top1) / len(learned_validation_top1)
            - sum(recency_validation_top1) / len(recency_validation_top1)
            >= spec.minimum_validation_top1_gain
        )
    if spec.maximum_train_to_screen_top1_drop is not None:
        learned_train_top1 = [
            value["train"]["retrieval"]["top1"]
            for value in all_results[spec.candidate_name]
        ]
        gates["retrieval_train_to_screen_top1_drop_bounded"] = (
            sum(learned_train_top1) / len(learned_train_top1)
            - sum(learned_top1) / len(learned_top1)
            <= spec.maximum_train_to_screen_top1_drop
        )
    result = {
        "schema": spec.schema,
        "decision": "candidate_advances_to_new_preregistration"
        if all(gates.values())
        else "reject_or_revise_without_holdout_access",
        "headroom": preflight,
        "models": all_results,
        "mechanism_controls": mechanism,
        "comparisons": comparisons,
        "gates": gates,
        "forbidden_holdouts_accessed": False,
    }
    _atomic_json(args.output / "RESULT.json", result)
    _atomic_json(args.output / "COMPLETED.json", {"complete": True, **result})
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    return run_experiment(args, config=RunConfig(), spec=V61_SPEC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()

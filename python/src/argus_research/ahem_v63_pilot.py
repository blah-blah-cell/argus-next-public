"""Non-promotional real-data pilot for the AHEM-v6.3 reader.

The pilot freezes a hash-verified v6.1 runtime artifact, trains only the v6.3
reader on chronological local ARGUSLR2 records, and reports the exact controls
needed to decide whether a larger source-disjoint run is worth buying.  Local
results are never promotion or publication evidence.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter
from typing import Any

import torch
from torch import Tensor, nn

from .ahem_v63 import (
    AhemV63Config,
    AhemV63FrozenBackbone,
    AhemV63Output,
    AhemV63QueryRotation,
    AhemV63ValuePermutation,
    ahem_v63_loss,
)
from .ahem_v63_smoke import _load_backbone
from .flow_v3_protocol import (
    ProtocolConfig,
    compile_local_development,
    compile_source_disjoint,
    materialize_batch,
)
from .flow_v6_objective import dynamic_loss


PILOT_SCHEMA = "argus.ahem-v6.3-local-pilot/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _journal_inventory(
    directory: Path, *, minimum_files: int = 5
) -> list[dict[str, object]]:
    if not directory.is_dir():
        raise ValueError("v6.3 pilot journal must be an ARGUSLR2 directory")
    files = sorted(directory.glob("*.v2.arguslr"))
    if len(files) < minimum_files:
        raise ValueError(
            f"v6.3 pilot requires at least {minimum_files} ARGUSLR2 journal files"
        )
    return [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]


def _materialize(protocol, partition: str, context_packets: int) -> tuple[Tensor, Tensor]:
    return materialize_batch(
        protocol.partitions[partition],
        protocol.coordinates[partition],
        context_packets,
    )


def _batches(count: int, batch_size: int, seed: int) -> list[list[int]]:
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    return [
        indices[start : start + batch_size]
        for start in range(0, count, batch_size)
    ]


def _forward_metrics(output: AhemV63Output, target: Tensor) -> dict[str, float]:
    sample_loss = dynamic_loss(output.prediction, target, reduction="none")
    backbone_loss = dynamic_loss(
        output.backbone_prediction, target, reduction="none"
    )
    candidate_costs = dynamic_loss(
        output.candidates,
        target[:, None].expand_as(output.candidates),
        reduction="none",
    ).masked_fill(~output.valid, torch.inf)
    selected = output.weights.argmax(-1)
    oracle = candidate_costs.argmin(-1)
    selected_cost = candidate_costs.gather(1, selected[:, None]).squeeze(1)
    oracle_cost = candidate_costs.amin(-1)
    expected_cost = (
        output.weights * candidate_costs.masked_fill(~output.valid, 0)
    ).sum(-1)
    tie_aware = selected_cost <= oracle_cost + 1e-7
    safe_weights = output.weights.clamp_min(1e-12)
    entropy = -(output.weights * safe_weights.log()).sum(-1)
    entropy_scale = output.valid.sum(-1).to(output.weights.dtype).log().clamp_min(1.0)
    return {
        "forecast_loss": float(sample_loss.mean()),
        "forecast_p95": float(torch.quantile(sample_loss, 0.95)),
        "backbone_loss": float(backbone_loss.mean()),
        "relative_loss_reduction_vs_backbone": float(
            (backbone_loss.mean() - sample_loss.mean())
            / backbone_loss.mean().clamp_min(1e-12)
        ),
        "exact_oracle_top1": float((selected == oracle).float().mean()),
        "tie_aware_oracle_top1": float(tie_aware.float().mean()),
        "hard_selected_proposal_loss": float(selected_cost.mean()),
        "soft_expected_proposal_loss": float(expected_cost.mean()),
        "oracle_proposal_loss": float(oracle_cost.mean()),
        "null_top1_rate": float((selected == 0).float().mean()),
        "mean_null_mass": float(output.weights[:, 0].mean()),
        "mean_non_null_mass": float(output.weights[:, 1:].sum(-1).mean()),
        "mean_maximum_weight": float(output.weights.max(-1).values.mean()),
        "mean_effective_candidates": float(entropy.exp().mean()),
        "mean_normalized_selection_entropy": float((entropy / entropy_scale).mean()),
        "mean_absolute_correction": float(
            (output.prediction - output.backbone_prediction).abs().mean()
        ),
    }


def _evaluate(
    model: AhemV63FrozenBackbone,
    contexts: Tensor,
    target: Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        output = model.forward_diagnostics(contexts.to(device))
        return _forward_metrics(output, target.to(device))


def _variant(
    source: AhemV63FrozenBackbone,
    kind: str,
    device: torch.device,
) -> AhemV63FrozenBackbone:
    config = source.config
    model_type: type[AhemV63FrozenBackbone] = AhemV63FrozenBackbone
    if kind == "recency":
        config = replace(config, reader="recency")
    elif kind == "none":
        config = replace(config, reader="none")
    elif kind == "query_rotation":
        model_type = AhemV63QueryRotation
    elif kind == "value_permutation":
        model_type = AhemV63ValuePermutation
    elif kind != "learned":
        raise ValueError(f"unknown v6.3 pilot variant: {kind}")
    variant = model_type(deepcopy(source.backbone), config).to(device)
    variant.load_state_dict(source.state_dict(), strict=True)
    return variant.eval()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
    return _sha256(path)


def run_pilot(
    journal: Path,
    backbone_path: Path,
    output: Path,
    *,
    context_packets: int = 512,
    samples: int = 16,
    epochs: int = 80,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    patience: int = 12,
    seed: int = 20260913,
    device: torch.device | None = None,
    validation_source: str | None = None,
    test_source: str | None = None,
) -> dict[str, Any]:
    if context_packets < 257:
        raise ValueError("v6.3 pilot context must be at least 257 packets")
    if min(samples, epochs, batch_size, patience) <= 0:
        raise ValueError("v6.3 pilot counts must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("v6.3 pilot learning rate must be finite and positive")
    selected_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    if (validation_source is None) != (test_source is None):
        raise ValueError(
            "validation_source and test_source must be supplied together"
        )
    source_disjoint = validation_source is not None
    inventory_before = _journal_inventory(
        journal, minimum_files=3 if source_disjoint else 5
    )
    torch.manual_seed(seed)
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    backbone, backbone_hash = _load_backbone(backbone_path)
    protocol_config = ProtocolConfig(
        context_packets=context_packets,
        local_packets=backbone.local_packets,
        train_samples=samples,
        validation_samples=samples,
        test_samples=samples,
        seed=seed,
        require_memory_match=True,
        sampling_policy="natural",
        minimum_completed_transitions=2,
    )
    protocol = (
        compile_source_disjoint(
            journal,
            validation_source=validation_source,
            test_source=test_source,
            config=protocol_config,
        )
        if validation_source is not None and test_source is not None
        else compile_local_development(journal, protocol_config)
    )
    tensors = {
        partition: _materialize(protocol, partition, context_packets)
        for partition in ("train", "validation", "test")
    }
    config = AhemV63Config(horizon=context_packets)
    candidate = AhemV63FrozenBackbone(backbone, config).to(selected_device).train()
    trainable = [parameter for parameter in candidate.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=5e-4
    )
    initial = _evaluate(candidate, *tensors["validation"], selected_device)
    if initial["mean_absolute_correction"] != 0:
        raise RuntimeError("v6.3 pilot did not begin at the exact frozen backbone")

    best_state = deepcopy(candidate.state_dict())
    best_validation = initial["forecast_loss"]
    best_epoch = -1
    stale = 0
    history: list[dict[str, float | int]] = []
    started = perf_counter()
    train_contexts, train_target = tensors["train"]
    for epoch in range(epochs):
        candidate.train()
        running_loss = 0.0
        seen = 0
        for indices in _batches(len(train_contexts), batch_size, seed + epoch):
            batch_context = train_contexts[indices].to(selected_device)
            batch_target = train_target[indices].to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            diagnostics = candidate.forward_diagnostics(batch_context)
            loss, _ = ahem_v63_loss(diagnostics, batch_target, config)
            if not torch.isfinite(loss):
                raise RuntimeError("v6.3 pilot produced a non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            running_loss += float(loss.detach()) * len(indices)
            seen += len(indices)
        validation = _evaluate(candidate, *tensors["validation"], selected_device)
        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch + 1,
            "training_objective": running_loss / seen,
            "validation_forecast_loss": validation["forecast_loss"],
            "validation_reduction_vs_backbone": validation[
                "relative_loss_reduction_vs_backbone"
            ],
        }
        history.append(epoch_metrics)
        print(
            json.dumps(
                {
                    "event": "epoch_complete",
                    "seed": seed,
                    **epoch_metrics,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if validation["forecast_loss"] < best_validation - 1e-8:
            best_validation = validation["forecast_loss"]
            best_epoch = epoch + 1
            best_state = deepcopy(candidate.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    candidate.load_state_dict(best_state, strict=True)
    candidate.eval()

    evaluations = {
        name: _evaluate(_variant(candidate, name, selected_device), *tensors["test"], selected_device)
        for name in (
            "learned",
            "recency",
            "none",
            "query_rotation",
            "value_permutation",
        )
    }
    learned = evaluations["learned"]
    recency = evaluations["recency"]
    journal_unchanged = inventory_before == _journal_inventory(
        journal, minimum_files=3 if source_disjoint else 5
    )
    if not journal_unchanged:
        raise RuntimeError("ARGUSLR2 input changed during the v6.3 pilot")

    checkpoint_path = output / "reader-best.pt"
    checkpoint_hash = _atomic_checkpoint(
        checkpoint_path,
        {
            "schema": PILOT_SCHEMA,
            "study_role": (
                "source_disjoint_exploratory_vm_training"
                if source_disjoint
                else "local_engineering_pilot_only"
            ),
            "promotion_eligible": False,
            "security_detector": False,
            "config": asdict(config),
            "state_dict": candidate.state_dict(),
            "backbone_sha256": backbone_hash,
            "protocol_coordinate_sha256": protocol.manifest["coordinate_sha256"],
            "best_epoch": best_epoch,
        },
    )
    report: dict[str, Any] = {
        "schema": PILOT_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "study_role": (
            "source_disjoint_exploratory_vm_training"
            if source_disjoint
            else "local_engineering_pilot_only"
        ),
        "publication_evidence": False,
        "promotion_eligible": False,
        "claim_boundary": (
            "Source-disjoint reused development traffic; this exploratory run can "
            "select or reject a design but cannot support confirmatory superiority "
            "or deployment claims."
            if source_disjoint
            else "Reused single-device traffic; this pilot can reject a bad design "
            "but cannot support superiority or deployment claims."
        ),
        "device": {
            "type": selected_device.type,
            "name": torch.cuda.get_device_name(selected_device)
            if selected_device.type == "cuda"
            else "CPU",
            "torch": torch.__version__,
        },
        "data": {
            "journal_directory": str(journal.resolve()),
            "immutable_during_run": journal_unchanged,
            "inventory": inventory_before,
            "protocol": protocol.manifest,
        },
        "model": {
            "config": asdict(config),
            "backbone_sha256": backbone_hash,
            "backbone_parameters": sum(
                value.numel() for value in candidate.backbone.parameters()
            ),
            "reader_trainable_parameters": sum(
                value.numel() for value in candidate.parameters() if value.requires_grad
            ),
            "total_parameters": sum(value.numel() for value in candidate.parameters()),
        },
        "training": {
            "seed": seed,
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "best_epoch": best_epoch,
            "seconds": perf_counter() - started,
            "history": history,
        },
        "validation_initial": initial,
        "test": evaluations,
        "local_decision_screen": {
            "forecast_improves_over_frozen_backbone": learned["forecast_loss"]
            < learned["backbone_loss"],
            "learned_forecast_beats_same_weights_recency": learned["forecast_loss"]
            < recency["forecast_loss"],
            "learned_hard_retrieval_beats_recency": learned[
                "hard_selected_proposal_loss"
            ]
            < recency["hard_selected_proposal_loss"],
            "learned_soft_retrieval_beats_recency": learned[
                "soft_expected_proposal_loss"
            ]
            < recency["soft_expected_proposal_loss"],
            "query_rotation_hurts": evaluations["query_rotation"]["forecast_loss"]
            > learned["forecast_loss"],
            "value_permutation_hurts": evaluations["value_permutation"][
                "forecast_loss"
            ]
            > learned["forecast_loss"],
            "selection_is_not_diffuse": learned["mean_effective_candidates"]
            < 4.0,
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_hash,
        },
    }
    _atomic_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-packets", type=int, default=512)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--validation-source")
    parser.add_argument("--test-source")
    args = parser.parse_args()
    device = None if args.device == "auto" else torch.device(args.device)
    report = run_pilot(
        args.journal,
        args.backbone,
        args.output,
        context_packets=args.context_packets,
        samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
        device=device,
        validation_source=args.validation_source,
        test_source=args.test_source,
    )
    print(json.dumps(report["local_decision_screen"], sort_keys=True))


if __name__ == "__main__":
    main()

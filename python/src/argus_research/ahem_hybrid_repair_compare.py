"""Bounded repair comparison on real source-disjoint development journals.

This is a post-result development experiment. All model checkpoints are locked
before the reused screen is materialized. It cannot admit a model to the app or
support a confirmatory claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import torch
from torch import nn

from .ahem_hybrid import AhemHybrid, HybridConfig, hybrid_loss
from .ahem_hybrid_compare import matched_hidden, parameter_count
from .ahem_hybrid_repair import AhemHybridRepair, repair_loss
from .ahem_pro_pilot import (
    ALLOWED_SOURCES,
    SCREEN_SOURCE,
    VALIDATION_SOURCE,
    _atomic_checkpoint,
    _atomic_json,
    _baseline_metrics,
    _batches,
    _inventory,
)
from .flow_v3_protocol import ProtocolConfig, compile_source_disjoint, materialize_batch
from .flow_v6_objective import dynamic_loss
from .pro_baselines import PersistenceResidualLstm

SCHEMA = "argus.ahem-hybrid-repair-comparison/v1"
SEEDS = (20260911, 20260912, 20260913)
MODEL_NAMES = (
    "repair_flow",
    "repair_recency",
    "repair_local",
    "repair_global",
    "original_hybrid",
    "residual_lstm_matched",
)


@dataclass(frozen=True)
class RepairCompareConfig:
    context_packets: int = 10_000
    train_samples: int = 512
    validation_samples: int = 128
    screen_samples: int = 128
    epochs: int = 12
    batch_size: int = 16
    learning_rate: float = 2e-4

    def validate(self):
        for name, value in asdict(self).items():
            if name != "learning_rate" and (type(value) is not int or value <= 0):
                raise ValueError("comparison integer settings must be positive")
        if self.context_packets not in (10_000, 30_000):
            raise ValueError("use a declared 10k or 30k context")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("invalid learning rate")


def construct(name: str, horizon: int):
    config = HybridConfig(horizon=horizon, transport=0.0)
    constructors = {
        "repair_flow": lambda: AhemHybridRepair(config),
        "repair_recency": lambda: AhemHybridRepair(
            HybridConfig(horizon=horizon, transport=0.0, reader="recency")
        ),
        "repair_local": lambda: AhemHybridRepair(
            HybridConfig(horizon=horizon, transport=0.0, reader="none")
        ),
        "repair_global": lambda: AhemHybridRepair(config, flow_local=False),
        "original_hybrid": lambda: AhemHybrid(config),
        "residual_lstm_matched": lambda: PersistenceResidualLstm(
            hidden_dim=matched_hidden()
        ),
    }
    if name not in constructors:
        raise ValueError(f"unknown repair comparison model: {name}")
    return constructors[name]()


def objective(model, x, y):
    if isinstance(model, AhemHybridRepair):
        return repair_loss(model.forward_diagnostics(x), y)[0]
    if isinstance(model, AhemHybrid):
        return hybrid_loss(model.forward_diagnostics(x), y)[0]
    return dynamic_loss(model(x), y)


def evaluate(model, values, config, device):
    model.eval()
    losses, retrieval = [], {}
    x, y = values
    with torch.inference_mode():
        for index in _batches(len(x), config.batch_size):
            bx, by = x[index].to(device), y[index].to(device)
            if isinstance(model, AhemHybridRepair):
                output = model.forward_diagnostics(bx)
                prediction = output.prediction
                costs = dynamic_loss(
                    output.candidates,
                    by[:, None].expand_as(output.candidates),
                    reduction="none",
                ).masked_fill(~output.valid, torch.inf)
                chosen = output.weights.argmax(-1, keepdim=True)
                selected = costs.gather(1, chosen)[:, 0]
                oracle = costs.amin(-1)
                diagnostics = {
                    "native_selected_regret": selected - oracle,
                    "native_tie_aware": (selected <= oracle + 1e-8).float(),
                    "abstention_fraction": (chosen[:, 0] == 0).float(),
                    "valid_memory_slots": output.valid[:, 1:].sum(-1).float(),
                }
                eligible = output.valid[:, 1:].any(-1)
                if eligible.any():
                    memory_costs = costs[:, 1:]
                    memory_choice = output.weights[:, 1:].argmax(-1, keepdim=True)
                    memory_selected = memory_costs.gather(1, memory_choice)[:, 0]
                    memory_oracle = memory_costs.amin(-1)
                    diagnostics.update(
                        {
                            "memory_only_selected_regret": (
                                memory_selected[eligible] - memory_oracle[eligible]
                            ),
                            "memory_only_tie_aware": (
                                memory_selected[eligible]
                                <= memory_oracle[eligible] + 1e-8
                            ).float(),
                            "recency_memory_regret": (
                                memory_costs[eligible, 0] - memory_oracle[eligible]
                            ),
                        }
                    )
                for key, value in diagnostics.items():
                    retrieval.setdefault(key, []).append(value.cpu())
            else:
                prediction = model(bx)
            losses.append(dynamic_loss(prediction, by, reduction="none").cpu())
    losses = torch.cat(losses)
    if not torch.isfinite(losses).all():
        raise FloatingPointError("non-finite repair evaluation")
    result = {
        "loss": float(losses.mean()),
        "samples": len(losses),
        "per_sample_loss": losses.tolist(),
    }
    if retrieval:
        result["retrieval"] = {
            key: {
                "mean": float(torch.cat(values).mean()),
                "samples": sum(len(value) for value in values),
            }
            for key, values in retrieval.items()
        }
    return result


def train_one(name, seed, training, validation, config, device, directory, binding):
    """Train without receiving screen tensors."""
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = construct(name, config.context_packets).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-4
    )
    initial = evaluate(model, validation, config, device)["loss"]
    best, best_epoch = initial, 0
    history = [{"epoch": 0, "validation_loss": initial}]

    def save():
        _atomic_checkpoint(
            directory / "best.pt",
            {
                "schema": SCHEMA,
                "run_binding": binding,
                "name": name,
                "seed": seed,
                "best_epoch": best_epoch,
                "trained": best_epoch > 0,
                "security_detector": False,
                "config": asdict(config),
                "state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            },
        )

    save()
    started = time.perf_counter()
    x, y = training
    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        for index in _batches(len(x), config.batch_size, seed=seed + epoch * 1_000_003):
            bx, by = x[index].to(device), y[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, bx, by)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite repair training loss: {name}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(index)
        value = evaluate(model, validation, config, device)["loss"]
        history.append(
            {
                "epoch": epoch,
                "validation_loss": value,
                "train_objective": total / len(x),
            }
        )
        if value < best:
            best, best_epoch = value, epoch
            save()
        _atomic_json(
            directory / "progress.json",
            {
                "schema": SCHEMA,
                "name": name,
                "seed": seed,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_loss": best,
                "complete": False,
            },
        )
    row = {
        "name": name,
        "seed": seed,
        "parameters": parameter_count(model),
        "initial_validation_loss": initial,
        "validation_loss": best,
        "best_epoch": best_epoch,
        "elapsed_training_validation_seconds": time.perf_counter() - started,
        "history": history,
        "checkpoint_sha256": hashlib.sha256(
            (directory / "best.pt").read_bytes()
        ).hexdigest(),
    }
    _atomic_json(directory / "metrics.json", row)
    _atomic_json(
        directory / "progress.json",
        {
            "schema": SCHEMA,
            "name": name,
            "seed": seed,
            "epoch": config.epochs,
            "best_epoch": best_epoch,
            "best_validation_loss": best,
            "complete": True,
        },
    )
    return row


def advance_decision(results):
    """Apply the frozen development gates without dropping seeds or ties."""
    by_name = {
        name: sorted(
            (row for row in results if row["name"] == name),
            key=lambda row: row["seed"],
        )
        for name in MODEL_NAMES
    }
    expected_seeds = list(SEEDS)
    if any(
        [row["seed"] for row in rows] != expected_seeds for rows in by_name.values()
    ):
        raise ValueError("advance decision requires every declared model and seed")
    if any("screen" not in row for rows in by_name.values() for row in rows):
        raise ValueError("advance decision requires locked screen results")
    means = {
        name: sum(row["screen"]["loss"] for row in rows) / len(rows)
        for name, rows in by_name.items()
    }
    repair = by_name["repair_flow"]

    def beats_every_seed(other):
        return all(
            candidate["screen"]["loss"] < comparator["screen"]["loss"]
            for candidate, comparator in zip(repair, by_name[other], strict=True)
        )

    gates = {
        "repair_flow_lower_mean_screen_loss_than_every_comparator": all(
            means["repair_flow"] < means[name]
            for name in MODEL_NAMES
            if name != "repair_flow"
        ),
        "repair_flow_beats_original_and_lstm_each_seed": all(
            beats_every_seed(name)
            for name in ("original_hybrid", "residual_lstm_matched")
        ),
        "repair_flow_beats_recency_and_local_each_seed": all(
            beats_every_seed(name) for name in ("repair_recency", "repair_local")
        ),
        "all_selected_checkpoints_trained": all(
            row["best_epoch"] > 0 for rows in by_name.values() for row in rows
        ),
    }
    return {
        "advance": all(gates.values()),
        "gates": gates,
        "mean_screen_loss": means,
        "tie_policy": "strict loss inequality; a tie fails",
        "scope": "development repair only; never publication or app admission",
    }


def coordinate_receipt(protocol):
    return {
        part: [
            {
                **asdict(coordinate),
                "source_file": protocol.partitions[part][
                    coordinate.record_index
                ].source_file,
                "segment_id": protocol.partitions[part][
                    coordinate.record_index
                ].segment_id,
                "flow_id": int(
                    protocol.partitions[part][coordinate.record_index].flow_ids[
                        coordinate.anchor_index
                    ]
                ),
            }
            for coordinate in coordinates
        ]
        for part, coordinates in protocol.coordinates.items()
    }


def load_completed_cell(directory, name, seed, config, binding):
    """Return a hash- and state-validated completed cell, otherwise no cell."""
    if not directory.exists():
        return None
    if directory.is_symlink():
        raise ValueError("repair cell directory cannot be a symlink")
    required = [
        directory / value for value in ("best.pt", "metrics.json", "progress.json")
    ]
    if not all(path.is_file() and not path.is_symlink() for path in required):
        return None
    progress = json.loads((directory / "progress.json").read_text())
    row = json.loads((directory / "metrics.json").read_text())
    if progress.get("complete") is not True:
        return None
    checkpoint_path = directory / "best.pt"
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    expected_row = {"name": name, "seed": seed, "checkpoint_sha256": digest}
    if any(row.get(key) != value for key, value in expected_row.items()):
        raise ValueError("completed repair cell metrics/hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_checkpoint = {
        "schema": SCHEMA,
        "run_binding": binding,
        "name": name,
        "seed": seed,
        "best_epoch": row.get("best_epoch"),
        "trained": row.get("best_epoch", 0) > 0,
        "config": asdict(config),
    }
    if any(checkpoint.get(key) != value for key, value in expected_checkpoint.items()):
        raise ValueError("completed repair checkpoint metadata mismatch")
    model = construct(name, config.context_packets)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("completed repair checkpoint has non-finite state")
    return {key: value for key, value in row.items() if key != "screen"}


def clear_incomplete_cell(directory, output):
    if not directory.exists():
        return
    resolved, root = directory.resolve(), output.resolve()
    if resolved.parent.parent != root or directory.is_symlink():
        raise ValueError("refusing unsafe incomplete-cell cleanup")
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise ValueError("refusing incomplete repair cell containing a symlink")
    shutil.rmtree(directory)


def validated_sources(journal):
    """Validate the exact inventory; the validator intentionally returns None."""
    _inventory(journal, ALLOWED_SOURCES)
    return set(ALLOWED_SOURCES)


def run(journal: Path, output: Path, config: RepairCompareConfig, *, resume=False):
    config.validate()
    if output.exists() and not resume:
        raise FileExistsError("repair comparison output already exists")
    sources = validated_sources(journal)
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
            seed=SEEDS[0],
            sampling_policy="natural",
            require_memory_match=True,
            minimum_completed_transitions=2,
        ),
    )
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    immutable_manifest = {
        "schema": SCHEMA,
        "study_role": "post_result_development_repair",
        "publication_eligible": False,
        "app_admission": False,
        "config": asdict(config),
        "seeds": list(SEEDS),
        "models": list(MODEL_NAMES),
        "protocol": protocol.manifest,
        "sources": sorted(sources),
        "research_source_sha256": digest.hexdigest(),
        "selection": "per_cell_best_validation_epoch_including_epoch_zero",
        "screen_policy": "lock_all_checkpoint_hashes_before_materialization",
        "runtime": {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    coordinates = coordinate_receipt(protocol)
    if output.exists():
        run_path, coordinate_path = output / "run.json", output / "coordinates.json"
        if not run_path.is_file() or not coordinate_path.is_file():
            raise ValueError("resumed repair output lacks its provenance")
        manifest = json.loads(run_path.read_text())
        if any(manifest.get(key) != value for key, value in immutable_manifest.items()):
            raise ValueError(
                "resumed repair output has a different experiment contract"
            )
        if json.loads(coordinate_path.read_text()) != coordinates:
            raise ValueError("resumed repair coordinates changed")
        binding = manifest.get("run_binding")
        unsigned = {
            key: value for key, value in manifest.items() if key != "run_binding"
        }
        if (
            not isinstance(binding, str)
            or binding
            != hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
        ):
            raise ValueError("resumed repair run binding is invalid")
    else:
        manifest = {
            **immutable_manifest,
            "created_utc": datetime.now(UTC).isoformat(),
        }
        binding = hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest()
        manifest["run_binding"] = binding
        output.mkdir(parents=True)
        _atomic_json(output / "run.json", manifest)
        _atomic_json(output / "coordinates.json", coordinates)
    data = {
        part: materialize_batch(
            protocol.partitions[part],
            protocol.coordinates[part],
            config.context_packets,
        )
        for part in ("train", "validation")
    }
    if not torch.cuda.is_available():
        raise RuntimeError("repair matrix requires CUDA")
    device = torch.device("cuda")
    results = []
    for seed in SEEDS:
        for name in MODEL_NAMES:
            directory = output / str(seed) / name
            row = load_completed_cell(directory, name, seed, config, binding)
            if row is None:
                clear_incomplete_cell(directory, output)
                row = train_one(
                    name,
                    seed,
                    data["train"],
                    data["validation"],
                    config,
                    device,
                    directory,
                    binding,
                )
            else:
                print(json.dumps({"resumed_cell": name, "seed": seed}), flush=True)
            results.append(row)
    lock = {
        "schema": SCHEMA,
        "run_binding": binding,
        "screen_targets_materialized": False,
        "checkpoints": [
            {
                key: row[key]
                for key in ("name", "seed", "best_epoch", "checkpoint_sha256")
            }
            for row in results
        ],
    }
    lock_path = output / "LOCK.json"
    if lock_path.exists():
        if lock_path.is_symlink() or json.loads(lock_path.read_text()) != lock:
            raise ValueError("existing repair lock differs from validated checkpoints")
    else:
        _atomic_json(lock_path, lock)
    data["test"] = materialize_batch(
        protocol.partitions["test"],
        protocol.coordinates["test"],
        config.context_packets,
    )
    for row in results:
        path = output / str(row["seed"]) / row["name"] / "best.pt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["checkpoint_sha256"]:
            raise ValueError("checkpoint changed after repair lock")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        expected = {
            "schema": SCHEMA,
            "run_binding": binding,
            "name": row["name"],
            "seed": row["seed"],
            "best_epoch": row["best_epoch"],
            "trained": row["best_epoch"] > 0,
            "config": asdict(config),
        }
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise ValueError("repair checkpoint metadata mismatch")
        model = construct(row["name"], config.context_packets).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        row["screen"] = evaluate(model, data["test"], config, device)
        _atomic_json(path.parent / "metrics.json", row)
    decision = advance_decision(results)
    result = {
        "schema": SCHEMA,
        "run_binding": binding,
        "publication_eligible": False,
        "confirmatory_claim_eligible": False,
        "app_admission": False,
        "baselines": _baseline_metrics(data),
        "models": results,
        "development_advance_decision": decision,
        "inference_warning": "reused development screen; overlapping windows; post-result repair",
    }
    _atomic_json(output / "RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--context-packets", type=int, choices=(10_000, 30_000), default=10_000
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(
        args.journal,
        args.output,
        RepairCompareConfig(args.context_packets),
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

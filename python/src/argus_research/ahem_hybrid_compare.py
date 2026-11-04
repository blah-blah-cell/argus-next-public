"""Bounded exploratory comparison. Real journals only; final holdouts excluded.

All checkpoints are selected on validation. Hybrid selection is written before
ANY screen evaluation. No deployment admission or confirmatory claim follows.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn

from .ahem_hybrid import AhemHybrid, HybridConfig, hybrid_loss
from .ahem_pro import AhemProNextPacket, ProConfig, pro_training_loss
from .ahem_pro_pilot import (
    ALLOWED_SOURCES,
    DEVELOPMENT_SOURCE_ALLOWLIST,
    SCREEN_SOURCE,
    VALIDATION_SOURCE,
    _atomic_checkpoint,
    _atomic_json,
    _baseline_metrics,
    _batches,
    _inventory,
)
from .ahem_v6 import (
    AhemV6StateTransitionNextPacket,
    FlowLaneLstmNextPacket,
    transition_slots,
)
from .flow_v3_protocol import ProtocolConfig, compile_source_disjoint, materialize_batch
from .flow_v6_objective import dynamic_loss, training_loss
from .flow_v6_development import _retrieval_metrics
from .pro_baselines import PersistenceResidualLstm

SCHEMA = "argus.ahem-hybrid-comparison/v1"
SEEDS = (20260909, 20260910, 20260911)
CANDIDATES = ("hybrid_absolute", "hybrid_blend", "hybrid_delta")
TRANSPORT = dict(zip(CANDIDATES, (0.0, 0.5, 1.0)))
MODEL_NAMES = (
    CANDIDATES
    + tuple(name + "_recency" for name in CANDIDATES)
    + (
        "hybrid_local",
        "v61",
        "pro_v21",
        "lstm_h85",
        "residual_lstm_h85",
        "residual_lstm_matched",
        "residual_lstm_h255",
    )
)


@dataclass(frozen=True)
class CompareConfig:
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


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def matched_hidden():
    # One recurrent layer and a two-linear-layer forecast head.
    target = parameter_count(AhemHybrid())
    return min(range(8, 256), key=lambda h: abs(5 * h * h + 59 * h + 2 - target))


def construct(name: str, horizon: int):
    if name in CANDIDATES:
        return AhemHybrid(HybridConfig(horizon=horizon, transport=TRANSPORT[name]))
    if name.endswith("_recency") and name[:-8] in CANDIDATES:
        return AhemHybrid(
            HybridConfig(
                horizon=horizon, transport=TRANSPORT[name[:-8]], reader="recency"
            )
        )
    constructors = {
        "hybrid_local": lambda: AhemHybrid(
            HybridConfig(horizon=horizon, reader="none")
        ),
        "v61": AhemV6StateTransitionNextPacket,
        "pro_v21": lambda: AhemProNextPacket(ProConfig(horizon_packets=horizon)),
        "lstm_h85": lambda: FlowLaneLstmNextPacket(hidden_dim=85),
        "residual_lstm_h85": lambda: PersistenceResidualLstm(hidden_dim=85),
        "residual_lstm_h255": PersistenceResidualLstm,
        "residual_lstm_matched": lambda: PersistenceResidualLstm(
            hidden_dim=matched_hidden()
        ),
    }
    if name not in constructors:
        raise ValueError(f"unknown comparison model: {name}")
    return constructors[name]()


def objective(model, x, y):
    if isinstance(model, AhemHybrid):
        return hybrid_loss(model.forward_diagnostics(x), y)[0]
    if isinstance(model, AhemProNextPacket):
        return pro_training_loss(model.forward_diagnostics(x), y)[0]
    if isinstance(model, AhemV6StateTransitionNextPacket):
        return training_loss(model.forward_diagnostics(x), y)[0]
    return dynamic_loss(model(x), y)


def hybrid_retrieval(output, x, y, capacity):
    """Separate native proposals from common observed-successor diagnostics.

    Tie-aware utility is NOT a semantic attack/event-identity retrieval label.
    The common metric excludes the null slot, preventing abstention inflation.
    """
    cost = dynamic_loss(
        output.candidates, y[:, None].expand_as(output.candidates), reduction="none"
    )
    cost = cost.masked_fill(~output.valid, torch.inf)
    chosen = output.weights.argmax(-1, keepdim=True)
    selected = cost.gather(1, chosen)[:, 0]
    oracle = cost.amin(-1)
    metrics = {
        "native_selected_regret": selected - oracle,
        "native_tie_aware": (selected <= oracle + 1e-8).float(),
        "abstention_fraction": (chosen[:, 0] == 0).float(),
        "mean_memory_gate": output.gate.mean(-1),
    }
    _, _, slots = transition_slots(x, max_transitions=capacity)
    raw = dynamic_loss(
        slots.successor[..., :2],
        y[:, None].expand(-1, slots.valid.shape[1], -1),
        reduction="none",
    )
    raw = raw.masked_fill(~slots.valid, torch.inf)
    eligible = slots.valid.any(-1) & output.valid[:, 1:].any(-1)
    if eligible.any():
        selected_memory = output.weights[:, 1:].argmax(-1, keepdim=True)
        picked = raw.gather(1, selected_memory)[:, 0]
        best = raw.amin(-1)
        metrics.update(
            {
                "memory_only_raw_successor_tie_aware": (
                    picked[eligible] <= best[eligible] + 1e-8
                ).float(),
                "memory_only_raw_successor_regret": picked[eligible] - best[eligible],
                "recency_raw_successor_regret": raw[eligible, 0] - best[eligible],
                "valid_memory_slots": slots.valid[eligible].sum(-1).float(),
            }
        )
    return metrics


def evaluate(model, values, config, device):
    model.eval()
    losses, metrics = [], {}
    x, y = values
    with torch.inference_mode():
        for index in _batches(len(x), config.batch_size):
            bx, by = x[index].to(device), y[index].to(device)
            if isinstance(model, AhemHybrid):
                out = model.forward_diagnostics(bx)
                prediction = out.prediction
                for key, value in hybrid_retrieval(
                    out, bx, by, model.config.capacity
                ).items():
                    metrics.setdefault(key, []).append(value.cpu())
            elif isinstance(model, AhemV6StateTransitionNextPacket):
                out = model.forward_diagnostics(bx)
                prediction = out.prediction
                for key, value in _retrieval_metrics(out, by).items():
                    metrics.setdefault("legacy_" + key, []).append(value.cpu())
            else:
                prediction = model(bx)
            losses.append(dynamic_loss(prediction, by, reduction="none").cpu())
    losses = torch.cat(losses)
    if not torch.isfinite(losses).all():
        raise FloatingPointError("non-finite evaluation")
    result = {
        "loss": float(losses.mean()),
        "samples": len(losses),
        "per_sample_loss": losses.tolist(),
    }
    if metrics:
        result["retrieval"] = {
            key: {"mean": float(torch.cat(v).mean()), "samples": sum(len(a) for a in v)}
            for key, v in metrics.items()
        }
    return result


def train_one(name, seed, training, validation, config, device, directory, binding):
    """Screen data is deliberately not an argument to training."""
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = construct(name, config.context_packets).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-4
    )
    initial = evaluate(model, validation, config, device)["loss"]
    best, best_epoch, history = initial, 0, [{"epoch": 0, "validation_loss": initial}]

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
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
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
                raise FloatingPointError(f"non-finite training loss: {name}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(index)
        value = evaluate(model, validation, config, device)["loss"]
        if value < best:
            best, best_epoch = value, epoch
            save()
        record = {
            "epoch": epoch,
            "validation_loss": value,
            "train_objective": total / len(x),
        }
        history.append(record)
        _atomic_json(
            directory / "progress.json", {"history": history, "status": "running"}
        )
        print(json.dumps({"model": name, "seed": seed, **record}), flush=True)
    result = {
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
    _atomic_json(directory / "progress.json", {"status": "training_complete", **result})
    return result


def select_candidate(results):
    # All seeds required; no dropping an inconvenient seed or late winner swap.
    averages = {}
    for name in CANDIDATES:
        rows = [row for row in results if row["name"] == name]
        if sorted(row["seed"] for row in rows) != sorted(SEEDS):
            raise ValueError("candidate selection requires every declared seed")
        averages[name] = sum(row["validation_loss"] for row in rows) / len(rows)
        if not math.isfinite(averages[name]):
            raise ValueError("non-finite selection metric")
    return {
        "selected": min(CANDIDATES, key=averages.get),
        "validation_mean_loss": averages,
        "screen_accessed": False,
        "tie_break": "declared_candidate_order",
    }


def run(journal: Path, output: Path, config: CompareConfig, sources=ALLOWED_SOURCES):
    config.validate()
    _inventory(journal, sources)
    if output.exists():
        raise ValueError("output exists: preserve earlier experiments")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "full real-data comparison requires CUDA; do not substitute fixtures"
        )
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
            sampling_policy="source_balanced_natural"
            if len(sources) > 3
            else "natural",
            require_memory_match=True,
            minimum_completed_transitions=2,
        ),
    )
    expected = set(sources) - {VALIDATION_SOURCE, SCREEN_SOURCE}
    if set(protocol.manifest["split"]["train_sources"]) != expected:
        raise ValueError("unexpected training sources")
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    manifest = {
        "schema": SCHEMA,
        "config": asdict(config),
        "seeds": SEEDS,
        "models": MODEL_NAMES,
        "hybrid_contract": HybridConfig(horizon=config.context_packets).contract(),
        "value_hypotheses": TRANSPORT,
        "matched_residual_lstm_hidden": matched_hidden(),
        "sources": sorted(sources),
        "protocol": protocol.manifest,
        "source_sha256": digest.hexdigest(),
        "created_utc": datetime.now(UTC).isoformat(),
        "publication_eligible": False,
        "study_role": "exploratory_fixed_matrix",
        "selection": "mean_validation_loss_over_all_seeds_before_screen",
        "objective_policy": "hybrid_regret_v1; v61_frozen; pro_v21_frozen; lstm_forecast_only",
        "runtime": {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    binding = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["run_binding"] = binding
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "run.json", manifest)
    _atomic_json(
        output / "coordinates.json",
        {
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
        },
    )
    data = {
        part: materialize_batch(
            protocol.partitions[part],
            protocol.coordinates[part],
            config.context_packets,
        )
        for part in ("train", "validation")
    }
    device = torch.device("cuda")
    results = []
    for seed in SEEDS:
        for name in MODEL_NAMES:
            results.append(
                train_one(
                    name,
                    seed,
                    data["train"],
                    data["validation"],
                    config,
                    device,
                    output / str(seed) / name,
                    binding,
                )
            )
    selection = select_candidate(results)
    _atomic_json(output / "selection.json", selection)
    # First materialization/evaluation of screen targets after locked selection.
    data["test"] = materialize_batch(
        protocol.partitions["test"],
        protocol.coordinates["test"],
        config.context_packets,
    )
    for row in results:
        path = output / str(row["seed"]) / row["name"] / "best.pt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["checkpoint_sha256"]:
            raise ValueError("checkpoint changed after selection")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        expected_metadata = {
            "run_binding": binding,
            "name": row["name"],
            "seed": row["seed"],
            "best_epoch": row["best_epoch"],
            "config": asdict(config),
            "schema": SCHEMA,
        }
        if any(
            checkpoint.get(key) != value for key, value in expected_metadata.items()
        ):
            raise ValueError("checkpoint metadata mismatch")
        if not all(
            torch.isfinite(value).all() for value in checkpoint["state_dict"].values()
        ):
            raise ValueError("non-finite checkpoint state")
        model = construct(row["name"], config.context_packets).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        row["screen"] = evaluate(model, data["test"], config, device)
        _atomic_json(path.parent / "metrics.json", row)
    selected = selection["selected"]
    comparisons = {}
    for name in MODEL_NAMES:
        if name == selected:
            continue
        differences = []
        for seed in SEEDS:
            chosen = next(
                row
                for row in results
                if row["name"] == selected and row["seed"] == seed
            )
            reference = next(
                row for row in results if row["name"] == name and row["seed"] == seed
            )
            a, b = chosen["screen"]["loss"], reference["screen"]["loss"]
            differences.append(
                {
                    "seed": seed,
                    "hybrid_minus_reference_loss": a - b,
                    "relative_loss_reduction": (b - a) / b if b > 0 else None,
                }
            )
        comparisons[name] = differences
    final = {
        "schema": SCHEMA,
        "run_binding": binding,
        "selection": selection,
        "publication_eligible": False,
        "confirmatory_claim_eligible": False,
        "baselines": _baseline_metrics(data),
        "models": results,
        "comparisons": comparisons,
        "inference_warning": "paired development losses, not independent packet significance or confirmation",
    }
    _atomic_json(output / "RESULT.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--context-packets", type=int, choices=(10_000, 30_000), default=10_000
    )
    for name in (
        "epochs",
        "train_samples",
        "validation_samples",
        "screen_samples",
        "batch_size",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int,
            default=getattr(CompareConfig(), name),
        )
    parser.add_argument(
        "--additional-training-source",
        action="append",
        default=[],
        choices=sorted(DEVELOPMENT_SOURCE_ALLOWLIST - ALLOWED_SOURCES),
    )
    args = vars(parser.parse_args())
    journal, output = args.pop("journal"), args.pop("output")
    sources = ALLOWED_SOURCES | frozenset(args.pop("additional_training_source"))
    run(journal, output, CompareConfig(**args), sources)


if __name__ == "__main__":
    main()

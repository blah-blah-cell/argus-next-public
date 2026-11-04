"""Secondary competitor comparison against the frozen AHEM v6.1 final result.

Trains parameter-matched competitors from scratch on the same development
sources with the same forecast-only contract, then evaluates them on the
same frozen Friday-23 coordinates.  This is post-first-access secondary
evidence: the holdout is no longer untouched after the one-attempt final
evaluation, and every output carries ``study_role=secondary_comparison``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .flow_v2_train import _source_provenance
from .flow_v3_protocol import (
    ProtocolConfig,
    compile_source_disjoint,
    materialize_batch,
)
from .flow_v6_competitors import construct_competitor
from .flow_v6_development import (
    SEEDS,
    VALIDATION_SOURCE as DEV_VALIDATION_SOURCE,
    SCREEN_SOURCE as DEV_SCREEN_SOURCE,
    _atomic_json,
    _atomic_checkpoint,
    _batches,
)
from .flow_v6_final_evaluate import (
    _check_final_inventory,
    _compile_final_protocol,
    _final_interval,
    _file_sha256,
    _FinalProtocol,
    _validate_plan,
)
from .flow_v6_objective import dynamic_loss

SCHEMA = "argus.ahem-v6-competitor-comparison/v1"
COMPETITORS = ("gru", "s4d", "long_tcn", "mamba", "dlinear")
CANDIDATE = "ahem_v6_state_transition"
PRIMARY_REFERENCE = "flow_lane_lstm"
BOOTSTRAP_RESAMPLES = 10_000
CHECKPOINT_SCHEMA = "argus.ahem-v6-competitor-checkpoint/v1"
EXPECTED_FROZEN_RESULT_SHA256 = (
    "fbb468ad92c54a98a6a94754458ff44cd21b944a146604959f3b8402e7e00a9a"
)

# The development sources used for v6.1 training (must match exactly).
DEV_SOURCES = frozenset(
    {
        "Wednesday-14-02-2018",
        "Thursday-15-02-2018",
        "Friday-16-02-2018",
        "Tuesday-20-02-2018",
        "Wednesday-21-02-2018",
        "Wednesday-28-02-2018",
        DEV_VALIDATION_SOURCE,
        DEV_SCREEN_SOURCE,
    }
)


def _check_dev_inventory(journal: Path) -> None:
    """Refuse anything but the eight development sources before reading."""
    paths = sorted(journal.glob("*.arguslr"))
    if not paths:
        raise ValueError("no development journals found")
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"development journal is not a regular file: {path.name}")
        if path.stem not in DEV_SOURCES:
            raise ValueError(
                f"competitor training accepts only development sources, found {path.name}"
            )
    stems = {path.stem for path in paths}
    if stems != DEV_SOURCES:
        missing = DEV_SOURCES - stems
        raise ValueError(
            f"development journal inventory is incomplete: {sorted(missing)}"
        )


@torch.inference_mode()
def _evaluate_model(
    model: nn.Module,
    records: list,
    coordinates: list,
    context_packets: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    losses: list = []
    model.eval()
    for batch in _batches(coordinates, batch_size):
        contexts, target = materialize_batch(records, batch, context_packets)
        contexts = contexts.to(device)
        target = target.to(device)
        prediction = model(contexts)
        per = dynamic_loss(prediction, target, reduction="none")
        if not torch.isfinite(per).all():
            raise FloatingPointError("non-finite competitor evaluation loss")
        losses.append(per.cpu())
    joined = torch.cat(losses)
    return {
        "samples": len(joined),
        "weighted_smooth_l1": float(joined.mean()),
        "per_sample_weighted_smooth_l1": joined.tolist(),
    }


def _train_competitor(
    name: str,
    seed: int,
    dev_protocol,
    config: dict[str, Any],
    device: torch.device,
    output: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train one competitor with forecast-only loss on the development split."""
    torch.manual_seed(seed)
    model = construct_competitor(name).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_validation = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    context_packets = config["context_packets"]
    batch_size = config["batch_size"]
    for epoch in range(1, config["epochs"] + 1):
        if stale >= config["patience"]:
            break
        model.train()

        epoch_losses = []
        for batch in _batches(
            dev_protocol.coordinates["train"],
            batch_size,
            seed + epoch * 1_000_003,
        ):
            contexts, target = materialize_batch(
                dev_protocol.partitions["train"], batch, context_packets
            )
            contexts = contexts.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = dynamic_loss(model(contexts), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {name} training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        validation = _evaluate_model(
            model,
            dev_protocol.partitions["validation"],
            dev_protocol.coordinates["validation"],
            context_packets,
            batch_size,
            device,
        )
        value = validation["weighted_smooth_l1"]
        improved = value < best_validation
        if improved:
            best_validation = value
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": sum(epoch_losses) / len(epoch_losses),
                "validation_weighted_smooth_l1": value,
                "improved": improved,
            }
        )
        print(
            json.dumps(
                {"model": name, "seed": seed, "epoch": epoch, "val_loss": value}
            ),
            flush=True,
        )
    model.load_state_dict(best_state, strict=True)
    model.to(device).eval()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_checkpoint(
        output / "best.pt",
        {
            "schema": CHECKPOINT_SCHEMA,
            "model": name,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation": best_validation,
            "state_dict": best_state,
        },
    )
    return model, {
        "model": name,
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "history": history,
    }


def _secondary_interval(interval: dict[str, Any]) -> dict[str, Any]:
    """Relabel the reused crossed-bootstrap estimator without claiming confirmation."""
    value = dict(interval)
    value["analysis_contract"] = "secondary_crossed_seed_whole_flow_cluster_ratio/v1"
    value["confirmatory_claim_evidence"] = False
    value["study_role"] = "secondary_comparison"
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("competitor output must be a fresh empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    source = _source_provenance()
    if source.get("available") is not True or source.get("git_dirty") is not False:
        raise RuntimeError("competitor comparison requires a clean Git snapshot")

    _check_dev_inventory(args.dev_journal)
    _check_final_inventory(args.final_journal)

    # Phase 1: compile the development protocol and train competitors.
    dev_config = ProtocolConfig(
        context_packets=10_000,
        train_samples=2_048,
        validation_samples=512,
        test_samples=512,
        seed=20260906,
        require_memory_match=True,
        sampling_policy="natural",
        minimum_completed_transitions=2,
    )
    dev_protocol = compile_source_disjoint(
        args.dev_journal,
        DEV_VALIDATION_SOURCE,
        DEV_SCREEN_SOURCE,
        dev_config,
    )
    training_config = {
        "context_packets": 10_000,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "epochs": 50,
        "patience": 8,
    }
    if args.quick:
        training_config["epochs"] = 3
        training_config["patience"] = 3

    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "study_role": "secondary_comparison",
        "publication_eligible": False,
        "description": (
            "Parameter-matched competitors trained on the same development "
            "sources and evaluated on the same frozen Friday-23 coordinates. "
            "Post-first-access secondary evidence; the holdout is no longer "
            "untouched."
        ),
        "competitors": list(COMPETITORS),
        "candidate": CANDIDATE,
        "frozen_reference": PRIMARY_REFERENCE,
        "seeds": list(SEEDS),
        "training_config": training_config,
        "source_provenance": source,
    }
    _atomic_json(args.output / "run.json", manifest)

    device = torch.device(args.device)
    training_results: dict[str, list[dict[str, Any]]] = {
        name: [] for name in COMPETITORS
    }
    for seed in SEEDS:
        for name in COMPETITORS:
            model_output = args.output / "models" / str(seed) / name
            model, info = _train_competitor(
                name,
                seed,
                dev_protocol,
                training_config,
                device,
                model_output,
            )
            training_results[name].append(info)

    # Phase 2: evaluate on the frozen Friday-23 coordinates.
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes.decode("utf-8"))
    _validate_plan(plan, hashlib.sha256(plan_bytes).hexdigest())
    final_manifest, final_partitions, final_coordinates = _compile_final_protocol(
        args.final_journal, plan
    )
    final_protocol = _FinalProtocol(final_partitions, final_coordinates, final_manifest)
    final_config = {
        "context_packets": int(plan["sampling"]["context_packets"]),
        "batch_size": 8,
    }
    eval_results: dict[str, list[dict[str, Any]]] = {name: [] for name in COMPETITORS}
    for seed in SEEDS:
        for name in COMPETITORS:
            checkpoint_path = args.output / "models" / str(seed) / name / "best.pt"
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            model = construct_competitor(name)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.to(device).eval()
            test = _evaluate_model(
                model,
                final_protocol.partitions["test"],
                final_protocol.coordinates["test"],
                final_config["context_packets"],
                final_config["batch_size"],
                device,
            )
            eval_results[name].append(
                {
                    "seed": seed,
                    "test": test,
                }
            )
            print(
                json.dumps(
                    {
                        "eval": name,
                        "seed": seed,
                        "test_weighted_smooth_l1": test["weighted_smooth_l1"],
                    }
                ),
                flush=True,
            )

    # Phase 3: comparison report.
    # Read the frozen AHEM result for the head-to-head comparison.
    frozen_hash = _file_sha256(args.frozen_result)
    if frozen_hash != EXPECTED_FROZEN_RESULT_SHA256:
        raise ValueError("frozen AHEM result bytes differ from the final evidence")
    frozen_result = json.loads(args.frozen_result.read_text(encoding="utf-8"))
    if (
        frozen_result.get("schema") != "argus.ahem-v6-final-evaluation-run/v2"
        or frozen_result.get("decision") != "supported"
        or frozen_result.get("confirmatory_claim_eligible") is not True
        or frozen_result.get("not_intrusion_detection_efficacy") is not True
        or frozen_result.get("plan_sha256") != hashlib.sha256(plan_bytes).hexdigest()
        or frozen_result.get("coordinate_sha256")
        != final_manifest["coordinate_sha256"]
    ):
        raise ValueError("frozen AHEM result does not match the compiled final protocol")
    for name in (CANDIDATE, PRIMARY_REFERENCE):
        if [row.get("seed") for row in frozen_result["models"][name]] != list(SEEDS):
            raise ValueError(f"frozen {name} seed order differs")
    candidate_test = [
        row["test"]["weighted_smooth_l1"] for row in frozen_result["models"][CANDIDATE]
    ]
    lstm_test = [
        row["test"]["weighted_smooth_l1"]
        for row in frozen_result["models"][PRIMARY_REFERENCE]
    ]
    comparison: dict[str, Any] = {}
    candidate_loss_rows = [
        row["test"]["per_sample_weighted_smooth_l1"]
        for row in frozen_result["models"][CANDIDATE]
    ]
    cluster_ids = frozen_result["primary"]["bootstrap_cluster_ids"]
    for competitor_index, name in enumerate(COMPETITORS):
        comp_test = [row["test"]["weighted_smooth_l1"] for row in eval_results[name]]
        candidate_vs_competitor = [
            (comp - candidate) / comp
            for candidate, comp in zip(candidate_test, comp_test, strict=True)
        ]
        vs_lstm = [
            (lstm - comp) / lstm
            for lstm, comp in zip(lstm_test, comp_test, strict=True)
        ]
        comparison[name] = {
            "parameters": training_results[name][0]["parameters"],
            "per_seed_test_loss": comp_test,
            "mean_test_loss": statistics.fmean(comp_test),
            "candidate_mean_relative_reduction_vs_competitor": statistics.fmean(
                candidate_vs_competitor
            ),
            "mean_relative_reduction_vs_lstm": statistics.fmean(vs_lstm),
            "candidate_per_seed_relative_reduction_vs_competitor": (
                candidate_vs_competitor
            ),
            "per_seed_vs_lstm": vs_lstm,
            "candidate_vs_competitor_interval": _secondary_interval(
                _final_interval(
                    candidate_loss_rows,
                    [
                        row["test"]["per_sample_weighted_smooth_l1"]
                        for row in eval_results[name]
                    ],
                    cluster_ids,
                    20261001 + competitor_index,
                )
            ),
        }
    result = {
        "schema": SCHEMA,
        "study_role": "secondary_comparison",
        "publication_eligible": False,
        "not_confirmatory": (
            "the holdout was opened by the one-attempt final evaluation; "
            "competitors were trained blind to it but this is not confirmatory"
        ),
        "frozen_result_sha256": frozen_hash,
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "coordinate_sha256": final_manifest["coordinate_sha256"],
        "source_provenance": source,
        "training": training_results,
        "evaluation": eval_results,
        "comparison": comparison,
        "summary": {
            "candidate_mean_test_loss": statistics.fmean(candidate_test),
            "lstm_mean_test_loss": statistics.fmean(lstm_test),
            "candidate_mean_relative_reduction_vs_lstm": (
                statistics.fmean(
                    (lstm - cand) / lstm
                    for lstm, cand in zip(lstm_test, candidate_test, strict=True)
                )
            ),
        },
    }
    _atomic_json(args.output / "RESULT.json", result)
    _atomic_json(args.output / "COMPLETED.json", {"complete": True})
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="the frozen final-evaluation plan (for coordinate compilation)",
    )
    parser.add_argument(
        "--dev-journal",
        type=Path,
        required=True,
        help="directory of the eight development ARGUSLR2 journals",
    )
    parser.add_argument(
        "--final-journal",
        type=Path,
        required=True,
        help="directory of the two untouched ARGUSLR2 journals (Thursday-22, Friday-23)",
    )
    parser.add_argument(
        "--frozen-result",
        type=Path,
        required=True,
        help="the frozen AHEM v6.1 final RESULT.json for head-to-head comparison",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="3-epoch smoke run for testing; not evidence",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

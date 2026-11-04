"""Frozen one-attempt final evaluation of AHEM-v6.1 on the untouched holdouts.

This runner opens the previously forbidden Thursday-22-02-2018 and
Friday-23-02-2018 sources exactly once.  It loads the existing v6.1 development
checkpoints, evaluates them without any training, and renders the predeclared
verdict from `research/flow-v6-final-evaluation-plan.json`.

No training happens here.  A rejected verdict ends the study's superiority
claim; it does not authorize a v6.3 revision under this plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .ahem_v6 import AhemV6StateTransitionNextPacket, transition_slots
from .continual_learning import iter_sequence_records_v2
from .flow_v2_protocol import _journal_manifest
from .flow_v2_train import _source_provenance
from .flow_v3_protocol import (
    ProtocolConfig,
    condition_record,
    deterministic_coordinates,
    materialize_batch,
)
from .flow_v6_development import (
    RunConfig,
    _atomic_json,
    _batches,
    _canonical,
    _construct,
    _evaluate,
    _mechanism_controls,
)

SCHEMA = "argus.ahem-v6-final-evaluation-run/v2"
PLAN_SCHEMA = "argus.ahem-v6-final-evaluation/v2"
# Updated only before first holdout access. It binds the CLI to the reviewed
# plan instead of accepting an arbitrary external JSON file with the same schema.
EXPECTED_PLAN_SHA256 = (
    "26fe0da9994591384a72e25f4ed543e1f2e6f11c06ad9fe75bc2bf0b22fa1b97"
)
VALIDATION_SOURCE = "Thursday-22-02-2018"
TEST_SOURCE = "Friday-23-02-2018"
FINAL_SOURCES = frozenset({VALIDATION_SOURCE, TEST_SOURCE})
SEEDS = (20260906, 20260907, 20260908)
MODELS = (
    "ahem_v6_state_transition",
    "ahem_v6_local_only",
    "ahem_v6_recency",
    "flow_lane_lstm",
)
CANDIDATE = "ahem_v6_state_transition"
PRIMARY_REFERENCE = "flow_lane_lstm"
CHECKPOINT_SCHEMA = "argus.ahem-v6-training-development/v2"
STRATA_BOUNDARIES = {"short": (1, 50), "medium": (50, 500), "long": (500, None)}
BOOTSTRAP_RESAMPLES = 10_000


def _validate_plan(plan: dict[str, Any], plan_sha256: str) -> None:
    if plan_sha256 != EXPECTED_PLAN_SHA256:
        raise ValueError("final plan bytes differ from the implementation-bound freeze")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("final evaluation requires the frozen v6 final plan")
    if plan.get("study_role") != "confirmatory_final_evaluation":
        raise ValueError("final plan has the wrong study role")
    if plan.get("one_attempt", {}).get("existing_ledger_refuses_rerun") is not True:
        raise ValueError("final plan must enforce one-attempt access")
    checkpoints = plan.get("checkpoints", {})
    expected_keys = {f"{seed}/{name}" for seed in SEEDS for name in MODELS}
    if checkpoints.get("seeds") != list(SEEDS):
        raise ValueError("final plan must contain every frozen seed exactly once")
    if checkpoints.get("models") != list(MODELS):
        raise ValueError("final plan must contain every frozen model exactly once")
    if checkpoints.get("candidate") != CANDIDATE:
        raise ValueError("final plan candidate changed")
    if checkpoints.get("primary_reference") != PRIMARY_REFERENCE:
        raise ValueError("final plan primary reference changed")
    hashes = checkpoints.get("expected_sha256", {})
    if set(hashes) != expected_keys or any(
        not isinstance(value, str) or len(value) != 64 for value in hashes.values()
    ):
        raise ValueError("final plan lacks the twelve frozen checkpoint hashes")
    uncertainty = plan.get("primary_endpoint", {}).get("uncertainty", {})
    if uncertainty.get("resamples") != BOOTSTRAP_RESAMPLES:
        raise ValueError("final plan bootstrap count changed")
    execution = plan.get("execution", {})
    if execution.get("device_type") not in {"cpu", "cuda"}:
        raise ValueError("final plan lacks a supported frozen device type")
    if type(execution.get("batch_size")) is not int or execution["batch_size"] < 2:
        raise ValueError("final plan requires a batch size of at least two")


def _preflight_device(device: torch.device) -> None:
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        try:
            torch.empty(1, device=device)
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(f"CUDA device preflight failed: {error}") from error
    elif device.type != "cpu":
        raise ValueError("final evaluation supports only CPU or CUDA")


def _load_runtime_evidence(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    expected = plan["secondary_endpoints"]["runtime_evidence"]
    file_hash = _file_sha256(path)
    if file_hash != expected["expected_sha256"]:
        raise ValueError("runtime evidence differs from the frozen audit artifact")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("schema") != expected["expected_schema"]:
        raise ValueError("runtime evidence has an incompatible schema")
    if evidence.get("quality_evidence") is not False:
        raise ValueError("runtime audit must not masquerade as quality evidence")
    return {"path": str(path), "sha256": file_hash, "schema": evidence["schema"]}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_final_inventory(journal: Path) -> None:
    """Refuse anything but the two untouched holdout sources before reading."""
    paths = sorted(journal.glob("*.arguslr"))
    if not paths:
        raise ValueError("no final-evaluation journals found")
    stems = {path.stem for path in paths}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"final journal is not a regular file: {path.name}")
        if path.stem not in FINAL_SOURCES:
            raise ValueError(
                f"final evaluation accepts only the frozen holdouts, found {path.name}"
            )
    if stems != FINAL_SOURCES:
        raise ValueError(f"final journal inventory is incomplete: {sorted(stems)}")


def _compile_final_protocol(
    journal: Path, plan: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, list], dict[str, list]]:
    """Compile validation/test partitions only; a final evaluation has no train."""
    sampling = plan["sampling"]
    config = ProtocolConfig(
        context_packets=int(sampling["context_packets"]),
        train_samples=1,
        validation_samples=int(sampling["validation_samples"]),
        test_samples=int(sampling["test_samples"]),
        seed=int(sampling["protocol_seed"]),
        require_memory_match=bool(sampling["require_memory_match"]),
        sampling_policy=sampling["sampling_policy"],
        minimum_completed_transitions=int(sampling["minimum_completed_transitions"]),
    )
    config.validate()
    raw_records = list(iter_sequence_records_v2(journal))
    if not raw_records:
        raise ValueError("no checksum-valid ARGUSLR2 records were found")
    available = {record.source_file for record in raw_records}
    if available != FINAL_SOURCES:
        raise ValueError(
            f"final journal sources differ from the frozen split: {sorted(available)}"
        )
    partitions: dict[str, list] = {"validation": [], "test": []}
    for raw in raw_records:
        partition = "validation" if raw.source_file == VALIDATION_SOURCE else "test"
        partitions[partition].append(condition_record(raw))
    segment_sets = {
        name: {record.segment_id for record in records}
        for name, records in partitions.items()
    }
    if segment_sets["validation"] & segment_sets["test"]:
        raise ValueError("a record segment appears in both final partitions")
    journal_manifest = _journal_manifest(journal, FINAL_SOURCES)
    if journal_manifest["provenance_complete"] is not True:
        problems = {
            member["name"]: member["provenance"]["problems"]
            for member in journal_manifest["members"]
            if member["provenance"]["problems"]
        }
        raise ValueError(f"final journals lack complete provenance: {problems}")
    coordinates: dict[str, list] = {}
    for index, (name, count) in enumerate(
        (("validation", config.validation_samples), ("test", config.test_samples))
    ):
        coordinates[name] = deterministic_coordinates(
            partitions[name],
            context_packets=config.context_packets,
            local_packets=config.local_packets,
            samples=count,
            seed=config.seed + index,
            require_memory_match=config.require_memory_match,
            sampling_policy=config.sampling_policy,
            minimum_completed_transitions=config.minimum_completed_transitions,
        )
    coordinate_payload = {
        name: [
            {
                "record_index": value.record_index,
                "anchor_index": value.anchor_index,
                "target_index": value.target_index,
            }
            for value in values
        ]
        for name, values in coordinates.items()
    }
    manifest = {
        "schema": "argus.flow-conditioned-final/v1",
        "plan": PLAN_SCHEMA,
        "study_role": plan["study_role"],
        "confirmatory_claim_evidence": True,
        "task": "evaluate_only_next_same_flow_dynamic_event_forecast",
        "config": {
            "context_packets": config.context_packets,
            "validation_samples": config.validation_samples,
            "test_samples": config.test_samples,
            "seed": config.seed,
            "sampling_policy": config.sampling_policy,
            "minimum_completed_transitions": config.minimum_completed_transitions,
            "require_memory_match": config.require_memory_match,
        },
        "split": {
            "method": "journal_source_file_disjoint_evaluate_only",
            "validation_source": VALIDATION_SOURCE,
            "test_source": TEST_SOURCE,
            "train_sources": [],
        },
        "journal": journal_manifest,
        "coordinate_sha256": _canonical(coordinate_payload),
    }
    return manifest, partitions, coordinates


def _load_frozen_checkpoint(
    checkpoints: Path, seed: int, name: str, expected_sha256: str
) -> tuple[nn.Module, str, dict[str, Any]]:
    path = checkpoints / "models" / str(seed) / name / "best.pt"
    if not path.is_file():
        raise ValueError(f"frozen checkpoint is missing: {path}")
    file_hash = _file_sha256(path)
    if file_hash != expected_sha256:
        raise ValueError(f"frozen checkpoint hash changed: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"{path} has an incompatible checkpoint schema")
    if checkpoint.get("model") != name or checkpoint.get("seed") != seed:
        raise ValueError(f"{path} belongs to another model or seed")
    model = _construct(name)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    identity = {
        "epoch": checkpoint.get("epoch"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_validation": checkpoint.get("best_validation"),
        "training_binding_sha256": checkpoint.get("training_binding"),
    }
    return model, file_hash, identity


def _persistence_prediction(contexts: Tensor) -> Tensor:
    """Forecast the query packet's own dynamic fields (data-only baseline)."""
    return contexts[:, -1, :2].clone()


def _hard_recency_prediction(contexts: Tensor) -> Tensor:
    """Forecast the newest completed transition's observed successor fields.

    Under the v6 slot contract the newest transition is (prev1 -> query), so
    its observed successor IS the query: this baseline coincides with
    persistence by construction and is reported separately only because the
    frozen plan names it.  Every final-evaluation coordinate guarantees at
    least two completed transitions; otherwise this fails closed through
    ``transition_slots``.
    """
    _, _, slots = transition_slots(contexts, max_transitions=16)
    if not slots.valid.any(dim=-1).all():
        raise ValueError("hard recency requires one transition per coordinate")
    newest = slots.valid.to(torch.int64).argmax(dim=-1)
    return (
        slots.successor[..., :2]
        .gather(1, newest[:, None, None].expand(-1, 1, 2))
        .squeeze(1)
    )


class _FinalProtocol:
    """Adapter exposing validation/test partitions to the audited evaluators."""

    def __init__(
        self,
        partitions: dict[str, list],
        coordinates: dict[str, list],
        manifest: dict[str, Any],
    ) -> None:
        self.partitions = {
            "train": [],
            "validation": partitions["validation"],
            "test": partitions["test"],
        }
        self.coordinates = {
            "train": [],
            "validation": coordinates["validation"],
            "test": coordinates["test"],
        }
        self.manifest = manifest


def _collect_decomposition(
    predictor,
    records: list,
    coordinates: list,
    config: RunConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Per-sample weighted loss, per-field losses, gaps, and flow clusters."""
    losses: list[Tensor] = []
    length_losses: list[Tensor] = []
    interarrival_losses: list[Tensor] = []
    gaps: list[int] = []
    clusters: list[str] = []
    for batch in _batches(coordinates, config.batch_size):
        contexts, target = materialize_batch(records, batch, config.context_packets)
        contexts = contexts.to(device)
        target = target.to(device)
        with torch.inference_mode():
            prediction = predictor(contexts)
        per_field = F.smooth_l1_loss(prediction, target, reduction="none")
        weighted = (per_field * prediction.new_tensor((1.0, 4.0))).sum(dim=-1) / 5.0
        if not torch.isfinite(weighted).all():
            raise FloatingPointError("non-finite decomposition loss")
        losses.append(weighted.cpu())
        length_losses.append(per_field[:, 0].detach().cpu())
        interarrival_losses.append(per_field[:, 1].detach().cpu())
        for coordinate in batch:
            gaps.append(coordinate.target_index - coordinate.anchor_index)
            flow = int(
                records[coordinate.record_index].flow_ids[coordinate.anchor_index]
            )
            clusters.append(f"{coordinate.record_index}:{flow}")
    return {
        "per_sample_weighted_smooth_l1": torch.cat(losses),
        "per_sample_log_frame_length_smooth_l1": torch.cat(length_losses),
        "per_sample_log_interarrival_smooth_l1": torch.cat(interarrival_losses),
        "gaps": gaps,
        "clusters": clusters,
    }


def _stratum_means(values: Tensor, gaps: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, (low, high) in STRATA_BOUNDARIES.items():
        index = [
            position
            for position, gap in enumerate(gaps)
            if gap >= low and (high is None or gap < high)
        ]
        if not index:
            raise ValueError(f"time-gap stratum {name} is empty; widen the strata")
        result[name] = float(values[index].mean())
        result[f"{name}_samples"] = len(index)
    return result


def _final_interval(
    candidate_losses: list[list[float]],
    reference_losses: list[list[float]],
    cluster_ids: list[str],
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Confirmatory crossed seed x whole-flow-cluster interval.

    Seeds and shared whole clusters are drawn independently, and every sampled
    cluster retains all of its coordinates. The ratio of aggregate losses is
    recomputed inside every draw; per-coordinate ratios are never averaged.
    """
    if not candidate_losses or len(candidate_losses) != len(reference_losses):
        raise ValueError("aligned nonempty candidate/reference seed rows are required")
    coordinate_count = len(cluster_ids)
    if coordinate_count == 0 or any(
        len(row) != coordinate_count
        for rows in (candidate_losses, reference_losses)
        for row in rows
    ):
        raise ValueError("loss rows and cluster identifiers must align")
    if any(
        not math.isfinite(value) or value < 0.0
        for rows in (candidate_losses, reference_losses)
        for row in rows
        for value in row
    ):
        raise ValueError("final losses must be finite and nonnegative")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(cluster_ids):
        groups[cluster].append(index)
    if len(groups) < 2:
        raise ValueError("one flow cluster cannot estimate between-cluster uncertainty")
    group_indices = list(groups.values())
    candidate_totals = [
        [sum(row[index] for index in group) for group in group_indices]
        for row in candidate_losses
    ]
    reference_totals = [
        [sum(row[index] for index in group) for group in group_indices]
        for row in reference_losses
    ]

    def reduction(seed: int, clusters: list[int]) -> float:
        candidate = sum(candidate_totals[seed][cluster] for cluster in clusters)
        reference = sum(reference_totals[seed][cluster] for cluster in clusters)
        if reference <= 0.0:
            raise ValueError("reference loss is zero; relative reduction is undefined")
        return (reference - candidate) / reference

    point_reductions = []
    for candidate_row, reference_row in zip(
        candidate_losses, reference_losses, strict=True
    ):
        candidate_mean = statistics.fmean(candidate_row)
        reference_mean = statistics.fmean(reference_row)
        if reference_mean <= 0.0:
            raise ValueError("reference loss is zero; relative reduction is undefined")
        point_reductions.append((reference_mean - candidate_mean) / reference_mean)
    point = statistics.fmean(point_reductions)
    seed_rng = random.Random(bootstrap_seed)
    cluster_rng = random.Random(bootstrap_seed ^ 0x5A17)
    sampled: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled_seeds = [
            seed_rng.randrange(len(candidate_losses)) for _ in candidate_losses
        ]
        sampled_clusters = [
            cluster_rng.randrange(len(group_indices)) for _ in group_indices
        ]
        sampled.append(
            statistics.fmean(
                reduction(seed, sampled_clusters) for seed in sampled_seeds
            )
        )
    sampled.sort()
    return {
        "analysis_contract": "confirmatory_crossed_seed_whole_flow_cluster_ratio/v2",
        "confirmatory_claim_evidence": True,
        "study_role": "confirmatory_final_evaluation",
        "mean_relative_reduction": point,
        "interval_95pct_low": sampled[int(0.025 * BOOTSTRAP_RESAMPLES)],
        "interval_95pct_high": sampled[
            min(BOOTSTRAP_RESAMPLES - 1, int(0.975 * BOOTSTRAP_RESAMPLES))
        ],
        "seed_count": len(candidate_losses),
        "cluster_count": len(group_indices),
        "coordinates_per_seed": coordinate_count,
        "resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": bootstrap_seed,
        "estimand": "equal-seed mean of ratios of coordinate-weighted mean losses",
        "cluster_definition": "segment-local query-anchor flow identity",
        "assumptions": [
            "seed and cluster factors are crossed; coordinates are aligned",
            "clusters are independent; entire within-cluster series is retained",
            "percentile interval is not an exact-coverage guarantee with three seeds",
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes.decode("utf-8"))
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    _validate_plan(plan, plan_sha256)
    execution = plan["execution"]
    if args.device != execution["device_type"]:
        raise ValueError("CLI device differs from the frozen final plan")
    if args.batch_size != execution["batch_size"]:
        raise ValueError("CLI batch size differs from the frozen final plan")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("final-evaluation output must be a fresh empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    source = _source_provenance()
    if source.get("available") is not True or source.get("git_dirty") is not False:
        raise RuntimeError("final evaluation requires a clean Git snapshot")
    _check_final_inventory(args.journal)
    sampling = plan["sampling"]
    config = RunConfig(
        context_packets=int(sampling["context_packets"]),
        batch_size=args.batch_size,
        utility_temperature=0.04,
    )
    device = torch.device(args.device)
    if execution.get("deterministic_algorithms") is not True:
        raise ValueError("final plan must require deterministic evaluation")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":4096:8",
        ":16:8",
    }:
        raise RuntimeError(
            "set CUBLAS_WORKSPACE_CONFIG=:4096:8 before the final CUDA process"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _preflight_device(device)

    checkpoint_hashes: dict[str, Any] = {}
    loaded: dict[tuple[int, str], tuple[nn.Module, dict[str, Any]]] = {}
    expected_hashes = plan["checkpoints"]["expected_sha256"]
    for seed in SEEDS:
        for name in MODELS:
            key = f"{seed}/{name}"
            model, file_hash, identity = _load_frozen_checkpoint(
                args.checkpoints, seed, name, expected_hashes[key]
            )
            # Catch an unavailable/index-invalid device and parameter transfer
            # failure before the irreversible holdout-access ledger is written.
            model.to(device).eval()
            loaded[(seed, name)] = (model, identity)
            checkpoint_hashes[key] = {"sha256": file_hash, **identity}

    runtime_evidence = _load_runtime_evidence(args.runtime_evidence, plan)

    # This ledger is intentionally written BEFORE checksums, parsing, protocol
    # compilation, coordinate construction, or any other holdout payload read.
    # Any failure after it is a spent final attempt.
    ledger = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "holdout_access_started",
        "study_role": "confirmatory_final_evaluation",
        "plan_sha256": plan_sha256,
        "holdout_inventory": sorted(f"{source}.arguslr" for source in FINAL_SOURCES),
        "checkpoints": checkpoint_hashes,
        "runtime_evidence": runtime_evidence,
        "source_provenance": source,
        "device": str(device),
        "batch_size": config.batch_size,
    }
    _atomic_json(args.output / "final-access.json", ledger)

    manifest, partitions, coordinates = _compile_final_protocol(args.journal, plan)
    _atomic_json(args.output / "protocol-manifest.json", manifest)
    protocol = _FinalProtocol(partitions, coordinates, manifest)

    per_seed: dict[str, list[dict[str, Any]]] = {name: [] for name in MODELS}
    mechanism: list[dict[str, Any]] = []
    for seed in SEEDS:
        for name in MODELS:
            model = loaded[(seed, name)][0]
            result = {
                "seed": seed,
                "validation": _evaluate(model, protocol, "validation", config, device),
                "test": _evaluate(model, protocol, "test", config, device),
            }
            per_seed[name].append(result)
            print(
                json.dumps(
                    {
                        "model": name,
                        "seed": seed,
                        "test_weighted_smooth_l1": result["test"]["weighted_smooth_l1"],
                    }
                ),
                flush=True,
            )
        candidate = loaded[(seed, CANDIDATE)][0]
        assert isinstance(candidate, AhemV6StateTransitionNextPacket)
        mechanism.append(_mechanism_controls(candidate, protocol, config, device))

    # Data-only baselines and per-feature/stratum decomposition.  The
    # decomposition re-derives the audited weighted loss; it must agree with
    # the primary evaluation to floating-point tolerance.
    decomposition: dict[str, Any] = {}
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for name in (CANDIDATE, PRIMARY_REFERENCE):
        rows = []
        for seed_index, seed in enumerate(SEEDS):
            model = loaded[(seed, name)][0]
            row = _collect_decomposition(
                model, partitions["test"], coordinates["test"], config, device
            )
            audited = per_seed[name][seed_index]["test"][
                "per_sample_weighted_smooth_l1"
            ]
            if not torch.allclose(
                row["per_sample_weighted_smooth_l1"],
                torch.tensor(audited, dtype=torch.float32),
                atol=1e-6,
                rtol=0.0,
            ):
                raise ValueError(
                    "decomposition disagrees with the audited primary evaluation"
                )
            rows.append(row)
        rows_by_model[name] = rows
    clusters = rows_by_model[CANDIDATE][0]["clusters"]
    for rows in rows_by_model.values():
        for row in rows:
            if row["clusters"] != clusters:
                raise ValueError("final coordinates are not aligned across seeds")
    candidate_loss_rows = [
        row["per_sample_weighted_smooth_l1"].to(torch.float64).tolist()
        for row in rows_by_model[CANDIDATE]
    ]
    reference_loss_rows = [
        row["per_sample_weighted_smooth_l1"].to(torch.float64).tolist()
        for row in rows_by_model[PRIMARY_REFERENCE]
    ]
    interval = _final_interval(
        candidate_loss_rows,
        reference_loss_rows,
        clusters,
        int(plan["primary_endpoint"]["uncertainty"]["bootstrap_seed"]),
    )
    gaps = rows_by_model[CANDIDATE][0]["gaps"]
    for name, rows in rows_by_model.items():
        decomposition[name] = {
            "per_seed_strata": [
                _stratum_means(row["per_sample_weighted_smooth_l1"], gaps)
                for row in rows
            ],
            "per_feature_mean": {
                field: float(torch.stack([row[key] for row in rows]).mean())
                for field, key in (
                    ("log_frame_length", "per_sample_log_frame_length_smooth_l1"),
                    (
                        "log_per_flow_interarrival_seconds",
                        "per_sample_log_interarrival_smooth_l1",
                    ),
                )
            },
        }

    data_only: dict[str, Any] = {
        "note": (
            "under the v6 slot contract hard_recency and persistence coincide "
            "by construction: the newest transition's observed successor is "
            "the query itself"
        )
    }
    for name, predictor in (
        ("persistence", _persistence_prediction),
        ("hard_recency", _hard_recency_prediction),
    ):
        collected = _collect_decomposition(
            predictor, partitions["test"], coordinates["test"], config, device
        )
        data_only[name] = {
            "weighted_smooth_l1": float(
                collected["per_sample_weighted_smooth_l1"].mean()
            ),
            **_stratum_means(collected["per_sample_weighted_smooth_l1"], gaps),
        }

    candidate_test = [statistics.fmean(row) for row in candidate_loss_rows]
    reference_test = [statistics.fmean(row) for row in reference_loss_rows]
    per_seed_reduction = [
        (reference - candidate) / reference
        for candidate, reference in zip(candidate_test, reference_test, strict=True)
    ]
    mean_reduction = statistics.fmean(per_seed_reduction)
    if not math.isclose(
        mean_reduction, interval["mean_relative_reduction"], abs_tol=1e-12
    ):
        raise ValueError("point estimate and interval estimand disagree")
    rotation_harm = [
        control["query_rotation"]["weighted_smooth_l1"]
        - row["test"]["weighted_smooth_l1"]
        for control, row in zip(mechanism, per_seed[CANDIDATE], strict=True)
    ]
    permutation_harm = [
        control["value_permutation"]["weighted_smooth_l1"]
        - row["test"]["weighted_smooth_l1"]
        for control, row in zip(mechanism, per_seed[CANDIDATE], strict=True)
    ]
    gates = {
        "mean_relative_reduction_at_least_5pct": mean_reduction >= 0.05,
        "interval_lower_bound_strictly_positive": interval["interval_95pct_low"] > 0.0,
        "query_rotation_hurts_every_seed": all(value > 0.0 for value in rotation_harm),
        "value_permutation_hurts_every_seed": all(
            value > 0.0 for value in permutation_harm
        ),
    }
    if not (
        gates["mean_relative_reduction_at_least_5pct"]
        and gates["interval_lower_bound_strictly_positive"]
    ):
        verdict = "rejected"
    else:
        verdict = "supported"
    mechanism_interpretation = (
        "corroborated"
        if gates["query_rotation_hurts_every_seed"]
        and gates["value_permutation_hurts_every_seed"]
        else "inconclusive"
    )
    result = {
        "schema": SCHEMA,
        "decision": verdict,
        "verdict_meaning": {
            "supported": "the predeclared primary forecast claim passes",
            "rejected": "a primary gate failed; this ends the study's superiority claim and does not authorize v6.3",
        }[verdict],
        "study_role": "confirmatory_final_evaluation",
        "reportable_regardless_of_verdict": True,
        "confirmatory_claim_eligible": True,
        "publication_acceptance_not_implied": True,
        "not_intrusion_detection_efficacy": True,
        "not_retrieval_novelty_claim": True,
        "plan_sha256": ledger["plan_sha256"],
        "coordinate_sha256": manifest["coordinate_sha256"],
        "protocol": manifest,
        "checkpoints": checkpoint_hashes,
        "models": per_seed,
        "mechanism_controls": mechanism,
        "primary": {
            "candidate": CANDIDATE,
            "reference": PRIMARY_REFERENCE,
            "per_seed_candidate_test_loss": candidate_test,
            "per_seed_reference_test_loss": reference_test,
            "per_seed_relative_reduction": per_seed_reduction,
            "mean_relative_reduction": mean_reduction,
            "bootstrap_cluster_ids": clusters,
            "interval": interval,
            "gates": {
                key: gates[key]
                for key in (
                    "mean_relative_reduction_at_least_5pct",
                    "interval_lower_bound_strictly_positive",
                )
            },
        },
        "secondary": {
            "mechanism_interpretation": mechanism_interpretation,
            "gates": {
                "query_rotation_hurts_every_seed": gates[
                    "query_rotation_hurts_every_seed"
                ],
                "value_permutation_hurts_every_seed": gates[
                    "value_permutation_hurts_every_seed"
                ],
            },
            "rotation_harm_per_seed": rotation_harm,
            "permutation_harm_per_seed": permutation_harm,
            "non_learned_baselines": data_only,
            "decomposition": decomposition,
            "runtime_evidence": runtime_evidence,
        },
        "evaluation_access_note": (
            "the run ledger preceded the first ARGUSLR payload read by this "
            "evaluator; earlier raw-PCAP conversion access is recorded in the "
            "converter sidecars and is not mislabelled as model evaluation"
        ),
    }
    _atomic_json(args.output / "RESULT.json", result)
    _atomic_json(
        args.output / "COMPLETED.json", {"complete": True, "decision": verdict}
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        required=True,
        help="v6.1 development run root containing models/<seed>/<name>/best.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--runtime-evidence",
        type=Path,
        required=True,
        help="completed-audit timing artifact embedded by SHA-256",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("batch size must be positive")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

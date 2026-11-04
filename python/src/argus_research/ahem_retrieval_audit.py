"""Offline retrieval diagnostics; target outcomes must never enter runtime queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import numpy as np
from torch import Tensor

from .flow_v6_objective import dynamic_loss


def query_coverage(previous: np.ndarray, following: np.ndarray, *, horizon: int = 10_000,
                   capacity: int = 16, local_packets: int = 128) -> dict[str, Any]:
    """Account for all observed packets, target censoring and causal cache availability.

    A v6 key requires predecessor -> source, with a completed source ->
    successor value. Thus a query needs two preceding same-flow packets for
    its first readable slot. Future links only define evaluation censoring.
    """
    previous, following = np.asarray(previous), np.asarray(following)
    if previous.ndim != 1 or previous.shape != following.shape or not len(previous):
        raise ValueError("nonempty aligned recurrence links are required")
    if not np.issubdtype(previous.dtype, np.integer) or not np.issubdtype(following.dtype, np.integer):
        raise ValueError("recurrence links must be integer indices")
    if min(horizon, capacity, local_packets) <= 0:
        raise ValueError("memory dimensions must be positive")
    indices = np.arange(len(previous))
    if np.any((previous < -1) | (previous >= indices)) or np.any(
        (following < -1) | ((following != -1) & ((following <= indices) | (following >= len(previous))))
    ):
        raise ValueError("recurrence links violate causal order")
    has_next = following >= 0
    if np.any(previous[following[has_next]] != indices[has_next]):
        raise ValueError("previous and following links disagree")
    start = np.maximum(0, indices - horizon + 1)
    cursor = indices.copy()
    active = np.ones(len(indices), dtype=bool)
    predecessors = np.zeros(len(indices), dtype=np.int32)
    for _ in range(capacity + 1):
        cursor = previous[np.maximum(cursor, 0)]
        active &= cursor >= start
        predecessors += active
    slots = np.maximum(0, predecessors - 1)
    historical = (indices >= horizon - 1) & (previous >= indices - horizon + 1)
    historical &= (previous <= indices - local_packets) & (predecessors >= 3) & has_next
    count = int(has_next.sum())
    return {
        "observed_packets": len(indices), "observed_successor_queries": count,
        "right_censored_packets_without_observed_successor": int((~has_next).sum()),
        "first_flow_packet_queries": int((has_next & (previous == -1)).sum()),
        "queries_without_readable_memory": int((has_next & (slots == 0)).sum()),
        "queries_with_one_slot": int((has_next & (slots == 1)).sum()),
        "queries_with_multiple_slots": int((has_next & (slots >= 2)).sum()),
        "memory_coverage_among_observed_successor_queries": (
            float((has_next & (slots > 0)).sum() / count) if count else None
        ),
        "historical_v61_eligible_queries": int(historical.sum()),
        "historical_v61_fraction_of_observed_successor_queries": (
            float(historical.sum() / count) if count else None
        ),
        "horizon": horizon, "capacity": capacity,
    }


def retrieval_quality(
    successor: Tensor, target: Tensor, valid: Tensor, weights: Tensor,
    *, tie_tolerance: float = 1e-8, consequential_regret: float = 0.01,
) -> dict[str, Any]:
    """Measure coverage and useful selection, including rows without memory.

    The oracle minimizes the existing weighted successor loss. This is an
    outcome-based diagnostic, not a ground-truth identity label or an online
    operation. No-memory rows contribute to coverage, not conditional accuracy.
    """
    if successor.ndim != 3 or successor.shape[-1] != 2:
        raise ValueError("successors must have shape [queries, slots, 2]")
    if successor.shape[0] == 0 or successor.shape[1] == 0:
        raise ValueError("at least one query and one allocated slot are required")
    if target.shape != (successor.shape[0], 2):
        raise ValueError("targets are not aligned")
    if valid.shape != successor.shape[:2] or valid.dtype != torch.bool or weights.shape != valid.shape:
        raise ValueError("slot masks and weights are not aligned")
    if any(not math.isfinite(value) or value < 0 for value in (tie_tolerance, consequential_regret)):
        raise ValueError("diagnostic tolerances must be finite and nonnegative")
    if not torch.isfinite(target).all() or not torch.isfinite(successor[valid]).all():
        raise ValueError("observed outcomes must be finite")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    if (weights[~valid] != 0).any():
        raise ValueError("invalid slots must carry zero weight")
    counts = valid.sum(-1)
    available = counts > 0
    if not torch.allclose(weights.sum(-1), available.to(weights.dtype), atol=1e-5):
        raise ValueError("weights must sum to one with memory and zero without it")
    losses = dynamic_loss(successor, target[:, None].expand_as(successor), reduction="none")
    losses = losses.masked_fill(~valid, torch.inf)
    oracle = losses.argmin(-1)
    selected = weights.masked_fill(~valid, -1).argmax(-1)
    recency = valid.to(torch.int64).argmax(-1)
    best = losses.gather(1, oracle[:, None]).squeeze(1)
    chosen = losses.gather(1, selected[:, None]).squeeze(1)
    recent = losses.gather(1, recency[:, None]).squeeze(1)
    # Avoid inf-inf for no-memory rows; these rows have no retrieval outcome.
    best = torch.where(available, best, torch.zeros_like(best))
    chosen = torch.where(available, chosen, torch.zeros_like(chosen))
    recent = torch.where(available, recent, torch.zeros_like(recent))
    tied = valid & ((losses - best[:, None]).abs() <= tie_tolerance)
    correct = tied.gather(1, selected[:, None]).squeeze(1)
    recency_correct = tied.gather(1, recency[:, None]).squeeze(1)
    informative = counts >= 2
    consequential = informative & ((recent - best) > consequential_regret)
    predicted = weights.masked_fill(~valid, -1).argsort(dim=-1, descending=True, stable=True)
    rank = (predicted.argsort(-1) + 1).masked_fill(~tied, valid.shape[1] + 1).amin(-1)

    def mean(values: Tensor, mask: Tensor) -> float | None:
        return float(values[mask].double().mean()) if mask.any() else None

    return {
        "schema": "argus.retrieval-quality/v1",
        "queries": len(target), "without_memory": int((~available).sum()),
        "single_candidate": int((counts == 1).sum()),
        "multiple_candidates": int(informative.sum()),
        "memory_coverage": float(available.double().mean()),
        "conditional_top1_canonical": mean((selected == oracle).float(), available),
        "conditional_top1_tie_aware": mean(correct.float(), available),
        "multi_candidate_top1_tie_aware": mean(correct.float(), informative),
        "recency_top1_tie_aware": mean(recency_correct.float(), informative),
        "oracle_mrr": mean(1.0 / rank.float(), informative),
        "top3_tie_aware": mean((rank <= 3).float(), informative),
        "random_top1_tie_aware": mean(tied.sum(-1) / counts.clamp_min(1), informative),
        "selected_oracle_regret": mean(chosen - best, informative),
        "recency_oracle_regret": mean(recent - best, informative),
        "consequential_queries": int(consequential.sum()),
        "consequential_top1_tie_aware": mean(correct.float(), consequential),
        "oracle_ties_mean": mean(tied.sum(-1).float(), informative),
        "tie_tolerance": tie_tolerance, "consequential_regret_threshold": consequential_regret,
    }


def audit_saved_result(payload: dict[str, Any], partitions: list[str]) -> dict[str, Any]:
    """Summarize recorded diagnostics without reaccessing captures or checkpoints."""
    rows = []
    for model, runs in payload.get("models", {}).items():
        for run in runs:
            for partition in partitions:
                result = run.get(partition)
                if not result or not ("retrieval" in result or "per_sample_retrieval" in result):
                    continue
                if "per_sample_retrieval" not in result:
                    recorded = result["retrieval"]
                    if any(not math.isfinite(value) for value in recorded.values()):
                        raise ValueError("saved retrieval diagnostics contain non-finite values")
                    rows.append({
                        "model": model, "seed": run["seed"], "partition": partition,
                        "samples": result["samples"], "available_detail": "aggregate_only",
                        "recorded_metrics": recorded,
                        "mrr_interpretation": (
                            "legacy mrr ranks selected item by oracle utility; not predicted-ranking MRR"
                            if "mrr" in recorded else "see recorded metric schema"
                        ),
                    })
                    continue
                metrics = result["per_sample_retrieval"]
                count = result["samples"]
                if count <= 0 or any(len(values) != count for values in metrics.values()):
                    raise ValueError("saved retrieval diagnostics have inconsistent sample counts")
                if any(not math.isfinite(value) for values in metrics.values() for value in values):
                    raise ValueError("saved retrieval diagnostics contain non-finite values")
                canonical = metrics["top1"]
                ties = metrics["top1_tie_aware"]
                chosen = metrics["selected_transition_loss"]
                recent = metrics["recency_transition_loss"]
                oracle = metrics["oracle_transition_loss"]
                consequential = [index for index in range(count) if recent[index] - oracle[index] > 0.01]
                rows.append({
                    "model": model, "seed": run["seed"], "partition": partition,
                    "samples": count, "canonical_top1": sum(canonical) / count,
                    "tie_aware_top1": sum(ties) / count,
                    "canonical_miss_but_equally_useful": sum(b > a for a, b in zip(canonical, ties)),
                    "mean_oracle_ties": sum(metrics["oracle_tie_count"]) / count,
                    "selected_loss": sum(chosen) / count,
                    "recency_loss": sum(recent) / count,
                    "oracle_loss": sum(oracle) / count,
                    "consequential_queries": len(consequential),
                    "consequential_tie_aware_top1": (
                        sum(ties[index] for index in consequential) / len(consequential)
                        if consequential else None
                    ),
                })
    if not rows:
        raise ValueError("no recorded retrieval diagnostics in requested partitions")
    return {
        "schema": "argus.saved-retrieval-audit/v1",
        "analysis_role": "retrospective_diagnostic_of_existing_artifacts",
        "source_schema": payload.get("schema"), "rows": rows,
        "unavailable_from_saved_result": [
            "coverage of all capture packets and no-memory queries",
            "full-history oracle versus retained-slot oracle",
            "byte and millisecond forecast errors",
            "independent capture-day uncertainty",
        ],
        "consequential_regret_threshold": 0.01,
        "threshold_role": "descriptive_only_not_a_promotion_gate",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--partitions", nargs="+", choices=("train", "validation", "screen", "test"),
                        default=["validation", "screen"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.result.read_bytes()
    report = audit_saved_result(json.loads(raw), args.partitions)
    report["source"] = {"path": str(args.result), "sha256": hashlib.sha256(raw).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    if args.output.exists() or temporary.exists():
        raise FileExistsError("audit output already exists")
    with temporary.open("xb") as destination:
        destination.write((json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

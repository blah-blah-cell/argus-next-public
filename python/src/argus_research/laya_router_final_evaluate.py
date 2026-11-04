"""One-shot locked-test comparison of Laya and the matched compact router."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from .laya_router_baseline_train import _counterfactuals
from .laya_router_vm_train import _load_manifest, _sha256, _typed_rows
from .laya_window_router import LayaWindowRouter
from .memory_models import CausalRouter


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, round(probability * (len(ordered) - 1))))
    return ordered[position]


def hierarchical_gain_interval(
    gains_by_group: dict[str, list[float]], seed: int, replicates: int = 10_000
) -> tuple[float, float]:
    if len(gains_by_group) < 3:
        raise ValueError("hierarchical interval requires at least three source groups")
    rng = random.Random(seed)
    groups = sorted(gains_by_group)
    estimates = []
    for _ in range(replicates):
        sampled = [rng.choice(groups) for _ in groups]
        values = []
        for group in sampled:
            members = gains_by_group[group]
            values.extend(rng.choice(members) for _ in members)
        estimates.append(sum(values) / len(values))
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def expected_calibration_error(
    confidences: list[float], correct: list[bool], bins: int = 15
) -> float:
    if not confidences or len(confidences) != len(correct):
        raise ValueError("calibration inputs are empty or misaligned")
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        members = [
            item
            for item, confidence in enumerate(confidences)
            if low <= confidence < high or (index == bins - 1 and confidence == 1.0)
        ]
        if members:
            mean_confidence = sum(confidences[item] for item in members) / len(members)
            accuracy = sum(correct[item] for item in members) / len(members)
            error += len(members) / len(confidences) * abs(accuracy - mean_confidence)
    return error


def _summary(selected: list[int], forecast: torch.Tensor, utility: torch.Tensor, names: list[str]) -> dict[str, Any]:
    indices = torch.tensor(selected, dtype=torch.long)
    selected_forecast = forecast.gather(1, indices[:, None]).squeeze(1)
    selected_utility = utility.gather(1, indices[:, None]).squeeze(1)
    oracle = utility.min(1).values
    best_fixed = int(utility.mean(0).argmin())
    return {
        "mean_forecast_loss": float(selected_forecast.mean()),
        "mean_utility": float(selected_utility.mean()),
        "mean_route_regret": float((selected_utility - oracle).mean()),
        "best_fixed_expert": names[best_fixed],
        "best_fixed_utility": float(utility[:, best_fixed].mean()),
        "gain_over_best_fixed": float(utility[:, best_fixed].mean() - selected_utility.mean()),
        "selection_rate": {
            name: selected.count(index) / len(selected) for index, name in enumerate(names)
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("final evaluation output already exists")
    prepared = _load_manifest(args.prepared)
    for name in ("test.jsonl", "test.counterfactual.jsonl"):
        path = args.prepared / name
        if not path.is_file() or _sha256(path) != prepared["files"].get(name):
            raise ValueError(f"locked {name} SHA-256 differs")
    names = [expert["expert_id"] for expert in prepared["experts"]]
    typed_rows = _typed_rows(args.prepared / "test.jsonl")
    private_rows = [
        json.loads(line)
        for line in (args.prepared / "test.counterfactual.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    if [row["id"] for row in typed_rows] != [row["id"] for row in private_rows]:
        raise ValueError("locked typed and counterfactual rows differ")
    features, forecast, utility = _counterfactuals(
        args.prepared / "test.counterfactual.jsonl", names
    )

    compact_payload = torch.load(args.compact, map_location="cpu", weights_only=True)
    compact_manifest = compact_payload.get("manifest")
    if not isinstance(compact_manifest, dict):
        raise ValueError("compact control manifest is absent")
    if compact_manifest.get("preparation_manifest_sha256") != _sha256(
        args.prepared / "manifest.json"
    ):
        raise ValueError("compact control used a different prepared dataset")
    if compact_manifest.get("experts") != names:
        raise ValueError("compact control expert order differs")
    compact = CausalRouter(24, len(names))
    compact.load_state_dict(compact_payload["state_dict"], strict=True)
    compact.eval()
    with torch.inference_mode():
        compact_started = time.perf_counter_ns()
        compact_probabilities = (
            compact(features) / float(compact_manifest["temperature"])
        ).softmax(1)
        compact_elapsed_ms = (time.perf_counter_ns() - compact_started) / 1_000_000.0
        compact_selected = compact_probabilities.argmax(1)
        fallback = names.index(compact_manifest["fallback_expert"])
        compact_selected[
            compact_probabilities.max(1).values
            < float(compact_manifest["confidence_threshold"])
        ] = fallback
    compact_indices = compact_selected.tolist()

    previous = os.environ.get("ARGUS_LAYA_CHECKPOINT")
    os.environ["ARGUS_LAYA_CHECKPOINT"] = str(args.laya.resolve())
    try:
        laya_router = LayaWindowRouter.from_environment(args.device)
    finally:
        if previous is None:
            os.environ.pop("ARGUS_LAYA_CHECKPOINT", None)
        else:
            os.environ["ARGUS_LAYA_CHECKPOINT"] = previous
    if laya_router.config is None or laya_router.agent is None:
        raise RuntimeError(f"Laya checkpoint was rejected: {laya_router.detail}")
    if list(laya_router.config.expert_ids) != names:
        raise ValueError("Laya expert order differs from prepared data")

    laya_indices = []
    laya_latencies = []
    abstained = 0
    active_confidences = []
    active_correct = []
    oracle_indices = utility.argmin(1).tolist()
    for index, row in enumerate(typed_rows):
        result = laya_router.route(row["state"], set(names))
        laya_latencies.append(float(result["execution_ms"]))
        if result["state"] == "active":
            selected = names.index(result["selected_expert"])
            laya_indices.append(selected)
            active_confidences.append(float(result["confidence"]))
            active_correct.append(selected == oracle_indices[index])
        else:
            abstained += 1
            laya_indices.append(compact_indices[index])

    compact_summary = _summary(compact_indices, forecast, utility, names)
    laya_summary = _summary(laya_indices, forecast, utility, names)
    compact_utilities = utility.gather(
        1, torch.tensor(compact_indices)[:, None]
    ).squeeze(1)
    laya_utilities = utility.gather(1, torch.tensor(laya_indices)[:, None]).squeeze(1)
    gains_by_group: dict[str, list[float]] = {}
    for index, row in enumerate(private_rows):
        gains_by_group.setdefault(row["source_group"], []).append(
            float(compact_utilities[index] - laya_utilities[index])
        )
    interval = hierarchical_gain_interval(gains_by_group, args.seed)
    latency_values = prepared["profiled_isolated_median_latency_ms"]
    execute_all_ms = sum(float(latency_values[name]) for name in names)
    laya_total_latencies = [
        laya_latencies[index] + float(latency_values[names[selected]])
        for index, selected in enumerate(laya_indices)
    ]
    compact_per_window_ms = compact_elapsed_ms / len(compact_indices)
    compact_total_latencies = [
        compact_per_window_ms + float(latency_values[names[selected]])
        for selected in compact_indices
    ]
    ece = (
        expected_calibration_error(active_confidences, active_correct)
        if active_confidences
        else 1.0
    )
    forecast_noninferior = laya_summary["mean_forecast_loss"] <= (
        compact_summary["mean_forecast_loss"] * 1.01
    )
    gates = {
        "utility_better_than_matched_compact": laya_summary["mean_utility"]
        < compact_summary["mean_utility"],
        "hierarchical_gain_ci_above_zero": interval[0] > 0,
        "regret_within_25_percent_of_best_fixed": laya_summary["mean_route_regret"]
        <= 0.25 * laya_summary["best_fixed_utility"],
        "ece_at_most_0_05": ece <= 0.05,
        "p95_total_latency_below_execute_all": _percentile(laya_total_latencies, 0.95)
        < execute_all_ms,
        "forecast_loss_noninferior_within_1_percent": forecast_noninferior,
    }
    result = {
        "schema": "argus.laya-router-final-evaluation/v1",
        "locked_test_opened": True,
        "prepared_manifest_sha256": _sha256(args.prepared / "manifest.json"),
        "laya_checkpoint_sha256": laya_router.config.weights_sha256,
        "compact_checkpoint_sha256": _sha256(args.compact),
        "source_groups": sorted(gains_by_group),
        "samples": len(private_rows),
        "laya": laya_summary,
        "matched_compact": compact_summary,
        "mean_utility_gain_over_compact": float(
            compact_utilities.mean() - laya_utilities.mean()
        ),
        "hierarchical_gain_95_interval": list(interval),
        "laya_abstention_rate": abstained / len(private_rows),
        "laya_active_ece_15_bin": ece,
        "latency_ms": {
            "laya_route_p50": statistics.median(laya_latencies),
            "laya_route_p95": _percentile(laya_latencies, 0.95),
            "laya_total_p95": _percentile(laya_total_latencies, 0.95),
            "matched_compact_total_p95": _percentile(compact_total_latencies, 0.95),
            "execute_all_profiled_sum": execute_all_ms,
        },
        "gates": gates,
        "admitted": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--laya", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

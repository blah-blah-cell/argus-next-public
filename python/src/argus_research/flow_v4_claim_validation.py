"""Consistency checks for frozen confirmatory report consumers, without test access."""

from __future__ import annotations

import math
from statistics import fmean, median
from typing import Any

from .flow_v2_train import _canonical_sha256
from .flow_v4_mechanism_eval import summarize
from .flow_v4_multiseed import CANDIDATE, LONG_GAP_BUCKETS, TRAINING_SEEDS
from .flow_v4_screen_report import _hash, _losses, _number


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _same(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-7)


def _gates(reported: Any, expected: dict[str, bool]) -> None:
    if (
        not isinstance(reported, dict)
        or any(type(value) is not bool for value in reported.values())
        or reported != expected
    ):
        raise ValueError("reported gates disagree with the underlying statistics")


def paired_statistics(value: dict[str, Any]) -> dict[str, Any]:
    paired = value.get("paired_weighted_forecast_loss", {})
    candidate = value.get("mean_candidate_loss")
    reference = value.get("mean_reference_loss")
    relative = value.get("mean_relative_improvement")
    mean = paired.get("mean_baseline_minus_candidate")
    lower, upper = paired.get("bootstrap_95pct_low"), paired.get("bootstrap_95pct_high")
    seed_means = paired.get("seed_means")
    if (
        not _number(candidate)
        or not _number(reference)
        or not all(_finite(item) for item in (relative, mean, lower, upper))
        or lower > upper
        or not isinstance(seed_means, list)
        or len(seed_means) != len(TRAINING_SEEDS)
        or not all(_finite(item) for item in seed_means)
        or not _same(mean, fmean(seed_means))
        or not _same(mean, reference - candidate)
        or not _same(relative, mean / max(reference, 1e-12))
    ):
        raise ValueError("paired comparison statistics are inconsistent or non-finite")
    return paired


def comparison(value: dict[str, Any], *, replication: bool = False) -> None:
    if (
        value.get("candidate") != CANDIDATE
        or value.get("training_seeds") != list(TRAINING_SEEDS)
        or value.get("study_role")
        != ("confirmatory_untouched_replication" if replication else "confirmatory")
        or value.get("hierarchical_bootstrap_resamples") != 10000
        or value.get("bootstrap_contract")
        != "seeds_then_flow_generations_then_coordinates"
        or not _hash(value.get("protocol_sha256"))
        or not _hash(value.get("coordinate_sha256"))
    ):
        raise ValueError(
            "comparison has the wrong identity, seeds or bootstrap contract"
        )
    overall = value if replication else value.get("overall", {})
    paired = paired_statistics(overall)
    expected = {
        "paired_interval_strictly_positive": paired["bootstrap_95pct_low"] > 0,
        "relative_improvement_at_least_five_percent": overall[
            "mean_relative_improvement"
        ]
        >= 0.05,
    }
    if not replication:
        if overall.get("samples_per_seed") != [512] * len(TRAINING_SEEDS):
            raise ValueError("primary comparison has the wrong sample inventory")
        counts = value.get("primary_bucket_counts", {})
        gaps = value.get("sufficiently_populated_long_gap_buckets", {})
        if (
            set(counts) != set(LONG_GAP_BUCKETS)
            or any(
                type(count) is not int or count < 0 or count > 512
                for count in counts.values()
            )
            or sum(counts.values()) > 512
            or set(gaps) != {name for name, count in counts.items() if count >= 50}
        ):
            raise ValueError(
                "comparison omitted or changed a populated long-gap bucket"
            )
        for name, gap in gaps.items():
            if gap.get("bucket") != name or gap.get("samples_per_seed") != [
                counts[name]
            ] * len(TRAINING_SEEDS):
                raise ValueError("long-gap sample inventory is inconsistent")
            paired_statistics(gap)
        expected.update(
            candidate_better_overall=paired["mean_baseline_minus_candidate"] > 0,
            all_populated_long_gap_means_positive=bool(gaps)
            and all(
                gap["paired_weighted_forecast_loss"]["mean_baseline_minus_candidate"]
                > 0
                for gap in gaps.values()
            ),
        )
    else:
        protocol = value.get("protocol", {})
        if (
            _canonical_sha256(protocol) != value["protocol_sha256"]
            or protocol.get("coordinate_sha256") != value["coordinate_sha256"]
            or protocol.get("schema") != "argus.ahem-v5-replication-protocol/v1"
            or protocol.get("preregistration_sha256")
            != value.get("preregistration_sha256")
            or protocol.get("source") != "Friday-WorkingHours"
            or protocol.get("dataset") != "CIC-IDS2017"
            or protocol.get("config")
            != {
                "context_packets": 10000,
                "local_packets": 128,
                "samples": 512,
                "seed": 20260901,
                "require_memory_match": True,
                "changed_event_fraction": 0.5,
            }
            or protocol.get("journal", {}).get("provenance_complete") is not True
            or protocol.get("journal", {}).get("datasets") != ["CIC-IDS2017"]
        ):
            raise ValueError("replication protocol is not bound to its result")
        losses = value.get("per_seed_losses", {})
        if set(losses) != {CANDIDATE, "lstm"}:
            raise ValueError("replication per-seed losses are missing")
        means = {}
        for model, by_seed in losses.items():
            if set(by_seed) != {str(seed) for seed in TRAINING_SEEDS}:
                raise ValueError("replication loss seeds differ")
            for samples in by_seed.values():
                if (
                    not isinstance(samples, list)
                    or len(samples) != 512
                    or not all(_number(item) for item in samples)
                ):
                    raise ValueError("replication losses are invalid")
            means[model] = [fmean(by_seed[str(seed)]) for seed in TRAINING_SEEDS]
        if (
            not _same(fmean(means[CANDIDATE]), value["mean_candidate_loss"])
            or not _same(fmean(means["lstm"]), value["mean_reference_loss"])
            or any(
                not _same(b - a, delta)
                for a, b, delta in zip(
                    means[CANDIDATE], means["lstm"], paired["seed_means"], strict=True
                )
            )
        ):
            raise ValueError("replication summaries disagree with saved losses")
    expected["comparison_passed"] = all(expected.values())
    _gates(value.get("gates"), expected)


def mechanism(value: dict[str, Any]) -> None:
    if value.get("study_role") != "confirmatory" or not _hash(
        value.get("candidate_checkpoint_sha256")
    ):
        raise ValueError("mechanism role or checkpoint is invalid")
    summary = value.get("summary", {})
    losses = summary.get("weighted_smooth_l1", {})
    names = (
        "candidate",
        "local_only",
        "recency",
        "query_rotation",
        "value_permutation",
    )
    if set(losses) != set(names) or not all(_number(item) for item in losses.values()):
        raise ValueError("mechanism losses are invalid")
    for name in ("query_rotation", "value_permutation"):
        result = value.get(name, {})
        _losses(result)
        if result["samples"] != 512 or result["weighted_smooth_l1"] != losses[name]:
            raise ValueError(
                "mechanism sample inventory or loss differs from its summary"
            )
    if any(
        value["query_rotation"][field] != value["value_permutation"][field]
        for field in ("sample_coordinates", "per_sample_persistence_weighted_smooth_l1")
    ):
        raise ValueError("mechanism corruption coordinates are not paired")
    expected = summarize(*(losses[name] for name in names))
    if summary != expected:
        raise ValueError("mechanism summary disagrees with its losses")
    _gates(summary.get("gates"), expected["gates"])
    if value.get("all_mechanism_gates_pass") is not all(expected["gates"].values()):
        raise ValueError("mechanism pass flag disagrees with its losses")


def runtime(value: dict[str, Any], *, checkpoint: str) -> None:
    process_ids = value.get("process_instance_ids")
    source = value.get("source_provenance", {})
    if (
        value.get("processes") != 5
        or not isinstance(process_ids, list)
        or len(process_ids) != 5
        or any(not isinstance(item, str) or not item for item in process_ids)
        or len(set(process_ids)) != 5
        or value.get("environment", {}).get("device") != "cpu"
        or not value.get("environment", {}).get("platform", "").startswith("Windows-")
        or source.get("available") is not True
        or source.get("git_dirty") is not False
        or not _hash(source.get("git_revision"), 40)
        or not value.get("implementation_sha256")
        or not all(_hash(item) for item in value["implementation_sha256"].values())
        or value.get("settings")
        != {
            "warmup": 50,
            "iterations": 200,
            "torch_threads": 1,
            "measurement_order": "alternating_forward_reverse",
        }
        or value.get("checkpoints", {}).get("ahem_v5", {}).get("sha256") != checkpoint
        or not _hash(value.get("checkpoints", {}).get("lstm", {}).get("sha256"))
        or not _number(value.get("max_stream_reference_abs_error"))
    ):
        raise ValueError("runtime aggregate violates the frozen measurement contract")
    for name in (
        "ahem_v5_stream_step",
        "ahem_v5_transition_write",
        "lstm_stateful_step",
    ):
        summary = value.get("summaries", {}).get(name, {})
        samples = summary.get("process_p95_ms")
        if (
            not isinstance(samples, list)
            or len(samples) != 5
            or not all(_number(item) for item in samples)
            or not _number(summary.get("median_process_p95_ms"))
            or not _same(summary["median_process_p95_ms"], median(samples))
        ):
            raise ValueError(
                "runtime summary is inconsistent or has invalid p95 values"
            )

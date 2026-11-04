"""Machine-generated pass/fail verdict for the frozen AHEM-v5 claim."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any

from .flow_v2_protocol import write_manifest
from .flow_v3_train import V4_PREREGISTRATION_SHA256
from .flow_v4_mechanism_eval import MECHANISM_SCHEMA
from .flow_v4_multiseed import MULTISEED_SCHEMA, TRAINING_SEEDS
from .flow_v4_runtime_replicates import AGGREGATE_SCHEMA
from .flow_v4_claim_validation import (
    _same,
    comparison as validate_comparison,
    mechanism as validate_mechanism,
    runtime as validate_runtime,
)

CLAIM_SCHEMA = "argus.ahem-v5-claim-report/v2"


def _load(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema:
        raise ValueError(f"{path.name} has an incompatible schema")
    if value.get("preregistration_sha256") != V4_PREREGISTRATION_SHA256:
        raise ValueError(f"{path.name} belongs to another preregistration")
    if value.get("publication_evidence") is not True:
        raise ValueError(f"{path.name} is not publication evidence")
    if value.get("study_role") == "confirmatory_untouched_replication":
        access_path = path.with_suffix(path.suffix + ".access.json")
        if not access_path.is_file():
            raise ValueError("replication has no completed test-access ledger")
        access = json.loads(access_path.read_text(encoding="utf-8"))
        if (
            access.get("schema") != "argus.ahem-v5-replication-access/v1"
            or access.get("status") != "completed"
            or access.get("protocol_sha256") != value.get("protocol_sha256")
            or access.get("coordinate_sha256") != value.get("coordinate_sha256")
            or access.get("result_sha256")
            != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("replication does not match its completed access ledger")
    return value


def build(
    *,
    lstm: Path,
    local: Path,
    recency: Path,
    gru: Path,
    tcn: Path,
    s4d: Path,
    persistence: Path,
    mechanisms: list[Path],
    runtimes: list[Path],
    replication: Path,
) -> dict[str, Any]:
    comparisons = {
        "lstm": _load(lstm, MULTISEED_SCHEMA),
        "ahem_v5_local_only": _load(local, MULTISEED_SCHEMA),
        "ahem_v5_recency": _load(recency, MULTISEED_SCHEMA),
    }
    secondary_comparisons = {
        "gru": _load(gru, MULTISEED_SCHEMA),
        "long_tcn": _load(tcn, MULTISEED_SCHEMA),
        "s4d": _load(s4d, MULTISEED_SCHEMA),
        "last_event_persistence": _load(persistence, MULTISEED_SCHEMA),
    }
    all_comparisons = {**comparisons, **secondary_comparisons}
    if any(value.get("reference") != name for name, value in all_comparisons.items()):
        raise ValueError("claim report comparison labels are inconsistent")
    for value in all_comparisons.values():
        validate_comparison(value)
        if not _same(
            value["overall"]["mean_candidate_loss"],
            comparisons["lstm"]["overall"]["mean_candidate_loss"],
        ):
            raise ValueError("comparisons disagree on the candidate observations")
    protocol_hashes = {
        value.get("protocol_sha256") for value in all_comparisons.values()
    }
    coordinate_hashes = {
        value.get("coordinate_sha256") for value in all_comparisons.values()
    }
    if len(protocol_hashes) != 1 or len(coordinate_hashes) != 1:
        raise ValueError("claim comparisons mix protocols or coordinates")

    mechanism_values = [_load(path, MECHANISM_SCHEMA) for path in mechanisms]
    if len(mechanism_values) != len(TRAINING_SEEDS) or sorted(
        int(value["training_seed"]) for value in mechanism_values
    ) != list(TRAINING_SEEDS):
        raise ValueError("claim report requires one mechanism result per seed")
    if any(
        value.get("protocol_sha256") not in protocol_hashes
        or value.get("coordinate_sha256") not in coordinate_hashes
        or type(value.get("training_seed")) is not int
        for value in mechanism_values
    ):
        raise ValueError("mechanism results use another protocol")
    for value in mechanism_values:
        validate_mechanism(value)
    mechanisms_by_seed = {value["training_seed"]: value for value in mechanism_values}
    for field in ("sample_coordinates", "per_sample_persistence_weighted_smooth_l1"):
        if any(
            value["query_rotation"][field]
            != mechanism_values[0]["query_rotation"][field]
            for value in mechanism_values[1:]
        ):
            raise ValueError("mechanism seeds have different test observations")
    candidate_losses = [
        mechanisms_by_seed[seed]["summary"]["weighted_smooth_l1"]["candidate"]
        for seed in TRAINING_SEEDS
    ]
    if not _same(
        fmean(candidate_losses), comparisons["lstm"]["overall"]["mean_candidate_loss"]
    ):
        raise ValueError("mechanism candidate differs from the paired comparison")
    for reference, name in (
        ("ahem_v5_local_only", "local_only"),
        ("ahem_v5_recency", "recency"),
    ):
        deltas = comparisons[reference]["overall"]["paired_weighted_forecast_loss"][
            "seed_means"
        ]
        if any(
            not _same(
                mechanisms_by_seed[seed]["summary"]["weighted_smooth_l1"][name]
                - candidate_losses[index],
                deltas[index],
            )
            for index, seed in enumerate(TRAINING_SEEDS)
        ):
            raise ValueError("mechanism baseline differs from the paired comparison")

    runtime_values = [_load(path, AGGREGATE_SCHEMA) for path in runtimes]
    expected_runtime_count = len(TRAINING_SEEDS) * 3
    if len(runtime_values) != expected_runtime_count:
        raise ValueError("claim report requires three runtime contexts for every seed")
    by_context: dict[int, list[float]] = {257: [], 10_000: [], 30_000: []}
    runtime_equivalence = []
    runtime_coordinates: set[tuple[int, int]] = set()
    for value in runtime_values:
        if (
            value.get("study_role") != "confirmatory_runtime_audit"
            or value.get("publication_evidence") is not True
            or value.get("protocol_sha256") not in protocol_hashes
            or value.get("coordinate_sha256") not in coordinate_hashes
            or type(value.get("training_seed")) is not int
            or value["training_seed"] not in TRAINING_SEEDS
            or type(value.get("context_packets")) is not int
        ):
            raise ValueError("claim report received non-confirmatory runtime evidence")
        validate_runtime(
            value,
            checkpoint=mechanisms_by_seed[value["training_seed"]][
                "candidate_checkpoint_sha256"
            ],
        )
        context = int(value["context_packets"])
        if context not in by_context:
            raise ValueError("runtime report uses an unfrozen context")
        by_context[context].extend(
            float(item)
            for item in value["summaries"]["ahem_v5_stream_step"]["process_p95_ms"]
        )
        runtime_coordinates.add((int(value["training_seed"]), context))
        runtime_equivalence.append(float(value["max_stream_reference_abs_error"]))
    if (
        len(
            {
                identity
                for value in runtime_values
                for identity in value["process_instance_ids"]
            }
        )
        != expected_runtime_count * 5
    ):
        raise ValueError("runtime evidence reuses a fresh process")
    if any(
        value.get(field) != runtime_values[0].get(field)
        for value in runtime_values[1:]
        for field in ("environment", "source_provenance", "implementation_sha256")
    ):
        raise ValueError("runtime evidence mixes implementations or environments")
    for seed in TRAINING_SEEDS:
        checkpoints = [
            value["checkpoints"]
            for value in runtime_values
            if value["training_seed"] == seed
        ]
        if any(value != checkpoints[0] for value in checkpoints[1:]):
            raise ValueError("runtime contexts mix checkpoints within a seed")
    runtime_summary = {
        str(context): {
            "process_p95_ms": values,
            "median_process_p95_ms": median(values),
            "passes_2ms": median(values) <= 2.0,
        }
        for context, values in by_context.items()
    }
    if any(len(values) != len(TRAINING_SEEDS) * 5 for values in by_context.values()):
        raise ValueError("runtime report set lacks five processes per seed and context")
    expected_runtime_coordinates = {
        (training_seed, context)
        for training_seed in TRAINING_SEEDS
        for context in by_context
    }
    if runtime_coordinates != expected_runtime_coordinates:
        raise ValueError("runtime report set duplicates or omits a seed/context pair")

    replication_value = _load(replication, MULTISEED_SCHEMA)
    if (
        replication_value.get("study_role") != "confirmatory_untouched_replication"
        or replication_value.get("candidate") != "ahem_v5_source_aligned"
        or replication_value.get("reference") != "lstm"
        or replication_value.get("training_seeds") != list(TRAINING_SEEDS)
    ):
        raise ValueError("claim report lacks the untouched replication role")
    validate_comparison(replication_value, replication=True)
    if (
        replication_value.get("primary_protocol_sha256") not in protocol_hashes
        or replication_value["protocol_sha256"] in protocol_hashes
        or replication_value["coordinate_sha256"] in coordinate_hashes
    ):
        raise ValueError(
            "replication is not separately bound to the same primary study"
        )

    gates = {
        # Post-audit retirement: internally consistent legacy evidence cannot
        # override the demonstrated flaws in this frozen analysis contract.
        "methodology_cleared_after_20260905_audit": False,
        "quality_vs_lstm": comparisons["lstm"]["gates"]["comparison_passed"],
        "quality_vs_local_only": comparisons["ahem_v5_local_only"]["gates"][
            "comparison_passed"
        ],
        "quality_vs_recency": comparisons["ahem_v5_recency"]["gates"][
            "comparison_passed"
        ],
        "candidate_beats_persistence": secondary_comparisons["last_event_persistence"][
            "gates"
        ]["candidate_better_overall"],
        "mechanism_all_five_seeds": all(
            value["all_mechanism_gates_pass"] for value in mechanism_values
        ),
        "runtime_all_contexts_under_2ms": all(
            value["passes_2ms"] for value in runtime_summary.values()
        ),
        "runtime_cache_equivalence": max(runtime_equivalence) <= 1e-6,
        "untouched_replication": replication_value["gates"]["comparison_passed"],
    }
    claim_passed = all(gates.values())
    return {
        "schema": CLAIM_SCHEMA,
        "preregistration_sha256": V4_PREREGISTRATION_SHA256,
        "protocol_sha256": next(iter(protocol_hashes)),
        "coordinate_sha256": next(iter(coordinate_hashes)),
        "gates": gates,
        "claim_passed": claim_passed,
        "publication_evidence": False,
        "methodology_blockers": [
            "nested resampling misrepresents shared test-flow uncertainty",
            "original cache audit does not cover compact sparse lanes",
            "conditional LSTM step timing excludes query-relative state construction",
            "a revised scientific claim requires a new prospective analysis freeze",
        ],
        "verdict": (
            "frozen_compound_claim_supported"
            if claim_passed
            else "frozen_compound_claim_rejected"
        ),
        "runtime": runtime_summary,
        "max_cache_equivalence_error": max(runtime_equivalence),
        "comparisons": comparisons,
        "secondary_comparisons": secondary_comparisons,
        "mechanism_results": mechanism_values,
        "replication": replication_value,
        "claim_limits": [
            "forecasting is not intrusion-detection efficacy",
            "no faster-than-LSTM claim",
            "no first-method claim",
            "results apply only to declared sources, task, and hardware",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--recency", type=Path, required=True)
    parser.add_argument("--gru", type=Path, required=True)
    parser.add_argument("--tcn", type=Path, required=True)
    parser.add_argument("--s4d", type=Path, required=True)
    parser.add_argument("--persistence", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, nargs=5, required=True)
    parser.add_argument("--runtime", type=Path, nargs=15, required=True)
    parser.add_argument("--replication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_manifest(
        args.output,
        build(
            lstm=args.lstm,
            local=args.local,
            recency=args.recency,
            gru=args.gru,
            tcn=args.tcn,
            s4d=args.s4d,
            persistence=args.persistence,
            mechanisms=args.mechanism,
            runtimes=args.runtime,
            replication=args.replication,
        ),
    )


if __name__ == "__main__":
    main()

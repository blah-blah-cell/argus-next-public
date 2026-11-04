"""Separate post-hoc crossed-bootstrap sensitivity on already-saved losses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .crossed_bootstrap import crossed_bootstrap
from .flow_v2_protocol import write_manifest
from .flow_v4_screen_report import _metric

CANDIDATE = "ahem_v5_source_aligned"


def overlap_clusters(coordinates: list[dict], context: int) -> list[tuple]:
    """Connected components of overlapping context-through-target intervals."""
    if type(context) is not int or context < 1:
        raise ValueError("positive context required")
    ordered = []
    for index, coordinate in enumerate(coordinates):
        anchor, target = coordinate["anchor_index"], coordinate["target_index"]
        if (
            type(anchor) is not int
            or type(target) is not int
            or not context - 1 <= anchor < target
        ):
            raise ValueError("invalid causal coordinate")
        ordered.append(
            (
                coordinate["source_file"],
                coordinate["segment_id"],
                anchor - context + 1,
                target,
                index,
            )
        )
    assignments = [()] * len(ordered)
    source_segment = None
    right = -1
    component = -1
    for source, segment, start, end, index in sorted(ordered):
        if (source, segment) != source_segment or start > right:
            component += 1
            source_segment = (source, segment)
            right = end
        else:
            right = max(right, end)
        assignments[index] = (source, segment, component)
    return assignments


def reanalyze(
    paths: list[Path], *, resamples: int = 10000, seed: int = 20260905
) -> dict:
    runs = [_metric(path) for path in paths]
    if not runs:
        raise ValueError("no saved diagnostic runs provided")
    paired_fields = ("sample_coordinates", "per_sample_persistence_weighted_smooth_l1")
    for run in runs[1:]:
        if any(
            run[field] != runs[0][field]
            for field in ("source_provenance", "protocol_sha256", "coordinate_sha256")
        ) or any(
            run["test"][field] != runs[0]["test"][field] for field in paired_fields
        ):
            raise ValueError("saved runs are not paired under one source/protocol")
    by_model: dict[str, dict[int, dict]] = {}
    for run in runs:
        model, training_seed = run["model"]["name"], run["training"]["seed"]
        if training_seed in by_model.setdefault(model, {}):
            raise ValueError("duplicate model/seed")
        by_model[model][training_seed] = run
    if CANDIDATE not in by_model or len(by_model) < 2:
        raise ValueError("candidate and at least one reference required")
    seeds = sorted(by_model[CANDIDATE])
    if any(sorted(group) != seeds for group in by_model.values()):
        raise ValueError("all models must share training seeds")
    coordinates = runs[0]["test"]["sample_coordinates"]
    clusterings = {
        "flow": [
            (c["source_file"], c["segment_id"], c["flow_generation"])
            for c in coordinates
        ],
        "overlapping_context_target_components": overlap_clusters(
            coordinates, runs[0]["protocol"]["config"]["context_packets"]
        ),
    }
    results = {}
    for reference in sorted(set(by_model) - {CANDIDATE}):
        differences = [
            [
                b - a
                for a, b in zip(
                    by_model[CANDIDATE][s]["test"]["per_sample_weighted_smooth_l1"],
                    by_model[reference][s]["test"]["per_sample_weighted_smooth_l1"],
                    strict=True,
                )
            ]
            for s in seeds
        ]
        results[reference] = {
            name: (
                crossed_bootstrap(
                    differences, cluster_ids=ids, resamples=resamples, seed=seed
                )
                if len(set(ids)) > 1
                else {
                    "status": "insufficient_independent_clusters",
                    "cluster_count": len(set(ids)),
                    "publication_evidence": False,
                }
            )
            for name, ids in clusterings.items()
        }
    return {
        "schema": "argus.v5-posthoc-crossed-sensitivity/v1",
        "publication_evidence": False,
        "study_role": "posthoc_sensitivity",
        "candidate": CANDIDATE,
        "training_seeds": seeds,
        "inputs": [
            {
                "path": str(p.resolve()),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in paths
        ],
        "results": results,
        "limitations": [
            "saved diagnostic losses only; no new holdout access or training",
            "shared draws repair the crossed-factor mismatch, not all dependence",
            "flow groups can share packet context; overlap components address that overlap only",
            "nonoverlap does not guarantee temporal independence or cross-day generalization",
            "historical intervals and the negative verdict are not overwritten",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    result = reanalyze(args.metrics, resamples=args.resamples)
    write_manifest(args.output, result)
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()

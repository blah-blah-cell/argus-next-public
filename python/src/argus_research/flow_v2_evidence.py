"""Validate and score independently labelled causal evidence retrieval.

Forecast loss, attention weights, and the chunk that minimizes a model's own
loss are not retrieval ground truth.  This module instead requires a frozen
task whose candidate and relevant evidence spans are bound to checksum-valid
ARGUSLR2 flow streams before predictions are admitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import fmean
from typing import Any

from .continual_learning import flow_sequences_from_record, iter_sequence_records_v2
from .flow_v2_protocol import FEATURE_SCHEMA, write_manifest

EVIDENCE_SCHEMA = "argus.flow-v2-causal-evidence-task/v1"
PREDICTION_SCHEMA = "argus.flow-v2-retrieval-predictions/v1"
REPORT_SCHEMA = "argus.flow-v2-retrieval-report/v1"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_sha256(task: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in task.items() if key != "task_sha256"}
    return _canonical_sha256(unsigned)


def _journal_index(journal: Path) -> dict[tuple[str, str, int], int]:
    index: dict[tuple[str, str, int], int] = {}
    for record in iter_sequence_records_v2(journal):
        for stream in flow_sequences_from_record(record):
            key = (stream.source_file, stream.segment_id, stream.flow_id)
            if key in index:
                raise ValueError(f"duplicate flow-stream identity in journal: {key}")
            index[key] = len(stream.features)
    if not index:
        raise ValueError("no checksum-valid ARGUSLR2 flow streams were found")
    return index


def validate_evidence_task(task: dict[str, Any], journal: Path) -> dict[str, Any]:
    if task.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError("unsupported causal evidence task schema")
    if task.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("evidence task uses another feature schema")
    expected_hash = task_sha256(task)
    if task.get("task_sha256") != expected_hash:
        raise ValueError("causal evidence task fingerprint is absent or invalid")

    labeling = task.get("labeling")
    if not isinstance(labeling, dict):
        raise ValueError("evidence task lacks labeling provenance")
    if labeling.get("model_outputs_used") is not False:
        raise ValueError("evidence labels must be independent of model outputs")
    if labeling.get("frozen_before_predictions") is not True:
        raise ValueError("evidence labels were not frozen before prediction")
    for field in ("method", "protocol", "conflict_resolution"):
        if not isinstance(labeling.get(field), str) or not labeling[field].strip():
            raise ValueError(f"labeling provenance lacks {field}")
    annotator_count = labeling.get("annotator_count")
    if not isinstance(annotator_count, int) or annotator_count < 1:
        raise ValueError("labeling provenance needs at least one annotator")

    streams = _journal_index(journal)
    items = task.get("items")
    if not isinstance(items, list) or len(items) < 2:
        raise ValueError("evidence task needs at least two queries")
    query_ids: set[str] = set()
    source_files: set[str] = set()
    for item in items:
        query_id = item.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("evidence query lacks query_id")
        if query_id in query_ids:
            raise ValueError(f"duplicate evidence query_id {query_id!r}")
        query_ids.add(query_id)
        key = (item.get("source_file"), item.get("segment_id"), item.get("flow_id"))
        if key not in streams:
            raise ValueError(f"evidence query references an absent flow stream: {key}")
        source_files.add(str(key[0]))
        query_index = item.get("query_index")
        if (
            not isinstance(query_index, int)
            or query_index < 1
            or query_index >= streams[key]
        ):
            raise ValueError(f"query {query_id!r} has an invalid query_index")

        candidates = item.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise ValueError(f"query {query_id!r} needs at least two candidates")
        candidate_ids: set[str] = set()
        for candidate in candidates:
            evidence_id = candidate.get("evidence_id")
            start = candidate.get("start_index")
            end = candidate.get("end_index")
            if not isinstance(evidence_id, str) or not evidence_id:
                raise ValueError(f"query {query_id!r} has an unnamed candidate")
            if evidence_id in candidate_ids:
                raise ValueError(
                    f"query {query_id!r} repeats candidate {evidence_id!r}"
                )
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or start >= end
                or end > query_index
            ):
                raise ValueError(f"candidate {evidence_id!r} is not causally prior")
            candidate_ids.add(evidence_id)

        relevant = item.get("relevant_evidence_ids")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(f"query {query_id!r} lacks relevant evidence")
        if len(relevant) != len(set(relevant)):
            raise ValueError(f"query {query_id!r} repeats relevant evidence")
        if not set(relevant) <= candidate_ids:
            raise ValueError(f"query {query_id!r} labels a non-candidate as relevant")

    review = task.get("review", {})
    journal_members = []
    for source_file in sorted(source_files):
        path = journal / f"{source_file}.arguslr"
        if not path.is_file():
            raise ValueError(f"evidence journal source file is absent: {source_file}")
        journal_members.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if task.get("journal_members") != journal_members:
        raise ValueError("causal evidence task is not bound to the journal bytes")
    publication_eligible = (
        review.get("status") == "approved"
        and isinstance(review.get("reviewer"), str)
        and bool(review.get("reviewer", "").strip())
        and isinstance(review.get("approved_utc"), str)
        and bool(review.get("approved_utc", "").strip())
    )
    return {
        "task_sha256": expected_hash,
        "queries": len(items),
        "source_files": sorted(source_files),
        "publication_eligible": publication_eligible,
        "publication_blockers": (
            [] if publication_eligible else ["independent label review is not approved"]
        ),
    }


def _score(
    task_items: list[dict[str, Any]],
    rankings: dict[str, list[str]],
    cutoffs: tuple[int, ...],
) -> dict[str, Any]:
    reciprocal_ranks: list[float] = []
    precision = {cutoff: [] for cutoff in cutoffs}
    recall = {cutoff: [] for cutoff in cutoffs}
    for item in task_items:
        relevant = set(item["relevant_evidence_ids"])
        ranking = rankings[item["query_id"]]
        first = next(
            (
                rank
                for rank, evidence_id in enumerate(ranking, 1)
                if evidence_id in relevant
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if first is None else 1.0 / first)
        for cutoff in cutoffs:
            hits = len(set(ranking[:cutoff]) & relevant)
            precision[cutoff].append(hits / cutoff)
            recall[cutoff].append(hits / len(relevant))
    return {
        "mrr": fmean(reciprocal_ranks),
        "precision_at_k": {
            str(cutoff): fmean(values) for cutoff, values in precision.items()
        },
        "recall_at_k": {
            str(cutoff): fmean(values) for cutoff, values in recall.items()
        },
    }


def evaluate_predictions(
    task: dict[str, Any],
    predictions: dict[str, Any],
    journal: Path,
    *,
    cutoffs: tuple[int, ...] = (1, 3, 5),
    random_repetitions: int = 1_000,
    seed: int = 20260825,
) -> dict[str, Any]:
    validation = validate_evidence_task(task, journal)
    if predictions.get("schema") != PREDICTION_SCHEMA:
        raise ValueError("unsupported retrieval prediction schema")
    if predictions.get("task_sha256") != validation["task_sha256"]:
        raise ValueError("predictions belong to another evidence task")
    if not isinstance(predictions.get("model"), str) or not predictions["model"]:
        raise ValueError("retrieval predictions do not identify a model")
    if random_repetitions < 100:
        raise ValueError("random baseline requires at least 100 repetitions")
    if not cutoffs or min(cutoffs) <= 0 or len(set(cutoffs)) != len(cutoffs):
        raise ValueError("retrieval cutoffs must be unique positive integers")

    task_items = task["items"]
    task_by_query = {item["query_id"]: item for item in task_items}
    prediction_items = predictions.get("items")
    if not isinstance(prediction_items, list):
        raise ValueError("retrieval predictions contain no items")
    rankings: dict[str, list[str]] = {}
    for item in prediction_items:
        query_id = item.get("query_id")
        ranking = item.get("ranked_evidence_ids")
        if query_id not in task_by_query or query_id in rankings:
            raise ValueError("prediction query set is duplicated or unknown")
        if not isinstance(ranking, list) or len(ranking) != len(set(ranking)):
            raise ValueError(f"query {query_id!r} has an invalid ranking")
        candidates = {
            value["evidence_id"] for value in task_by_query[query_id]["candidates"]
        }
        if set(ranking) != candidates:
            raise ValueError(
                f"query {query_id!r} ranking is not a candidate permutation"
            )
        rankings[query_id] = ranking
    if set(rankings) != set(task_by_query):
        raise ValueError("prediction query set is incomplete")

    recency = {
        item["query_id"]: [
            value["evidence_id"]
            for value in sorted(
                item["candidates"], key=lambda value: value["end_index"], reverse=True
            )
        ]
        for item in task_items
    }
    rng = random.Random(seed)
    random_scores = []
    for _ in range(random_repetitions):
        random_rankings = {}
        for item in task_items:
            values = [value["evidence_id"] for value in item["candidates"]]
            rng.shuffle(values)
            random_rankings[item["query_id"]] = values
        random_scores.append(_score(task_items, random_rankings, cutoffs))
    random_baseline = {
        "mrr": fmean(value["mrr"] for value in random_scores),
        "precision_at_k": {
            str(cutoff): fmean(
                value["precision_at_k"][str(cutoff)] for value in random_scores
            )
            for cutoff in cutoffs
        },
        "recall_at_k": {
            str(cutoff): fmean(
                value["recall_at_k"][str(cutoff)] for value in random_scores
            )
            for cutoff in cutoffs
        },
    }
    ordered_queries = sorted(rankings)
    shuffled = {
        query_id: rankings[ordered_queries[(index + 1) % len(ordered_queries)]]
        for index, query_id in enumerate(ordered_queries)
    }
    return {
        "schema": REPORT_SCHEMA,
        "task": "causal_evidence_retrieval_only",
        "not_intrusion_detection_efficacy": True,
        "task_sha256": validation["task_sha256"],
        "model": predictions["model"],
        "queries": len(task_items),
        "cutoffs": list(cutoffs),
        "model_metrics": _score(task_items, rankings, cutoffs),
        "baselines": {
            "recency": _score(task_items, recency, cutoffs),
            "random_mean": random_baseline,
            "query_rotation_negative_control": _score(task_items, shuffled, cutoffs),
            "random_repetitions": random_repetitions,
            "seed": seed,
        },
        "publication_eligible_labels": validation["publication_eligible"],
        "publication_blockers": validation["publication_blockers"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-repetitions", type=int, default=1_000)
    args = parser.parse_args()
    task = json.loads(args.task.read_text(encoding="utf-8"))
    if args.predictions is None:
        result = validate_evidence_task(task, args.journal)
    else:
        predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
        result = evaluate_predictions(
            task,
            predictions,
            args.journal,
            random_repetitions=args.random_repetitions,
        )
    write_manifest(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

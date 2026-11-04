"""Build a real-trace, causally keyed cache-recall probe without synthetic labels."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path

import numpy as np


def build(
    inputs: list[Path], output: Path, minimum_gap: int, maximum_gap: int
) -> dict[str, object]:
    current_rows = []
    cached_rows = []
    ages = []
    values = []
    is_test = []
    source_ids = []
    query_indices = []
    eligible_queries = 0
    retained_queries = 0
    for source_index, path in enumerate(inputs):
        source = np.load(path)
        features = source["features"].astype(np.float32)
        flow_ids = source["flow_ids"]
        split = int(len(features) * 0.8)
        cache: dict[int, deque[tuple[int, np.ndarray]]] = defaultdict(
            lambda: deque(maxlen=256)
        )
        complete_history: dict[int, deque[tuple[int, np.ndarray]]] = defaultdict(deque)
        for index, (feature, flow_id) in enumerate(
            zip(features, flow_ids, strict=True)
        ):
            history = cache[int(flow_id)]
            full_history = complete_history[int(flow_id)]
            complete_prior = next(
                (
                    (prior_index, prior_feature)
                    for prior_index, prior_feature in reversed(full_history)
                    if minimum_gap <= index - prior_index <= maximum_gap
                ),
                None,
            )
            prior = next(
                (
                    (prior_index, prior_feature)
                    for prior_index, prior_feature in reversed(history)
                    if minimum_gap <= index - prior_index <= maximum_gap
                ),
                None,
            )
            if complete_prior is not None:
                eligible_queries += 1
            if prior is not None:
                retained_queries += 1
                prior_index, prior_feature = prior
                current_rows.append(feature)
                cached_rows.append(prior_feature)
                ages.append((index - prior_index) / maximum_gap)
                values.append(prior_feature[0])  # earlier observed log packet length
                is_test.append(index >= split)
                source_ids.append(source_index)
                query_indices.append(index)
            history.append((index, feature))
            full_history.append((index, feature))
    if not current_rows:
        raise ValueError("no same-flow recall pairs in the requested temporal gap")
    train_values = np.asarray(values)[~np.asarray(is_test)]
    if not len(train_values):
        raise ValueError("no train-split recall pairs")
    threshold = float(np.median(train_values))
    labels = (np.asarray(values) >= threshold).astype(np.int64)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        current=np.asarray(current_rows, dtype=np.float32),
        cached=np.asarray(cached_rows, dtype=np.float32),
        age=np.asarray(ages, dtype=np.float32),
        label=labels,
        is_test=np.asarray(is_test, dtype=np.bool_),
        source_id=np.asarray(source_ids, dtype=np.int16),
        query_index=np.asarray(query_indices, dtype=np.int64),
    )
    result = {
        "benchmark": "real_trace_keyed_cache_recall",
        "not_intrusion_detection_efficacy": True,
        "inputs": [str(path) for path in inputs],
        "minimum_global_packet_gap": minimum_gap,
        "maximum_global_packet_gap": maximum_gap,
        "cache_entries_per_flow": 256,
        "eligible_same_flow_queries": eligible_queries,
        "retained_same_flow_queries": retained_queries,
        "bounded_cache_retention_recall": retained_queries / eligible_queries
        if eligible_queries
        else None,
        "recall_pairs": len(labels),
        "train_pairs": int((~np.asarray(is_test)).sum()),
        "test_pairs": int(np.asarray(is_test).sum()),
        "target": "whether the causally cached earlier same-flow packet is at-or-above the training median log length",
        "threshold_fit_scope": "chronological_train_suffix_excluded",
    }
    output.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-gap", type=int, default=10_000)
    parser.add_argument("--maximum-gap", type=int, default=30_000)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.input, args.output, args.minimum_gap, args.maximum_gap), indent=2
        )
    )


if __name__ == "__main__":
    main()

"""Real-trace recall probe for a bounded raw-ring plus chunk-summary cache."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path

import numpy as np


def build(
    inputs: list[Path],
    output: Path,
    minimum_gap: int,
    maximum_gap: int,
    chunk_size: int,
    chunk_capacity: int,
) -> dict[str, object]:
    current_rows: list[np.ndarray] = []
    cached_rows: list[np.ndarray] = []
    ages: list[float] = []
    target_values: list[float] = []
    is_test: list[bool] = []
    source_ids: list[int] = []
    query_indices: list[int] = []
    eligible = retained = 0
    for source_id, path in enumerate(inputs):
        source = np.load(path)
        features = source["features"].astype(np.float32)
        ids = source["flow_ids"]
        split = int(len(features) * 0.8)
        active: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        bounded: dict[int, deque[tuple[int, int, np.ndarray]]] = defaultdict(
            lambda: deque(maxlen=chunk_capacity)
        )
        complete: dict[int, deque[tuple[int, int, np.ndarray]]] = defaultdict(deque)
        for index, (feature, flow_id) in enumerate(zip(features, ids, strict=True)):
            key = int(flow_id)
            full_match = next(
                (
                    (end, summary)
                    for _, end, summary in reversed(complete[key])
                    if minimum_gap <= index - end <= maximum_gap
                ),
                None,
            )
            bounded_match = next(
                (
                    (end, summary)
                    for _, end, summary in reversed(bounded[key])
                    if minimum_gap <= index - end <= maximum_gap
                ),
                None,
            )
            if full_match is not None:
                eligible += 1
            if bounded_match is not None:
                retained += 1
                end, summary = bounded_match
                current_rows.append(feature)
                cached_rows.append(summary)
                ages.append((index - end) / maximum_gap)
                target_values.append(float(summary[0]))
                is_test.append(index >= split)
                source_ids.append(source_id)
                query_indices.append(index)
            active[key].append((index, feature))
            if len(active[key]) == chunk_size:
                packets = active[key]
                start, end = packets[0][0], packets[-1][0]
                # Mean/max/min preserve a compact, causal evidence summary.
                values = np.stack([packet for _, packet in packets])
                summary = (
                    values.mean(axis=0) + values.max(axis=0) + values.min(axis=0)
                ) / 3.0
                complete[key].append((start, end, summary))
                bounded[key].append((start, end, summary))
                active[key].clear()
    if not current_rows:
        raise ValueError("no chunk recall pairs at this temporal gap")
    train_values = np.asarray(target_values)[~np.asarray(is_test)]
    threshold = float(np.median(train_values))
    labels = (np.asarray(target_values) >= threshold).astype(np.int64)
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
        "benchmark": "real_trace_chunk_cache_recall",
        "not_intrusion_detection_efficacy": True,
        "minimum_global_packet_gap": minimum_gap,
        "maximum_global_packet_gap": maximum_gap,
        "chunk_size_packets": chunk_size,
        "chunk_capacity_per_flow": chunk_capacity,
        "eligible_chunk_queries": eligible,
        "retained_chunk_queries": retained,
        "bounded_chunk_retention_recall": retained / eligible if eligible else None,
        "recall_pairs": len(labels),
        "train_pairs": int((~np.asarray(is_test)).sum()),
        "test_pairs": int(np.asarray(is_test).sum()),
        "target": "whether the causally retrieved chunk's mean/max/min packet-size summary is at-or-above its train median",
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
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--chunk-capacity", type=int, default=512)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.input,
                args.output,
                args.minimum_gap,
                args.maximum_gap,
                args.chunk_size,
                args.chunk_capacity,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

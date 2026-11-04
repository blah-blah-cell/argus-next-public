"""Training-journal census and nested sampling; no training or target input to reader."""

from __future__ import annotations

import random
from typing import Any

import torch

from .ahem_fifo_teacher import build_cache
from .ahem_nlm_loop_20261019 import Entry, Retention, encode_record, source_metrics


def nested_coordinates(
    population: list[tuple[int, int]],
    original: list[tuple[int, int]],
    count: int,
    seed: int,
) -> list[tuple[int, int]]:
    if len(set(population)) != len(population) or len(set(original)) != len(original):
        raise ValueError("duplicate coordinates")
    if not set(original) <= set(population) or not len(original) <= count <= len(
        population
    ):
        raise ValueError("nested sample outside observed-successor population")
    remaining = sorted(set(population) - set(original))
    extra = random.Random(seed).sample(remaining, count - len(original))
    return list(original) + sorted(extra)


@torch.no_grad()
def record_forecasts(
    reader: Any, readout: Any, scale: torch.Tensor, record: Any
) -> dict:
    """Batched source-only encoding then FIFO chronological admission; offline diagnostic."""
    encoded = encode_record(reader, record, batch=32)
    x = record.features
    n = len(x)
    previous = record.previous_same_flow.tolist()
    following = record.next_same_flow
    anchors = torch.nonzero(following >= 0).flatten()
    sources = torch.full((n, 16), -1, dtype=torch.long)
    successors = torch.full_like(sources, -1)
    valid = torch.zeros(n, 16, dtype=torch.bool)
    keys = encoded["keys"].numpy()
    footprints = encoded["footprint"].tolist()
    lanes: dict[int, Retention] = {}
    for ordinal, flow in enumerate(record.flow_ids.tolist()):
        lane = lanes.setdefault(flow, Retention("fifo"))
        source = previous[ordinal]
        before = previous[source] if source >= 0 else -1
        incoming = (
            Entry(source, ordinal, before, footprints[source], keys[source])
            if before >= max(0, ordinal - 9999)
            else None
        )
        entries = lane.update(ordinal, incoming)
        if entries:
            count = len(entries)
            sources[ordinal, :count] = torch.tensor([e.source for e in entries])
            successors[ordinal, :count] = torch.tensor([e.successor for e in entries])
            valid[ordinal, :count] = True
    chunks = []
    for start in range(0, len(anchors), 128):
        index = anchors[start : start + 128]
        mask, src, nxt = valid[index], sources[index], successors[index]
        values = x[nxt.clamp_min(0), :2] * mask[..., None]
        age = reader._normalized_gap(
            src, index[:, None], torch.full((len(index),), 10000.0)
        )
        scores = reader._scores(
            encoded["keys"][index], encoded["keys"][src.clamp_min(0)], age
        )
        scores = scores.masked_fill(~mask, -1e4)
        weights, raw, _, prediction = readout(
            scores, mask, values, x[index, :2], encoded["phase"][index]
        )
        corrected = prediction + (1 - scale) * (x[index, :2] - prediction)
        prediction = torch.where((index < 127)[:, None], corrected, prediction)
        chunks.append(
            {
                "anchors": index,
                "prediction": prediction,
                "weights": weights,
                "raw": raw,
                "valid": mask,
                "source_ordinals": src,
                "successor_ordinals": nxt,
                "successors": values,
                "context": encoded["phase"][index],
                "current": x[index, :2],
                "target": x[following[index], :2],
            }
        )
    if not chunks:
        raise ValueError("record has no observed-successor anchors")
    result = {key: torch.cat([c[key] for c in chunks]) for key in chunks[0]}
    result["packet_count"] = n
    result["no_target_count"] = n - len(anchors)
    _, counts = torch.unique(record.flow_ids, return_counts=True)
    result["singleton_flows"] = int((counts == 1).sum())
    return result


def make_cache(rows: dict, coordinates: list[tuple[int, int]], records: list) -> dict:
    saved = {
        "arms": {
            "fifo": {
                key: rows[key]
                for key in (
                    "prediction",
                    "weights",
                    "valid",
                    "source_ordinals",
                    "successor_ordinals",
                    "successors",
                )
            }
        },
        "coordinates": torch.tensor(coordinates, dtype=torch.long),
        "context": rows["context"],
        "current": rows["current"],
        "target": rows["target"],
        "original_parent_prediction": rows["prediction"],
    }
    return build_cache(saved, records, 2.5)


def describe(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    return source_metrics(prediction.double().numpy(), target.double().numpy())


def population_summary(rows: dict) -> dict:
    valid_count = rows["valid"].sum(1)
    groups = {
        "all": torch.ones(len(valid_count), dtype=torch.bool),
        "no_memory": valid_count == 0,
        "one_candidate": valid_count == 1,
        "multiple_candidates": valid_count > 1,
        "short_global_history": rows["anchors"] < 127,
        "full_10000_prefix": rows["anchors"] >= 9999,
    }
    result = {}
    for group, mask in groups.items():
        result[group] = {
            "queries": int(mask.sum()),
            "models": {
                name: describe(rows[name][mask], rows["target"][mask])
                for name in ("prediction", "current", "ewma")
            }
            if mask.any()
            else {},
        }
    result["strata_overlap"] = True
    result["census_not_independent_replication"] = True
    return result

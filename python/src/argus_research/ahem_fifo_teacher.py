"""Causally checked FIFO cache and target-free centered score application."""

from __future__ import annotations

import torch

from .ahem_score_span import causal_features, score_cap, score_inputs


def build_cache(
    saved: dict, records: list, beta: float = 2.5
) -> dict[str, torch.Tensor]:
    arm = saved["arms"]["fifo"]
    source_current = torch.zeros_like(arm["successors"])
    for index, (record_index, anchor) in enumerate(saved["coordinates"].tolist()):
        record = records[record_index]
        valid = arm["valid"][index]
        source = arm["source_ordinals"][index, valid]
        successor = arm["successor_ordinals"][index, valid]
        if (
            (source < 0).any()
            or (source >= anchor).any()
            or (anchor - source > 10000).any()
            or (successor > anchor).any()
            or (successor <= source).any()
        ):
            raise ValueError("FIFO transition not causally completed inside horizon")
        if (
            not torch.equal(record.next_same_flow[source], successor)
            or not (record.flow_ids[source] == record.flow_ids[anchor]).all()
            or not torch.equal(
                record.features[successor, :2], arm["successors"][index, valid]
            )
        ):
            raise ValueError("FIFO source/successor flow or physical values differ")
        target = int(record.next_same_flow[anchor])
        if (
            target <= anchor
            or not torch.equal(record.features[anchor, :2], saved["current"][index])
            or not torch.equal(record.features[target, :2], saved["target"][index])
        ):
            raise ValueError("FIFO forecast current/target binding differs")
        source_current[index, valid] = record.features[source, :2]
    scores = score_inputs(arm["weights"], arm["valid"], beta)
    ordinal = saved["coordinates"][:, 1]
    features = causal_features(
        scores,
        ordinal,
        arm["source_ordinals"],
        source_current,
        arm["successors"],
        saved["current"],
        saved["context"],
    )
    return {
        "features": features,
        "scores": scores,
        "valid": arm["valid"],
        "successors": arm["successors"],
        "current": saved["current"],
        "context": saved["context"],
        "ordinal": ordinal,
        "startup_parent": saved["original_parent_prediction"],
        "target": saved["target"],
        "reference": arm["prediction"],
        "coordinates": saved["coordinates"],
    }


def centered_attention(
    scores: torch.Tensor,
    hidden: torch.Tensor,
    valid: torch.Tensor,
    theta: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    h = hidden.double() @ theta.double()
    base = (beta * scores.double()).masked_fill(~valid, -1e4).softmax(1) * valid
    base = base / base.sum(1, keepdim=True).clamp_min(1e-30)
    mean = (base * h).sum(1, keepdim=True)
    cap = score_cap(scores, valid, True).double()
    correction = cap * ((h - mean) / cap).tanh()
    probability = (beta * (scores.double() + correction)).masked_fill(
        ~valid, -1e4
    ).softmax(1) * valid
    probability = probability / probability.sum(1, keepdim=True).clamp_min(1e-30)
    return probability.to(scores.dtype), correction

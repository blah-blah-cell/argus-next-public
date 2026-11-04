"""Target-free pooled FIFO adapter application; training lives in the runner."""

from __future__ import annotations

import torch

from .ahem_context_teacher import ContextTeacherReadout
from .ahem_fifo_teacher import centered_attention
from .ahem_forecast_teacher import from_weights, inputs
from .ahem_score_span import causal_features, score_inputs


def projected_cache(
    projected: dict[str, torch.Tensor],
    parent_prediction: torch.Tensor,
    parent_weights: torch.Tensor,
    beta: float,
) -> dict[str, torch.Tensor]:
    """Use only audited observed projection fields; no future target argument."""
    valid = projected["phase_valid"]
    current = projected["persistence"]
    scores = score_inputs(parent_weights, valid, beta)
    source = projected["source_ordinals"]
    ordinal = projected["anchor_ordinals"]
    if (valid & ((source < 0) | (source >= ordinal[:, None]))).any():
        raise ValueError("noncausal projected source")
    features = causal_features(
        scores,
        ordinal,
        source,
        projected["key_phase"][..., :2],
        projected["successor_dynamic"],
        current,
        projected["query_phase"],
    )
    return {
        "features": features,
        "scores": scores,
        "valid": valid,
        "successors": projected["successor_dynamic"],
        "current": current,
        "context": projected["query_phase"],
        "ordinal": ordinal,
        "startup_parent": parent_prediction,
        "reference": parent_prediction,
    }


def apply(
    model: ContextTeacherReadout,
    cache: dict[str, torch.Tensor],
    theta: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """No label or coordinate lookup: only the eight causal inference fields."""
    if theta is None:
        prediction, weights, _ = model(*inputs(cache))
        return prediction, weights
    hidden = model.reranker[:2](
        model.ranking_features(cache["features"], cache["context"])
    )
    weights, _ = centered_attention(
        cache["scores"], hidden, cache["valid"], theta, model.beta
    )
    prediction, _ = from_weights(model.head, cache, weights)
    return prediction, weights

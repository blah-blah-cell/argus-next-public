"""Independent utility targets and pre-training gates for AHEM-v6."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .ahem_v6 import AhemV6Output


def dynamic_loss(
    prediction: Tensor, target: Tensor, *, reduction: str = "mean"
) -> Tensor:
    if prediction.shape[-1] != 2 or target.shape[-1] != 2:
        raise ValueError("dynamic loss expects length and interarrival fields")
    values = F.smooth_l1_loss(prediction, target, reduction="none")
    values = (values * prediction.new_tensor((1.0, 4.0))).sum(dim=-1) / 5.0
    return values.mean() if reduction == "mean" else values


def frozen_utility_distribution(
    successor_dynamic: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    temperature: float = 0.04,
) -> tuple[Tensor, Tensor]:
    """Define relevance from observed successor outcomes, never model attention.

    This target is legal only during training/development evaluation.  It uses
    the forecast target to label which already-observed transition would have
    been most useful, and must never be computed by online inference.
    """
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("utility temperature must be finite and positive")
    if successor_dynamic.ndim != 3 or successor_dynamic.size(-1) != 2:
        raise ValueError("successor dynamics must be [batch, slots, 2]")
    if target.shape != (successor_dynamic.size(0), 2):
        raise ValueError("utility target has an incompatible shape")
    if valid.shape != successor_dynamic.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("utility validity mask is incompatible")
    slot_loss = dynamic_loss(
        successor_dynamic,
        target[:, None].expand_as(successor_dynamic),
        reduction="none",
    ).masked_fill(~valid, float("inf"))
    logits = (-slot_loss / temperature).masked_fill(~valid, -1e4)
    distribution = logits.softmax(dim=-1) * valid.to(logits.dtype)
    distribution = distribution / distribution.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return distribution, slot_loss


def retrieval_cross_entropy(
    retrieval_logits: Tensor, target_distribution: Tensor, valid: Tensor
) -> Tensor:
    if retrieval_logits.shape != target_distribution.shape or valid.shape != (
        retrieval_logits.shape
    ):
        raise ValueError("retrieval objective tensors have incompatible shapes")
    informative = valid.sum(dim=-1) >= 2
    if not informative.any():
        return retrieval_logits.sum() * 0.0
    log_probabilities = retrieval_logits.masked_fill(~valid, -1e4).log_softmax(dim=-1)
    cross_entropy = -(target_distribution * log_probabilities).sum(dim=-1)
    # Frozen v6.1 used a floor of 1.0. Changing this training objective after
    # the checkpoints were selected would make the recorded run irreproducible.
    # A log(2) floor belongs to a separately versioned future objective.
    normalizer = valid.sum(dim=-1).to(retrieval_logits.dtype).log().clamp_min(1.0)
    return (cross_entropy[informative] / normalizer[informative]).mean()


def utility_gap_ranking(
    retrieval_logits: Tensor,
    slot_loss: Tensor,
    valid: Tensor,
    *,
    margin: float = 0.10,
) -> Tensor:
    """Rank the oracle above consequential alternatives.

    Unlike unweighted top-1 supervision, this objective gives little weight to
    alternatives whose observed successor is almost as useful as the oracle
    and more weight to mistakes with high forecast regret.
    """
    if margin < 0.0 or not math.isfinite(margin):
        raise ValueError("ranking margin must be finite and non-negative")
    if retrieval_logits.shape != slot_loss.shape or valid.shape != slot_loss.shape:
        raise ValueError("ranking tensors have incompatible shapes")
    informative = valid.sum(dim=-1) >= 2
    if not informative.any():
        return retrieval_logits.sum() * 0.0
    oracle_index = slot_loss.argmin(dim=-1)
    oracle_loss = slot_loss.gather(1, oracle_index[:, None])
    oracle_logit = retrieval_logits.gather(1, oracle_index[:, None])
    regret = (slot_loss - oracle_loss).clamp_min(0.0).masked_fill(~valid, 0.0)
    negatives = valid & (regret > 1e-8)
    scaled_regret = regret / regret.masked_fill(~negatives, 0.0).amax(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    violations = F.softplus(retrieval_logits - oracle_logit + margin)
    numerator = (violations * scaled_regret * negatives).sum(dim=-1)
    denominator = (scaled_regret * negatives).sum(dim=-1).clamp_min(1e-8)
    usable = informative & negatives.any(dim=-1)
    if not usable.any():
        return retrieval_logits.sum() * 0.0
    return (numerator[usable] / denominator[usable]).mean()


def training_loss(
    output: AhemV6Output,
    target: Tensor,
    *,
    utility_temperature: float = 0.04,
    ranking_weight: float = 0.0,
    ranking_margin: float = 0.10,
) -> tuple[Tensor, dict[str, Tensor]]:
    if ranking_weight < 0.0 or not math.isfinite(ranking_weight):
        raise ValueError("ranking weight must be finite and non-negative")
    relevance, slot_loss = frozen_utility_distribution(
        output.successor_dynamic,
        target,
        output.valid,
        temperature=utility_temperature,
    )
    forecast = dynamic_loss(output.prediction, target)
    local = dynamic_loss(output.local_prediction, target)
    memory = dynamic_loss(output.memory_candidate, target)
    retrieval = retrieval_cross_entropy(
        output.retrieval_logits, relevance, output.valid
    )
    ranking = (
        utility_gap_ranking(
            output.retrieval_logits, slot_loss, output.valid, margin=ranking_margin
        )
        if ranking_weight > 0.0
        else forecast.new_zeros(())
    )
    local_example = dynamic_loss(output.local_prediction, target, reduction="none")
    memory_example = dynamic_loss(output.memory_candidate, target, reduction="none")
    gate_target = torch.sigmoid(
        (local_example.detach() - memory_example.detach()) / 0.05
    )
    gate = F.binary_cross_entropy(
        output.memory_gate.squeeze(-1).clamp(1e-6, 1.0 - 1e-6), gate_target
    )
    total = (
        forecast
        + 0.10 * local
        + 0.10 * memory
        + 0.35 * retrieval
        + ranking_weight * ranking
        + 0.10 * gate
    )
    components = {
        "forecast": forecast.detach(),
        "local": local.detach(),
        "memory": memory.detach(),
        "retrieval": retrieval.detach(),
        "gate": gate.detach(),
    }
    if ranking_weight > 0.0:
        components["ranking"] = ranking.detach()
    return total, components


@dataclass(frozen=True)
class Headroom:
    samples: int
    oracle_loss: float
    recency_loss: float
    relative_oracle_gain: float
    oracle_not_recency_rate: float
    informative_rate: float
    normalized_target_entropy: float


def headroom(
    successor_dynamic: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    utility_temperature: float = 0.04,
) -> Headroom:
    relevance, losses = frozen_utility_distribution(
        successor_dynamic, target, valid, temperature=utility_temperature
    )
    usable = valid.any(dim=-1)
    if not usable.any():
        raise ValueError("headroom requires at least one valid transition")
    oracle_index = losses.argmin(dim=-1)
    oracle = losses.gather(1, oracle_index[:, None]).squeeze(1)
    # Slots are newest first, so the first valid slot is the recency control.
    recency_index = valid.to(torch.int64).argmax(dim=-1)
    recency = losses.gather(1, recency_index[:, None]).squeeze(1)
    informative = valid.sum(dim=-1) >= 2
    entropy = -(relevance * relevance.clamp_min(1e-8).log()).sum(dim=-1)
    entropy = entropy / valid.sum(dim=-1).to(entropy.dtype).log().clamp_min(1.0)
    oracle_mean = oracle[usable].mean()
    recency_mean = recency[usable].mean()
    return Headroom(
        samples=int(usable.sum()),
        oracle_loss=float(oracle_mean),
        recency_loss=float(recency_mean),
        relative_oracle_gain=float(
            (recency_mean - oracle_mean) / recency_mean.clamp_min(1e-8)
        ),
        oracle_not_recency_rate=float(
            (oracle_index[usable] != recency_index[usable]).float().mean()
        ),
        informative_rate=float(informative[usable].float().mean()),
        normalized_target_entropy=float(entropy[usable].mean()),
    )

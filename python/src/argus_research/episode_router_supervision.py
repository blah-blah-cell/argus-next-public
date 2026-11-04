"""Counterfactual supervision for the causal packet-episode router.

The router never receives the future packet or an expert's prediction as input.
Those values are used only here, offline, to determine which non-empty subset of
the frozen experts would have produced the best quality/compute trade-off.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .ahem_v32_benchmark import _weighted_loss
from .packet_episode_graph import CausalEpisode, RouterBundle

SUPERVISION_SCHEMA = "argus.episode-router-supervision/v2"
EVIDENCE_SCHEMA = "argus.causal-episode-evidence/v2"
QUESTION_SCHEMA = "argus.episode-router-questions/v2"
COMPACT_FEATURE_SCHEMA = "argus.compact-episode-router-features/v2"
_EXPERT_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def expert_subsets(expert_ids: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Enumerate all non-empty subsets in a stable, manifest-defined order."""
    ordered = tuple(expert_ids)
    if not ordered or len(ordered) > 8:
        raise ValueError("episode routing requires between one and eight experts")
    if len(set(ordered)) != len(ordered):
        raise ValueError("expert identifiers must be unique")
    if any(not _EXPERT_ID.fullmatch(value) for value in ordered):
        raise ValueError("expert identifier is unsafe")
    return tuple(
        subset
        for width in range(1, len(ordered) + 1)
        for subset in itertools.combinations(ordered, width)
    )


def subset_id(subset: Sequence[str]) -> str:
    if not subset:
        raise ValueError("expert subset must not be empty")
    if any(not _EXPERT_ID.fullmatch(value) for value in subset):
        raise ValueError("expert identifier is unsafe")
    return "+".join(subset)


@dataclass(frozen=True)
class SubsetOutcome:
    subset_id: str
    experts: tuple[str, ...]
    forecast_loss: float
    measured_latency_ms: float
    normalized_compute: float
    utility: float


@dataclass(frozen=True)
class RouterSupervision:
    schema: str
    subset_order: tuple[str, ...]
    oracle_subset: str
    soft_targets: tuple[float, ...]
    oracle_margin: float
    abstain_target: bool
    outcomes: tuple[SubsetOutcome, ...]


def counterfactual_subset_outcomes(
    predictions: Mapping[str, Tensor],
    target: Tensor,
    measured_subset_latency_ms: Mapping[str, float],
    *,
    compute_weight: float,
) -> tuple[SubsetOutcome, ...]:
    """Score every subset using a fixed uniform ensemble and measured latency.

    Uniform averaging is intentional: learning ensemble weights on the locked
    test set would leak target information. More sophisticated weights may be
    fitted on train and frozen in a later manifest, but are not inferred here.
    """
    if compute_weight < 0 or not math.isfinite(compute_weight):
        raise ValueError("compute_weight must be finite and non-negative")
    if target.shape[-1:] != (12,) or not torch.isfinite(target).all():
        raise ValueError("target must contain finite twelve-field packet features")
    ids = tuple(predictions)
    subsets = expert_subsets(ids)
    for expert_id, prediction in predictions.items():
        if prediction.shape != target.shape:
            raise ValueError(f"{expert_id} prediction shape differs from target")
        if not torch.isfinite(prediction).all():
            raise ValueError(f"{expert_id} prediction contains non-finite values")
    subset_names = tuple(subset_id(values) for values in subsets)
    if set(measured_subset_latency_ms) != set(subset_names):
        raise ValueError("measured latency must cover every expert subset exactly")
    latencies = [float(measured_subset_latency_ms[name]) for name in subset_names]
    if any(not math.isfinite(value) or value <= 0 for value in latencies):
        raise ValueError("measured subset latencies must be finite and positive")
    latency_reference = min(latencies)

    output: list[SubsetOutcome] = []
    for subset, name, latency in zip(subsets, subset_names, latencies, strict=True):
        ensemble = torch.stack([predictions[expert] for expert in subset]).mean(0)
        loss = float(_weighted_loss(ensemble, target))
        normalized_compute = latency / latency_reference
        output.append(
            SubsetOutcome(
                subset_id=name,
                experts=subset,
                forecast_loss=loss,
                measured_latency_ms=latency,
                normalized_compute=normalized_compute,
                utility=loss + compute_weight * normalized_compute,
            )
        )
    return tuple(output)


def build_supervision(
    outcomes: Sequence[SubsetOutcome],
    *,
    temperature: float,
    identity_health: str,
) -> RouterSupervision:
    """Convert counterfactual utilities to a calibrated distribution target."""
    if not outcomes:
        raise ValueError("counterfactual outcomes are empty")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    names = tuple(value.subset_id for value in outcomes)
    if len(set(names)) != len(names):
        raise ValueError("counterfactual subset identifiers are not unique")
    utilities = torch.tensor([value.utility for value in outcomes], dtype=torch.float64)
    if not torch.isfinite(utilities).all():
        raise ValueError("counterfactual utilities contain non-finite values")
    shifted = utilities - utilities.min()
    targets = torch.softmax(-shifted / temperature, dim=0)
    ranking = torch.argsort(utilities)
    margin = (
        float(utilities[ranking[1]] - utilities[ranking[0]])
        if len(outcomes) > 1
        else float("inf")
    )
    return RouterSupervision(
        schema=SUPERVISION_SCHEMA,
        subset_order=names,
        oracle_subset=names[int(ranking[0])],
        soft_targets=tuple(float(value) for value in targets),
        oracle_margin=margin,
        abstain_target=identity_health != "nominal",
        outcomes=tuple(outcomes),
    )


def episode_evidence(
    anchor: CausalEpisode,
    bundle: RouterBundle,
    episodes: Mapping[str, CausalEpisode],
) -> dict[str, object]:
    """Serialize causal structure without identity, endpoint, or future leakage."""
    if bundle.anchor_episode_id != anchor.episode_id:
        raise ValueError("bundle anchor differs")
    if bundle.decision_timestamp_ns != anchor.end_timestamp_ns:
        raise ValueError("bundle decision time differs from anchor")
    if bundle.decision_source_ordinal != anchor.packet_ordinals[-1]:
        raise ValueError("bundle decision ordinal differs from anchor")
    if set(bundle.episode_ids) - set(episodes):
        raise ValueError("bundle episode is absent from the episode map")
    ordered = [episodes[episode_id] for episode_id in bundle.episode_ids]
    if any(value.end_timestamp_ns > bundle.decision_timestamp_ns for value in ordered):
        raise ValueError("episode evidence includes future observations")
    if any(
        value.episode_id != anchor.episode_id
        and value.packet_ordinals[-1] >= bundle.decision_source_ordinal
        for value in ordered
    ):
        raise ValueError("episode evidence includes a future source ordinal")

    relation_by_id = {value.episode_id: value for value in bundle.relations}
    history = []
    for episode in ordered:
        if episode.episode_id == anchor.episode_id:
            continue
        relation = relation_by_id.get(episode.episode_id)
        if relation is None:
            raise ValueError("historical episode lacks a relationship")
        history.append(
            {
                "age_seconds": relation.age_ns / 1_000_000_000.0,
                "relation_score": relation.score,
                "relation_kinds": list(relation.kinds),
                "protocol": episode.protocol,
                "boundary_reason": episode.boundary_reason,
                "identity_health": episode.identity_health,
                "summary": dict(episode.summary),
            }
        )
    return {
        "schema": EVIDENCE_SCHEMA,
        "decision": "online_next_packet_forecast_routing",
        "future_packets_in_input": False,
        "expert_predictions_in_input": False,
        "raw_identifiers_in_input": False,
        "identity_health": bundle.identity_health,
        "anchor": {
            "protocol": anchor.protocol,
            "boundary_reason": anchor.boundary_reason,
            "identity_health": anchor.identity_health,
            "summary": dict(anchor.summary),
        },
        "history": history,
        "bundle_summary": {
            "episode_count": len(bundle.episode_ids),
            "packet_count": len(bundle.packet_ids),
            "history_count": len(history),
        },
    }


def routing_questions_v2(expert_ids: Sequence[str]) -> dict[str, dict[str, object]]:
    """Return the exact question schema shared by v2 training and runtime."""
    subsets = expert_subsets(expert_ids)
    criteria = {
        subset_id(values): (
            "Run " + ", ".join(values) + " and uniformly combine their next-packet forecasts"
        )
        for values in subsets
    }
    questions: dict[str, dict[str, object]] = {
        "route_subset": {
            "type": "choice",
            "instructions": (
                "Select the frozen expert subset with the lowest expected "
                "source-disjoint forecast loss plus measured compute cost. "
                "This is compute routing, not attack detection."
            ),
            "criteria": criteria,
        },
        "compute_budget": {
            "type": "choice",
            "instructions": "Choose the smallest sufficient measured-compute tier.",
            "criteria": {
                "fast": "one expert",
                "balanced": "two experts",
                "deep": "three or more experts",
            },
        },
        "abstain": {
            "type": "choice",
            "instructions": (
                "Abstain when capture loss, truncation, reordering, or malformed "
                "identity makes a routing decision untrustworthy."
            ),
            "criteria": {
                "yes": "return the frozen safe fallback",
                "no": "use the selected calibrated subset",
            },
        },
    }
    return questions


def compact_episode_features(evidence: Mapping[str, object]) -> tuple[float, ...]:
    """Project v2 evidence into the frozen 24-field compact-gate contract."""
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError("compact router evidence schema differs")
    anchor = evidence.get("anchor")
    history = evidence.get("history")
    bundle = evidence.get("bundle_summary")
    if not isinstance(anchor, dict) or not isinstance(history, list) or not isinstance(bundle, dict):
        raise ValueError("compact router evidence is incomplete")

    def finite(value: object, name: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"compact router {name} is non-finite")
        return number

    def summary(row: Mapping[str, object]) -> tuple[float, float, float, float]:
        value = row.get("summary")
        if not isinstance(value, dict):
            raise ValueError("compact router episode summary is absent")
        return (
            finite(value.get("packets", 0), "packet count"),
            finite(value.get("duration_seconds", 0), "duration"),
            finite(value.get("mean_packet_bytes", 0), "mean packet bytes"),
            finite(value.get("maximum_packet_bytes", 0), "maximum packet bytes"),
        )

    anchor_packets, anchor_duration, anchor_mean, anchor_maximum = summary(anchor)
    if min(anchor_packets, anchor_duration, anchor_mean, anchor_maximum) < 0:
        raise ValueError("compact router anchor summary is negative")
    protocol = anchor.get("protocol")
    if protocol not in {"tcp", "udp", "other"}:
        raise ValueError("compact router protocol differs")

    scores: list[float] = []
    ages: list[float] = []
    historical: list[tuple[float, float, float, float]] = []
    relation_fractions = {"same_conversation": 0, "shared_endpoint": 0, "same_protocol": 0}
    degraded = 0
    for row in history:
        if not isinstance(row, dict):
            raise ValueError("compact router history row differs")
        score = finite(row.get("relation_score"), "relationship score")
        age = finite(row.get("age_seconds"), "relationship age")
        if score < 0 or age < 0:
            raise ValueError("compact router relationship metric is negative")
        kinds = row.get("relation_kinds")
        if not isinstance(kinds, list) or any(not isinstance(value, str) for value in kinds):
            raise ValueError("compact router relationship kinds differ")
        scores.append(score)
        ages.append(age)
        historical.append(summary(row))
        degraded += row.get("identity_health") != "nominal"
        for name in relation_fractions:
            relation_fractions[name] += name in kinds

    count = len(history)
    denominator = max(1, count)

    def average(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    hist_packets = [value[0] for value in historical]
    hist_duration = [value[1] for value in historical]
    hist_mean_bytes = [value[2] for value in historical]
    boundary = anchor.get("boundary_reason")
    values = (
        math.log1p(anchor_packets),
        math.log1p(anchor_duration),
        math.log1p(anchor_mean),
        math.log1p(anchor_maximum),
        float(protocol == "tcp"),
        float(protocol == "udp"),
        float(protocol == "other"),
        float(anchor.get("identity_health") != "nominal"),
        math.log1p(finite(bundle.get("episode_count", 0), "bundle episodes")),
        math.log1p(finite(bundle.get("packet_count", 0), "bundle packets")),
        math.log1p(finite(bundle.get("history_count", 0), "bundle history")),
        average(scores),
        max(scores, default=0.0),
        average(ages),
        max(ages, default=0.0),
        relation_fractions["same_conversation"] / denominator,
        relation_fractions["shared_endpoint"] / denominator,
        relation_fractions["same_protocol"] / denominator,
        math.log1p(average(hist_packets)),
        math.log1p(average(hist_duration)),
        math.log1p(average(hist_mean_bytes)),
        degraded / denominator,
        float(boundary in {"packet_limit", "duration_limit", "idle_timeout"}),
        float(boundary == "causal_change_point"),
    )
    if len(values) != 24 or not all(math.isfinite(value) for value in values):
        raise RuntimeError("compact episode feature contract is invalid")
    return values


def supervision_json(supervision: RouterSupervision) -> dict[str, object]:
    """Return a JSON-safe representation with no tensors or non-finite values."""
    payload = asdict(supervision)
    if math.isinf(supervision.oracle_margin):
        payload["oracle_margin"] = None
    return payload


CERO_COMPACT_EVIDENCE_SCHEMA = "argus.cero-compact-evidence/v2"


def compact_cero_evidence(
    evidence: Mapping[str, object],
    *,
    max_neighbors: int = 2,
    round_digits: int = 2,
    include_audit_metadata: bool = False,
) -> dict[str, object]:
    """Prune episode graph evidence to <= 116 state tokens while preserving all edge semantics.

    Reduces prompt sequence length from 531 tokens down to <= 180 tokens with zero quantization.
    When include_audit_metadata is False (default for prompt evaluation), static invariant fields
    (audit assertions) are omitted to minimize attention complexity.
    """
    if evidence.get("schema") not in {EVIDENCE_SCHEMA, CERO_COMPACT_EVIDENCE_SCHEMA}:
        raise ValueError("compact CERO evidence requires v2 episode evidence schema")
    anchor = evidence.get("anchor")
    if not isinstance(anchor, dict):
        raise ValueError("compact CERO evidence requires an anchor dictionary")
    a_summary = anchor.get("summary")
    if not isinstance(a_summary, dict):
        raise ValueError("compact CERO anchor requires summary dictionary")
    history = evidence.get("history")
    if not isinstance(history, list):
        raise ValueError("compact CERO evidence requires a history list")

    # Sort history by relation score descending, then recency (age ascending)
    sorted_history = sorted(
        history,
        key=lambda h: (-float(h.get("relation_score", 0.0)), float(h.get("age_seconds", 0.0))),
    )

    compact_edges = []
    for h in sorted_history[:max_neighbors]:
        h_s = h.get("summary", {})
        compact_edges.append({
            "dt": round(float(h.get("age_seconds", 0.0)), round_digits),
            "sc": round(float(h.get("relation_score", 0.0)), 2),
            "k": list(h.get("relation_kinds", [])),
            "p": h.get("protocol", "other"),
            "s": [
                int(h_s.get("packets", 0)),
                round(float(h_s.get("duration_seconds", 0.0)), round_digits),
                int(round(float(h_s.get("mean_packet_bytes", 0.0)))),
                int(h_s.get("maximum_packet_bytes", 0)),
            ],
        })

    payload: dict[str, object] = {
        "anc": {
            "p": anchor.get("protocol", "other"),
            "b": anchor.get("boundary_reason", "unknown"),
            "s": [
                int(a_summary.get("packets", 0)),
                round(float(a_summary.get("duration_seconds", 0.0)), round_digits),
                int(round(float(a_summary.get("mean_packet_bytes", 0.0)))),
                int(a_summary.get("maximum_packet_bytes", 0)),
            ],
        },
        "edges": compact_edges,
        "hlth": evidence.get("identity_health", "nominal"),
    }
    if include_audit_metadata:
        payload["schema"] = CERO_COMPACT_EVIDENCE_SCHEMA
        payload["future_packets_in_input"] = False
        payload["raw_identifiers_in_input"] = False
    return payload

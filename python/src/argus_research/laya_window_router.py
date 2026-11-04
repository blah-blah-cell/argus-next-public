"""Optional Laya router for causal network windows.

This module does not turn Laya into a packet detector.  It adapts a completed
causal window into a compact, payload-free state and asks an explicitly
fine-tuned Laya checkpoint which *forecast* experts are worth executing.  A
generic Laya checkpoint is deliberately rejected: expert routing needs an
Argus-specific, calibrated artifact and source-disjoint supervision.

The runtime is optional and local-only.  No package import, model download, or
network access occurs unless ``ARGUS_LAYA_CHECKPOINT`` names an existing local
checkpoint directory with a valid Argus manifest.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
import os
import shutil
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from importlib.metadata import version as distribution_version
from itertools import pairwise
from pathlib import Path
from time import perf_counter_ns
from typing import Protocol

import torch

EVIDENCE_SCHEMA = "argus.causal-window-evidence/v1"
ROUTER_SCHEMA = "argus.laya-window-router/v1"
QUESTION_SCHEMA = "argus.laya-expert-routing-questions/v1"
EPISODE_ROUTER_SCHEMA = "argus.laya-episode-router/v2"
EPISODE_ROUTER_SOURCE = "argus_cero_episode_router"
EPISODE_EVIDENCE_SCHEMA = "argus.causal-episode-evidence/v2"
EPISODE_QUESTION_SCHEMA = "argus.episode-router-questions/v2"
MANIFEST_FILE = "argus-laya-router.json"
WEIGHTS_FILE = "model.safetensors"
MIN_WINDOW_PACKETS = 32
MAX_EXPERTS = 12


class LayaAgent(Protocol):
    """The small portion of the Laya API used at the integration boundary."""

    def predict(
        self, state: dict[str, object], questions: dict[str, dict[str, object]]
    ) -> dict[str, object]: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_probability(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} is not a probability")
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is not numeric") from error
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field} is outside [0, 1]")
    return probability


def _choice_answer(
    answer: object, options: list[str], field: str
) -> tuple[str, float, dict[str, float]]:
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise TypeError(f"{field} answer is invalid")
    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in options:
        raise ValueError(f"{field} selected an unavailable option")
    confidence = _finite_probability(answer.get("confidence"), f"{field}.confidence")
    raw_probabilities = answer.get("probabilities")
    if not isinstance(raw_probabilities, dict) or set(raw_probabilities) != set(
        options
    ):
        raise ValueError(f"{field}.probabilities options differ")
    probabilities = {
        option: _finite_probability(
            raw_probabilities[option], f"{field}.probabilities.{option}"
        )
        for option in options
    }
    total = sum(probabilities.values())
    if not math.isfinite(total) or abs(total - 1.0) > 0.01:
        raise ValueError(f"{field}.probabilities do not sum to one")
    if probabilities[choice] + 1e-6 < max(probabilities.values()):
        raise ValueError(f"{field}.choice is not the maximum-probability option")
    return choice, confidence, probabilities


def _noul_answer(answer: object, field: str) -> tuple[float, float]:
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        raise TypeError(f"{field} answer is invalid")
    return (
        _finite_probability(answer.get("noul"), f"{field}.noul"),
        _finite_probability(answer.get("confidence"), f"{field}.confidence"),
    )


def _rounded(value: torch.Tensor) -> float:
    return round(float(value), 6)


def _quantile(values: torch.Tensor, probability: float) -> float:
    if values.numel() == 0:
        return 0.0
    return _rounded(torch.quantile(values.float(), probability))


def causal_window_evidence(
    features: torch.Tensor, rows: list[dict[str, str]], history_scope: str
) -> dict[str, object]:
    """Build bounded, payload-free evidence from already-observed packets.

    ``features`` follows ``packet-causal-v1``.  Every statistic is computed
    only from the supplied past/current window.  No labels, completed-flow
    facts, future packets, addresses, or payload bytes are exposed to Laya.
    """
    if features.ndim != 2 or features.shape[1] != 12:
        raise ValueError("Laya evidence requires packet-causal-v1 [N, 12] features")
    if len(features) != len(rows) or not rows:
        raise ValueError("Laya evidence requires one metadata row per packet")
    if not torch.isfinite(features).all():
        raise ValueError("Laya evidence contains non-finite features")

    timestamps = [int(row["monotonic_timestamp_ns"]) for row in rows]
    ordinals = [int(row["source_capture_ordinal"]) for row in rows]
    if any(right < left for left, right in pairwise(timestamps)):
        raise ValueError("Laya evidence timestamps are not monotonic")
    if any(right <= left for left, right in pairwise(ordinals)):
        raise ValueError("Laya evidence ordinals are not strictly increasing")

    packet_lengths = torch.expm1(features[:, 0]).clamp(min=0.0)
    interarrival = torch.expm1(features[:, 1]).clamp(min=0.0, max=60.0)
    duration_seconds = max(0.0, (timestamps[-1] - timestamps[0]) / 1_000_000_000)
    flow_counts: dict[int, int] = {}
    for row in rows:
        flow_id = int(row["opaque_flow_id"])
        flow_counts[flow_id] = flow_counts.get(flow_id, 0) + 1
    largest_flow_fraction = max(flow_counts.values()) / len(rows)

    def phase_summary(phase: torch.Tensor) -> dict[str, object]:
        if phase.numel() == 0:
            return {"packets": 0}
        return {
            "packets": len(phase),
            "mean_packet_bytes": _rounded(torch.expm1(phase[:, 0]).mean()),
            "mean_interarrival_seconds": _rounded(
                torch.expm1(phase[:, 1]).clamp(max=60.0).mean()
            ),
            "tcp_syn_fraction": _rounded(phase[:, 6].mean()),
            "tcp_rst_fraction": _rounded(phase[:, 9].mean()),
        }

    phases = torch.tensor_split(features, 3)
    packet_count = len(rows)
    return {
        "schema": EVIDENCE_SCHEMA,
        "scope": history_scope,
        "decision_time": "online_window_end",
        "payload_included": False,
        "window": {
            "packets": packet_count,
            "flows": len(flow_counts),
            "duration_seconds": round(duration_seconds, 6),
            "packets_per_second": round(packet_count / max(duration_seconds, 1e-6), 6),
            "largest_flow_fraction": round(largest_flow_fraction, 6),
        },
        "packet_length_bytes": {
            "mean": _rounded(packet_lengths.mean()),
            "p95": _quantile(packet_lengths, 0.95),
            "max": _rounded(packet_lengths.max()),
        },
        "interarrival_seconds": {
            "mean": _rounded(interarrival.mean()),
            "p95": _quantile(interarrival, 0.95),
            "max": _rounded(interarrival.max()),
        },
        "protocol_fractions": {
            "ipv4": _rounded(features[:, 2].mean()),
            "ipv6": _rounded(features[:, 3].mean()),
            "tcp": _rounded(features[:, 4].mean()),
            "udp": _rounded(features[:, 5].mean()),
        },
        "tcp_flag_fractions": {
            "syn": _rounded(features[:, 6].mean()),
            "ack": _rounded(features[:, 7].mean()),
            "fin": _rounded(features[:, 8].mean()),
            "rst": _rounded(features[:, 9].mean()),
            "push": _rounded(features[:, 10].mean()),
            "urg": _rounded(features[:, 11].mean()),
        },
        "temporal_phases": {
            "early": phase_summary(phases[0]),
            "middle": phase_summary(phases[1]),
            "late": phase_summary(phases[2]),
        },
    }


@dataclass(frozen=True)
class LayaRouterConfig:
    checkpoint: Path
    model_id: str
    laya_version: str
    expert_ids: tuple[str, ...]
    top_k: int
    minimum_selection_probability: float
    abstain_probability: float
    weights_sha256: str
    training_provenance: str
    training_provenance_sha256: str
    schema: str = ROUTER_SCHEMA
    expert_sha256: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, checkpoint: Path) -> LayaRouterConfig:
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_dir():
            raise ValueError("configured Laya checkpoint directory does not exist")
        manifest_path = checkpoint / MANIFEST_FILE
        if not manifest_path.is_file():
            raise ValueError(f"{MANIFEST_FILE} is absent")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Laya router manifest is unreadable") from error
        schema = manifest.get("schema")
        if schema not in {ROUTER_SCHEMA, EPISODE_ROUTER_SCHEMA}:
            raise ValueError("Laya router manifest schema differs")
        episode_v2 = schema == EPISODE_ROUTER_SCHEMA
        required = {
            "schema": schema,
            "feature_schema": EPISODE_EVIDENCE_SCHEMA if episode_v2 else EVIDENCE_SCHEMA,
            "question_schema": EPISODE_QUESTION_SCHEMA if episode_v2 else QUESTION_SCHEMA,
            "task": (
                "causal_episode_cost_aware_subset_routing"
                if episode_v2 else "cost_aware_forecast_expert_routing"
            ),
            "security_detector": False,
            "calibrated": True,
            "trained_out_of_fold": True,
            "calibration_source_disjoint": True,
        }
        for key, expected in required.items():
            if manifest.get(key) != expected:
                raise ValueError(f"Laya router manifest {key} differs")
        if episode_v2 and manifest.get("routes_expert_subsets") is not True:
            raise ValueError("Laya episode router does not declare subset routing")

        expert_value = manifest.get("expert_ids")
        if not isinstance(expert_value, list) or not expert_value:
            raise ValueError("Laya router manifest has no expert_ids")
        expert_ids = tuple(str(value) for value in expert_value)
        if (
            len(expert_ids) > MAX_EXPERTS
            or len(set(expert_ids)) != len(expert_ids)
            or any(
                not value
                or len(value) > 64
                or any(
                    not (character.isalnum() or character in "-_.")
                    for character in value
                )
                for value in expert_ids
            )
        ):
            raise ValueError("Laya router manifest expert_ids are invalid")

        top_k = int(manifest.get("top_k", 0))
        if not 1 <= top_k <= len(expert_ids):
            raise ValueError("Laya router manifest top_k is invalid")
        minimum = _finite_probability(
            manifest.get("minimum_selection_probability"),
            "minimum_selection_probability",
        )
        if minimum <= 0.0:
            raise ValueError("minimum_selection_probability must be greater than zero")
        abstain = _finite_probability(
            manifest.get("abstain_probability"), "abstain_probability"
        )
        model_id = manifest.get("model_id")
        laya_version = manifest.get("laya_version")
        provenance = manifest.get("training_provenance")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Laya router manifest model_id is absent")
        if not isinstance(laya_version, str) or not laya_version.strip():
            raise ValueError("Laya router manifest laya_version is absent")
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError("Laya router manifest training_provenance is absent")
        provenance_relative = Path(provenance)
        if provenance_relative.is_absolute() or ".." in provenance_relative.parts:
            raise ValueError("Laya router training_provenance path is unsafe")
        provenance_path = (checkpoint / provenance_relative).resolve()
        if (
            not provenance_path.is_relative_to(checkpoint)
            or not provenance_path.is_file()
        ):
            raise ValueError("Laya router training_provenance file is absent")
        provenance_hash = _sha256(provenance_path)
        if manifest.get("training_provenance_sha256") != provenance_hash:
            raise ValueError("Laya router training_provenance SHA-256 differs")
        expert_hashes: dict[str, str] = {}
        if episode_v2:
            provenance_data = json.loads(provenance_path.read_text(encoding="utf-8"))
            raw_hashes = provenance_data.get("expert_sha256")
            if not isinstance(raw_hashes, dict) or set(raw_hashes) != set(expert_ids):
                raise ValueError("Laya episode expert provenance differs")
            if any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in raw_hashes.values()
            ):
                raise ValueError("Laya episode expert hashes are invalid")
            expert_hashes = dict(raw_hashes)

        weights = checkpoint / WEIGHTS_FILE
        encoder = checkpoint / "encoder"
        tokenizer = checkpoint / "tokenizer"
        laya_config = checkpoint / "rl_agent_config.json"
        if (
            not weights.is_file()
            or not encoder.is_dir()
            or not tokenizer.is_dir()
            or not laya_config.is_file()
        ):
            raise ValueError("Laya checkpoint is not fully local")
        expected_hash = manifest.get("model_safetensors_sha256")
        actual_hash = _sha256(weights)
        if expected_hash != actual_hash:
            raise ValueError("Laya model.safetensors SHA-256 differs")

        # Laya 0.3.4 repairs these tokenizer values by writing to disk.  Argus
        # keeps model artifacts immutable and rejects checkpoints needing repair.
        tokenizer_config = tokenizer / "tokenizer_config.json"
        if not tokenizer_config.is_file():
            raise ValueError("Laya tokenizer_config.json is absent")
        token_data = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        if not episode_v2 and token_data.get("tokenizer_class") in (None, "TokenizersBackend"):
            raise ValueError("Laya tokenizer config requires an in-place repair")
        if not episode_v2 and isinstance(token_data.get("extra_special_tokens"), list):
            raise TypeError("Laya tokenizer special tokens require an in-place repair")

        return cls(
            checkpoint=checkpoint,
            model_id=model_id,
            laya_version=laya_version,
            expert_ids=expert_ids,
            top_k=top_k,
            minimum_selection_probability=minimum,
            abstain_probability=abstain,
            weights_sha256=actual_hash,
            training_provenance=provenance,
            training_provenance_sha256=provenance_hash,
            schema=schema,
            expert_sha256=expert_hashes,
        )


class LayaWindowRouter:
    """Validated, fail-closed adapter around one local Laya checkpoint."""

    def __init__(
        self, config: LayaRouterConfig | None, agent: LayaAgent | None, detail: str
    ) -> None:
        self.config = config
        self.agent = agent
        self.detail = detail
        self.last_route: dict[str, object] | None = None

    @classmethod
    def from_environment(
        cls, device: str = "auto", checkpoint: Path | None = None
    ) -> LayaWindowRouter:
        configured = (
            str(checkpoint)
            if checkpoint is not None
            else (
                os.environ.get("ARGUS_CERO_CHECKPOINT", "").strip()
                or os.environ.get("ARGUS_LAYA_CHECKPOINT", "").strip()
            )
        )
        if not configured:
            return cls(None, None, "not configured")
        try:
            configured_path = Path(configured)
            if not configured_path.is_absolute():
                raise ValueError("Argus CERO checkpoint path must be absolute")
            config = LayaRouterConfig.load(configured_path)
            module = importlib.import_module("laya")
            installed_version = distribution_version("laya")
            if installed_version != config.laya_version:
                raise ValueError("installed Laya version differs from router manifest")
            load = getattr(module, "load", None)
            if not callable(load):
                raise TypeError("installed laya package has no load function")
            requested_device = None if device == "auto" else device
            # Laya 0.3.4 rewrites tokenizer metadata on load.  The v2 trainer
            # emitted TokenizersBackend, so stage a disposable local copy and
            # leave the hash-verified research artifact untouched.
            if config.schema == EPISODE_ROUTER_SCHEMA:
                with tempfile.TemporaryDirectory(
                    prefix="argus-laya-runtime-", dir=config.checkpoint.parent
                ) as stage_name:
                    stage = Path(stage_name)
                    shutil.copytree(config.checkpoint / "encoder", stage / "encoder")
                    shutil.copytree(config.checkpoint / "tokenizer", stage / "tokenizer")
                    shutil.copy2(
                        config.checkpoint / "rl_agent_config.json",
                        stage / "rl_agent_config.json",
                    )
                    source_weights = config.checkpoint / WEIGHTS_FILE
                    staged_weights = stage / WEIGHTS_FILE
                    try:
                        os.link(source_weights, staged_weights)
                    except OSError:
                        shutil.copyfile(source_weights, staged_weights)
                    with redirect_stdout(io.StringIO()):
                        agent = load(str(stage), device=requested_device)
            else:
                with redirect_stdout(io.StringIO()):
                    agent = load(str(config.checkpoint), device=requested_device)
            return cls(config, agent, "local calibrated router admitted")
        except Exception as error:  # noqa: BLE001 - optional boundary is quarantined
            return cls(None, None, f"rejected: {error}")

    @classmethod
    def for_test(cls, config: LayaRouterConfig, agent: LayaAgent) -> LayaWindowRouter:
        """Inject a protocol implementation; never presented as model evidence."""
        return cls(config, agent, "test interface injected")

    def status(self) -> dict[str, object]:
        if self.config is None or self.agent is None:
            loading = self.detail.startswith("loading ")
            return {
                "state": "loading" if loading else "unavailable",
                "detail": self.detail,
                "configured": (
                    self.detail != "not configured"
                    or bool(os.environ.get("ARGUS_CERO_CHECKPOINT", "").strip())
                    or bool(os.environ.get("ARGUS_LAYA_CHECKPOINT", "").strip())
                ),
                "network_downloads": False,
            }
        return {
            "state": "ready",
            "detail": self.detail,
            "configured": True,
            "model_id": self.config.model_id,
            "laya_version": self.config.laya_version,
            "checkpoint_sha256": self.config.weights_sha256,
            "feature_schema": (
                EPISODE_EVIDENCE_SCHEMA
                if self.config.schema == EPISODE_ROUTER_SCHEMA else EVIDENCE_SCHEMA
            ),
            "question_schema": (
                EPISODE_QUESTION_SCHEMA
                if self.config.schema == EPISODE_ROUTER_SCHEMA else QUESTION_SCHEMA
            ),
            "router_schema": self.config.schema,
            "expert_ids": list(self.config.expert_ids),
            "top_k": self.config.top_k,
            "training_provenance": self.config.training_provenance,
            "training_provenance_sha256": self.config.training_provenance_sha256,
            "network_downloads": False,
            "last_route": self.last_route,
        }

    def route(
        self,
        evidence: dict[str, object],
        available_experts: set[str],
    ) -> dict[str, object]:
        if self.config is None or self.agent is None:
            raise RuntimeError("Laya router is unavailable")
        if self.config.schema == EPISODE_ROUTER_SCHEMA:
            return self._route_episode(evidence, available_experts)
        if evidence.get("schema") != EVIDENCE_SCHEMA:
            raise ValueError("Laya evidence schema differs")
        trained = [
            expert for expert in self.config.expert_ids if expert in available_experts
        ]
        if not trained:
            raise ValueError("no trained Laya expert is currently available")

        questions = routing_questions(trained)

        started_ns = perf_counter_ns()
        # Laya emits device-fallback warnings on stdout.  The persistent Argus
        # worker reserves stdout for its JSON-line protocol, so capture them.
        with redirect_stdout(io.StringIO()):
            raw = self.agent.predict(evidence, questions)
        latency_ms = (perf_counter_ns() - started_ns) / 1_000_000.0
        if not isinstance(raw, dict) or not isinstance(raw.get("answers"), dict):
            raise TypeError("Laya response has no answers object")
        if not isinstance(raw.get("model"), str) or not str(raw["model"]).strip():
            raise ValueError("Laya response model identity is absent")
        answers = raw["answers"]
        if set(answers) != set(questions):
            raise ValueError("Laya response answer identifiers differ")
        primary_name, primary_confidence, primary_probabilities = _choice_answer(
            answers.get("route_primary"), trained, "route_primary"
        )
        budget_options = ["fast", "balanced", "deep"]
        budget_name, budget_confidence, budget_probabilities = _choice_answer(
            answers.get("compute_budget"), budget_options, "compute_budget"
        )
        abstain_probability, abstain_confidence = _noul_answer(
            answers.get("abstain"), "abstain"
        )

        scores: dict[str, float] = {}
        confidences: dict[str, float] = {}
        for index, expert in enumerate(trained):
            scores[expert], confidences[expert] = _noul_answer(
                answers.get(f"use_expert_{index}"), f"use_expert_{index}"
            )

        selected: list[str] = []
        if abstain_probability < self.config.abstain_probability:
            candidate_limit = 1 if budget_name == "fast" else self.config.top_k
            selected = [
                expert
                for expert, score in sorted(
                    scores.items(), key=lambda item: (-item[1], item[0])
                )
                if score >= self.config.minimum_selection_probability
                and confidences[expert] >= self.config.minimum_selection_probability
            ][:candidate_limit]
        if primary_name in selected:
            selected.remove(primary_name)
            selected.insert(0, str(primary_name))

        if not selected:
            result: dict[str, object] = {
                "state": "abstained",
                "selected_experts": [],
                "eligible_experts": trained,
                "confidence": 0.0,
                "abstain_probability": abstain_probability,
                "abstain_confidence": abstain_confidence,
                "compute_budget": budget_name,
                "compute_budget_confidence": budget_confidence,
                "execution_ms": latency_ms,
                "reason": "confidence gate or abstention gate did not admit a route",
            }
        else:
            total = sum(scores[expert] for expert in selected)
            if not math.isfinite(total) or total <= 0.0:
                raise ValueError("selected Laya expert weights have no positive mass")
            weights = {expert: scores[expert] / total for expert in selected}
            result = {
                "state": "active",
                "source": "laya_local_window_router",
                "selected_expert": selected[0],
                "selected_experts": selected,
                "eligible_experts": trained,
                "confidence": min(
                    primary_confidence,
                    budget_confidence,
                    *(confidences[expert] for expert in selected),
                ),
                "abstain_probability": abstain_probability,
                "abstain_confidence": abstain_confidence,
                "compute_budget": budget_name,
                "compute_budget_confidence": budget_confidence,
                "compute_budget_probabilities": budget_probabilities,
                "expert_weights": weights,
                "expert_selection_probabilities": scores,
                "primary_probabilities": primary_probabilities,
                "execution_ms": latency_ms,
                "security_router": False,
                "task": "cost_aware_forecast_expert_routing",
                "window_level": True,
                "model_id": self.config.model_id,
                "laya_version": self.config.laya_version,
                "checkpoint_sha256": self.config.weights_sha256,
                "training_provenance": self.config.training_provenance,
                "training_provenance_sha256": (self.config.training_provenance_sha256),
                "feature_schema": EVIDENCE_SCHEMA,
                "question_schema": QUESTION_SCHEMA,
            }
        self.last_route = result
        return result

    def _route_episode(
        self, evidence: dict[str, object], available_experts: set[str]
    ) -> dict[str, object]:
        """Use the exact v2 question set and uniform subset weights from training."""
        from .episode_router_supervision import routing_questions_v2, expert_subsets, subset_id

        if self.config is None or self.agent is None:
            raise RuntimeError("Laya episode router is unavailable")
        if evidence.get("schema") != EPISODE_EVIDENCE_SCHEMA:
            raise ValueError("Laya episode evidence schema differs")
        if set(self.config.expert_ids) - available_experts:
            raise ValueError("a frozen episode-router expert is unavailable")
        if evidence.get("identity_health") != "nominal":
            result: dict[str, object] = {
                "state": "abstained", "reason": "episode identity is degraded",
                "selected_experts": [], "eligible_experts": list(self.config.expert_ids),
                "security_router": False, "execution_ms": 0.0,
            }
            self.last_route = result
            return result
        questions = routing_questions_v2(self.config.expert_ids)
        started_ns = perf_counter_ns()
        with redirect_stdout(io.StringIO()):
            raw = self.agent.predict(evidence, questions)
        latency_ms = (perf_counter_ns() - started_ns) / 1_000_000.0
        if not isinstance(raw, dict) or not isinstance(raw.get("answers"), dict):
            raise TypeError("Laya response has no answers object")
        if not isinstance(raw.get("model"), str) or not raw["model"].strip():
            raise ValueError("Laya response model identity is absent")
        answers = raw["answers"]
        if set(answers) != set(questions):
            raise ValueError("Laya episode answer identifiers differ")
        subset_options = [subset_id(s) for s in expert_subsets(self.config.expert_ids)]
        choice, confidence, probabilities = _choice_answer(
            answers["route_subset"], subset_options, "route_subset"
        )
        budget, budget_confidence, _ = _choice_answer(
            answers["compute_budget"], ["fast", "balanced", "deep"], "compute_budget"
        )
        abstain, abstain_confidence, abstain_probs = _choice_answer(
            answers["abstain"], ["yes", "no"], "abstain"
        )
        selected = choice.split("+")
        expected_budget = "fast" if len(selected) == 1 else "balanced" if len(selected) == 2 else "deep"
        admitted = (
            abstain == "no"
            and abstain_probs["yes"] < self.config.abstain_probability
            and confidence >= self.config.minimum_selection_probability
            and probabilities[choice] >= self.config.minimum_selection_probability
            and budget == expected_budget
        )
        result = {
            "state": "active" if admitted else "abstained",
            "source": EPISODE_ROUTER_SOURCE,
            "selected_expert": selected[0] if admitted else None,
            "selected_experts": selected if admitted else [],
            "eligible_experts": list(self.config.expert_ids),
            "expert_weights": {name: 1.0 / len(selected) for name in selected} if admitted else {},
            "confidence": confidence,
            "subset_probabilities": probabilities,
            "abstain_probability": abstain_probs["yes"],
            "abstain_confidence": abstain_confidence,
            "compute_budget": budget,
            "compute_budget_confidence": budget_confidence,
            "execution_ms": latency_ms,
            "security_router": False,
            "task": "causal_episode_cost_aware_subset_routing",
            "episode_level": True,
            "model_id": self.config.model_id,
            "checkpoint_sha256": self.config.weights_sha256,
            "feature_schema": EPISODE_EVIDENCE_SCHEMA,
            "question_schema": EPISODE_QUESTION_SCHEMA,
            "reason": None if admitted else "calibration, abstention, or budget gate did not admit a route",
        }
        self.last_route = result
        return result


def routing_questions(trained: list[str]) -> dict[str, dict[str, object]]:
    """Return the single canonical question schema used in training and runtime."""
    if not trained or len(set(trained)) != len(trained):
        raise ValueError("routing questions require unique expert identifiers")
    if any(not isinstance(expert, str) or not expert.strip() for expert in trained):
        raise ValueError("routing questions contain an invalid expert identifier")
    questions: dict[str, dict[str, object]] = {
        "route_primary": {
            "type": "choice",
            "instructions": (
                "Select the forecast expert with the lowest expected "
                "source-disjoint loss for this causal network window under "
                "the compute budget. This is routing, not attack detection."
            ),
            "criteria": {
                expert: f"Execute the calibrated {expert} forecast expert"
                for expert in trained
            },
        },
        "compute_budget": {
            "type": "choice",
            "instructions": (
                "Choose the smallest compute profile expected to preserve "
                "forecast quality for this window."
            ),
            "criteria": {
                "fast": "one expert",
                "balanced": "up to the calibrated top-k experts",
                "deep": "the calibrated top-k experts when uncertainty warrants it",
            },
        },
        "abstain": {
            "type": "noul",
            "instructions": (
                "Should routing abstain because this window is out of "
                "distribution or lacks enough evidence?"
            ),
        },
    }
    for index, expert in enumerate(trained):
        questions[f"use_expert_{index}"] = {
            "type": "noul",
            "instructions": (
                f"Should forecast expert {expert} execute for this causal "
                "window after accounting for its measured latency and loss?"
            ),
        }

    return questions

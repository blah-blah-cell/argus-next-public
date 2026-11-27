from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch

from argus_research.experimental_cnn import (
    CONTEXT_PACKETS,
    LONG_CONTEXT_PACKETS,
    MODEL_TYPES,
    TemporalEnsemble,
    adapt_runtime_rows,
    adapt_runtime_rows_v6,
)
from argus_research.laya_window_router import (
    EPISODE_ROUTER_SCHEMA,
    EPISODE_ROUTER_SOURCE,
    LayaRouterConfig,
    LayaWindowRouter,
)
from argus_research.live_episode_router import build_runtime_episode_decision


def _write_interleaved_csv(path: Path, packets: int = 320) -> None:
    """Write two interleaved TCP flows with distinct global (1s) vs per-flow (2s) deltas."""
    fields = [
        "schema_id",
        "schema_version",
        "source_capture_ordinal",
        "monotonic_timestamp_ns",
        "opaque_flow_id",
        "packet_length_bytes",
        "captured_length_bytes",
        "ip_version",
        "transport_protocol",
        "packets_lost_before",
        "is_truncated",
        "is_reordered",
        "tcp_syn",
        "tcp_ack",
        "tcp_fin",
        "tcp_rst",
        "tcp_push",
        "tcp_urg",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx in range(packets):
            flow_id = idx % 2
            writer.writerow(
                {
                    "schema_id": "argus.causal-flow",
                    "schema_version": 1,
                    "source_capture_ordinal": idx + 1,
                    "monotonic_timestamp_ns": idx * 1_000_000_000,
                    "opaque_flow_id": flow_id,
                    "packet_length_bytes": 100 + flow_id * 200,
                    "captured_length_bytes": 100 + flow_id * 200,
                    "ip_version": 4,
                    "transport_protocol": "tcp",
                    "packets_lost_before": 0,
                    "is_truncated": 0,
                    "is_reordered": 0,
                    "tcp_syn": 1 if idx < 2 else 0,
                    "tcp_ack": 1 if idx >= 2 else 0,
                    "tcp_fin": 0,
                    "tcp_rst": 0,
                    "tcp_push": 0,
                    "tcp_urg": 0,
                }
            )


class _SpyV1Model(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.seen_inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen_inputs.append(x.detach().cpu().clone())
        return self.inner(x)


class _EpisodeStubAgent:
    def predict(
        self, state: dict[str, object], questions: dict[str, dict[str, object]]
    ) -> dict[str, object]:
        choices = list(questions["route_subset"]["criteria"])
        choice = "cnn_cse+lstm_cse"
        probs = {name: 0.2 / (len(choices) - 1) for name in choices}
        probs[choice] = 0.8
        return {
            "model": "episode-stub",
            "answers": {
                "route_subset": {
                    "type": "choice",
                    "choice": choice,
                    "confidence": 0.9,
                    "probabilities": probs,
                },
                "compute_budget": {
                    "type": "choice",
                    "choice": "balanced",
                    "confidence": 0.9,
                    "probabilities": {"fast": 0.05, "balanced": 0.9, "deep": 0.05},
                },
                "abstain": {
                    "type": "choice",
                    "choice": "no",
                    "confidence": 0.95,
                    "probabilities": {"yes": 0.05, "no": 0.95},
                },
            },
        }


def _build_episode_routed_ensemble(
    tmp_path: Path, model_names: tuple[str, ...]
) -> tuple[TemporalEnsemble, dict[str, _SpyV1Model]]:
    for name in model_names:
        file_name, build_model, input_context, contract = MODEL_TYPES[name]
        payload: dict[str, object] = {
            "state_dict": build_model().state_dict(),
            "feature_schema": "packet-causal-v1",
            "supervision": "next_packet_only",
            "security_detector": False,
        }
        if contract is not None and contract[0] == "expert":
            payload.update(
                {
                    "schema": "argus.temporal-expert-checkpoint/v1",
                    "expert": contract[1],
                    "input_context_packets": input_context,
                }
            )
        torch.save(payload, tmp_path / file_name)

    ensemble = TemporalEnsemble(tmp_path, device="cpu")
    spies: dict[str, _SpyV1Model] = {}
    for name in model_names:
        model, ckpt, ctx = ensemble.models[name]
        spy = _SpyV1Model(model).eval()
        spies[name] = spy
        ensemble.models[name] = (spy, ckpt, ctx)

    config = LayaRouterConfig(
        checkpoint=tmp_path,
        model_id="argus.cero-v2-parity-test",
        laya_version="0.3.4",
        expert_ids=("cnn_cse", "lstm_cse", "gru_cse", "tcn_cse"),
        top_k=4,
        minimum_selection_probability=0.7,
        abstain_probability=0.6,
        weights_sha256="a" * 64,
        training_provenance="test-fixture",
        training_provenance_sha256="b" * 64,
        schema=EPISODE_ROUTER_SCHEMA,
    )
    ensemble.laya_router = LayaWindowRouter.for_test(config, _EpisodeStubAgent())
    ensemble._laya_verified_experts = {"cnn_cse", "lstm_cse", "gru_cse", "tcn_cse"}
    return ensemble, spies


@pytest.mark.parametrize("boost_mode", [False, True])
def test_packet_causal_v1_models_receive_global_history_not_v6_features_when_episode_route_active(
    tmp_path: Path, boost_mode: bool
) -> None:
    capture = tmp_path / "interleaved.csv"
    _write_interleaved_csv(capture, packets=320)

    legacy_features, rows = adapt_runtime_rows(capture)
    v6_features = adapt_runtime_rows_v6(legacy_features, rows)
    episode_decision = build_runtime_episode_decision(v6_features, rows, "parity:test")
    assert episode_decision is not None
    # Verify test fixture actually creates a divergence between global (1s) and per-flow (2s) deltas.
    assert legacy_features[-1, 1].item() == pytest.approx(math.log1p(1.0))
    assert v6_features[-1, 1].item() == pytest.approx(math.log1p(2.0))

    model_names = (
        "cnn_cse",
        "lstm_cse",
        "gru_cse",
        "tcn_cse",
        "cnn_local",
        "ahem_fixed",
        "tcn_long",
    )
    ensemble, spies = _build_episode_routed_ensemble(tmp_path, model_names)

    result = ensemble.infer(capture, "parity:test", boost_mode=boost_mode)

    assert result["router"]["source"] == EPISODE_ROUTER_SOURCE
    assert result["router"]["state"] == "active"

    expected_executed = (
        set(model_names) if boost_mode else {"cnn_cse", "lstm_cse", "cnn_local", "ahem_fixed", "tcn_long"}
    )
    for name in model_names:
        forecast = result["models"][name]
        if name not in expected_executed:
            assert forecast["state"] == "skipped_by_router"
            assert not spies[name].seen_inputs
            continue

        assert forecast["state"] == "forecast_only"
        assert forecast["feature_schema"] == "packet-causal-v1"
        assert forecast["contract"] == "packet-causal-v1/next_global_packet"
        assert forecast["contract"] != "packet-causal-flow-v2/next_same_flow_packet"
        assert "predicted_features" not in forecast
        assert "observed_features" not in forecast
        assert forecast["observed_log_interarrival"] == pytest.approx(math.log1p(1.0))

        assert len(spies[name].seen_inputs) == 1
        actual_input = spies[name].seen_inputs[0][0]
        ctx = forecast["context_packets"]
        expected_input = ensemble.history[-ctx - 1 : -1]
        assert torch.allclose(actual_input, expected_input)
        # Ensure it does NOT match v6_features or same-flow context_indices.
        if name in {"cnn_cse", "lstm_cse", "gru_cse", "tcn_cse"}:
            v6_ctx_len = CONTEXT_PACKETS if name == "cnn_cse" else min(128, len(episode_decision.context_indices))
            v6_slice = v6_features[list(episode_decision.context_indices[-v6_ctx_len:])]
            if actual_input.shape == v6_slice.shape:
                assert not torch.allclose(actual_input, v6_slice)

    fusion = result["forecast_fusion"]
    assert fusion["legacy_consensus"]["state"] == "forecast_only"
    assert fusion["legacy_consensus"]["contract"] == "packet-causal-v1/next_global_packet"
    assert fusion["episode_consensus"]["state"] == "unavailable"


def test_packet_causal_v1_retry_path_uses_global_history_when_episode_route_active(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "interleaved_retry.csv"
    _write_interleaved_csv(capture, packets=320)
    ensemble, spies = _build_episode_routed_ensemble(tmp_path, ("cnn_cse", "lstm_cse"))

    class _FailOnceSpy(torch.nn.Module):
        def __init__(self, base_spy: _SpyV1Model) -> None:
            super().__init__()
            self.base_spy = base_spy
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated transient failure")
            return self.base_spy(x)

    retry_spy = _FailOnceSpy(spies["cnn_cse"]).eval()
    _, ckpt, ctx = ensemble.models["cnn_cse"]
    ensemble.models["cnn_cse"] = (retry_spy, ckpt, ctx)

    result = ensemble.infer(capture, "parity:retry", boost_mode=False)
    forecast = result["models"]["cnn_cse"]
    assert forecast["state"] == "forecast_only"
    assert forecast["contract"] == "packet-causal-v1/next_global_packet"
    assert forecast["feature_schema"] == "packet-causal-v1"
    assert len(spies["cnn_cse"].seen_inputs) == 1
    assert torch.allclose(
        spies["cnn_cse"].seen_inputs[0][0],
        ensemble.history[-CONTEXT_PACKETS - 1 : -1],
    )
    assert LONG_CONTEXT_PACKETS >= 128

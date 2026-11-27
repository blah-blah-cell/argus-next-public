"""Unit tests for the pure-Python ``compact-router-v1`` runtime and integration."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from argus_research.compact_router import (
    COMPACT_ROUTER_SCHEMA,
    COMPACT_ROUTER_SOURCE,
    EXPECTED_FEATURE_COUNT,
    FEATURE_NAMES,
    MAX_ARTIFACT_BYTES,
    CompactRouterConfig,
    CompactWindowRouter,
    causal_summary_features_pure,
    causal_window_evidence_pure,
    default_compact_router_path,
    extract_compact_router_features,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "research" / "evidence" / "compact-router-v1.json"


def _sample_window(count: int = 128) -> tuple[list[list[float]], list[dict[str, str]]]:
    features: list[list[float]] = []
    rows: list[dict[str, str]] = []
    for idx in range(count):
        features.append(
            [
                4.2 + (idx % 5) * 0.3,
                0.0 if idx == 0 else 0.05,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0 if idx % 16 == 0 else 0.0,
                1.0,
                0.0,
                0.0,
                0.25 if idx % 4 == 0 else 0.0,
                0.0,
            ]
        )
        rows.append(
            {
                "monotonic_timestamp_ns": str(idx * 50_000_000),
                "source_capture_ordinal": str(idx + 1),
                "opaque_flow_id": str(idx % 4),
            }
        )
    return features, rows


def test_real_artifact_loads_in_pure_python_without_torch_or_numpy() -> None:
    assert ARTIFACT_PATH.is_file(), f"Missing tracked artifact: {ARTIFACT_PATH}"
    raw = ARTIFACT_PATH.read_bytes()
    assert len(raw) < MAX_ARTIFACT_BYTES

    script = f"""
import importlib.abc, sys
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        root = fullname.split('.')[0]
        if root in ('torch', 'numpy', 'sklearn'):
            raise ImportError(f'blocked {{fullname}}')
        return None
sys.meta_path.insert(0, BlockHeavy())
sys.path.insert(0, {str(REPO_ROOT / 'python' / 'src')!r})
from pathlib import Path
from argus_research.compact_router import CompactWindowRouter, COMPACT_ROUTER_SCHEMA
router = CompactWindowRouter.from_path(Path({str(ARTIFACT_PATH)!r}))
assert router.ready, router.detail
assert router.config.schema == COMPACT_ROUTER_SCHEMA
res = router.route_features([0.0] * 60)
assert res['state'] == 'active'
assert res['requires_pytorch'] is False
assert res['requires_gpu'] is False
assert 0.0 <= res['abstain_probability'] <= 1.0
print('PURE_PYTHON_OK')
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "PURE_PYTHON_OK" in proc.stdout


def test_real_artifact_routing_and_abstain_scoring() -> None:
    router = CompactWindowRouter.from_path(ARTIFACT_PATH)
    assert router.ready
    assert router.config is not None
    assert router.config.schema == COMPACT_ROUTER_SCHEMA
    assert router.config.fallback_expert == "lstm_cse"
    assert router.config.confidence_threshold == pytest.approx(0.5)
    assert router.config.artifact_bytes < MAX_ARTIFACT_BYTES
    assert default_compact_router_path().resolve() == ARTIFACT_PATH.resolve()

    status = router.status()
    assert status["state"] == "ready"
    assert status["route_tree_nodes"] == 29
    assert status["requires_gpu"] is False
    assert status["requires_pytorch"] is False

    # High-confidence leaf path: cf_0 > 4.52719402, late_log1p_bytes > 7.09102869,
    # cf_0 > 6.48742318, late_log1p_bytes > 7.17297244 -> leaf 28 (lstm_cse ~0.968)
    high_conf_vec = [0.0] * EXPECTED_FEATURE_COUNT
    high_conf_vec[36] = 7.0  # cf_0
    high_conf_vec[32] = 7.5  # late_log1p_bytes
    res = router.route_features(high_conf_vec)
    assert res["state"] == "active"
    assert res["source"] == COMPACT_ROUTER_SOURCE
    assert res["selected_expert"] == "lstm_cse"
    assert res["used_fallback"] is False
    assert res["fallback_reason"] is None
    assert res["confidence"] > 0.95
    assert 0.0 <= res["abstain_probability"] <= 1.0
    assert isinstance(res["abstained"], bool)

    # High-confidence cnn_cse leaf path: cf_0 <= 4.527, cf_2 <= 0.5, cf_22 <= 0.393,
    # largest_flow_frac > 0.96875 -> leaf 5 (cnn_cse ~0.8857)
    cnn_vec = [0.0] * EXPECTED_FEATURE_COUNT
    cnn_vec[36] = 2.0
    cnn_vec[38] = 0.0
    cnn_vec[58] = 0.1
    cnn_vec[4] = 0.99
    cnn_res = router.route_features(cnn_vec)
    assert cnn_res["selected_expert"] == "cnn_cse"
    assert cnn_res["used_fallback"] is False
    assert cnn_res["confidence"] == pytest.approx(0.885714, abs=1e-5)

    # Unavailable top expert must fail closed to lstm_cse
    fallback_unavail = router.route_features(
        cnn_vec, available_experts={"lstm_cse", "gru_cse"}
    )
    assert fallback_unavail["selected_expert"] == "lstm_cse"
    assert fallback_unavail["raw_selected_expert"] == "cnn_cse"
    assert fallback_unavail["used_fallback"] is True
    assert fallback_unavail["fallback_reason"] == "selected_expert_unavailable"


def test_confidence_gate_triggers_fail_closed_fallback_to_lstm_cse() -> None:
    router = CompactWindowRouter.from_path(ARTIFACT_PATH)
    assert router.ready
    # Leaf 12 has max probability 0.480519 < 0.5 threshold (gru_cse):
    # cf_0 <= 4.527 (go left->1), cf_2 > 0.5 (go right->9),
    # flag_fin <= 0.0390625 (go left->10), cf_0 > 4.21210027 (go right->12)
    low_conf_vec = [0.0] * EXPECTED_FEATURE_COUNT
    low_conf_vec[36] = 4.4  # cf_0
    low_conf_vec[38] = 1.0  # cf_2
    low_conf_vec[17] = 0.0  # flag_fin
    res = router.route_features(low_conf_vec)
    assert res["state"] == "active"
    assert res["raw_selected_expert"] == "gru_cse"
    assert res["confidence"] < 0.5
    assert res["selected_expert"] == "lstm_cse"
    assert res["used_fallback"] is True
    assert res["fallback_reason"] == "confidence_below_threshold"
    assert res["expert_weights"] == {"lstm_cse": 1.0}

    # Optional abstain-gate fallback
    abstain_fb = router.route_features(
        low_conf_vec, abstain_threshold=0.0, fallback_on_abstain=True
    )
    assert abstain_fb["abstained"] is True
    assert abstain_fb["selected_expert"] == "lstm_cse"


def test_pure_python_window_feature_extraction_and_routing() -> None:
    features, rows = _sample_window(128)
    evidence = causal_window_evidence_pure(features, rows, "test:window")
    summary = causal_summary_features_pure(features)
    vec = extract_compact_router_features(evidence, summary)
    assert len(vec) == len(FEATURE_NAMES) == EXPECTED_FEATURE_COUNT

    router = CompactWindowRouter.from_path(ARTIFACT_PATH)
    res = router.route_window(features, rows, "test:window")
    assert res["state"] == "active"
    assert res["selected_expert"] in {"cnn_cse", "lstm_cse", "gru_cse", "tcn_cse"}
    assert 0.0 <= res["abstain_probability"] <= 1.0


def test_fail_closed_on_tampered_or_invalid_artifact(tmp_path: Path) -> None:
    good_data = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    bad_schema = dict(good_data, schema="argus.legacy-router/v0")
    p_schema = tmp_path / "bad_schema.json"
    p_schema.write_text(json.dumps(bad_schema), encoding="utf-8")
    r_schema = CompactWindowRouter.from_path(p_schema)
    assert not r_schema.ready
    assert "schema differs" in r_schema.detail

    bad_det = dict(good_data, security_detector=True)
    p_det = tmp_path / "bad_det.json"
    p_det.write_text(json.dumps(bad_det), encoding="utf-8")
    with pytest.raises(ValueError, match="security_detector"):
        CompactRouterConfig.load(p_det)

    # Tampered digest check
    with pytest.raises(ValueError, match="SHA-256 differs"):
        CompactRouterConfig.load(ARTIFACT_PATH, expected_sha256="0" * 64)

    # Oversized artifact > 50 KB
    oversized = tmp_path / "huge.json"
    oversized.write_bytes(b" " * MAX_ARTIFACT_BYTES)
    with pytest.raises(ValueError, match="exceeds <50 KB"):
        CompactRouterConfig.load(oversized)


def test_temporal_ensemble_routes_via_compact_router_when_laya_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch
    from argus_research.experimental_cnn import MODEL_TYPES, TemporalEnsemble

    monkeypatch.setenv("ARGUS_COMPACT_ROUTER_PATH", str(ARTIFACT_PATH))
    monkeypatch.delenv("ARGUS_CERO_CHECKPOINT", raising=False)
    monkeypatch.delenv("ARGUS_LAYA_CHECKPOINT", raising=False)

    for name in ("cnn_cse", "lstm_cse"):
        file_name, build_model, input_context, contract = MODEL_TYPES[name]
        assert contract is not None
        payload = {
            "state_dict": build_model().state_dict(),
            "feature_schema": "packet-causal-v1",
            "supervision": "next_packet_only",
            "security_detector": False,
            "schema": "argus.temporal-expert-checkpoint/v1",
            "expert": contract[1],
            "input_context_packets": input_context,
        }
        torch.save(payload, tmp_path / file_name)

    capture = tmp_path / "window.csv"
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
    with capture.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx in range(160):
            writer.writerow(
                {
                    "schema_id": "argus.causal-flow",
                    "schema_version": 1,
                    "source_capture_ordinal": idx + 1,
                    "monotonic_timestamp_ns": idx * 10_000_000,
                    "opaque_flow_id": idx % 3,
                    "packet_length_bytes": 120 + (idx % 5) * 40,
                    "captured_length_bytes": 120 + (idx % 5) * 40,
                    "ip_version": 4,
                    "transport_protocol": "tcp",
                    "packets_lost_before": 0,
                    "is_truncated": 0,
                    "is_reordered": 0,
                    "tcp_syn": 1 if idx < 3 else 0,
                    "tcp_ack": 1 if idx >= 3 else 0,
                    "tcp_fin": 0,
                    "tcp_rst": 0,
                    "tcp_push": 0,
                    "tcp_urg": 0,
                }
            )

    ensemble = TemporalEnsemble(tmp_path, device="cpu")
    assert ensemble.compact_router.ready
    result = ensemble.infer(capture, "compact:test", boost_mode=False)
    assert result["router"]["source"] == COMPACT_ROUTER_SOURCE
    assert result["router"]["state"] == "active"
    assert result["router"]["selected_expert"] in {"cnn_cse", "lstm_cse"}
    assert "abstain_probability" in result["router"]
    assert result["compact_router"]["state"] == "ready"


def test_compact_router_forged_valid_schema_rejected_by_pinned_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good_data = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    forged = dict(good_data)
    forged["route_tree"] = {
        "node_count": 1,
        "nodes": [{"leaf": True, "probs": [1.0, 0.0, 0.0, 0.0]}],
    }
    forged_path = tmp_path / "forged.json"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")

    # Default unpinned load rejects tampered valid-schema artifacts.
    rejected = CompactWindowRouter.from_path(forged_path)
    assert not rejected.ready
    assert "pinned" in rejected.detail

    monkeypatch.setenv("ARGUS_COMPACT_ROUTER_PATH", str(forged_path))
    monkeypatch.delenv("ARGUS_COMPACT_ROUTER_ALLOW_UNPINNED", raising=False)
    monkeypatch.delenv("ARGUS_COMPACT_ROUTER_SHA256", raising=False)
    env_rejected = CompactWindowRouter.from_environment()
    assert not env_rejected.ready
    assert "pinned" in env_rejected.detail

    # Explicit opt-in allows custom artifacts during research/testing.
    admitted = CompactWindowRouter.from_path(forged_path, allow_unpinned=True)
    assert admitted.ready


def test_compact_router_extreme_features_fail_closed_and_validate_threshold() -> None:
    router = CompactWindowRouter.from_path(ARTIFACT_PATH)
    assert router.ready

    # Extreme finite inputs that previously overflowed (x-m)/s to +/-inf -> NaN logit.
    extreme = [0.0] * EXPECTED_FEATURE_COUNT
    extreme[37] = 1e308
    extreme[48] = 1e308
    prob = router.predict_abstain_probability(extreme)
    assert prob == 1.0
    res = router.route_features(extreme, fallback_on_abstain=True)
    assert res["abstain_probability"] == 1.0
    assert res["abstained"] is True
    assert res["used_fallback"] is True
    assert res["selected_expert"] == "lstm_cse"

    # Non-finite or out-of-range abstain_threshold must raise ValueError.
    for bad_threshold in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValueError, match="abstain_threshold"):
            router.route_features([0.0] * EXPECTED_FEATURE_COUNT, abstain_threshold=bad_threshold)


def test_compact_router_expm1_clamping_prevents_overflow_error() -> None:
    features, rows = _sample_window(64)
    features[0][0] = 800.0  # extreme log1p packet length
    features[0][1] = 800.0  # extreme log1p interarrival delay
    evidence = causal_window_evidence_pure(features, rows, "test:overflow")
    assert evidence["interarrival_seconds"]["max"] == pytest.approx(60.0)
    assert evidence["packet_length_bytes"]["max"] > 0.0


def test_compact_router_deep_tree_iterative_validation_no_recursion_error(
    tmp_path: Path,
) -> None:
    good_data = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    # Build a 1,050-node acyclic chain (<50 KB) that exceeds Python's default recursion limit.
    depth = 1050
    nodes: list[dict[str, object]] = []
    for i in range(depth - 1):
        nodes.append({"leaf": False, "f": 0, "th": 0, "l": i + 1, "r": depth - 1})
    nodes.append({"leaf": True, "probs": [0.1, 0.7, 0.1, 0.1]})
    deep_data = {
        "schema": good_data["schema"],
        "security_detector": False,
        "requires_gpu": False,
        "requires_pytorch": False,
        "experts": good_data["experts"],
        "fallback_expert": good_data["fallback_expert"],
        "confidence_threshold": good_data["confidence_threshold"],
        "feature_names": good_data["feature_names"],
        "route_tree": {"node_count": depth, "nodes": nodes},
        "abstain_head": {
            "type": "temperature_scaled_logistic",
            "temperature": 1.0,
            "intercept": 0.0,
            "coef": [0] * EXPECTED_FEATURE_COUNT,
            "scaler_mean": [0] * EXPECTED_FEATURE_COUNT,
            "scaler_scale": [1] * EXPECTED_FEATURE_COUNT,
        },
    }
    encoded = json.dumps(deep_data, separators=(",", ":")).encode("utf-8")
    assert len(encoded) < MAX_ARTIFACT_BYTES
    deep_path = tmp_path / "deep.json"
    deep_path.write_bytes(encoded)

    cfg = CompactRouterConfig.load(deep_path, allow_unpinned=True)
    assert len(cfg.route_tree_nodes) == depth


def test_compact_router_exact_probability_tie_fails_closed(tmp_path: Path) -> None:
    good_data = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    tied = dict(
        good_data,
        route_tree={
            "node_count": 1,
            "nodes": [{"leaf": True, "probs": [0.5, 0.5, 0.0, 0.0]}],
        },
    )
    tied_path = tmp_path / "tied.json"
    tied_path.write_text(json.dumps(tied), encoding="utf-8")
    router = CompactWindowRouter.from_path(tied_path, allow_unpinned=True)
    assert router.ready
    res = router.route_features([0.0] * EXPECTED_FEATURE_COUNT)
    assert res["selected_expert"] == "lstm_cse"
    assert res["used_fallback"] is True
    assert res["fallback_reason"] == "exact_probability_tie"


def test_from_environment_falls_back_to_default_when_model_dir_lacks_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ARGUS_COMPACT_ROUTER_PATH", raising=False)
    monkeypatch.delenv("ARGUS_DISABLE_COMPACT_ROUTER", raising=False)
    router = CompactWindowRouter.from_environment(model_dir=tmp_path)
    assert router.ready, router.detail
    assert router.config is not None
    assert router.config.artifact_path.resolve() == ARTIFACT_PATH.resolve()

    # Explicit disable switch still isolates tests cleanly.
    monkeypatch.setenv("ARGUS_DISABLE_COMPACT_ROUTER", "1")
    disabled = CompactWindowRouter.from_environment(model_dir=tmp_path)
    assert not disabled.ready


def test_temporal_ensemble_causal_alignment_and_short_segment_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import math
    import torch
    from argus_research.experimental_cnn import MODEL_TYPES, TemporalEnsemble

    monkeypatch.delenv("ARGUS_COMPACT_ROUTER_PATH", raising=False)
    monkeypatch.delenv("ARGUS_DISABLE_COMPACT_ROUTER", raising=False)
    monkeypatch.delenv("ARGUS_CERO_CHECKPOINT", raising=False)
    monkeypatch.delenv("ARGUS_LAYA_CHECKPOINT", raising=False)

    for name in ("cnn_cse", "lstm_cse"):
        file_name, build_model, input_context, contract = MODEL_TYPES[name]
        assert contract is not None
        payload = {
            "state_dict": build_model().state_dict(),
            "feature_schema": "packet-causal-v1",
            "supervision": "next_packet_only",
            "security_detector": False,
            "schema": "argus.temporal-expert-checkpoint/v1",
            "expert": contract[1],
            "input_context_packets": input_context,
        }
        torch.save(payload, tmp_path / file_name)

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

    def _write_csv(path: Path, start_idx: int, count: int) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for offset in range(count):
                idx = start_idx + offset
                pkt_len = 100 + idx * 3
                writer.writerow(
                    {
                        "schema_id": "argus.causal-flow",
                        "schema_version": 1,
                        "source_capture_ordinal": idx + 1,
                        "monotonic_timestamp_ns": idx * 10_000_000,
                        "opaque_flow_id": idx % 3,
                        "packet_length_bytes": pkt_len,
                        "captured_length_bytes": pkt_len,
                        "ip_version": 4,
                        "transport_protocol": "tcp",
                        "packets_lost_before": 0,
                        "is_truncated": 0,
                        "is_reordered": 0,
                        "tcp_syn": 1 if offset == 0 else 0,
                        "tcp_ack": 1 if offset > 0 else 0,
                        "tcp_fin": 0,
                        "tcp_rst": 0,
                        "tcp_push": 0,
                        "tcp_urg": 0,
                    }
                )

    ensemble = TemporalEnsemble(tmp_path, device="cpu")
    assert ensemble.compact_router.ready

    captured_calls: list[tuple[list[list[float]], list[dict[str, str]]]] = []
    original_route_window = ensemble.compact_router.route_window

    def _spy_route_window(packet_features, rows, history_scope, available_experts=None, **kwargs):
        captured_calls.append((list(packet_features), list(rows)))
        return original_route_window(
            packet_features,
            rows,
            history_scope,
            available_experts=available_experts,
            **kwargs,
        )

    monkeypatch.setattr(ensemble.compact_router, "route_window", _spy_route_window)

    # Segment 1: 160 packets (>= 129) — verify exact 1-to-1 causal alignment (zero +1 shift).
    seg1 = tmp_path / "seg1.csv"
    _write_csv(seg1, 0, 160)
    res1 = ensemble.infer(seg1, "align:scope", boost_mode=False)
    assert res1["router"]["source"] == COMPACT_ROUTER_SOURCE
    assert len(captured_calls) == 1
    w1, r1 = captured_calls[0]
    assert len(w1) == 128 and len(r1) == 128
    # Last row in segment 1 is ordinal 160 (excluded target); routed window must end at ordinal 159.
    assert int(r1[-1]["source_capture_ordinal"]) == 159
    for vec, row in zip(w1, r1, strict=True):
        expected_len = float(row["packet_length_bytes"])
        assert math.expm1(vec[0]) == pytest.approx(expected_len, rel=1e-4)

    # Segment 2: 60 packets (< 128) after warmup — must NOT be silently skipped.
    seg2 = tmp_path / "seg2.csv"
    _write_csv(seg2, 160, 60)
    res2 = ensemble.infer(seg2, "align:scope", boost_mode=False)
    assert res2["router"]["source"] == COMPACT_ROUTER_SOURCE
    assert len(captured_calls) == 2
    w2, r2 = captured_calls[1]
    assert len(w2) == 59 and len(r2) == 59
    assert int(r2[-1]["source_capture_ordinal"]) == 219
    for vec, row in zip(w2, r2, strict=True):
        expected_len = float(row["packet_length_bytes"])
        assert math.expm1(vec[0]) == pytest.approx(expected_len, rel=1e-4)

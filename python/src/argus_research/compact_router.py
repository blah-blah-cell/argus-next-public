"""Pure-Python runtime loader and evaluator for ``compact-router-v1``.

This module executes the <50 KB pure-JSON decision-tree router and
temperature-scaled logistic abstention head with zero PyTorch, NumPy,
scikit-learn, or GPU dependencies.  All validation is fail-closed: tampered,
oversized, schema-mismatched, or structurally invalid artifacts are rejected
without executing any route, and low-confidence windows fail closed to the
artifact's declared ``fallback_expert`` (``lstm_cse``).

This router selects among causal packet-forecasting experts under compute
budgets.  It is not a security detector.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Sequence

COMPACT_ROUTER_SCHEMA = "argus.compact-router/v1"
COMPACT_ROUTER_SOURCE = "compact_router_v1"
MAX_ARTIFACT_BYTES = 50 * 1024  # 51,200 bytes (< 50 KB)
MIN_WINDOW_PACKETS = 32
DEFAULT_ABSTAIN_THRESHOLD = 0.5
DEFAULT_ARTIFACT_RELATIVE_PATH = Path("research/evidence/compact-router-v1.json")
PINNED_COMPACT_ROUTER_SHA256 = (
    "ee4c12344e05e4ee1b3530412e2a689c2819c548c818c8e77d0cbadbf690dfce"
)
_MAX_LOG1P_BYTES = 30.0
_MAX_LOG1P_DELAY = math.log1p(60.0)

FEATURE_NAMES: tuple[str, ...] = (
    "window_packets",
    "window_flows",
    "window_duration_s",
    "window_pps",
    "largest_flow_frac",
    "log1p_len_mean",
    "log1p_len_p95",
    "log1p_len_max",
    "log1p_ia_mean",
    "log1p_ia_p95",
    "log1p_ia_max",
    "frac_ipv4",
    "frac_ipv6",
    "frac_tcp",
    "frac_udp",
    "flag_syn",
    "flag_ack",
    "flag_fin",
    "flag_rst",
    "flag_push",
    "flag_urg",
    "early_packets",
    "early_log1p_bytes",
    "early_log1p_ia",
    "early_syn",
    "early_rst",
    "mid_packets",
    "mid_log1p_bytes",
    "mid_log1p_ia",
    "mid_syn",
    "mid_rst",
    "late_packets",
    "late_log1p_bytes",
    "late_log1p_ia",
    "late_syn",
    "late_rst",
    *[f"cf_{i}" for i in range(24)],
)
EXPECTED_FEATURE_COUNT = len(FEATURE_NAMES)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, not bool")
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is not numeric") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{field} is non-finite")
    return parsed


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _linear_quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = probability * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] + frac * (ordered[hi] - ordered[lo])


def _safe_packet_bytes(log1p_bytes: float) -> float:
    return max(0.0, math.expm1(min(_MAX_LOG1P_BYTES, max(0.0, log1p_bytes))))


def _safe_interarrival_seconds(log1p_delay: float) -> float:
    return min(60.0, max(0.0, math.expm1(min(_MAX_LOG1P_DELAY, max(0.0, log1p_delay)))))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def extract_compact_router_features(
    state_dict: dict[str, Any], cf_features: Sequence[float]
) -> list[float]:
    """Extract the exact 60 causal pre-expert features from window evidence + cf_0..23."""
    if len(cf_features) != 24:
        raise ValueError("compact router requires exactly 24 causal summary features")
    w = state_dict["window"]
    pl = state_dict["packet_length_bytes"]
    ia = state_dict["interarrival_seconds"]
    pf = state_dict["protocol_fractions"]
    tf = state_dict["tcp_flag_fractions"]
    tp = state_dict["temporal_phases"]

    feats = [
        _finite_float(w["packets"], "window.packets"),
        _finite_float(w["flows"], "window.flows"),
        _finite_float(w["duration_seconds"], "window.duration_seconds"),
        _finite_float(w["packets_per_second"], "window.packets_per_second"),
        _finite_float(w["largest_flow_fraction"], "window.largest_flow_fraction"),
        math.log1p(max(0.0, _finite_float(pl["mean"], "packet_length_bytes.mean"))),
        math.log1p(max(0.0, _finite_float(pl["p95"], "packet_length_bytes.p95"))),
        math.log1p(max(0.0, _finite_float(pl["max"], "packet_length_bytes.max"))),
        math.log1p(max(0.0, _finite_float(ia["mean"], "interarrival_seconds.mean"))),
        math.log1p(max(0.0, _finite_float(ia["p95"], "interarrival_seconds.p95"))),
        math.log1p(max(0.0, _finite_float(ia["max"], "interarrival_seconds.max"))),
        _finite_float(pf["ipv4"], "protocol_fractions.ipv4"),
        _finite_float(pf["ipv6"], "protocol_fractions.ipv6"),
        _finite_float(pf["tcp"], "protocol_fractions.tcp"),
        _finite_float(pf["udp"], "protocol_fractions.udp"),
        _finite_float(tf["syn"], "tcp_flag_fractions.syn"),
        _finite_float(tf["ack"], "tcp_flag_fractions.ack"),
        _finite_float(tf["fin"], "tcp_flag_fractions.fin"),
        _finite_float(tf["rst"], "tcp_flag_fractions.rst"),
        _finite_float(tf["push"], "tcp_flag_fractions.push"),
        _finite_float(tf["urg"], "tcp_flag_fractions.urg"),
    ]
    for phase in ("early", "middle", "late"):
        p = tp[phase]
        feats.extend(
            [
                _finite_float(p.get("packets", 0), f"temporal_phases.{phase}.packets"),
                math.log1p(
                    max(
                        0.0,
                        _finite_float(
                            p.get("mean_packet_bytes", 0.0),
                            f"temporal_phases.{phase}.mean_packet_bytes",
                        ),
                    )
                ),
                math.log1p(
                    max(
                        0.0,
                        _finite_float(
                            p.get("mean_interarrival_seconds", 0.0),
                            f"temporal_phases.{phase}.mean_interarrival_seconds",
                        ),
                    )
                ),
                _finite_float(
                    p.get("tcp_syn_fraction", 0.0),
                    f"temporal_phases.{phase}.tcp_syn_fraction",
                ),
                _finite_float(
                    p.get("tcp_rst_fraction", 0.0),
                    f"temporal_phases.{phase}.tcp_rst_fraction",
                ),
            ]
        )
    for idx, value in enumerate(cf_features):
        feats.append(_finite_float(value, f"cf_{idx}"))
    return feats


def causal_window_evidence_pure(
    packet_features: Sequence[Sequence[float]],
    rows: Sequence[dict[str, str]],
    history_scope: str,
) -> dict[str, Any]:
    """Pure-Python counterpart to ``laya_window_router.causal_window_evidence``."""
    count = len(packet_features)
    if count == 0 or count != len(rows):
        raise ValueError("compact router evidence requires one metadata row per packet")
    normalized: list[list[float]] = []
    for idx, vec in enumerate(packet_features):
        if len(vec) != 12:
            raise ValueError("compact router requires packet-causal-v1 [N, 12] features")
        parsed = [_finite_float(v, f"features[{idx}][{c}]") for c, v in enumerate(vec)]
        normalized.append(parsed)

    timestamps = [int(row["monotonic_timestamp_ns"]) for row in rows]
    ordinals = [int(row["source_capture_ordinal"]) for row in rows]
    if any(right < left for left, right in pairwise(timestamps)):
        raise ValueError("compact router timestamps are not monotonic")
    if any(right <= left for left, right in pairwise(ordinals)):
        raise ValueError("compact router ordinals are not strictly increasing")

    packet_lengths = [_safe_packet_bytes(vec[0]) for vec in normalized]
    interarrivals = [_safe_interarrival_seconds(vec[1]) for vec in normalized]
    duration_seconds = max(0.0, (timestamps[-1] - timestamps[0]) / 1_000_000_000.0)
    flow_counts: dict[int, int] = {}
    for row in rows:
        flow_id = int(row["opaque_flow_id"])
        flow_counts[flow_id] = flow_counts.get(flow_id, 0) + 1
    largest_flow_fraction = max(flow_counts.values()) / count

    def _phase_summary(slice_rows: list[list[float]]) -> dict[str, Any]:
        if not slice_rows:
            return {"packets": 0}
        return {
            "packets": len(slice_rows),
            "mean_packet_bytes": round(
                _mean([_safe_packet_bytes(v[0]) for v in slice_rows]), 6
            ),
            "mean_interarrival_seconds": round(
                _mean([_safe_interarrival_seconds(v[1]) for v in slice_rows]), 6
            ),
            "tcp_syn_fraction": round(_mean([v[6] for v in slice_rows]), 6),
            "tcp_rst_fraction": round(_mean([v[9] for v in slice_rows]), 6),
        }

    # Matches torch.tensor_split(features, 3)
    base, rem = divmod(count, 3)
    s0 = base + (1 if rem > 0 else 0)
    s1 = s0 + base + (1 if rem > 1 else 0)
    phases = (normalized[:s0], normalized[s0:s1], normalized[s1:])

    return {
        "schema": "argus.causal-window-evidence/v1",
        "scope": history_scope,
        "decision_time": "online_window_end",
        "payload_included": False,
        "window": {
            "packets": count,
            "flows": len(flow_counts),
            "duration_seconds": round(duration_seconds, 6),
            "packets_per_second": round(count / max(duration_seconds, 1e-6), 6),
            "largest_flow_fraction": round(largest_flow_fraction, 6),
        },
        "packet_length_bytes": {
            "mean": round(_mean(packet_lengths), 6),
            "p95": round(_linear_quantile(packet_lengths, 0.95), 6),
            "max": round(max(packet_lengths), 6),
        },
        "interarrival_seconds": {
            "mean": round(_mean(interarrivals), 6),
            "p95": round(_linear_quantile(interarrivals, 0.95), 6),
            "max": round(max(interarrivals), 6),
        },
        "protocol_fractions": {
            "ipv4": round(_mean([v[2] for v in normalized]), 6),
            "ipv6": round(_mean([v[3] for v in normalized]), 6),
            "tcp": round(_mean([v[4] for v in normalized]), 6),
            "udp": round(_mean([v[5] for v in normalized]), 6),
        },
        "tcp_flag_fractions": {
            "syn": round(_mean([v[6] for v in normalized]), 6),
            "ack": round(_mean([v[7] for v in normalized]), 6),
            "fin": round(_mean([v[8] for v in normalized]), 6),
            "rst": round(_mean([v[9] for v in normalized]), 6),
            "push": round(_mean([v[10] for v in normalized]), 6),
            "urg": round(_mean([v[11] for v in normalized]), 6),
        },
        "temporal_phases": {
            "early": _phase_summary(phases[0]),
            "middle": _phase_summary(phases[1]),
            "late": _phase_summary(phases[2]),
        },
    }


def causal_summary_features_pure(
    packet_features: Sequence[Sequence[float]],
) -> list[float]:
    """Pure-Python counterpart to ``memory_models.causal_router_features`` (24 dims)."""
    if not packet_features:
        raise ValueError("causal_summary_features_pure requires at least one packet")
    last = [_finite_float(v, f"last[{c}]") for c, v in enumerate(packet_features[-1])]
    if len(last) != 12:
        raise ValueError("expected 12 packet-causal-v1 features per packet")
    prefix = packet_features[:-1] if len(packet_features) > 1 else packet_features
    prefix_mean = [
        _mean([_finite_float(row[c], f"prefix[{r}][{c}]") for r, row in enumerate(prefix)])
        for c in range(12)
    ]
    return last + prefix_mean


def _validate_tree_nodes(
    nodes: object, num_experts: int, num_features: int
) -> tuple[dict[str, Any], ...]:
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("compact router route_tree.nodes is empty")
    node_count = len(nodes)
    validated: list[dict[str, Any]] = []
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict) or not isinstance(node.get("leaf"), bool):
            raise ValueError(f"route_tree node {idx} is malformed")
        if node["leaf"]:
            raw_probs = node.get("probs")
            if not isinstance(raw_probs, list) or len(raw_probs) != num_experts:
                raise ValueError(f"leaf node {idx} probs length differs from experts")
            probs = [_finite_float(p, f"nodes[{idx}].probs") for p in raw_probs]
            if any(p < 0.0 or p > 1.0 for p in probs):
                raise ValueError(f"leaf node {idx} has out-of-range probability")
            if abs(sum(probs) - 1.0) > 1e-3:
                raise ValueError(f"leaf node {idx} probabilities do not sum to 1")
            validated.append({"leaf": True, "probs": tuple(probs)})
        else:
            f_idx = node.get("f")
            left = node.get("l")
            right = node.get("r")
            if (
                isinstance(f_idx, bool)
                or not isinstance(f_idx, int)
                or not 0 <= f_idx < num_features
            ):
                raise ValueError(f"internal node {idx} feature index out of bounds")
            if (
                isinstance(left, bool)
                or isinstance(right, bool)
                or not isinstance(left, int)
                or not isinstance(right, int)
                or not (0 <= left < node_count and 0 <= right < node_count)
            ):
                raise ValueError(f"internal node {idx} child index out of bounds")
            threshold = _finite_float(node.get("th"), f"nodes[{idx}].th")
            validated.append(
                {"leaf": False, "f": f_idx, "th": threshold, "l": left, "r": right}
            )

    # Verify the tree is acyclic from root 0 using an explicit iterative stack
    # so valid deep trees within the <50 KB limit never hit Python's recursion cap.
    visit_state = [0] * node_count  # 0=unvisited, 1=visiting, 2=visited
    stack: list[tuple[int, int]] = [(0, 0)]
    while stack:
        node_idx, phase = stack.pop()
        if phase == 1:
            visit_state[node_idx] = 2
            continue
        if visit_state[node_idx] == 2:
            continue
        if visit_state[node_idx] == 1:
            raise ValueError("route_tree contains a cycle")
        visit_state[node_idx] = 1
        stack.append((node_idx, 1))
        entry = validated[node_idx]
        if not entry["leaf"]:
            for child in (entry["r"], entry["l"]):
                if visit_state[child] == 1:
                    raise ValueError("route_tree contains a cycle")
                if visit_state[child] == 0:
                    stack.append((child, 0))
    return tuple(validated)


def _validate_abstain_head(
    raw: object, num_features: int
) -> tuple[float, float, tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if not isinstance(raw, dict):
        raise ValueError("compact router abstain_head is absent")
    if raw.get("type") != "temperature_scaled_logistic":
        raise ValueError("unsupported compact router abstain_head type")
    temperature = _finite_float(raw.get("temperature"), "abstain_head.temperature")
    if temperature <= 0.0:
        raise ValueError("abstain_head.temperature must be positive")
    intercept = _finite_float(raw.get("intercept"), "abstain_head.intercept")

    def _vec(key: str, *, strictly_positive: bool = False) -> tuple[float, ...]:
        values = raw.get(key)
        if not isinstance(values, list) or len(values) != num_features:
            raise ValueError(f"abstain_head.{key} length differs from feature count")
        parsed = tuple(_finite_float(v, f"abstain_head.{key}[{i}]") for i, v in enumerate(values))
        if strictly_positive and any(v <= 0.0 for v in parsed):
            raise ValueError(f"abstain_head.{key} must be strictly positive")
        return parsed

    coef = _vec("coef")
    mean = _vec("scaler_mean")
    scale = _vec("scaler_scale", strictly_positive=True)
    return temperature, intercept, coef, mean, scale


@dataclass(frozen=True)
class CompactRouterConfig:
    """Immutable, validated representation of a ``compact-router-v1`` artifact."""

    artifact_path: Path
    artifact_sha256: str
    artifact_bytes: int
    schema: str
    experts: tuple[str, ...]
    fallback_expert: str
    confidence_threshold: float
    feature_names: tuple[str, ...]
    route_tree_nodes: tuple[dict[str, Any], ...]
    abstain_temperature: float
    abstain_intercept: float
    abstain_coef: tuple[float, ...]
    abstain_scaler_mean: tuple[float, ...]
    abstain_scaler_scale: tuple[float, ...]
    prepared_manifest_sha256: str | None = None

    @classmethod
    def load(
        cls,
        artifact_path: Path,
        *,
        expected_sha256: str | None = None,
        allow_unpinned: bool = False,
    ) -> CompactRouterConfig:
        resolved = artifact_path.resolve()
        if not resolved.is_file():
            raise ValueError(f"compact router artifact does not exist: {resolved}")
        raw_bytes = resolved.read_bytes()
        if len(raw_bytes) >= MAX_ARTIFACT_BYTES:
            raise ValueError(
                f"compact router artifact {len(raw_bytes)} B exceeds <50 KB limit"
            )
        digest = _sha256_bytes(raw_bytes)
        if expected_sha256 is not None and digest != expected_sha256.lower():
            raise ValueError("compact router artifact SHA-256 differs")
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("compact router artifact JSON is invalid") from error
        if not isinstance(payload, dict):
            raise ValueError("compact router artifact must be a JSON object")

        if payload.get("schema") != COMPACT_ROUTER_SCHEMA:
            raise ValueError("compact router schema differs")
        if payload.get("security_detector") is not False:
            raise ValueError("compact router security_detector must be false")
        if payload.get("requires_gpu") is not False:
            raise ValueError("compact router requires_gpu must be false")
        if payload.get("requires_pytorch") is not False:
            raise ValueError("compact router requires_pytorch must be false")

        raw_experts = payload.get("experts")
        if (
            not isinstance(raw_experts, list)
            or not raw_experts
            or any(not isinstance(e, str) or not e.strip() for e in raw_experts)
            or len(set(raw_experts)) != len(raw_experts)
        ):
            raise ValueError("compact router experts list is invalid")
        experts = tuple(raw_experts)

        fallback_expert = payload.get("fallback_expert")
        if not isinstance(fallback_expert, str) or fallback_expert not in experts:
            raise ValueError("compact router fallback_expert is not in experts")

        confidence_threshold = _finite_float(
            payload.get("confidence_threshold"), "confidence_threshold"
        )
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("compact router confidence_threshold outside [0, 1]")

        raw_features = payload.get("feature_names")
        if raw_features != list(FEATURE_NAMES):
            raise ValueError("compact router feature_names contract differs")

        route_tree = payload.get("route_tree")
        if not isinstance(route_tree, dict):
            raise ValueError("compact router route_tree is absent")
        nodes = _validate_tree_nodes(
            route_tree.get("nodes"), len(experts), EXPECTED_FEATURE_COUNT
        )
        if route_tree.get("node_count") != len(nodes):
            raise ValueError("compact router route_tree.node_count differs")

        temp, intercept, coef, mean, scale = _validate_abstain_head(
            payload.get("abstain_head"), EXPECTED_FEATURE_COUNT
        )
        prepared_hash = payload.get("prepared_manifest_sha256")
        if prepared_hash is not None and not isinstance(prepared_hash, str):
            raise ValueError("prepared_manifest_sha256 must be a string or null")

        if (
            expected_sha256 is None
            and not allow_unpinned
            and digest != PINNED_COMPACT_ROUTER_SHA256
        ):
            raise ValueError(
                "compact router artifact SHA-256 differs from pinned compact-router-v1 digest"
            )

        return cls(
            artifact_path=resolved,
            artifact_sha256=digest,
            artifact_bytes=len(raw_bytes),
            schema=COMPACT_ROUTER_SCHEMA,
            experts=experts,
            fallback_expert=fallback_expert,
            confidence_threshold=confidence_threshold,
            feature_names=FEATURE_NAMES,
            route_tree_nodes=nodes,
            abstain_temperature=temp,
            abstain_intercept=intercept,
            abstain_coef=coef,
            abstain_scaler_mean=mean,
            abstain_scaler_scale=scale,
            prepared_manifest_sha256=prepared_hash,
        )


def default_compact_router_path() -> Path:
    """Return the canonical repo-tracked ``research/evidence/compact-router-v1.json`` path."""
    return Path(__file__).resolve().parents[3] / DEFAULT_ARTIFACT_RELATIVE_PATH


class CompactWindowRouter:
    """Validated, fail-closed, pure-Python runtime evaluator for ``compact-router-v1``."""

    def __init__(self, config: CompactRouterConfig | None, detail: str) -> None:
        self.config = config
        self.detail = detail
        self.last_route: dict[str, object] | None = None

    @property
    def ready(self) -> bool:
        return self.config is not None

    @classmethod
    def from_path(
        cls,
        artifact_path: Path,
        *,
        expected_sha256: str | None = None,
        allow_unpinned: bool = False,
    ) -> CompactWindowRouter:
        try:
            config = CompactRouterConfig.load(
                artifact_path,
                expected_sha256=expected_sha256,
                allow_unpinned=allow_unpinned,
            )
            return cls(config, "local pure-JSON compact router admitted")
        except Exception as error:  # noqa: BLE001 - fail-closed runtime boundary
            return cls(None, f"rejected: {error}")

    @classmethod
    def from_environment(
        cls,
        artifact_path: Path | None = None,
        model_dir: Path | None = None,
        *,
        expected_sha256: str | None = None,
        allow_unpinned: bool = False,
    ) -> CompactWindowRouter:
        disable_flag = os.environ.get("ARGUS_DISABLE_COMPACT_ROUTER", "").strip().lower()
        if disable_flag in {"1", "true", "yes"}:
            return cls(None, "disabled by environment")
        env_path = os.environ.get("ARGUS_COMPACT_ROUTER_PATH", "").strip()
        if env_path.lower() in {"none", "disabled", "off"}:
            return cls(None, "disabled by environment")
        env_sha256 = os.environ.get("ARGUS_COMPACT_ROUTER_SHA256", "").strip() or None
        env_allow_unpinned = (
            os.environ.get("ARGUS_COMPACT_ROUTER_ALLOW_UNPINNED", "").strip().lower()
            in {"1", "true", "yes"}
        )
        effective_sha256 = expected_sha256 or env_sha256
        effective_allow_unpinned = allow_unpinned or env_allow_unpinned
        if artifact_path is not None:
            candidate = artifact_path
        elif env_path:
            candidate = Path(env_path)
            if not candidate.is_absolute():
                return cls(None, "rejected: ARGUS_COMPACT_ROUTER_PATH must be absolute")
        else:
            candidates: list[Path] = []
            if model_dir is not None:
                candidates.append(model_dir / "compact-router-v1.json")
            candidates.append(default_compact_router_path())
            existing = next((p for p in candidates if p.is_file()), None)
            if existing is None:
                return cls(None, "not configured")
            candidate = existing
        return cls.from_path(
            candidate,
            expected_sha256=effective_sha256,
            allow_unpinned=effective_allow_unpinned,
        )

    def status(self) -> dict[str, object]:
        if self.config is None:
            return {
                "state": "unavailable",
                "detail": self.detail,
                "configured": self.detail != "not configured",
                "requires_gpu": False,
                "requires_pytorch": False,
                "security_detector": False,
            }
        return {
            "state": "ready",
            "detail": self.detail,
            "configured": True,
            "router_schema": self.config.schema,
            "artifact_path": str(self.config.artifact_path),
            "artifact_sha256": self.config.artifact_sha256,
            "artifact_bytes": self.config.artifact_bytes,
            "experts": list(self.config.experts),
            "fallback_expert": self.config.fallback_expert,
            "confidence_threshold": self.config.confidence_threshold,
            "route_tree_nodes": len(self.config.route_tree_nodes),
            "abstain_temperature": self.config.abstain_temperature,
            "requires_gpu": False,
            "requires_pytorch": False,
            "security_detector": False,
            "last_route": self.last_route,
        }

    def predict_tree_probabilities(self, features: Sequence[float]) -> tuple[float, ...]:
        if self.config is None:
            raise RuntimeError("compact router is unavailable")
        if len(features) != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"compact router requires {EXPECTED_FEATURE_COUNT} features, got {len(features)}"
            )
        nodes = self.config.route_tree_nodes
        idx = 0
        while True:
            node = nodes[idx]
            if node["leaf"]:
                return node["probs"]
            val = _finite_float(features[node["f"]], f"features[{node['f']}]")
            idx = node["l"] if val <= node["th"] else node["r"]

    def predict_abstain_probability(self, features: Sequence[float]) -> float:
        if self.config is None:
            raise RuntimeError("compact router is unavailable")
        if len(features) != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"compact router requires {EXPECTED_FEATURE_COUNT} features, got {len(features)}"
            )
        logit = self.config.abstain_intercept
        for i, (x_raw, m_val, s_val, w_val) in enumerate(
            zip(
                features,
                self.config.abstain_scaler_mean,
                self.config.abstain_scaler_scale,
                self.config.abstain_coef,
                strict=True,
            )
        ):
            x_val = _finite_float(x_raw, f"features[{i}]")
            z_val = (x_val - m_val) / s_val
            if not math.isfinite(z_val):
                return 1.0
            term = z_val * w_val
            if not math.isfinite(term):
                return 1.0
            logit += term
            if not math.isfinite(logit):
                return 1.0
        scaled = logit / self.config.abstain_temperature
        if not math.isfinite(scaled):
            return 1.0
        prob = _sigmoid(max(-700.0, min(700.0, scaled)))
        return prob if math.isfinite(prob) else 1.0

    def route_features(
        self,
        features: Sequence[float],
        available_experts: set[str] | None = None,
        *,
        abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
        fallback_on_abstain: bool = False,
    ) -> dict[str, object]:
        if self.config is None:
            raise RuntimeError("compact router is unavailable")
        abstain_threshold = _finite_float(abstain_threshold, "abstain_threshold")
        if not 0.0 <= abstain_threshold <= 1.0:
            raise ValueError("abstain_threshold must be finite and in [0.0, 1.0]")
        started_ns = perf_counter_ns()
        validated_features = [
            _finite_float(v, f"features[{i}]") for i, v in enumerate(features)
        ]
        if len(validated_features) != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"compact router requires {EXPECTED_FEATURE_COUNT} features, got {len(validated_features)}"
            )

        eligible = (
            list(self.config.experts)
            if available_experts is None
            else [e for e in self.config.experts if e in available_experts]
        )
        if not eligible:
            result: dict[str, object] = {
                "state": "unavailable",
                "source": COMPACT_ROUTER_SOURCE,
                "reason": "no trained compact-router expert is currently available",
                "eligible_experts": [],
                "selected_experts": [],
                "execution_ms": (perf_counter_ns() - started_ns) / 1_000_000.0,
                "security_router": False,
                "requires_gpu": False,
                "requires_pytorch": False,
            }
            self.last_route = result
            return result

        probs = self.predict_tree_probabilities(validated_features)
        abstain_prob = self.predict_abstain_probability(validated_features)
        abstained = (not math.isfinite(abstain_prob)) or (
            abstain_prob >= abstain_threshold
        )

        best_prob = max(probs)
        top_indices = [i for i, p in enumerate(probs) if p == best_prob]
        best_idx = top_indices[0]
        is_exact_tie = len(top_indices) > 1
        raw_expert = self.config.experts[best_idx]
        confidence = float(best_prob)

        fallback = (
            self.config.fallback_expert
            if self.config.fallback_expert in eligible
            else eligible[0]
        )

        fallback_reason: str | None = None
        if is_exact_tie:
            selected_expert = fallback
            used_fallback = True
            fallback_reason = "exact_probability_tie"
        elif confidence < self.config.confidence_threshold:
            selected_expert = fallback
            used_fallback = True
            fallback_reason = "confidence_below_threshold"
        elif fallback_on_abstain and abstained:
            selected_expert = fallback
            used_fallback = True
            fallback_reason = "calibrated_abstain_gate"
        elif raw_expert not in eligible:
            selected_expert = fallback
            used_fallback = True
            fallback_reason = "selected_expert_unavailable"
        else:
            selected_expert = raw_expert
            used_fallback = False

        raw_prob_map = {
            name: float(probs[i]) for i, name in enumerate(self.config.experts)
        }
        if used_fallback:
            weights = {selected_expert: 1.0}
        else:
            eligible_mass = sum(raw_prob_map[e] for e in eligible)
            if eligible_mass > 0.0:
                weights = {e: raw_prob_map[e] / eligible_mass for e in eligible}
            else:
                weights = {selected_expert: 1.0}

        latency_ms = (perf_counter_ns() - started_ns) / 1_000_000.0
        result = {
            "state": "active",
            "source": COMPACT_ROUTER_SOURCE,
            "router_schema": self.config.schema,
            "artifact_sha256": self.config.artifact_sha256,
            "artifact_bytes": self.config.artifact_bytes,
            "selected_expert": selected_expert,
            "selected_experts": [selected_expert],
            "raw_selected_expert": raw_expert,
            "eligible_experts": eligible,
            "fallback_expert": fallback,
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason,
            "confidence": confidence,
            "confidence_threshold": self.config.confidence_threshold,
            "abstain_probability": abstain_prob,
            "abstain_threshold": abstain_threshold,
            "abstained": abstained,
            "expert_probabilities": raw_prob_map,
            "expert_weights": weights,
            "execution_ms": latency_ms,
            "security_router": False,
            "requires_gpu": False,
            "requires_pytorch": False,
            "task": "cost_aware_forecast_expert_routing",
        }
        self.last_route = result
        return result

    def route_evidence(
        self,
        evidence: dict[str, Any],
        cf_features: Sequence[float],
        available_experts: set[str] | None = None,
        *,
        abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
        fallback_on_abstain: bool = False,
    ) -> dict[str, object]:
        features = extract_compact_router_features(evidence, cf_features)
        return self.route_features(
            features,
            available_experts,
            abstain_threshold=abstain_threshold,
            fallback_on_abstain=fallback_on_abstain,
        )

    def route_window(
        self,
        packet_features: Sequence[Sequence[float]],
        rows: Sequence[dict[str, str]],
        history_scope: str,
        available_experts: set[str] | None = None,
        *,
        abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
        fallback_on_abstain: bool = False,
    ) -> dict[str, object]:
        if len(packet_features) < MIN_WINDOW_PACKETS:
            raise ValueError(
                f"compact router requires at least {MIN_WINDOW_PACKETS} packets"
            )
        evidence = causal_window_evidence_pure(packet_features, rows, history_scope)
        summary = causal_summary_features_pure(packet_features)
        return self.route_evidence(
            evidence,
            summary,
            available_experts,
            abstain_threshold=abstain_threshold,
            fallback_on_abstain=fallback_on_abstain,
        )

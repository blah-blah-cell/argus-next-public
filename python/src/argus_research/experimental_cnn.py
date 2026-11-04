"""Run the transferred CNN as an explicitly experimental packet forecaster.

This adapter is intentionally not a detector. The Rust capture path now retains
all twelve fields used by packet-causal-v1, including observed TCP flags.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from threading import Thread
from time import perf_counter_ns, time_ns

import torch

from . import ahem_v3 as frozen_v61_v3
from . import ahem_v4 as frozen_v61_v4
from . import ahem_v61_frozen, laya_window_router
from .live_episode_router import (
    RuntimeEpisodeDecision, build_runtime_episode_decision,
    runtime_flow_identity_status,
)
from .ahem_pretrain import (
    AhemNextPacket,
    CausalCnnNextPacket,
    DilatedCnnNextPacket,
    GruNextPacket,
    LongTcnNextPacket,
    LstmNextPacket,
)
from .ahem_profiles import get_profile
from .ahem_v3 import AhemV3MemoryCache, AhemV31NextPacket, AhemV32NextPacket
from .ahem_v6 import AhemV6StateTransitionNextPacket
from .ahem_v63 import AhemV63Config, AhemV63FrozenBackbone
from .continual_learning import DailyReplayJournal
from .flow_v2_features import FEATURE_CONTRACT_SHA256
from .flow_v6_objective import dynamic_loss
from .memory_models import CausalRouter, causal_router_features
from .package_ahem_v61 import (
    FINAL_RESULT_SHA256 as AHEM_V61_FINAL_RESULT_SHA256,
    MANIFEST_SCHEMA as AHEM_V61_MANIFEST_SCHEMA,
    MODEL_ID as AHEM_V61_MODEL_ID,
    RUNTIME_SCHEMA as AHEM_V61_RUNTIME_SCHEMA,
    SOURCE_CHECKPOINT_SHA256 as AHEM_V61_SOURCE_CHECKPOINT_SHA256,
)


def ahem_v61_implementation_sha256() -> str:
    """Bind the frozen v6.1 model and its dependencies to the original contract."""
    digest = hashlib.sha256()
    for original_name, module in (
        ("ahem_v3.py", frozen_v61_v3),
        ("ahem_v4.py", frozen_v61_v4),
        ("ahem_v6.py", ahem_v61_frozen),
    ):
        source = Path(module.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
        digest.update(original_name.encode())
        digest.update(b"\0")
        digest.update(source.encode())
        digest.update(b"\0")
    return digest.hexdigest()

CONTEXT_PACKETS = 31
LONG_CONTEXT_PACKETS = 1024
MAX_HISTORY_PACKETS = 30_000
AHEM_V61_CONTEXT_PACKETS = 10_000
AHEM_V61_FILE = "ahem_v61_forecast_only.pt"
AHEM_V61_ONNX_FILE = "ahem_v61.onnx"
AHEM_V63_EXPERIMENTAL_FILE = (
    "challengers/ahem_v63_experimental_seed_20260914.pt"
)
AHEM_V63_EXPERIMENTAL_SHA256 = (
    "fef9c75861b33699009ec4fe717d1d83b94a86954a2b59cd5b602fdd63605bf5"
)
AHEM_V63_BACKBONE_SHA256 = (
    "2457076fd716e4c342c2d359e0046b46ea574aa79ce4a596c2291c2e87943c9e"
)
AHEM_V63_PROTOCOL_SHA256 = (
    "79d51caca2fcc4269ba16dca2a194dc3ab846329fb715d8b78e12c6f47be45cb"
)
AHEM_E2E_MODEL_ID = "ahem_e2e_experimental"
AHEM_E2E_CHECKPOINT_SHA256 = "f4ed02499db772df25a5273d182b31c70ddebc1c782a45e787f39cc4dec27593"
AHEM_E2E_CHECKPOINT = "challengers/ahem_e2e_experimental_step3808.pt"
AHEM_E2E_CONFIG = {
    "width": 256, "key_dim": 64, "phase_width": 128,
    "local_packets": 128, "slots": 16, "horizon": 10_000,
}


class RecoveredAhemShadow:
    """Bounded, CPU-only exploratory inference; never a routed expert or detector."""

    def __init__(self, checkpoint: Path) -> None:
        # Validate the bytes before weights-only deserialization or allocation.
        if not checkpoint.is_file():
            raise ValueError("recovered checkpoint absent; not bundled with installed app")
        if checkpoint.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("recovered checkpoint exceeds size bound")
        if _sha256(checkpoint) != AHEM_E2E_CHECKPOINT_SHA256:
            raise ValueError("recovered checkpoint SHA-256 differs")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("schema") != "argus.ahem-e2e-inference-weights/v1":
            raise ValueError("recovered checkpoint schema differs")
        if payload.get("config") != AHEM_E2E_CONFIG:
            raise ValueError("recovered checkpoint configuration differs")
        if payload.get("step") != 3808 or payload.get("seed") != 20261018:
            raise ValueError("recovered checkpoint identity differs")
        required_metadata = {
            "source_checkpoint_sha256": "d206c62997c283bd050d4a29e09988a1aa431e73b4fbd1729bb0f2c94ce3549f",
            "security_detector": False,
            "publication_evidence": False,
            "promotion_eligible": False,
            "evaluation_status": "unvalidated_recovered_incomplete_run",
        }
        if any(payload.get(key) != expected for key, expected in required_metadata.items()):
            raise ValueError("recovered checkpoint inference contract differs")
        if set(payload) != {"schema", "config", "step", "seed", "state_dict", *required_metadata}:
            raise ValueError("recovered checkpoint contains unexpected metadata")
        state = payload.get("state_dict")
        if not isinstance(state, dict) or not state or any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or not torch.isfinite(value).all()
            for value in state.values()
        ):
            raise ValueError("recovered checkpoint weights absent or non-finite")
        from .ahem_endtoend_fast_20261103 import FastEndToEndAhem

        self.model = FastEndToEndAhem(**AHEM_E2E_CONFIG)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False)

    @staticmethod
    def status(state: str, **details: object) -> dict[str, object]:
        return {
            "state": state, "experimental_shadow": True,
            "promotion_eligible": False, "deployable": False,
            "publication_evidence": False, "security_detector": False,
            "security_verdict": None, "calibrated": False,
            "affects_alerts": False, "routing_eligible": False,
            "checkpoint_sha256": AHEM_E2E_CHECKPOINT_SHA256,
            "checkpoint_step": 3808, "selected_seed": 20261018,
            "runtime_backend": "pytorch", "execution_provider": "PyTorch CPU",
            "cache_scope": "capture_or_segment_local",
            "feature_schema": "packet-causal-flow-v2",
            "task": "next_same_flow_dynamic_event_forecast",
            "evidence_limit": "recovered incomplete research run; not validated for deployment",
            **details,
        }

    def run(self, features: torch.Tensor, rows: list[dict[str, str]]) -> dict[str, object]:
        if len(features) != len(rows) or features.ndim != 2 or features.shape[1] != 12:
            raise ValueError("recovered shadow requires aligned 12-field features")
        if runtime_flow_identity_status(rows)["state"] == "invalid":
            return self.status("unavailable", reason="invalid flow identity")
        if not len(rows):
            return self.status("warming_up", context_packets=0)
        # Never retain unbounded capture history, and never use the target packet
        # when scoring the earlier query. Truncation makes early rows protected.
        rows = rows[-AHEM_E2E_CONFIG["horizon"]:]
        features = features[-len(rows):].cpu()
        if not torch.isfinite(features).all():
            raise ValueError("recovered shadow features are non-finite")
        previous = torch.full((len(rows),), -1, dtype=torch.long)
        last_by_flow: dict[str, int] = {}
        for index, row in enumerate(rows):
            flow = row["opaque_flow_id"]
            previous[index] = last_by_flow.get(flow, -1)
            last_by_flow[flow] = index

        def forecast(anchor: int) -> dict[str, object] | None:
            if anchor < AHEM_E2E_CONFIG["local_packets"] - 1:
                return None
            with torch.inference_mode():
                output = self.model(
                    features[:anchor + 1], previous[:anchor + 1],
                    torch.tensor([anchor]), features[anchor:anchor + 1, :2],
                )
            # The checkpoint expects a separate frozen-parent fallback. Do not
            # impersonate that parent: protected rows are reported as warmup.
            if bool(output["protected"][0]):
                return None
            prediction = output["prediction"][0]
            if not torch.isfinite(prediction).all():
                raise ValueError("recovered shadow prediction is non-finite")
            valid = output["valid"][0]
            sources = output["source_ordinals"][0][valid].tolist()
            return {
                "predicted_features": prediction.tolist(),
                "context_packets": anchor + 1,
                "selected_source_ordinals": [int(rows[index]["source_capture_ordinal"]) for index in sources],
                "memory_weights": output["weights"][0][valid].tolist(),
            }

        pending = forecast(len(rows) - 1)
        prior = int(previous[-1])
        scored = forecast(prior) if prior >= 0 else None
        result = self.status("forecast_only" if scored else "warming_up", forecasts_scored=int(scored is not None))
        if pending is not None:
            result["pending_next_same_flow"] = pending
        if scored is not None:
            prediction = torch.tensor(scored["predicted_features"])
            result.update(scored)
            result.update({
                "forecast_error": float(dynamic_loss(prediction, features[-1, :2])),
                "predicted_log_packet_length": float(prediction[0]),
                "predicted_log_interarrival": float(prediction[1]),
                "observed_log_packet_length": float(features[-1, 0]),
                "observed_log_interarrival": float(features[-1, 1]),
            })
        return result


def _torch_cpu_threads() -> int:
    """Bound opt-in PyTorch CPU parallelism; the balanced desktop sets one."""
    raw = os.environ.get("ARGUS_TORCH_CPU_THREADS", "1")
    try:
        threads = int(raw)
    except ValueError:
        return 1
    return threads if 1 <= threads <= 8 else 1


def _smooth_l1_scalar(prediction: float, target: float) -> float:
    difference = abs(prediction - target)
    return 0.5 * difference * difference if difference < 1.0 else difference - 0.5


def evidence_weighted_consensus(
    forecasts: dict[str, object], router: dict[str, object]
) -> dict[str, object]:
    """Blend only contract-compatible forecasts using causal router weights.

    AHEM v6.1 is deliberately excluded: it forecasts the next packet on the
    same flow, while legacy experts forecast the next globally observed packet.
    Mixing those targets would manufacture a meaningless score.
    """
    episode_route = router.get("source") == laya_window_router.EPISODE_ROUTER_SOURCE
    if router.get("used_fallback") is True and isinstance(
        router.get("selected_expert"), str
    ):
        declared = {str(router["selected_expert"]): 1.0}
        weight_source = "calibrated_router_fallback"
    else:
        declared = router.get("expert_weights", {})
        weight_source = "causal_learned_router_probabilities"
    if not isinstance(declared, dict):
        declared = {}
    candidates: list[tuple[str, dict[str, object], float]] = []
    for name, raw_weight in declared.items():
        result = forecasts.get(name)
        if not isinstance(result, dict) or result.get("state") != "forecast_only":
            continue
        if result.get("experimental_shadow") is True:
            continue
        if episode_route and result.get("contract") != "packet-causal-flow-v2/next_same_flow_packet":
            continue
        if not episode_route and result.get("contract") == "packet-causal-flow-v2/next_same_flow_packet":
            continue
        try:
            weight = max(0.0, float(raw_weight))
            float(result["predicted_log_packet_length"])
            float(result["predicted_log_interarrival"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(weight) and weight > 0.0:
            candidates.append((name, result, weight))
    if not candidates:
        return {
            "state": "unavailable",
            "reason": "no contract-compatible routed forecasts completed",
            "cross_contract_blending": False,
        }
    total = sum(weight for _, _, weight in candidates)
    weights = {name: weight / total for name, _, weight in candidates}
    if episode_route:
        predicted = torch.tensor(
            [
                sum(weights[name] * float(result["predicted_features"][index])
                    for name, result, _ in candidates)
                for index in range(12)
            ]
        )
        observed = torch.tensor(candidates[0][1]["observed_features"])
        field_weights = torch.tensor([1.0, 4.0] + [0.5] * 10)
        error = float((
            torch.nn.functional.smooth_l1_loss(predicted, observed, reduction="none")
            * field_weights
        ).sum() / field_weights.sum())
        return {
            "state": "forecast_only",
            "contract": "packet-causal-flow-v2/next_same_flow_packet",
            "weight_source": "frozen_uniform_episode_subset",
            "model_weights": weights,
            "predicted_log_packet_length": float(predicted[0]),
            "predicted_log_interarrival": float(predicted[1]),
            "observed_log_packet_length": float(observed[0]),
            "observed_log_interarrival": float(observed[1]),
            "forecast_error": error,
            "cross_contract_blending": False,
        }
    predicted_length = sum(
        weights[name] * float(result["predicted_log_packet_length"])
        for name, result, _ in candidates
    )
    predicted_time = sum(
        weights[name] * float(result["predicted_log_interarrival"])
        for name, result, _ in candidates
    )
    first = candidates[0][1]
    observed_length = float(first["observed_log_packet_length"])
    observed_time = float(first["observed_log_interarrival"])
    return {
        "state": "forecast_only",
        "contract": "packet-causal-v1/next_global_packet",
        "weight_source": weight_source,
        "model_weights": weights,
        "predicted_log_packet_length": predicted_length,
        "predicted_log_interarrival": predicted_time,
        "observed_log_packet_length": observed_length,
        "observed_log_interarrival": observed_time,
        "forecast_error": 0.5
        * (
            _smooth_l1_scalar(predicted_length, observed_length)
            + _smooth_l1_scalar(predicted_time, observed_time)
        ),
        "cross_contract_blending": False,
    }


def _elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_ahem_v31_manifest(checkpoint: Path) -> str:
    path = checkpoint.with_suffix(checkpoint.suffix + ".manifest.json")
    if not path.is_file():
        raise ValueError("AHEM-v3.1 manifest is absent")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = _sha256(checkpoint)
    checks = {
        "format": "argus-model-artifact",
        "format_version": 1,
        "feature_schema_id": "packet-causal-v1",
        "feature_schema_version": 1,
        "model_id": "argus.ahem-v3.1.forecaster",
        "artifact_sha256": expected_hash,
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise ValueError(f"AHEM-v3.1 manifest {key} differs")
    return expected_hash


def adapt_runtime_rows(
    path: Path, previous_timestamp_ns: int | None = None
) -> tuple[torch.Tensor, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    features: list[list[float]] = []
    previous_timestamp = previous_timestamp_ns
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "schema_id",
            "schema_version",
            "source_capture_ordinal",
            "monotonic_timestamp_ns",
            "opaque_flow_id",
            "ip_version",
            "transport_protocol",
            "packet_length_bytes",
            "tcp_syn",
            "tcp_ack",
            "tcp_fin",
            "tcp_rst",
            "tcp_push",
            "tcp_urg",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "runtime feature export lacks the experimental adapter fields"
            )
        for row in reader:
            if row["schema_id"] != "argus.causal-flow" or row["schema_version"] != "1":
                raise ValueError("unsupported runtime feature schema")
            timestamp = int(row["monotonic_timestamp_ns"])
            delta_seconds = (
                0.0
                if previous_timestamp is None
                else max(
                    0.0, min((timestamp - previous_timestamp) / 1_000_000_000.0, 60.0)
                )
            )
            previous_timestamp = timestamp
            ip_version = int(row["ip_version"])
            protocol = row["transport_protocol"]
            features.append(
                [
                    math.log1p(int(row["packet_length_bytes"])),
                    math.log1p(delta_seconds),
                    float(ip_version == 4),
                    float(ip_version == 6),
                    float(protocol == "tcp"),
                    float(protocol == "udp"),
                    float(int(row["tcp_syn"])),
                    float(int(row["tcp_ack"])),
                    float(int(row["tcp_fin"])),
                    float(int(row["tcp_rst"])),
                    float(int(row["tcp_push"])),
                    float(int(row["tcp_urg"])),
                ]
            )
            rows.append(row)
    if not features:
        return torch.empty((0, 12), dtype=torch.float32), rows
    return torch.tensor(features, dtype=torch.float32), rows


def adapt_runtime_rows_v6(
    features: torch.Tensor, rows: list[dict[str, str]]
) -> torch.Tensor:
    """Apply the frozen v2 per-flow interarrival contract within one capture."""
    if len(features) != len(rows):
        raise ValueError("v6 adapter rows and features have different lengths")
    adapted = features.clone()
    previous_by_flow: dict[int, int] = {}
    interarrivals: list[float] = []
    for row in rows:
        flow_id = int(row["opaque_flow_id"])
        timestamp = int(row["monotonic_timestamp_ns"])
        previous = previous_by_flow.get(flow_id)
        delta_seconds = (
            0.0
            if previous is None
            else max(0.0, min((timestamp - previous) / 1_000_000_000.0, 60.0))
        )
        interarrivals.append(math.log1p(delta_seconds))
        previous_by_flow[flow_id] = timestamp
    adapted[:, 1] = torch.tensor(interarrivals, dtype=torch.float32)
    return adapted


def load_checkpoint(checkpoint: Path) -> CausalCnnNextPacket:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("feature_schema") != "packet-causal-v1":
        raise ValueError("CNN checkpoint has the wrong feature schema")
    if payload.get("supervision") != "next_packet_only":
        raise ValueError("CNN checkpoint is not a next-packet forecaster")
    model = CausalCnnNextPacket(12)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


MODEL_TYPES = {
    "ahem_fixed": (
        "ahem_fixed_real_pcap_30k.pt",
        lambda: AhemNextPacket(12),
        LONG_CONTEXT_PACKETS,
        None,
    ),
    "cnn_local": (
        "cnn_local_real_pcap_30k.pt",
        lambda: CausalCnnNextPacket(12),
        CONTEXT_PACKETS,
        None,
    ),
    "lstm": (
        "lstm_real_pcap_30k.pt",
        lambda: LstmNextPacket(12),
        LONG_CONTEXT_PACKETS,
        None,
    ),
    "tcn_long": (
        "tcn_long_real_pcap_30k.pt",
        lambda: LongTcnNextPacket(12),
        LONG_CONTEXT_PACKETS,
        None,
    ),
    "ahem_v31": (
        "ahem_v31_forecast_only.pt",
        lambda: AhemV31NextPacket(12),
        MAX_HISTORY_PACKETS,
        None,
    ),
    "cnn_cse": (
        "cnn_cse_cic_ids2018_best.pt",
        lambda: DilatedCnnNextPacket(12),
        CONTEXT_PACKETS,
        ("expert", "cnn"),
    ),
    "lstm_cse": (
        "lstm_cse_cic_ids2018_best.pt",
        lambda: LstmNextPacket(12),
        MAX_HISTORY_PACKETS,
        ("expert", "lstm"),
    ),
    "gru_cse": (
        "gru_cse_cic_ids2018_best.pt",
        lambda: GruNextPacket(12, 104),
        MAX_HISTORY_PACKETS,
        ("expert", "gru"),
    ),
    "tcn_cse": (
        "tcn_cse_cic_ids2018_best.pt",
        lambda: LongTcnNextPacket(12, 48),
        MAX_HISTORY_PACKETS,
        ("expert", "tcn"),
    ),
    "ahem_flash": (
        "ahem_flash_cse_cic_ids2018_best.pt",
        lambda: get_profile("flash").build(),
        MAX_HISTORY_PACKETS,
        ("profile", "flash"),
    ),
    "ahem_pro": (
        "ahem_pro_cse_cic_ids2018_best.pt",
        lambda: get_profile("pro").build(),
        MAX_HISTORY_PACKETS,
        ("profile", "pro"),
    ),
}


class AhemV31StreamingMemory:
    """Incremental delayed writer for the bounded AHEM-v3.1 chunk cache."""

    def __init__(self, model: AhemV31NextPacket, version: str) -> None:
        self.model = model
        self.version = version
        capacity = (MAX_HISTORY_PACKETS - model.local_packets) // model.chunk_size
        self.cache = AhemV3MemoryCache(capacity)
        self.delay = torch.empty((0, model.feature_dim), dtype=torch.float32)
        self.pending = torch.empty((0, model.feature_dim), dtype=torch.float32)

    def reset(self) -> None:
        self.cache = AhemV3MemoryCache(self.cache.capacity)
        self.delay = torch.empty((0, self.model.feature_dim), dtype=torch.float32)
        self.pending = torch.empty((0, self.model.feature_dim), dtype=torch.float32)

    def append(self, packets: torch.Tensor) -> None:
        combined = torch.cat((self.delay, packets))
        retained = self.model.local_packets + 1
        matured = max(0, len(combined) - retained)
        if matured:
            self.pending = torch.cat((self.pending, combined[:matured]))
        self.delay = combined[matured:]
        completed = len(self.pending) // self.model.chunk_size
        if completed:
            count = completed * self.model.chunk_size
            with torch.inference_mode():
                states = self.model.encode_completed_chunks(self.pending[:count][None])[
                    0
                ]
            for state in states:
                self.cache.append(state, version=self.version)
            self.pending = self.pending[count:]

    def forecast(self) -> dict[str, object]:
        required = self.model.local_packets + 1
        if len(self.delay) < required or self.cache.states is None:
            return {
                "state": "warming_up",
                "observed_packets": len(self.delay),
                "required_packets": required + self.model.chunk_size,
                "cached_chunks": self.cache.chunk_count,
            }
        local = self.delay[-required:-1]
        target = self.delay[-1]
        with torch.inference_mode():
            output = self.model.forward_from_cache(
                local[None], self.cache, version=self.version
            )
        prediction = output.prediction[0]
        return {
            "state": "forecast_only",
            "context_packets": (
                self.cache.chunk_count * self.model.chunk_size
                + self.model.local_packets
            ),
            "forecast_error": float(
                torch.nn.functional.smooth_l1_loss(prediction, target)
            ),
            "predicted_log_packet_length": float(prediction[0]),
            "observed_log_packet_length": float(target[0]),
            "predicted_log_interarrival": float(prediction[1]),
            "observed_log_interarrival": float(target[1]),
            "cache": {
                "version": self.version,
                "chunks": self.cache.chunk_count,
                "capacity_chunks": self.cache.capacity,
                "bytes": self.cache.chunk_count * self.model.width * 4,
                "pending_packets": len(self.pending),
                "selected_chunks": output.selected_chunks[0].tolist(),
                "selected_weights": output.selected_weights[0].tolist(),
            },
        }


class AhemV32StreamingMemory:
    """Incremental transition cache matching the trained AHEM-v3.2 contract."""

    def __init__(self, model: AhemV32NextPacket, version: str) -> None:
        self.model = model
        self.version = version
        capacity = (MAX_HISTORY_PACKETS - model.local_packets - 1) // model.chunk_size
        self.cache = AhemV3MemoryCache(capacity)
        self.delay = torch.empty((0, model.feature_dim), dtype=torch.float32)
        self.pending = torch.empty((0, model.feature_dim), dtype=torch.float32)

    def reset(self) -> None:
        self.cache = AhemV3MemoryCache(self.cache.capacity)
        self.delay = torch.empty((0, self.model.feature_dim), dtype=torch.float32)
        self.pending = torch.empty((0, self.model.feature_dim), dtype=torch.float32)

    def append(self, packets: torch.Tensor) -> None:
        combined = torch.cat((self.delay, packets))
        retained = self.model.local_packets + 1
        matured = max(0, len(combined) - retained)
        if matured:
            self.pending = torch.cat((self.pending, combined[:matured]))
        self.delay = combined[matured:]
        completed = max(0, (len(self.pending) - 1) // self.model.chunk_size)
        if completed:
            consumed = completed * self.model.chunk_size
            with torch.inference_mode():
                states = self.model.encode_completed_chunks(
                    self.pending[: consumed + 1][None]
                )[0]
            for state in states:
                self.cache.append(state, version=self.version)
            self.pending = self.pending[consumed:]

    def forecast(self) -> dict[str, object]:
        required = self.model.local_packets + 1
        if len(self.delay) < required or self.cache.states is None:
            return {
                "state": "warming_up",
                "observed_packets": len(self.delay),
                "required_packets": required + self.model.chunk_size,
                "cached_chunks": self.cache.chunk_count,
            }
        local = self.delay[-required:-1]
        target = self.delay[-1]
        with torch.inference_mode():
            output = self.model.forward_from_cache(
                local[None], self.cache, version=self.version
            )
        prediction = output.prediction[0]
        return {
            "state": "forecast_only",
            "context_packets": (
                self.cache.chunk_count * self.model.chunk_size
                + self.model.local_packets
            ),
            "forecast_error": float(
                torch.nn.functional.smooth_l1_loss(prediction, target)
            ),
            "predicted_log_packet_length": float(prediction[0]),
            "observed_log_packet_length": float(target[0]),
            "predicted_log_interarrival": float(prediction[1]),
            "observed_log_interarrival": float(target[1]),
            "cache": {
                "version": self.version,
                "chunks": self.cache.chunk_count,
                "capacity_chunks": self.cache.capacity,
                "bytes": self.cache.chunk_count * self.model.memory_width * 4,
                "pending_packets": len(self.pending),
                "selected_chunks": output.selected_chunks[0].tolist(),
                "selected_weights": output.selected_weights[0].tolist(),
                "blend_is_learned": False,
                "memory_blend_fraction": float(output.memory_gate[0, 0]),
            },
        }


class AhemV61CaptureMemory:
    """Exact bounded v6.1 cache for one capture with stable opaque flow IDs.

    Desktop live segments currently mint flow IDs independently.  This object is
    therefore deliberately rebuilt for every exported segment instead of
    inventing cross-segment identity from network endpoints.
    """

    def __init__(
        self,
        model: ahem_v61_frozen.AhemV6StateTransitionNextPacket,
        artifact_hash: str,
        source_checkpoint_hash: str,
        onnx_session: object | None = None,
        onnx_artifact_hash: str | None = None,
    ) -> None:
        self.model = model
        self.artifact_hash = artifact_hash
        self.source_checkpoint_hash = source_checkpoint_hash
        self.onnx_session = onnx_session
        self.onnx_artifact_hash = onnx_artifact_hash

    def run(
        self, features: torch.Tensor, rows: list[dict[str, str]]
    ) -> dict[str, object]:
        flow_lanes: dict[int, list[tuple[int, torch.Tensor]]] = {}
        flow_transitions: dict[
            int,
            list[
                tuple[
                    int,
                    torch.Tensor,
                    int,
                    torch.Tensor,
                    int,
                    torch.Tensor,
                ]
            ],
        ] = {}
        pending: dict[int, dict[str, object]] = {}
        evaluated = 0
        issued = 0
        last_eval_event: (
            tuple[
                int,
                dict[str, object],
                int,
                torch.Tensor,
                dict[str, str],
            ]
            | None
        ) = None
        last_pending_item: dict[str, object] | None = None
        context_span = features.new_tensor([float(AHEM_V61_CONTEXT_PACKETS)])

        with torch.inference_mode():
            all_local_encoded = (
                self.model.local.forward_sequence(features[None])[0]
                if self.onnx_session is None
                else None
            )
            for index, (packet, row) in enumerate(zip(features, rows, strict=True)):
                flow_id = int(row["opaque_flow_id"])
                prior_pending = pending.pop(flow_id, None)
                if prior_pending is not None:
                    evaluated += 1
                    last_eval_event = (flow_id, prior_pending, index, packet, row)

                lane = flow_lanes.setdefault(flow_id, [])
                transitions = flow_transitions.setdefault(flow_id, [])
                if len(lane) >= 2:
                    predecessor_ordinal, predecessor = lane[-2]
                    source_ordinal, source = lane[-1]
                    transitions.append(
                        (
                            predecessor_ordinal,
                            predecessor,
                            source_ordinal,
                            source,
                            index,
                            packet,
                        )
                    )
                    oldest = index - AHEM_V61_CONTEXT_PACKETS + 1
                    transitions[:] = [
                        t for t in transitions if t[2] >= oldest
                    ][-self.model.max_transitions :]
                lane.append((index, packet))
                del lane[:-2]

                if (
                    index + 1 < self.model.local_packets
                    or len(transitions) < 2
                ):
                    continue
                predecessor_ordinal, query_predecessor = lane[-2]
                query_ordinal, query_current = lane[-1]
                pending_item = {
                    "index": index,
                    "local_encoded": (
                        all_local_encoded[index : index + 1]
                        if all_local_encoded is not None
                        else None
                    ),
                    "query_tuple": (
                        predecessor_ordinal,
                        query_predecessor,
                        query_ordinal,
                        query_current,
                    ),
                    "transitions": list(transitions),
                    "flow_id": flow_id,
                    "anchor_ordinal": int(row["source_capture_ordinal"]),
                }
                pending[flow_id] = pending_item
                issued += 1
                last_pending_item = pending_item

            def _compute_forecast(item: dict[str, object]) -> dict[str, object]:
                idx = int(item["index"])
                local_enc = item["local_encoded"]
                if self.onnx_session is None:
                    assert isinstance(local_enc, torch.Tensor)
                p_ord, q_pred, q_ord, q_curr = item["query_tuple"]
                trans = item["transitions"]
                assert isinstance(trans, list)
                device = features.device
                q_gap = self.model._normalized_gap(
                    torch.tensor([p_ord], device=device),
                    torch.tensor([q_ord], device=device),
                    context_span,
                )
                preds = torch.stack([t[1] for t in trans])
                srcs = torch.stack([t[3] for t in trans])
                succs = torch.stack([t[5] for t in trans])
                p_ords = torch.tensor([float(t[0]) for t in trans], device=device)
                s_ords = torch.tensor([float(t[2]) for t in trans], device=device)
                t_ords = torch.tensor([float(t[4]) for t in trans], device=device)

                state_gaps = self.model._normalized_gap(p_ords, s_ords, context_span)
                trans_gaps = self.model._normalized_gap(s_ords, t_ords, context_span)

                keys = self.model.project_state(preds, srcs, state_gaps)[None]
                values = self.model.project_value(srcs, succs, trans_gaps)[None]
                successors = succs[:, :2][None]
                source_ordinals = torch.tensor(
                    [[t[2] for t in trans]], dtype=torch.int64, device=device
                )
                valid = torch.ones(
                    source_ordinals.shape, dtype=torch.bool, device=device
                )
                if self.onnx_session is not None:
                    pad = self.model.max_transitions - len(trans)
                    padded_keys = torch.nn.functional.pad(keys, (0, 0, 0, pad))
                    padded_values = torch.nn.functional.pad(values, (0, 0, 0, pad))
                    padded_successors = torch.nn.functional.pad(successors, (0, 0, 0, pad))
                    padded_ordinals = torch.nn.functional.pad(source_ordinals, (0, pad))
                    padded_valid = torch.nn.functional.pad(valid, (0, pad))
                    feed = {
                        "local_packets": features[idx + 1 - self.model.local_packets : idx + 1][None],
                        "query_predecessor": q_pred[None],
                        "query_current": q_curr[None],
                        "keys": padded_keys,
                        "values": padded_values,
                        "successor_dynamic": padded_successors,
                        "source_ordinals": padded_ordinals,
                        "valid": padded_valid,
                        "anchor_ordinals": torch.tensor([idx], dtype=torch.int64, device=device),
                        "context_spans": context_span,
                        "query_gap": q_gap,
                    }
                    onnx_feed = {
                        name: tensor.detach().cpu().contiguous().numpy()
                        for name, tensor in feed.items()
                    }
                    prediction, retrieval_weights, memory_gate = self.onnx_session.run(
                        ["prediction", "retrieval_weights", "memory_gate"], onnx_feed
                    )
                    if (
                        prediction.shape != (1, 2)
                        or retrieval_weights.shape != (1, self.model.max_transitions)
                        or memory_gate.shape != (1, 1)
                    ):
                        raise ValueError("AHEM ONNX output shape differs from the frozen contract")
                    pred = torch.from_numpy(prediction[0].copy()).to(device)
                    weights = torch.from_numpy(retrieval_weights[0, : len(trans)].copy())
                    gate = float(memory_gate[0, 0])
                    if (
                        not torch.isfinite(pred).all()
                        or not torch.isfinite(weights).all()
                        or not 0.0 <= gate <= 1.0
                        or bool((weights < 0).any())
                        or bool((weights > 1).any())
                    ):
                        raise ValueError("AHEM ONNX output violates the forecast contract")
                else:
                    query = self.model.project_state(
                        q_pred[None], q_curr[None], q_gap
                    )
                    output = self.model._read(
                        local_enc,
                        query,
                        keys,
                        values,
                        successors,
                        source_ordinals,
                        valid,
                        torch.tensor([idx], dtype=torch.int64, device=device),
                        context_span,
                    )
                    pred = output.prediction[0].clone()
                    weights = output.retrieval_weights[0]
                    gate = float(output.memory_gate[0, 0])
                selected_count = min(3, len(trans))
                selected = weights.topk(selected_count).indices.tolist()
                return {
                    "prediction": pred,
                    "memory_gate": gate,
                    "selected_source_ordinals": [
                        trans[pos][2] for pos in selected
                    ],
                    "selected_weights": [
                        float(weights[pos]) for pos in selected
                    ],
                    "cache_entries": len(trans),
                    "flow_id": item["flow_id"],
                    "anchor_ordinal": item["anchor_ordinal"],
                }

            last_evaluation = None
            if last_eval_event is not None:
                (
                    _,
                    eval_pending_item,
                    _,
                    eval_packet,
                    eval_row,
                ) = last_eval_event
                evaluated_forecast = _compute_forecast(eval_pending_item)
                pred_tensor = evaluated_forecast["prediction"]
                assert isinstance(pred_tensor, torch.Tensor)
                err = float(dynamic_loss(pred_tensor, eval_packet[:2]))
                last_evaluation = {
                    "forecast_error": err,
                    "predicted_log_packet_length": float(pred_tensor[0]),
                    "observed_log_packet_length": float(eval_packet[0]),
                    "predicted_log_interarrival": float(pred_tensor[1]),
                    "observed_log_interarrival": float(eval_packet[1]),
                    "evaluated_target_ordinal": int(
                        eval_row["source_capture_ordinal"]
                    ),
                    "memory_gate": evaluated_forecast["memory_gate"],
                    "selected_source_ordinals": evaluated_forecast[
                        "selected_source_ordinals"
                    ],
                    "selected_weights": evaluated_forecast["selected_weights"],
                    "cache_entries": evaluated_forecast["cache_entries"],
                }

            last_pending = None
            if last_pending_item is not None:
                last_pending = _compute_forecast(last_pending_item)

        common: dict[str, object] = {
            "checkpoint": AHEM_V61_FILE,
            "checkpoint_sha256": self.artifact_hash,
            "onnx_artifact_sha256": self.onnx_artifact_hash,
            "source_checkpoint_sha256": self.source_checkpoint_hash,
            "security_verdict": None,
            "security_detector": False,
            "calibrated": False,
            "task": "next_same_flow_dynamic_event_forecast",
            "feature_schema": "packet-causal-flow-v2",
            "loss_contract": "weighted_smooth_l1_1_to_4",
            "cache_scope": "capture_or_segment_local",
            "cache_reset_reason": "opaque_flow_ids_are_not_stable_across_live_segments",
            "context_packets": min(len(features), AHEM_V61_CONTEXT_PACKETS),
            "processed_packets": len(features),
            "flows_seen": len(flow_lanes),
            "forecasts_issued": issued,
            "eligible_forecasts": evaluated,
            "forecasts_scored": 1 if last_evaluation is not None else 0,
            "pending_forecasts": len(pending),
        }
        if last_pending is not None:
            common["pending_next_same_flow"] = {
                key: value
                for key, value in last_pending.items()
                if key != "prediction"
            } | {
                "predicted_log_packet_length": float(last_pending["prediction"][0]),
                "predicted_log_interarrival": float(last_pending["prediction"][1]),
            }
        if last_evaluation is None:
            common.update(
                {
                    "state": "warming_up",
                    "required_local_packets": self.model.local_packets,
                    "required_completed_transitions": 2,
                }
            )
            return common
        common.update(last_evaluation)
        common["state"] = "forecast_only"
        common["cache"] = {
            "version": "ahem-v6.1",
            "entries": last_evaluation["cache_entries"],
            "capacity_entries": self.model.max_transitions,
            "horizon_packets": AHEM_V61_CONTEXT_PACKETS,
            "selected_source_ordinals": last_evaluation["selected_source_ordinals"],
            "selected_weights": last_evaluation["selected_weights"],
            "memory_gate": last_evaluation["memory_gate"],
            "scope": "capture_or_segment_local",
        }
        return common


class AhemV63CaptureShadow:
    """Evaluate the rejected v6.3 candidate without affecting live decisions."""

    def __init__(
        self,
        model: AhemV63FrozenBackbone,
        artifact_hash: str,
        selected_seed: int,
    ) -> None:
        self.model = model
        self.artifact_hash = artifact_hash
        self.selected_seed = selected_seed

    def _context(
        self,
        features: torch.Tensor,
        rows: list[dict[str, str]],
        end: int,
        query_flow: int,
    ) -> tuple[torch.Tensor, int]:
        start = max(0, end - self.model.config.horizon)
        relation = features.new_tensor(
            [
                float(int(row["opaque_flow_id"]) == query_flow)
                for row in rows[start:end]
            ]
        )[:, None]
        return torch.cat((features[start:end], relation), dim=1), start

    def _forecast(
        self,
        features: torch.Tensor,
        rows: list[dict[str, str]],
        end: int,
        query_flow: int,
    ) -> dict[str, object] | None:
        context, start = self._context(features, rows, end, query_flow)
        if len(context) < self.model.backbone.local_packets:
            return None
        with torch.inference_mode():
            diagnostics = self.model.forward_diagnostics(context[None])
        memory_valid = diagnostics.valid[0, 1:]
        memory_weights = diagnostics.weights[0, 1:]
        valid_indices = memory_valid.nonzero().flatten()
        selected_ordinals: list[int] = []
        selected_weights: list[float] = []
        if len(valid_indices):
            count = min(3, len(valid_indices))
            ranked = valid_indices[
                memory_weights[valid_indices].topk(count).indices
            ]
            for index in ranked.tolist():
                context_index = int(diagnostics.source_ordinals[0, index])
                selected_ordinals.append(
                    int(rows[start + context_index]["source_capture_ordinal"])
                )
                selected_weights.append(float(memory_weights[index]))
        prediction = diagnostics.prediction[0]
        return {
            "prediction": prediction,
            "context_packets": len(context),
            "query_flow": query_flow,
            "cache": {
                "version": "ahem-v6.3-experimental-seed-20260914",
                "entries": int(memory_valid.sum()),
                "capacity_entries": self.model.config.capacity,
                "horizon_packets": self.model.config.horizon,
                "selected_source_ordinals": selected_ordinals,
                "selected_weights": selected_weights,
                "null_weight": float(diagnostics.weights[0, 0]),
                "memory_gate": float(1.0 - diagnostics.weights[0, 0]),
                "selected_action": int(diagnostics.weights[0].argmax()),
                "scope": "capture_or_segment_local",
            },
        }

    def run(
        self, features: torch.Tensor, rows: list[dict[str, str]]
    ) -> dict[str, object]:
        query_flow = int(rows[-1]["opaque_flow_id"])
        pending = self._forecast(features, rows, len(features), query_flow)
        common: dict[str, object] = {
            "checkpoint": Path(AHEM_V63_EXPERIMENTAL_FILE).name,
            "checkpoint_sha256": self.artifact_hash,
            "selected_seed": self.selected_seed,
            "study_role": "source_disjoint_exploratory_vm_training",
            "promotion_eligible": False,
            "publication_evidence": False,
            "experimental_shadow": True,
            "security_verdict": None,
            "security_detector": False,
            "calibrated": False,
            "task": "next_same_flow_dynamic_event_forecast",
            "feature_schema": "packet-causal-flow-v2",
            "loss_contract": "weighted_smooth_l1_1_to_4",
        }
        if pending is not None:
            prediction = pending["prediction"]
            assert isinstance(prediction, torch.Tensor)
            common["pending_next_same_flow"] = {
                "predicted_log_packet_length": float(prediction[0]),
                "predicted_log_interarrival": float(prediction[1]),
                "query_flow": pending["query_flow"],
                "context_packets": pending["context_packets"],
            }

        prior_query = next(
            (
                index
                for index in range(len(rows) - 2, -1, -1)
                if int(rows[index]["opaque_flow_id"]) == query_flow
            ),
            None,
        )
        evaluated = (
            self._forecast(features, rows, prior_query + 1, query_flow)
            if prior_query is not None
            else None
        )
        if evaluated is None:
            common.update(
                {
                    "state": "warming_up",
                    "context_packets": (
                        int(pending["context_packets"])
                        if pending is not None
                        else min(len(features), self.model.config.horizon)
                    ),
                    "required_local_packets": self.model.backbone.local_packets,
                    "forecasts_scored": 0,
                }
            )
            return common

        prediction = evaluated.pop("prediction")
        assert isinstance(prediction, torch.Tensor)
        target = features[-1, :2]
        common.update(
            {
                "state": "forecast_only",
                "context_packets": evaluated["context_packets"],
                "forecast_error": float(dynamic_loss(prediction, target)),
                "predicted_log_packet_length": float(prediction[0]),
                "observed_log_packet_length": float(target[0]),
                "predicted_log_interarrival": float(prediction[1]),
                "observed_log_interarrival": float(target[1]),
                "forecasts_scored": 1,
                "cache": evaluated["cache"],
            }
        )
        return common


class TemporalEnsemble:
    """Persistent, bounded history shared by every compatible forecaster."""

    @staticmethod
    def _select_device(
        requested: str,
    ) -> tuple[torch.device, str, str, float | None, float | None]:
        if requested in ("auto", "cuda") and torch.cuda.is_available():
            try:
                device = torch.device("cuda:0")
                name = torch.cuda.get_device_name(0)
                total_mb = (
                    torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                )
                free_mb = torch.cuda.mem_get_info(0)[0] / (1024 * 1024)
                test_tensor = torch.zeros(1, device=device)
                del test_tensor
                torch.cuda.empty_cache()
                return device, name, "PyTorch CUDA", total_mb, free_mb
            except Exception:  # noqa: BLE001 - fallback if CUDA initialization fails
                pass
        return torch.device("cpu"), "CPU", "PyTorch CPU", None, None

    def __init__(
        self,
        model_dir: Path,
        device: str = "auto",
        disabled_models: frozenset[str] = frozenset(),
        micro_batch_size: int = 32,
        learning_directory: Path | None = None,
        user_disabled_models: frozenset[str] = frozenset(),
        enable_ahem_e2e_shadow: bool = False,
    ) -> None:
        self.model_dir = model_dir
        if learning_directory is not None and not learning_directory.is_absolute():
            raise ValueError("learning_directory must be an absolute path")
        self.learning_directory = (
            learning_directory if learning_directory is not None else model_dir.parent / "learning"
        )
        self.micro_batch_size = max(1, micro_batch_size)
        (
            self.device,
            self.device_name,
            self.execution_provider,
            self.vram_total_mb,
            self.vram_free_mb,
        ) = self._select_device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(_torch_cpu_threads())
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        self.history = torch.empty((0, 12), dtype=torch.float32)
        self.history_scope: str | None = None
        self.last_timestamp_ns: int | None = None
        self.models: dict[str, tuple[torch.nn.Module, Path, int]] = {}
        self.disabled_models = disabled_models
        self.user_disabled_models = user_disabled_models
        self.ahem_e2e_shadow: RecoveredAhemShadow | None = None
        self.rejected: dict[str, str] = {}
        self.router: CausalRouter | None = None
        self.router_manifest: dict[str, object] | None = None
        self.laya_router = laya_window_router.LayaWindowRouter(
            None, None, "not configured"
        )
        self._laya_verified_experts: set[str] | None = None
        self.ahem_v31_memory: AhemV31StreamingMemory | None = None
        self.ahem_v32_memories: dict[str, AhemV32StreamingMemory] = {}
        self.ahem_v61: tuple[ahem_v61_frozen.AhemV6StateTransitionNextPacket, str, str] | None = None
        self.ahem_v61_onnx: tuple[object, str] | None = None
        self.ahem_v63_experimental: tuple[AhemV63FrozenBackbone, str, int] | None = None
        # Simulation and imported captures must not create or prune the live journal.
        # Instantiate it only when a trusted live-segment key is supplied.
        self.daily_learning: DailyReplayJournal | None = None
        for name, (file_name, build_model, context, contract) in MODEL_TYPES.items():
            if name in user_disabled_models:
                self.rejected[name] = "disabled by user"
                continue
            checkpoint = model_dir / file_name
            if not checkpoint.is_file():
                self.rejected[name] = "artifact absent"
                continue
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if payload.get("feature_schema") != "packet-causal-v1":
                    raise ValueError("feature schema is not packet-causal-v1")
                if payload.get("supervision") != "next_packet_only":
                    raise ValueError("artifact is not next-packet-only")
                if contract is not None:
                    contract_kind, identity = contract
                    if payload.get("security_detector") is not False:
                        raise ValueError("security truthfulness flag is absent")
                    if contract_kind == "expert":
                        if (
                            payload.get("schema")
                            != "argus.temporal-expert-checkpoint/v1"
                        ):
                            raise ValueError(
                                "temporal expert checkpoint schema differs"
                            )
                        if payload.get("expert") != identity:
                            raise ValueError("temporal expert identity differs")
                    elif contract_kind == "profile":
                        profile = get_profile(identity)
                        if payload.get("schema") != "argus.ahem-training-checkpoint/v1":
                            raise ValueError("AHEM checkpoint schema differs")
                        if payload.get("profile") != profile.manifest():
                            raise ValueError("AHEM profile differs")
                model = build_model()
                model.load_state_dict(payload["state_dict"], strict=True)
                if (
                    self.device.type == "cuda"
                    and name not in disabled_models
                    and name != "ahem_v31"
                    and (contract is None or contract[0] != "profile")
                ):
                    try:
                        model.to(self.device)
                    except Exception:  # noqa: BLE001
                        model.to("cpu")
                model.eval()
                if name == "ahem_v31":
                    artifact_hash = _validate_ahem_v31_manifest(checkpoint)
                    if payload.get("artifact_type") != "causal_temporal_forecaster":
                        raise ValueError("AHEM-v3.1 artifact type is not declared")
                    if payload.get("security_detector") is not False:
                        raise ValueError(
                            "AHEM-v3.1 security truthfulness flag is absent"
                        )
                    if payload.get("architecture") != "ahem-v3.1-adaptive-chunk-memory":
                        raise ValueError("AHEM-v3.1 architecture identifier differs")
                    self.ahem_v31_memory = AhemV31StreamingMemory(model, artifact_hash)
                elif contract is not None and contract[0] == "profile":
                    if not isinstance(model, AhemV32NextPacket):
                        raise ValueError("AHEM profile did not build AHEM-v3.2")
                    self.ahem_v32_memories[name] = AhemV32StreamingMemory(
                        model, _sha256(checkpoint)
                    )
                self.models[name] = (model, checkpoint, context)
            except Exception as error:  # noqa: BLE001 - one bad optional artifact is quarantined
                self.rejected[name] = str(error)
        for name, loader in (("ahem_v61", self._load_ahem_v61), ("ahem_v63_experimental", self._load_ahem_v63_experimental)):
            if name in user_disabled_models:
                self.rejected[name] = "disabled by user"
            else:
                loader()
        if AHEM_E2E_MODEL_ID in user_disabled_models:
            self.rejected[AHEM_E2E_MODEL_ID] = "disabled by user"
        elif enable_ahem_e2e_shadow:
            checkpoint = model_dir / "challengers" / "ahem_e2e_experimental_step3808.pt"
            if not checkpoint.is_file():
                checkpoint = Path(__file__).resolve().parents[3] / "models" / AHEM_E2E_CHECKPOINT
            try:
                self.ahem_e2e_shadow = RecoveredAhemShadow(checkpoint)
            except Exception as error:  # noqa: BLE001 - optional shadow quarantined
                self.rejected[AHEM_E2E_MODEL_ID] = str(error)
        self._load_router()
        configured_laya = (
            os.environ.get("ARGUS_CERO_CHECKPOINT", "").strip()
            or os.environ.get("ARGUS_LAYA_CHECKPOINT", "").strip()
        )
        local_v2 = (
            model_dir.parent
            / "checkpoints"
            / "laya-episode-router-v2-20260923"
            / "router-v2-final-mb32-stable-20260923"
        )
        checkpoint = Path(configured_laya) if configured_laya else local_v2
        if configured_laya or local_v2.is_dir():
            self.laya_router = laya_window_router.LayaWindowRouter(
                None, None, "loading local Argus CERO checkpoint"
            )

            def load_optional_router() -> None:
                self.laya_router = laya_window_router.LayaWindowRouter.from_environment(
                    device=device, checkpoint=checkpoint
                )

            Thread(target=load_optional_router, name="argus-cero-loader", daemon=True).start()

    def _laya_status(self) -> dict[str, object]:
        status = self.laya_router.status()
        status["routing_eligible_experts"] = (
            list(self.laya_router.config.expert_ids)
            if self.laya_router.config is not None
            else []
        )
        status["expert_execution_venues"] = {
            name: "native_onnx" if name in self.disabled_models else "python_pytorch"
            for name in sorted(self.models)
        }
        excluded: list[dict[str, str]] = []
        if self.ahem_e2e_shadow is not None:
            excluded.append({"expert_id": AHEM_E2E_MODEL_ID, "reason": "experimental shadow is not promotion eligible"})
        if self.ahem_v61 is not None:
            excluded.append(
                {
                    "expert_id": "ahem_v61",
                    "reason": (
                        "different next-same-flow forecast contract; runs as the "
                        "primary research forecaster"
                    ),
                }
            )
        if self.ahem_v63_experimental is not None:
            excluded.append(
                {
                    "expert_id": "ahem_v63_experimental",
                    "reason": "experimental shadow is not promotion eligible",
                }
            )
        status["excluded_runtime_models"] = excluded
        return status

    def _load_ahem_v61(self) -> None:
        checkpoint = self.model_dir / AHEM_V61_FILE
        manifest_path = checkpoint.with_suffix(checkpoint.suffix + ".manifest.json")
        if not checkpoint.is_file():
            self.rejected["ahem_v61"] = "artifact absent"
            return
        try:
            if not manifest_path.is_file():
                raise ValueError("hash-bound runtime manifest is absent")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifact_hash = _sha256(checkpoint)
            required_manifest = {
                "schema": AHEM_V61_MANIFEST_SCHEMA,
                "model_id": AHEM_V61_MODEL_ID,
                "artifact_sha256": artifact_hash,
                "source_checkpoint_sha256": AHEM_V61_SOURCE_CHECKPOINT_SHA256,
                "final_result_sha256": AHEM_V61_FINAL_RESULT_SHA256,
                "security_detector": False,
                "feature_schema": "packet-causal-flow-v2",
                "model_implementation_sha256": ahem_v61_implementation_sha256(),
                "model_implementation_files": [
                    "ahem_v3.py",
                    "ahem_v4.py",
                    "ahem_v6.py",
                ],
            }
            for key, expected in required_manifest.items():
                if manifest.get(key) != expected:
                    raise ValueError(f"runtime manifest {key} differs")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            required_payload = {
                "schema": AHEM_V61_RUNTIME_SCHEMA,
                "model_id": AHEM_V61_MODEL_ID,
                "feature_schema": "packet-causal-flow-v2",
                "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
                "supervision": "next_same_flow_dynamic_event_forecast_only",
                "security_detector": False,
                "calibrated_security_verdict": False,
                "selected_seed": 20260906,
                "source_checkpoint_sha256": AHEM_V61_SOURCE_CHECKPOINT_SHA256,
                "final_result_sha256": AHEM_V61_FINAL_RESULT_SHA256,
                "model_implementation_sha256": ahem_v61_implementation_sha256(),
                "model_implementation_files": [
                    "ahem_v3.py",
                    "ahem_v4.py",
                    "ahem_v6.py",
                ],
            }
            for key, expected in required_payload.items():
                if payload.get(key) != expected:
                    raise ValueError(f"runtime artifact {key} differs")
            model = ahem_v61_frozen.AhemV6StateTransitionNextPacket()
            model.load_state_dict(payload["state_dict"], strict=True)
            if self.device.type == "cuda":
                try:
                    model.to(self.device)
                except Exception:  # noqa: BLE001 - quarantine device failure to CPU
                    model.to("cpu")
            model.eval()
            if payload.get("parameters") != sum(
                parameter.numel() for parameter in model.parameters()
            ):
                raise ValueError("runtime artifact parameter count differs")
            self.ahem_v61 = (
                model,
                artifact_hash,
                AHEM_V61_SOURCE_CHECKPOINT_SHA256,
            )
            self._load_ahem_v61_onnx(artifact_hash)
        except Exception as error:  # noqa: BLE001 - optional model is quarantined
            self.rejected["ahem_v61"] = str(error)

    def _load_ahem_v61_onnx(self, checkpoint_hash: str) -> None:
        artifact = self.model_dir / AHEM_V61_ONNX_FILE
        manifest_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
        try:
            import onnxruntime as ort

            if not artifact.is_file() or not manifest_path.is_file():
                raise FileNotFoundError("AHEM v6.1 ONNX artifact or manifest is absent")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            required = {
                "schema": "argus.ahem-v61-onnx-manifest/v1",
                "model_id": AHEM_V61_MODEL_ID,
                "artifact_sha256": _sha256(artifact),
                "source_checkpoint_sha256": checkpoint_hash,
                "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
                "model_implementation_sha256": ahem_v61_implementation_sha256(),
                "feature_schema_id": "packet-causal-flow-v2",
                "security_detector": False,
                "calibrated_security_verdict": False,
            }
            for key, expected in required.items():
                if manifest.get(key) != expected:
                    raise ValueError(f"ONNX manifest {key} differs")
            parity = manifest.get("parity_verification", {})
            if parity.get("verified") is not True or float(parity.get("max_abs_error", 1.0)) > 1e-4:
                raise ValueError("ONNX export parity is not verified")
            options = ort.SessionOptions()
            options.intra_op_num_threads = _torch_cpu_threads()
            session = ort.InferenceSession(
                str(artifact), sess_options=options, providers=["CPUExecutionProvider"]
            )
            expected_inputs = {
                "local_packets", "query_predecessor", "query_current", "keys", "values",
                "successor_dynamic", "source_ordinals", "valid", "anchor_ordinals",
                "context_spans", "query_gap",
            }
            expected_outputs = {"prediction", "retrieval_weights", "memory_gate"}
            if {item.name for item in session.get_inputs()} != expected_inputs:
                raise ValueError("AHEM ONNX input contract differs")
            if {item.name for item in session.get_outputs()} != expected_outputs:
                raise ValueError("AHEM ONNX diagnostic output contract differs")
            self.ahem_v61_onnx = (session, required["artifact_sha256"])
        except Exception as error:  # noqa: BLE001 - explicit PyTorch fallback
            self.rejected["ahem_v61_onnx"] = str(error)

    def _load_ahem_v63_experimental(self) -> None:
        checkpoint = self.model_dir / AHEM_V63_EXPERIMENTAL_FILE
        if not checkpoint.is_file():
            self.rejected["ahem_v63_experimental"] = "artifact absent"
            return
        try:
            artifact_hash = _sha256(checkpoint)
            if artifact_hash != AHEM_V63_EXPERIMENTAL_SHA256:
                raise ValueError("experimental artifact SHA-256 differs")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            required = {
                "schema": "argus.ahem-v6.3-local-pilot/v1",
                "study_role": "source_disjoint_exploratory_vm_training",
                "promotion_eligible": False,
                "security_detector": False,
                "backbone_sha256": AHEM_V63_BACKBONE_SHA256,
                "protocol_coordinate_sha256": AHEM_V63_PROTOCOL_SHA256,
                "best_epoch": 4,
            }
            for key, expected in required.items():
                if payload.get(key) != expected:
                    raise ValueError(f"experimental artifact {key} differs")
            config_value = payload.get("config")
            if not isinstance(config_value, dict):
                raise ValueError("experimental artifact config is absent")
            config = AhemV63Config(**config_value)
            if config.horizon != AHEM_V61_CONTEXT_PACKETS:
                raise ValueError("experimental artifact context differs")
            state = payload.get("state_dict")
            if not isinstance(state, dict) or not state:
                raise ValueError("experimental artifact state is absent")
            if not all(
                isinstance(value, torch.Tensor) and torch.isfinite(value).all()
                for value in state.values()
            ):
                raise ValueError("experimental artifact state is non-finite")
            model = AhemV63FrozenBackbone(
                AhemV6StateTransitionNextPacket(), config
            )
            model.load_state_dict(state, strict=True)
            if sum(parameter.numel() for parameter in model.parameters()) != 47_454:
                raise ValueError("experimental artifact parameter count differs")
            if self.device.type == "cuda":
                try:
                    model.to(self.device)
                except Exception:  # noqa: BLE001 - optional shadow may use CPU
                    model.to("cpu")
            model.eval()
            self.ahem_v63_experimental = (model, artifact_hash, 20260914)
        except Exception as error:  # noqa: BLE001 - challenger is fail-closed
            self.rejected["ahem_v63_experimental"] = str(error)

    def _load_router(self) -> None:
        path = self.model_dir / "temporal_router_real_live.pt"
        if not path.is_file():
            self.rejected["temporal_router"] = "artifact absent"
            return
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            manifest = payload["manifest"]
            if manifest.get("artifact_type") != "causal_temporal_forecaster_router":
                raise ValueError("wrong router artifact type")
            if manifest.get("security_detector") is not False:
                raise ValueError("router truthfulness flag is absent")
            hashes = manifest["expert_sha256"]
            expert_names = list(manifest["experts"])
            if not expert_names:
                raise ValueError("router manifest has no experts")
            for name in expert_names:
                if name not in self.models:
                    raise ValueError(f"router expert {name} is unavailable")
                _, checkpoint, _ = self.models[name]
                if hashes.get(name) != _sha256(checkpoint):
                    raise ValueError(f"{name} checksum differs from router training")
            router = CausalRouter(24, len(expert_names))
            router.load_state_dict(payload["state_dict"], strict=True)
            if self.device.type == "cuda":
                try:
                    router.to(self.device)
                except Exception:  # noqa: BLE001
                    router.to("cpu")
            router.eval()
            self.router = router
            self.router_manifest = manifest
        except Exception as error:  # noqa: BLE001 - optional router must not crash inference
            self.rejected["temporal_router"] = str(error)

    def _fallback_to_cpu(self, reason: str) -> None:
        """Gracefully migrate all models and execution to CPU following a GPU failure.

        Preserves ring-buffer history, temporal timestamps, scopes, and streaming
        memory completely intact without dropped packets or corrupted states.
        """
        if self.device.type == "cpu" and "fallback" in self.execution_provider:
            return
        self.device = torch.device("cpu")
        self.device_name = "CPU"
        self.execution_provider = f"PyTorch CPU (fallback: {reason})"
        self.vram_total_mb = None
        self.vram_free_mb = None
        self.vram_allocated_mb = None
        self.vram_reserved_mb = None
        torch.set_num_threads(_torch_cpu_threads())
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

        for name, (model, _, _) in list(self.models.items()):
            try:
                model.to("cpu")
                model.eval()
            except Exception:  # noqa: BLE001
                pass

        if self.router is not None:
            try:
                self.router.to("cpu")
                self.router.eval()
            except Exception:  # noqa: BLE001
                pass

        if self.ahem_v61 is not None:
            model, _, _ = self.ahem_v61
            try:
                model.to("cpu")
                model.eval()
            except Exception:  # noqa: BLE001
                pass
        if self.ahem_v63_experimental is not None:
            model, _, _ = self.ahem_v63_experimental
            try:
                model.to("cpu")
                model.eval()
            except Exception:  # noqa: BLE001
                pass

    def _stage_to_device(
        self,
        tensor: torch.Tensor,
        device: torch.device,
        micro_batch_size: int | None = None,
    ) -> torch.Tensor:
        """Micro-batched host-to-device transfer amortizing PCIe transfer overhead.

        Uses pinned host memory and asynchronous DMA copies in configurable
        slices (16-32 packets default) to eliminate PCIe transfer bottlenecks.
        Supports 1D, 2D, and 3D context tensors. Fails safe to unpinned copy on
        pinned memory pressure, and safe to CPU on device failure.
        """
        if device.type != "cuda" or tensor.is_cuda:
            return tensor.to(device)
        if tensor.dim() == 0 or tensor.numel() == 0:
            return tensor.to(device)

        mbs = micro_batch_size or self.micro_batch_size
        tensor_contig = tensor.contiguous()
        is_pinned = False
        try:
            if not tensor_contig.is_pinned():
                pinned = tensor_contig.pin_memory()
                is_pinned = True
            else:
                pinned = tensor_contig
                is_pinned = True
        except Exception:  # noqa: BLE001 - host pinned memory pool may be exhausted
            pinned = tensor_contig
            is_pinned = False

        try:
            dst = torch.empty(pinned.shape, dtype=pinned.dtype, device=device)
            stream = torch.cuda.current_stream(device)
            if pinned.dim() == 3 and pinned.shape[0] == 1:
                # 3D context tensor: (1, seq_len, features)
                seq_len = pinned.shape[1]
                if seq_len <= mbs:
                    dst.copy_(pinned, non_blocking=is_pinned)
                else:
                    for start in range(0, seq_len, mbs):
                        end = min(start + mbs, seq_len)
                        dst[:, start:end].copy_(
                            pinned[:, start:end], non_blocking=is_pinned
                        )
            else:
                # 1D or 2D tensor: (N, features) or (N,)
                n_items = pinned.shape[0] if pinned.dim() > 0 else 1
                if n_items <= mbs:
                    dst.copy_(pinned, non_blocking=is_pinned)
                else:
                    for start in range(0, n_items, mbs):
                        end = min(start + mbs, n_items)
                        dst[start:end].copy_(
                            pinned[start:end], non_blocking=is_pinned
                        )
            stream.synchronize()
            return dst
        except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as error:
            self._fallback_to_cpu(f"stage_to_device failure: {error}")
            return tensor.to("cpu")

    def recover_device(self) -> bool:
        """Attempt to re-initialize and migrate back to CUDA if the GPU has recovered."""
        if not torch.cuda.is_available():
            return False
        try:
            device, name, provider, total_mb, free_mb = self._select_device("cuda")
            if device.type != "cuda":
                return False
            # Verify driver health with a probe tensor
            probe = torch.zeros(1, device=device)
            del probe
            torch.cuda.empty_cache()

            self.device = device
            self.device_name = name
            self.execution_provider = provider
            self.vram_total_mb = total_mb
            self.vram_free_mb = free_mb
            for name_m, (model, _, _) in self.models.items():
                if (
                    name_m not in self.disabled_models
                    and name_m != "ahem_v31"
                    and name_m not in self.ahem_v32_memories
                ):
                    try:
                        model.to(device)
                        model.eval()
                    except Exception:  # noqa: BLE001
                        pass
            if self.router is not None:
                try:
                    self.router.to(device)
                    self.router.eval()
                except Exception:  # noqa: BLE001
                    pass
            if self.ahem_v61 is not None:
                try:
                    self.ahem_v61[0].to(device)
                    self.ahem_v61[0].eval()
                except Exception:  # noqa: BLE001
                    pass
            if self.ahem_v63_experimental is not None:
                try:
                    self.ahem_v63_experimental[0].to(device)
                    self.ahem_v63_experimental[0].eval()
                except Exception:  # noqa: BLE001
                    pass
            return True
        except Exception:  # noqa: BLE001
            return False

    def _route(
        self,
        window_features: torch.Tensor | None = None,
        rows: list[dict[str, str]] | None = None,
        history_scope: str | None = None,
        episode_decision: RuntimeEpisodeDecision | None = None,
    ) -> tuple[str | None, dict[str, object]]:
        started_ns = perf_counter_ns()
        laya_attempt: dict[str, object] | None = None
        if (
            self.laya_router.config is not None
            and self.laya_router.config.schema == laya_window_router.EPISODE_ROUTER_SCHEMA
            and self.laya_router.agent is not None
        ):
            if episode_decision is None:
                laya_attempt = {
                    "state": "warming_up",
                    "reason": "no completed same-flow episode with 128 prior packets in this segment",
                }
            else:
                try:
                    if self._laya_verified_experts is None:
                        expected = self.laya_router.config.expert_sha256
                        self._laya_verified_experts = {
                            name for name, (_, checkpoint, _) in self.models.items()
                            if name in expected
                            and name not in self.disabled_models
                            and _sha256(checkpoint) == expected[name]
                        }
                    laya_attempt = self.laya_router.route(
                        episode_decision.evidence, self._laya_verified_experts
                    )
                    laya_attempt["decision_ordinal"] = episode_decision.decision_ordinal
                    laya_attempt["target_ordinal"] = episode_decision.target_ordinal
                    laya_attempt["bundle_id"] = episode_decision.bundle_id
                    if laya_attempt.get("state") == "active":
                        selected = laya_attempt.get("selected_expert")
                        if not isinstance(selected, str):
                            raise ValueError("active Laya episode route has no selected expert")
                        return selected, laya_attempt
                except Exception as error:  # noqa: BLE001 - optional route is quarantined
                    laya_attempt = {
                        "state": "error", "detail": str(error),
                        "execution_ms": _elapsed_ms(started_ns),
                    }
        elif (
            self.laya_router.config is not None
            and self.laya_router.agent is not None
            and window_features is not None
            and rows is not None
            and history_scope is not None
            and len(window_features) >= laya_window_router.MIN_WINDOW_PACKETS
        ):
            try:
                evidence = laya_window_router.causal_window_evidence(
                    window_features, rows, history_scope
                )
                laya_attempt = self.laya_router.route(
                    evidence, set(self.models)
                )
                if laya_attempt.get("state") == "active":
                    selected = laya_attempt.get("selected_expert")
                    if not isinstance(selected, str):
                        raise ValueError("active Laya route has no selected expert")
                    return selected, laya_attempt
            except Exception as error:  # noqa: BLE001 - optional route is quarantined
                laya_attempt = {
                    "state": "error",
                    "detail": str(error),
                    "execution_ms": _elapsed_ms(started_ns),
                }
        if (
            self.router is None
            or self.router_manifest is None
            or len(self.history) < 129
        ):
            result: dict[str, object] = {
                "state": "unavailable_or_warming_up",
                "execution_ms": _elapsed_ms(started_ns),
            }
            if laya_attempt is not None:
                result["laya_attempt"] = laya_attempt
            return None, result
        context = self.history[-129:-1].unsqueeze(0)
        router_device = next(self.router.parameters()).device
        probabilities = None
        with torch.inference_mode():
            try:
                context_input = self._stage_to_device(context, router_device)
                probabilities = (
                    self.router(causal_router_features(context_input))
                    / float(self.router_manifest["temperature"])
                ).softmax(1)[0]
                if router_device.type == "cuda":
                    torch.cuda.synchronize()
                probabilities = probabilities.cpu()
            except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as error:
                if router_device.type == "cuda" or self.device.type == "cuda":
                    self._fallback_to_cpu(f"router cuda failure: {error}")
                try:
                    probabilities = (
                        self.router(causal_router_features(context))
                        / float(self.router_manifest["temperature"])
                    ).softmax(1)[0].cpu()
                except Exception:  # noqa: BLE001
                    probabilities = None

        fallback = str(self.router_manifest["fallback_expert"])
        threshold = float(self.router_manifest["confidence_threshold"])
        expert_names = list(self.router_manifest["experts"])
        available = [name for name in expert_names if name not in self.user_disabled_models]
        if not available:
            return None, {"state": "unavailable", "reason": "no enabled router experts"}
        if fallback in self.user_disabled_models:
            fallback = available[0]
        if probabilities is not None:
            probabilities = probabilities.clone()
            for index, name in enumerate(expert_names):
                if name in self.user_disabled_models:
                    probabilities[index] = 0
            probabilities /= probabilities.sum().clamp_min(1e-12)

        if probabilities is None or not expert_names:
            result = {
                "state": "fallback_route_due_to_error",
                "selected_expert": fallback,
                "confidence": 0.0,
                "confidence_threshold": threshold,
                "used_fallback": True,
                "security_router": False,
                "task": "next_packet_forecast_compute_routing",
                "expert_weights": {fallback: 1.0},
                "execution_ms": _elapsed_ms(started_ns),
            }
            if laya_attempt is not None:
                result["laya_attempt"] = laya_attempt
            return fallback, result

        selected = int(probabilities.argmax())
        confidence = float(probabilities[selected])
        name = fallback if confidence < threshold else expert_names[selected]
        result = {
            "state": "active",
            "selected_expert": name,
            "raw_selected_expert": expert_names[selected],
            "confidence": confidence,
            "confidence_threshold": threshold,
            "used_fallback": confidence < threshold,
            "security_router": False,
            "task": "next_packet_forecast_compute_routing",
            "expert_weights": {
                expert: float(probabilities[index])
                for index, expert in enumerate(expert_names)
            },
            "execution_ms": _elapsed_ms(started_ns),
        }
        if laya_attempt is not None:
            result["laya_attempt"] = laya_attempt
        return name, result

    def _learning_status_without_collection(self) -> dict[str, object] | None:
        if self.daily_learning is not None:
            return self.daily_learning.status()
        return None

    def infer(
        self,
        features_csv: Path,
        history_scope: str,
        *,
        predict: bool = True,
        learning_segment_key: str | None = None,
        micro_batch_size: int | None = None,
        boost_mode: bool = False,
    ) -> dict[str, object]:
        request_started_ns = perf_counter_ns()
        eff_micro_batch_size = max(1, micro_batch_size or self.micro_batch_size)
        if (
            not history_scope
            or len(history_scope) > 256
            or any(character.isspace() for character in history_scope)
        ):
            raise ValueError("history_scope must be a non-empty opaque identifier")
        history_reset = history_scope != self.history_scope
        history_reset_reason = "scope_changed" if history_reset else None
        if history_reset:
            self.history = torch.empty((0, 12), dtype=torch.float32)
            self.last_timestamp_ns = None
            self.history_scope = history_scope
            if self.ahem_v31_memory is not None:
                self.ahem_v31_memory.reset()
            for memory in self.ahem_v32_memories.values():
                memory.reset()
        adapter_started_ns = perf_counter_ns()
        features, rows = adapt_runtime_rows(features_csv, self.last_timestamp_ns)
        v6_features = adapt_runtime_rows_v6(features, rows)
        flow_identity = runtime_flow_identity_status(rows)
        adapter_ms = _elapsed_ms(adapter_started_ns)
        if rows:
            first_timestamp_ns = int(rows[0]["monotonic_timestamp_ns"])
            if (
                self.last_timestamp_ns is not None
                and first_timestamp_ns < self.last_timestamp_ns
            ):
                self.history = torch.empty((0, 12), dtype=torch.float32)
                adapter_started_ns = perf_counter_ns()
                features, rows = adapt_runtime_rows(features_csv)
                v6_features = adapt_runtime_rows_v6(features, rows)
                adapter_ms += _elapsed_ms(adapter_started_ns)
                history_reset = True
                history_reset_reason = "clock_rollback"
                if self.ahem_v31_memory is not None:
                    self.ahem_v31_memory.reset()
                for memory in self.ahem_v32_memories.values():
                    memory.reset()
            learning_started_ns = perf_counter_ns()
            if learning_segment_key is not None:
                if self.daily_learning is None:
                    self.daily_learning = DailyReplayJournal(self.learning_directory)
                learning_status = self.daily_learning.append(learning_segment_key, features, rows)
            else:
                learning_status = self._learning_status_without_collection()
            learning_ms = _elapsed_ms(learning_started_ns)
            memory_started_ns = perf_counter_ns()
            prior_count = len(self.history)
            self.history = torch.cat((self.history, features))[-MAX_HISTORY_PACKETS:]
            if self.ahem_v31_memory is not None:
                self.ahem_v31_memory.append(features)
            for memory in self.ahem_v32_memories.values():
                memory.append(features)
            memory_update_ms = _elapsed_ms(memory_started_ns)
            self.last_timestamp_ns = int(rows[-1]["monotonic_timestamp_ns"])
        else:
            learning_started_ns = perf_counter_ns()
            learning_status = self._learning_status_without_collection()
            learning_ms = _elapsed_ms(learning_started_ns)
            memory_update_ms = 0.0

        if len(self.history) <= CONTEXT_PACKETS:
            forecasts = {
                name: {
                    "state": "warming_up",
                    "context_packets": len(self.history),
                    "required_packets": 32,
                }
                for name in self.models
            }
            if self.ahem_e2e_shadow is not None:
                forecasts[AHEM_E2E_MODEL_ID] = self.ahem_e2e_shadow.status("warming_up")
            if self.ahem_v61 is not None:
                forecasts["ahem_v61"] = {
                    "state": "warming_up",
                    "reason": "accumulating_temporal_context",
                    "context_packets": len(self.history),
                    "required_packets": 32,
                }
            if self.ahem_v63_experimental is not None:
                forecasts["ahem_v63_experimental"] = {
                    "state": "warming_up",
                    "experimental_shadow": True,
                    "promotion_eligible": False,
                    "context_packets": len(self.history),
                    "required_packets": 32,
                }
            total_inf_ms = _elapsed_ms(request_started_ns)
            return {
                "mode": "temporal_ensemble_forecast_only",
                "security_verdict": None,
                "calibrated": False,
                "prediction_performed": False,
                "performance_mode": "boost" if boost_mode else "balanced",
                "source_contract": "model_specific_versioned_forecast_contracts",
                "runtime_contract": "argus.causal-flow/v1",
                "execution_provider": self.execution_provider,
                "device": str(self.device),
                "device_name": self.device_name,
                "vram_allocated_mb": None,
                "vram_reserved_mb": None,
                "vram_total_mb": self.vram_total_mb,
                "vram_free_mb": None,
                "history_scope": self.history_scope,
                "flow_identity": flow_identity,
                "history_packets": len(self.history),
                "history_reset": history_reset,
                "history_reset_reason": history_reset_reason,
                "router": {"state": "warming_up"},
                "laya_router": self._laya_status(),
                "consensus": {
                    "state": "warming_up",
                    "reason": "accumulating_temporal_context",
                },
                "ahem_primary": forecasts.get("ahem_v61", {"state": "unavailable"}),
                "models": forecasts,
                "rejected_models": self.rejected,
                "learning": learning_status,
                "completed_at_unix_ms": time_ns() // 1_000_000,
                "latency_breakdown": {
                    "adapter_ms": adapter_ms,
                    "learning_ms": learning_ms,
                    "memory_update_ms": memory_update_ms,
                    "router_ms": 0.0,
                    "total_inference_ms": total_inf_ms,
                    "feature_prep_ms": round(adapter_ms, 2),
                    "causal_memory_ms": round(memory_update_ms, 2),
                    "inference_ms": 0.0,
                    "total_ms": round(total_inf_ms, 2),
                    "micro_batch_size": eff_micro_batch_size,
                },
            }
        forecasts: dict[str, object] = {}
        episode_prepare_ms = 0.0
        route_total_ms = 0.0
        forecast_total_ms = 0.0
        if predict:
            # Keep the reported accelerator fail-closed even when every optional
            # checkpoint is absent or still warming up. Otherwise a dead CUDA
            # context can remain advertised indefinitely because no model reaches
            # the normal staging path that detects the failure.
            if self.device.type == "cuda":
                try:
                    self._stage_to_device(
                        features[-1:], self.device, eff_micro_batch_size
                    )
                except Exception as error:  # noqa: BLE001 - device probe boundary
                    self._fallback_to_cpu(f"cuda liveness probe failed: {error}")
            episode_decision = None
            episode_error = None
            if (
                self.laya_router.config is not None
                and self.laya_router.config.schema == laya_window_router.EPISODE_ROUTER_SCHEMA
            ):
                episode_started_ns = perf_counter_ns()
                if flow_identity["state"] == "invalid":
                    episode_error = f"invalid flow identity: {flow_identity['reason']}"
                else:
                    try:
                        episode_decision = build_runtime_episode_decision(
                            v6_features, rows, history_scope
                        )
                    except Exception as error:  # noqa: BLE001 - optional route is quarantined
                        episode_error = str(error)
                episode_prepare_ms = _elapsed_ms(episode_started_ns)
            route_started_ns = perf_counter_ns()
            selected_expert, router_result = self._route(
                features, rows, history_scope, episode_decision
            )
            route_total_ms = _elapsed_ms(route_started_ns)
            if episode_error is not None:
                router_result["episode_adapter_error"] = episode_error
            episode_route_active = (
                router_result.get("source") == laya_window_router.EPISODE_ROUTER_SOURCE
                and router_result.get("state") == "active"
                and episode_decision is not None
            )
            routed_experts_value = router_result.get("eligible_experts")
            routed_experts = (
                {str(name) for name in routed_experts_value}
                if isinstance(routed_experts_value, list)
                else (
                    set(self.router_manifest["experts"])
                    if self.router_manifest is not None
                    else set()
                )
            )
            selected_experts_value = router_result.get("selected_experts")
            selected_experts = (
                {str(name) for name in selected_experts_value}
                if isinstance(selected_experts_value, list)
                else ({selected_expert} if selected_expert is not None else set())
            )
            forecast_started_ns = perf_counter_ns()
            with torch.inference_mode():
                gpu_window: torch.Tensor | None = None
                gpu_window_len: int = 0
                if self.device.type == "cuda":
                    max_ctx_needed = 0
                    for name, (model, _, requested_context) in self.models.items():
                        if name in self.disabled_models:
                            continue
                        if name == "ahem_v31" or name in self.ahem_v32_memories:
                            continue
                        if (
                            not boost_mode
                            and selected_experts
                            and name in routed_experts
                            and name not in selected_experts
                        ):
                            continue
                        try:
                            if next(model.parameters()).device.type == "cuda":
                                ctx = min(requested_context, len(self.history) - 1)
                                if ctx > max_ctx_needed:
                                    max_ctx_needed = ctx
                        except Exception:  # noqa: BLE001
                            pass
                    if max_ctx_needed > 0:
                        try:
                            raw_window = self.history[-max_ctx_needed - 1 : -1]
                            gpu_window = self._stage_to_device(
                                raw_window, self.device, eff_micro_batch_size
                            )
                            gpu_window_len = max_ctx_needed
                        except Exception as error:  # noqa: BLE001
                            self._fallback_to_cpu(f"history stage failure: {error}")
                            gpu_window = None

                for name, (model, checkpoint, requested_context) in self.models.items():
                    if name in self.disabled_models:
                        continue
                    if name == "ahem_v31" and self.ahem_v31_memory is not None:
                        model_started_ns = perf_counter_ns()
                        forecast = self.ahem_v31_memory.forecast()
                        forecast["execution_ms"] = _elapsed_ms(model_started_ns)
                        forecast["checkpoint"] = checkpoint.name
                        forecast["checkpoint_sha256"] = self.ahem_v31_memory.version
                        forecast["runtime_backend"] = "pytorch"
                        forecast["execution_provider"] = "PyTorch CPU"
                        forecasts[name] = forecast
                        continue
                    if name in self.ahem_v32_memories:
                        model_started_ns = perf_counter_ns()
                        forecast = self.ahem_v32_memories[name].forecast()
                        forecast["execution_ms"] = _elapsed_ms(model_started_ns)
                        forecast["checkpoint"] = checkpoint.name
                        forecast["checkpoint_sha256"] = self.ahem_v32_memories[
                            name
                        ].version
                        forecast["runtime_backend"] = "pytorch"
                        forecast["execution_provider"] = "PyTorch CPU"
                        forecasts[name] = forecast
                        continue
                    if (
                        not boost_mode
                        and selected_experts
                        and name in routed_experts
                        and name not in selected_experts
                    ):
                        forecasts[name] = {"state": "skipped_by_router"}
                        continue
                    episode_expert = episode_route_active and name in selected_experts
                    context = (
                        31 if name == "cnn_cse" else 128
                    ) if episode_expert else min(requested_context, len(self.history) - 1)
                    if context < (65 if name == "ahem_fixed" else 2):
                        forecasts[name] = {
                            "state": "warming_up",
                            "context_packets": context,
                        }
                        continue
                    target = v6_features[-1] if episode_expert else self.history[-1]
                    model_device = next(model.parameters()).device
                    if episode_expert:
                        assert episode_decision is not None
                        context_source = v6_features[
                            list(episode_decision.context_indices[-context:])
                        ].unsqueeze(0)
                        context_input = self._stage_to_device(
                            context_source, model_device, eff_micro_batch_size
                        )
                    elif (
                        self.device.type == "cuda"
                        and model_device.type == "cuda"
                        and gpu_window is not None
                        and gpu_window.is_cuda
                        and gpu_window_len >= context
                    ):
                        context_input = gpu_window[-context:].unsqueeze(0)
                    else:
                        context_input = self._stage_to_device(
                            self.history[-context - 1 : -1].unsqueeze(0),
                            model_device,
                            eff_micro_batch_size,
                        )
                    model_started_ns = perf_counter_ns()
                    try:
                        prediction = model(context_input)[0]
                        if model_device.type == "cuda":
                            torch.cuda.synchronize()
                        prediction = prediction.cpu()
                    except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as error:
                        if model_device.type == "cuda" or self.device.type == "cuda":
                            self._fallback_to_cpu(f"model {name} cuda failure: {error}")
                            model_device = torch.device("cpu")
                            gpu_window = None
                        try:
                            retry_context = (
                                context_source if episode_expert
                                else self.history[-context - 1 : -1].unsqueeze(0)
                            )
                            prediction = model(retry_context)[0].cpu()
                        except Exception as cpu_error:  # noqa: BLE001
                            forecasts[name] = {
                                "state": "error",
                                "error": str(cpu_error),
                                "execution_ms": _elapsed_ms(model_started_ns),
                            }
                            continue
                    execution_ms = _elapsed_ms(model_started_ns)
                    if episode_expert:
                        field_weights = torch.tensor([1.0, 4.0] + [0.5] * 10)
                        error = float((
                            torch.nn.functional.smooth_l1_loss(
                                prediction, target, reduction="none"
                            ) * field_weights
                        ).sum() / field_weights.sum())
                    else:
                        error = torch.nn.functional.smooth_l1_loss(
                            prediction, target
                        ).item()
                    forecasts[name] = {
                        "state": "forecast_only",
                        "checkpoint": checkpoint.name,
                        "checkpoint_sha256": _sha256(checkpoint),
                        "context_packets": context,
                        "forecast_error": error,
                        **({
                            "predicted_features": prediction.tolist(),
                            "observed_features": target.tolist(),
                        } if episode_expert else {}),
                        "contract": (
                            "packet-causal-flow-v2/next_same_flow_packet"
                            if episode_expert else "packet-causal-v1/next_global_packet"
                        ),
                        "predicted_log_packet_length": float(prediction[0]),
                        "observed_log_packet_length": float(target[0]),
                        "predicted_log_interarrival": float(prediction[1]),
                        "observed_log_interarrival": float(target[1]),
                        "execution_ms": execution_ms,
                        "runtime_backend": "pytorch",
                        "execution_provider": (
                            "PyTorch CUDA"
                            if model_device.type == "cuda"
                            else "PyTorch CPU"
                        ),
                    }
            if self.ahem_v61 is not None:
                model, artifact_hash, source_hash = self.ahem_v61
                model_started_ns = perf_counter_ns()
                use_onnx = self.ahem_v61_onnx is not None and not boost_mode
                desired_device = torch.device("cpu") if use_onnx else self.device
                model_device = next(model.parameters()).device
                try:
                    if model_device != desired_device:
                        model.to(desired_device)
                        model.eval()
                        model_device = desired_device
                    v6_device_features = (
                        self._stage_to_device(
                            v6_features, model_device, eff_micro_batch_size
                        )
                        if model_device.type == "cuda"
                        else v6_features
                    )
                    forecast = AhemV61CaptureMemory(
                        model,
                        artifact_hash,
                        source_hash,
                        onnx_session=self.ahem_v61_onnx[0] if use_onnx else None,
                        onnx_artifact_hash=self.ahem_v61_onnx[1] if use_onnx else None,
                    ).run(v6_device_features, rows)
                except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as error:
                    if use_onnx:
                        self.rejected["ahem_v61_onnx"] = f"runtime failure: {error}"
                        self.ahem_v61_onnx = None
                        forecast_fallback_reason = str(error)
                    else:
                        forecast_fallback_reason = None
                    if not use_onnx and model_device.type == "cuda":
                        self._fallback_to_cpu(f"ahem_v61 cuda failure: {error}")
                    model_device = torch.device("cpu")
                    try:
                        forecast = AhemV61CaptureMemory(
                            model, artifact_hash, source_hash
                        ).run(v6_features, rows)
                        if forecast_fallback_reason is not None:
                            forecast["onnx_fallback_reason"] = forecast_fallback_reason
                    except Exception as cpu_error:  # noqa: BLE001
                        forecast = {
                            "state": "error",
                            "error": str(cpu_error),
                        }
                forecast["execution_ms"] = _elapsed_ms(model_started_ns)
                forecast["runtime_backend"] = (
                    "pytorch+onnxruntime"
                    if use_onnx and self.ahem_v61_onnx is not None
                    else "pytorch"
                )
                forecast["execution_provider"] = (
                    "PyTorch CPU projection + ONNX Runtime CPU"
                    if forecast["runtime_backend"] == "pytorch+onnxruntime"
                    else "PyTorch CUDA" if model_device.type == "cuda" else "PyTorch CPU"
                )
                forecasts["ahem_v61"] = forecast
            if self.ahem_v63_experimental is not None:
                model, artifact_hash, selected_seed = self.ahem_v63_experimental
                model_started_ns = perf_counter_ns()
                model_device = next(model.parameters()).device
                try:
                    v63_features = (
                        self._stage_to_device(
                            v6_features, model_device, eff_micro_batch_size
                        )
                        if model_device.type == "cuda"
                        else v6_features
                    )
                    forecast = AhemV63CaptureShadow(
                        model, artifact_hash, selected_seed
                    ).run(v63_features, rows)
                    if model_device.type == "cuda":
                        torch.cuda.synchronize()
                except Exception as error:  # noqa: BLE001 - shadow is optional
                    if model_device.type == "cuda" or self.device.type == "cuda":
                        self._fallback_to_cpu(f"ahem_v63 shadow cuda failure: {error}")
                    model_device = torch.device("cpu")
                    try:
                        forecast = AhemV63CaptureShadow(
                            model, artifact_hash, selected_seed
                        ).run(v6_features, rows)
                    except Exception as cpu_error:  # noqa: BLE001
                        forecast = {
                            "state": "error",
                            "experimental_shadow": True,
                            "promotion_eligible": False,
                            "security_verdict": None,
                            "error": str(cpu_error),
                        }
                forecast["execution_ms"] = _elapsed_ms(model_started_ns)
                forecast["runtime_backend"] = "pytorch"
                forecast["execution_provider"] = (
                    "PyTorch CUDA"
                    if model_device.type == "cuda"
                    else "PyTorch CPU"
                )
                forecasts["ahem_v63_experimental"] = forecast
            if self.ahem_e2e_shadow is not None:
                shadow_started = perf_counter_ns()
                try:
                    shadow_result = self.ahem_e2e_shadow.run(v6_features, rows)
                except Exception as error:  # noqa: BLE001 - no effect on admitted forecasts
                    shadow_result = self.ahem_e2e_shadow.status("error", error=str(error))
                shadow_result["execution_ms"] = _elapsed_ms(shadow_started)
                forecasts[AHEM_E2E_MODEL_ID] = shadow_result
            forecast_total_ms = _elapsed_ms(forecast_started_ns)
        else:
            selected_expert = None
            router_result = {"state": "deferred"}
            forecasts = {
                name: {
                    "state": "history_updated",
                    "context_packets": min(context, len(self.history)),
                }
                for name, (_, _, context) in self.models.items()
            }
            if self.ahem_e2e_shadow is not None:
                forecasts[AHEM_E2E_MODEL_ID] = self.ahem_e2e_shadow.status("history_updated")
            if self.ahem_v61 is not None:
                forecasts["ahem_v61"] = {
                    "state": "history_updated",
                    "context_packets": min(
                        len(v6_features), AHEM_V61_CONTEXT_PACKETS
                    ),
                    "cache_scope": "capture_or_segment_local",
                    "security_detector": False,
                }
            if self.ahem_v63_experimental is not None:
                forecasts["ahem_v63_experimental"] = {
                    "state": "history_updated",
                    "context_packets": min(
                        len(v6_features), AHEM_V61_CONTEXT_PACKETS
                    ),
                    "experimental_shadow": True,
                    "promotion_eligible": False,
                    "security_detector": False,
                }
        completed_at_unix_ms = time_ns() // 1_000_000
        vram_allocated_mb = None
        vram_reserved_mb = None
        vram_free_mb = None
        if self.device.type == "cuda":
            try:
                vram_allocated_mb = (
                    torch.cuda.memory_allocated(self.device) / (1024 * 1024)
                )
                vram_reserved_mb = (
                    torch.cuda.memory_reserved(self.device) / (1024 * 1024)
                )
                vram_free_mb = (
                    torch.cuda.mem_get_info(self.device)[0] / (1024 * 1024)
                )
            except Exception:  # noqa: BLE001
                self._fallback_to_cpu("cuda query failed")
        total_inf_ms = _elapsed_ms(request_started_ns)
        models_inf_ms = sum(
            float(f.get("execution_ms", 0.0))
            for f in forecasts.values()
            if isinstance(f, dict) and "execution_ms" in f
        )
        latency_breakdown = {
            "adapter_ms": adapter_ms,
            "learning_ms": learning_ms,
            "memory_update_ms": memory_update_ms,
            "episode_prepare_ms": episode_prepare_ms,
            "route_total_ms": route_total_ms,
            "router_ms": router_result.get("execution_ms", 0.0),
            "forecast_total_ms": forecast_total_ms,
            "total_inference_ms": total_inf_ms,
            "feature_prep_ms": round(adapter_ms, 2),
            "causal_memory_ms": round(memory_update_ms, 2),
            "inference_ms": round(models_inf_ms, 2),
            "total_ms": round(total_inf_ms, 2),
            "micro_batch_size": eff_micro_batch_size,
        }
        consensus = evidence_weighted_consensus(forecasts, router_result)
        episode_consensus = (
            consensus
            if router_result.get("source") == laya_window_router.EPISODE_ROUTER_SOURCE
            else {"state": "unavailable", "reason": "episode router did not admit a route"}
        )
        legacy_consensus = (
            {"state": "not_applicable", "reason": "episode route uses a same-flow target"}
            if router_result.get("source") == laya_window_router.EPISODE_ROUTER_SOURCE
            else consensus
        )
        ahem_primary = forecasts.get("ahem_v61", {"state": "unavailable"})
        return {
            "mode": "temporal_ensemble_forecast_only",
            "security_verdict": None,
            "calibrated": False,
            "prediction_performed": predict,
            "performance_mode": "boost" if boost_mode else "balanced",
            "source_contract": "model_specific_versioned_forecast_contracts",
            "runtime_contract": "argus.causal-flow/v1",
            "execution_provider": self.execution_provider,
            "device": str(self.device),
            "device_name": self.device_name,
            "vram_allocated_mb": vram_allocated_mb,
            "vram_reserved_mb": vram_reserved_mb,
            "vram_total_mb": self.vram_total_mb,
            "vram_free_mb": vram_free_mb,
            "latency_breakdown": latency_breakdown,
            "model_contracts": {
                "legacy_temporal_experts": "packet-causal-v1/next_packet_only",
                "ahem_v61": (
                    "packet-causal-flow-v2/next_same_flow_dynamic_event_forecast_only"
                ),
                "ahem_v63_experimental": (
                    "packet-causal-flow-v2/next_same_flow_dynamic_event_forecast_only; "
                    "shadow_only_not_promotion_eligible"
                ),
            },
            "adapter": {
                "id": "argus.experimental.packet-causal-adapter/v1",
                "exact_or_reconstructible_fields": 12,
                "total_fields": 12,
                "missing_fields": [],
            },
            "history_packets": len(self.history),
            "history_scope": history_scope,
            "flow_identity": flow_identity,
            "history_reset": history_reset,
            "history_reset_reason": history_reset_reason,
            "new_packets": len(features),
            "prior_history_packets": prior_count,
            "latest_packet": {
                "source_capture_ordinal": int(rows[-1]["source_capture_ordinal"]),
                "opaque_flow_id": int(rows[-1]["opaque_flow_id"]),
            },
            "models": forecasts,
            "forecast_fusion": {
                "legacy_consensus": legacy_consensus,
                "episode_consensus": episode_consensus,
                "primary_research_model": "ahem_v61",
                "primary_research_state": (
                    ahem_primary.get("state", "unavailable")
                    if isinstance(ahem_primary, dict)
                    else "unavailable"
                ),
                "policy": "AHEM v6.1 is primary within its same-flow contract; Argus CERO combines only selected v2 same-flow experts",
                "cross_contract_blending": False,
            },
            "rejected_models": self.rejected,
            "router": router_result,
            "laya_router": self._laya_status(),
            "daily_learning": learning_status,
            "telemetry": {
                "adapter_ms": adapter_ms,
                "learning_ms": learning_ms,
                "memory_update_ms": memory_update_ms,
                "episode_prepare_ms": episode_prepare_ms,
                "route_total_ms": route_total_ms,
                "router_ms": router_result.get("execution_ms", 0.0),
                "forecast_total_ms": forecast_total_ms,
                "total_inference_ms": total_inf_ms,
                "completed_at_unix_ms": completed_at_unix_ms,
                "executed_models": sum(
                    forecast.get("state") == "forecast_only"
                    for forecast in forecasts.values()
                ),
                "execution_provider": self.execution_provider,
                "device": str(self.device),
                "device_name": self.device_name,
                "micro_batch_size": eff_micro_batch_size,
                "vram_allocated_mb": vram_allocated_mb,
                "vram_reserved_mb": vram_reserved_mb,
                "vram_total_mb": self.vram_total_mb,
                "vram_free_mb": vram_free_mb,
            },
        }


def run_loaded(
    model: CausalCnnNextPacket,
    checkpoint: Path,
    features_csv: Path,
    batch_size: int = 256,
) -> dict[str, object]:
    features, rows = adapt_runtime_rows(features_csv)
    errors: list[torch.Tensor] = []
    latest_prediction: torch.Tensor | None = None
    latest_target: torch.Tensor | None = None
    with torch.inference_mode():
        for start in range(CONTEXT_PACKETS, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            contexts = torch.stack(
                [
                    features[index - CONTEXT_PACKETS : index]
                    for index in range(start, stop)
                ]
            )
            targets = features[start:stop]
            predictions = model(contexts)
            latest_prediction = predictions[-1].cpu()
            latest_target = targets[-1].cpu()
            errors.append(
                torch.nn.functional.smooth_l1_loss(
                    predictions, targets, reduction="none"
                ).mean(1)
            )
    packet_errors = torch.cat(errors)
    if latest_prediction is None or latest_target is None:
        raise ValueError("CNN produced no predictions")
    top_count = min(10, len(packet_errors))
    top_values, top_indices = torch.topk(packet_errors, top_count)
    top = []
    for error, relative_index in zip(
        top_values.tolist(), top_indices.tolist(), strict=True
    ):
        row = rows[relative_index + CONTEXT_PACKETS]
        top.append(
            {
                "source_capture_ordinal": int(row["source_capture_ordinal"]),
                "opaque_flow_id": int(row["opaque_flow_id"]),
                "forecast_error": error,
            }
        )
    return {
        "mode": "experimental_forecast_only",
        "security_verdict": None,
        "calibrated": False,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
        "source_contract": "packet-causal-v1/next_packet_only",
        "runtime_contract": "argus.causal-flow/v1",
        "adapter": {
            "id": "argus.experimental.packet-causal-adapter/v1",
            "exact_or_reconstructible_fields": 12,
            "total_fields": 12,
            "missing_fields": [],
        },
        "context_packets": CONTEXT_PACKETS,
        "evaluated_predictions": len(packet_errors),
        "mean_forecast_error": float(packet_errors.mean()),
        "p95_forecast_error": float(torch.quantile(packet_errors, 0.95)),
        "latest_forecast": {
            "source_capture_ordinal": int(rows[-1]["source_capture_ordinal"]),
            "opaque_flow_id": int(rows[-1]["opaque_flow_id"]),
            "predicted_log_packet_length": float(latest_prediction[0]),
            "observed_log_packet_length": float(latest_target[0]),
            "predicted_log_interarrival": float(latest_prediction[1]),
            "observed_log_interarrival": float(latest_target[1]),
        },
        "top_forecast_errors": top,
    }


def run(
    checkpoint: Path, features_csv: Path, batch_size: int = 256
) -> dict[str, object]:
    return run_loaded(load_checkpoint(checkpoint), checkpoint, features_csv, batch_size)


def serve(checkpoint: Path, batch_size: int) -> None:
    model = load_checkpoint(checkpoint)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = run_loaded(
                model, checkpoint, Path(request["features"]), batch_size
            )
        except Exception as error:  # noqa: BLE001 - protocol returns an explicit error envelope
            response = {"error": str(error)}
        print(json.dumps(response), flush=True)


def serve_ensemble(
    model_dir: Path,
    device: str = "auto",
    disabled_models: frozenset[str] = frozenset(),
    micro_batch_size: int = 32,
    learning_directory: Path | None = None,
    user_disabled_models: frozenset[str] = frozenset(),
    enable_ahem_e2e_shadow: bool = False,
) -> None:
    # The optional router loads on a background thread and its dependency
    # redirects sys.stdout while restoring tokenizer metadata. Keep the JSONL
    # protocol bound to the original stream so a status reply is never lost.
    protocol_stdout = sys.stdout
    # Tiny batch-one CPU operators are faster and less disruptive to the UI
    # without a large OpenMP worker pool competing for the desktop scheduler.
    if device == "cpu":
        torch.set_num_threads(_torch_cpu_threads())
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    ensemble = TemporalEnsemble(
        model_dir,
        device=device,
        disabled_models=disabled_models,
        micro_batch_size=micro_batch_size,
        learning_directory=learning_directory,
        user_disabled_models=user_disabled_models,
        enable_ahem_e2e_shadow=enable_ahem_e2e_shadow,
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            action = request.get("action")
            if action == "recover_device" or request.get("recover_device"):
                recovered = ensemble.recover_device()
                response = {
                    "action": "recover_device",
                    "recovered": recovered,
                    "device": str(ensemble.device),
                    "device_name": ensemble.device_name,
                    "execution_provider": ensemble.execution_provider,
                }
            elif action == "device_status":
                response = {
                    "device": str(ensemble.device),
                    "device_name": ensemble.device_name,
                    "execution_provider": ensemble.execution_provider,
                    "vram_total_mb": ensemble.vram_total_mb,
                    "vram_free_mb": ensemble.vram_free_mb,
                    "torch_cpu_threads": torch.get_num_threads(),
                    "episode_router": ensemble._laya_status(),
                }
            else:
                response = ensemble.infer(
                    Path(request["features"]),
                    str(request["history_scope"]),
                    predict=bool(request.get("predict", True)),
                    learning_segment_key=request.get("learning_segment_key"),
                    micro_batch_size=request.get("micro_batch_size"),
                    boost_mode=bool(request.get("boost_mode", False)),
                )
        except Exception as error:  # noqa: BLE001 - protocol returns an explicit error envelope
            response = {"error": str(error)}
        print(json.dumps(response), file=protocol_stdout, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--ensemble-server", type=Path)
    parser.add_argument(
        "--learning-directory", type=Path,
        help="Absolute writable learning directory supplied by the desktop data-root policy",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--skip-model",
        action="append",
        default=[],
        help="Model delegated to another admitted runtime; may be repeated",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=32,
        help="Host-to-device transfer micro-batch size (default: 32)",
    )
    parser.add_argument("--disable-model", action="append", default=[],
                        help="User-disabled model; never load, execute, or route to it")
    parser.add_argument("--enable-ahem-e2e-shadow", action="store_true",
                        help="Explicitly opt into incomplete recovered AHEM research shadow (CPU only)")
    args = parser.parse_args()
    if args.ensemble_server is not None:
        serve_ensemble(
            args.ensemble_server,
            device=args.device,
            disabled_models=frozenset(args.skip_model),
            micro_batch_size=args.micro_batch_size,
            learning_directory=args.learning_directory,
            user_disabled_models=frozenset(args.disable_model),
            enable_ahem_e2e_shadow=args.enable_ahem_e2e_shadow,
        )
    elif args.server:
        if args.checkpoint is None:
            parser.error("--checkpoint is required with --server")
        serve(args.checkpoint, args.batch_size)
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for one-shot inference")
        if args.features is None:
            parser.error("--features is required unless --server is used")
        print(json.dumps(run(args.checkpoint, args.features, args.batch_size)))


if __name__ == "__main__":
    main()

"""Reproducible AHEM-v3.4 cache-native versus LSTM runtime benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from torch import Tensor

from .ahem_pretrain import LstmNextPacket
from .ahem_v3 import AhemV34FlowIndexedNextPacket
from .flow_v2_protocol import write_manifest
from .flow_v3_protocol import PREREGISTRATION_SHA256
from .flow_v3_smoke import CANDIDATE, SMOKE_SCHEMA
from .flow_v3_train import CHECKPOINT_SCHEMA, RUN_SCHEMA

BENCHMARK_SCHEMA = "argus.flow-v3-cache-runtime/v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    function: Callable[[], Tensor],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        _synchronize(device)
        latencies = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            function()
            _synchronize(device)
            latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return {
        "p50_ms": _percentile(latencies, 0.50),
        "p95_ms": _percentile(latencies, 0.95),
        "p99_ms": _percentile(latencies, 0.99),
        "mean_ms": fmean(latencies),
        "samples_ms": latencies,
    }


def _checkpoint(path: Path, expected_model: str) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema") not in {
        SMOKE_SCHEMA,
        "argus.flow-conditioned-global-smoke/v2",
        CHECKPOINT_SCHEMA,
    }:
        raise ValueError(f"{path.name} is not a compatible flow-v3 checkpoint")
    if value.get("model") != expected_model:
        raise ValueError(f"{path.name} is not the expected {expected_model} model")
    if not isinstance(value.get("state_dict"), dict):
        raise TypeError(f"{path.name} has no state dictionary")
    return value


def _run_manifest(path: Path, checkpoint: Path, expected_model: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != RUN_SCHEMA:
        raise ValueError(f"{path.name} is not a flow-v3 run manifest")
    if value.get("model", {}).get("name") != expected_model:
        raise ValueError(f"{path.name} belongs to another model")
    access = value.get("test_access", {})
    if access.get("status") != "completed":
        raise ValueError(f"{path.name} has no completed test-access ledger")
    if access.get("best_checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError(f"{path.name} does not bind the supplied checkpoint")
    return value


def _structural_context(length: int, device: torch.device) -> Tensor:
    """Build value-independent benchmark input with a realistic sparse flow lane."""
    if length < 128:
        raise ValueError("runtime context lengths must be at least 128 packets")
    generator = torch.Generator(device="cpu").manual_seed(20260829 + length)
    packets = torch.randn((1, length, 13), generator=generator)
    packets[..., 1] = packets[..., 1].abs()
    packets[..., -1] = 0.0
    positions = torch.linspace(0, length - 1, 17).round().to(torch.int64).unique()
    packets[:, positions, -1] = 1.0
    packets[:, -1, -1] = 1.0
    return packets.to(device)


def _indexed_inputs(
    context: Tensor, model: AhemV34FlowIndexedNextPacket
) -> tuple[Tensor, ...]:
    length = context.size(1)
    positions = torch.arange(length, device=context.device)[None]
    selected = (
        positions.masked_fill(context[..., -1] <= 0.5, -1)
        .topk(min(length, model.max_flow_packets), dim=1)
        .values
    )
    lane = context.gather(
        1, selected.clamp_min(0)[..., None].expand(-1, -1, model.feature_dim)
    )[..., : model.packet_dim]
    return (
        context[:, -model.local_packets :],
        lane,
        selected,
        selected >= 0,
        context.new_full((context.size(0),), float(length)),
    )


def _cuda_peak(function: Callable[[], Tensor], device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        function()
    _synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def run(
    ahem_checkpoint: Path,
    lstm_checkpoint: Path,
    output: Path,
    *,
    lengths: list[int],
    device: torch.device,
    warmup: int,
    iterations: int,
    threads: int = 1,
    ahem_run: Path | None = None,
    lstm_run: Path | None = None,
) -> dict[str, Any]:
    if min(warmup, iterations, threads) <= 0:
        raise ValueError("warmup and iterations must be positive")
    if not lengths:
        raise ValueError("at least one context length is required")
    torch.set_num_threads(threads)
    ahem_state = _checkpoint(ahem_checkpoint, CANDIDATE)
    lstm_state = _checkpoint(lstm_checkpoint, "lstm")
    if (ahem_run is None) != (lstm_run is None):
        raise ValueError("both run manifests are required for confirmatory timing")
    run_manifests = None
    if ahem_run is not None and lstm_run is not None:
        run_manifests = {
            "ahem": _run_manifest(ahem_run, ahem_checkpoint, CANDIDATE),
            "lstm": _run_manifest(lstm_run, lstm_checkpoint, "lstm"),
        }
        if (
            run_manifests["ahem"]["training"]["seed"]
            != run_manifests["lstm"]["training"]["seed"]
        ):
            raise ValueError("runtime comparison mixes training seeds")
        if (
            run_manifests["ahem"]["protocol_sha256"]
            != run_manifests["lstm"]["protocol_sha256"]
        ):
            raise ValueError("runtime comparison mixes protocols")
    ahem_coordinates = ahem_state.get("coordinate_sha256") or ahem_state.get(
        "protocol", {}
    ).get("coordinate_sha256")
    lstm_coordinates = lstm_state.get("coordinate_sha256") or lstm_state.get(
        "protocol", {}
    ).get("coordinate_sha256")
    if ahem_coordinates != lstm_coordinates:
        raise ValueError(
            "AHEM and LSTM checkpoints use different evaluation coordinates"
        )
    ahem = AhemV34FlowIndexedNextPacket().to(device).eval()
    lstm = LstmNextPacket(13, output_dim=2).to(device).eval()
    ahem.load_state_dict(ahem_state["state_dict"], strict=True)
    lstm.load_state_dict(lstm_state["state_dict"], strict=True)

    measurements: dict[str, Any] = {}
    for length in lengths:
        context = _structural_context(length, device)
        indexed = _indexed_inputs(context, ahem)
        with torch.inference_mode():
            full_prediction = ahem(context)
            indexed_prediction = ahem.forward_from_indexed_lane(*indexed).prediction
            if length > 1:
                _, recurrent_state = lstm.encoder(context[:, :-1])
            else:
                recurrent_state = None
        equivalence_error = float((full_prediction - indexed_prediction).abs().max())
        if equivalence_error > 1e-6:
            raise RuntimeError("cache-native inference differs from full-context AHEM")

        def cached_call(indexed=indexed):
            return ahem.forward_from_indexed_lane(*indexed).prediction

        def full_call(context=context):
            return ahem(context)

        def lstm_call(context=context, recurrent_state=recurrent_state):
            return lstm.forward_step(context[:, -1], recurrent_state)[0]

        cached = _measure(
            cached_call, device=device, warmup=warmup, iterations=iterations
        )
        full = _measure(full_call, device=device, warmup=warmup, iterations=iterations)
        recurrent = _measure(
            lstm_call, device=device, warmup=warmup, iterations=iterations
        )
        measurements[str(length)] = {
            "ahem_cache_native": cached,
            "ahem_full_context_reference": full,
            "lstm_stateful_step": recurrent,
            "cache_native_equivalence_max_abs_error": equivalence_error,
            "lstm_p50_divided_by_ahem_cache_p50": (
                recurrent["p50_ms"] / cached["p50_ms"]
            ),
            "input_contract_bytes": {
                "ahem_local_ring": int(indexed[0].numel() * indexed[0].element_size()),
                "ahem_flow_lane": int(indexed[1].numel() * indexed[1].element_size()),
                "lstm_packet": int(context[:, -1].numel() * context.element_size()),
                "lstm_recurrent_state": (
                    0
                    if recurrent_state is None
                    else int(
                        sum(
                            value.numel() * value.element_size()
                            for value in recurrent_state
                        )
                    )
                ),
            },
            "cuda_peak_allocated_bytes": {
                "ahem_cache_native": _cuda_peak(cached_call, device),
                "lstm_stateful_step": _cuda_peak(lstm_call, device),
            },
        }
    longest = measurements[str(max(lengths))]
    result = {
        "schema": BENCHMARK_SCHEMA,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "created_utc": datetime.now(UTC).isoformat(),
        "study_role": (
            "confirmatory"
            if run_manifests
            and {value["study_role"] for value in run_manifests.values()}
            == {"confirmatory"}
            else "exploratory_runtime_screen"
        ),
        "publication_evidence": bool(
            run_manifests
            and all(
                value.get("publication_eligible") for value in run_manifests.values()
            )
        ),
        "task": "model_forward_compute_scaling_only",
        "checkpoints": {
            "ahem": {"path": str(ahem_checkpoint), "sha256": _sha256(ahem_checkpoint)},
            "lstm": {"path": str(lstm_checkpoint), "sha256": _sha256(lstm_checkpoint)},
        },
        "parameters": {
            "ahem": sum(value.numel() for value in ahem.parameters()),
            "lstm": sum(value.numel() for value in lstm.parameters()),
        },
        "training_seed": (
            run_manifests["ahem"]["training"]["seed"] if run_manifests else None
        ),
        "protocol_sha256": (
            run_manifests["ahem"]["protocol_sha256"] if run_manifests else None
        ),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "settings": {
            "warmup": warmup,
            "iterations": iterations,
            "torch_threads": threads,
        },
        "measurements": measurements,
        "longest_context_gate": {
            "context_packets": max(lengths),
            "cache_native_faster_than_stateful_lstm": (
                longest["lstm_p50_divided_by_ahem_cache_p50"] > 1.0
            ),
            "speedup": longest["lstm_p50_divided_by_ahem_cache_p50"],
        },
        "limitations": [
            "inputs are deterministic structural tensors because dense model compute is value-independent",
            "quality is measured separately on real ARGUSLR2 windows",
            "excludes capture, parsing, cache update, queueing, and feature extraction",
            "LSTM timing is one packet with carried hidden/cell state; prefix warm-up is excluded",
            "single-device exploratory screen; independent replication is required",
        ],
    }
    write_manifest(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ahem-checkpoint", type=Path, required=True)
    parser.add_argument("--lstm-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ahem-run", type=Path)
    parser.add_argument("--lstm-run", type=Path)
    parser.add_argument("--lengths", type=int, nargs="+", default=[257, 10_000, 30_000])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    result = run(
        args.ahem_checkpoint,
        args.lstm_checkpoint,
        args.output,
        lengths=args.lengths,
        device=torch.device(args.device),
        warmup=args.warmup,
        iterations=args.iterations,
        threads=args.threads,
        ahem_run=args.ahem_run,
        lstm_run=args.lstm_run,
    )
    print(result["longest_context_gate"])


if __name__ == "__main__":
    main()

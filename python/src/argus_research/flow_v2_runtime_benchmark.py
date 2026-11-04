"""Fresh-process CPU benchmark for a frozen flow-v2 checkpoint."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from torch import Tensor

from .flow_v2_protocol import (
    FEATURE_SCHEMA,
    ProtocolConfig,
    WindowCoordinate,
    compile_protocol,
    materialize_batch,
    write_manifest,
)
from .flow_v2_train import (
    CHECKPOINT_SCHEMA,
    RUN_SCHEMA,
    _canonical_sha256,
    _file_sha256,
    _model_input,
    _source_provenance,
    model_spec,
)

BENCHMARK_SCHEMA = "argus.flow-v2-runtime-benchmark/v1"


def _memory_snapshot() -> dict[str, int | None]:
    if sys.platform == "win32":

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        success = ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        if success:
            return {
                "working_set_bytes": int(counters.working_set_size),
                "peak_working_set_bytes": int(counters.peak_working_set_size),
            }
    peak: int | None = None
    try:
        import resource

        maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak = maximum if sys.platform == "darwin" else maximum * 1024
    except (ImportError, ValueError):
        pass
    current: int | None = None
    statm = Path("/proc/self/statm")
    if statm.is_file():
        fields = statm.read_text(encoding="ascii").split()
        if len(fields) >= 2:
            current = int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    return {"working_set_bytes": current, "peak_working_set_bytes": peak}


def _representative_coordinates(
    coordinates: list[WindowCoordinate], count: int
) -> list[WindowCoordinate]:
    if count <= 0:
        raise ValueError("runtime benchmark requires positive sample count")
    chosen: list[WindowCoordinate] = []
    used_streams: set[int] = set()
    for coordinate in coordinates:
        if coordinate.stream_index not in used_streams:
            chosen.append(coordinate)
            used_streams.add(coordinate.stream_index)
            if len(chosen) == count:
                return chosen
    for coordinate in coordinates:
        if coordinate not in chosen:
            chosen.append(coordinate)
            if len(chosen) == count:
                break
    return chosen


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _state_contract(model: torch.nn.Module, context_packets: int) -> dict[str, Any]:
    parameters = sum(value.numel() for value in model.parameters())
    parameter_bytes = sum(
        value.numel() * value.element_size() for value in model.state_dict().values()
    )
    contract: dict[str, Any] = {
        "parameters": parameters,
        "parameter_and_buffer_bytes": parameter_bytes,
        "input_tensor_bytes_batch1_float32": context_packets * 12 * 4,
    }
    if (
        getattr(model, "uses_transition_memory", False)
        and hasattr(model, "memory_width")
        and hasattr(model, "local_packets")
    ):
        local_packets = int(getattr(model, "local_packets"))
        chunk_size = int(getattr(model, "chunk_size"))
        memory_width = int(getattr(model, "memory_width"))
        chunks = max(0, (context_packets - local_packets - 1) // chunk_size)
        contract["ahem_cache"] = {
            "chunks": chunks,
            "entry_width_float32": memory_width,
            "cache_bytes_float32": chunks * memory_width * 4,
            "local_lane_bytes_float32": local_packets * 12 * 4,
            "allocator_overhead_included": False,
        }
    return contract


def _materialized_samples(
    protocol: Any,
    coordinates: list[WindowCoordinate],
    context_packets: int,
    spec: Any,
) -> list[Tensor]:
    samples = []
    for coordinate in coordinates:
        context, _ = materialize_batch(
            protocol.partitions["test"], [coordinate], context_packets
        )
        samples.append(_model_input(context, spec))
    return samples


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_manifest = json.loads(args.run.read_text(encoding="utf-8"))
    if run_manifest.get("schema") != RUN_SCHEMA:
        raise ValueError("runtime benchmark requires a flow-v2 run manifest")
    benchmark_source = _source_provenance()
    training_source = run_manifest.get("source_provenance", {})
    if (
        training_source.get("available") is True
        and benchmark_source.get("available") is True
        and training_source.get("git_revision") != benchmark_source.get("git_revision")
    ):
        raise ValueError("runtime benchmark source revision differs from training")
    frozen = run_manifest["protocol"]
    config = ProtocolConfig(**frozen["config"])
    protocol = compile_protocol(
        args.journal,
        frozen["split"]["validation_source"],
        frozen["split"]["test_source"],
        config,
    )
    protocol_sha256 = _canonical_sha256(protocol.manifest)
    if protocol_sha256 != run_manifest["protocol_sha256"]:
        raise ValueError("runtime journal no longer matches the frozen protocol")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("runtime benchmark checkpoint has an incompatible schema")
    if checkpoint.get("protocol_sha256") != protocol_sha256:
        raise ValueError("runtime checkpoint belongs to another protocol")
    if checkpoint.get("coordinate_sha256") != protocol.manifest["coordinate_sha256"]:
        raise ValueError("runtime checkpoint uses other evaluation coordinates")
    if checkpoint.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("runtime checkpoint uses another feature schema")
    if checkpoint.get("training_config_sha256") != run_manifest["training"].get(
        "config_sha256"
    ):
        raise ValueError("runtime checkpoint uses another training configuration")
    if _file_sha256(args.checkpoint) != run_manifest.get("test_access", {}).get(
        "best_checkpoint_sha256"
    ):
        raise ValueError("runtime checkpoint bytes differ from confirmatory evidence")
    model_name = run_manifest["model"]["name"]
    if checkpoint.get("model") != model_name:
        raise ValueError("runtime checkpoint belongs to another model")

    spec = model_spec(model_name, config.context_packets)
    before_model = _memory_snapshot()
    model = spec.constructor().cpu().eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    after_model = _memory_snapshot()
    torch.set_num_threads(args.threads)
    coordinates = _representative_coordinates(
        protocol.coordinates["test"],
        min(args.windows, len(protocol.coordinates["test"])),
    )
    samples = _materialized_samples(protocol, coordinates, config.context_packets, spec)
    if not samples:
        raise ValueError("runtime benchmark has no frozen test samples")

    with torch.inference_mode():
        started = time.perf_counter_ns()
        model(samples[0])
        first_forward_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        for sample in samples[: min(3, len(samples))]:
            model(sample)
        latencies = []
        for _ in range(args.rounds):
            for sample in samples:
                started = time.perf_counter_ns()
                model(sample)
                latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
    after_benchmark = _memory_snapshot()
    mean_ms = fmean(latencies)
    result = {
        "schema": BENCHMARK_SCHEMA,
        "task": "causal_per_flow_next_packet_forecast_only",
        "not_intrusion_detection_efficacy": True,
        "protocol_sha256": protocol_sha256,
        "coordinate_sha256": protocol.manifest["coordinate_sha256"],
        "model": model_name,
        "training_seed": run_manifest["training"]["seed"],
        "training_source_provenance": training_source,
        "benchmark_source_provenance": benchmark_source,
        "checkpoint": {
            "path": str(args.checkpoint),
            "epoch": checkpoint["epoch"],
            "best_epoch": checkpoint["best_epoch"],
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": args.threads,
        },
        "sample_contract": {
            "windows": len(samples),
            "rounds": args.rounds,
            "measurements": len(latencies),
            "unique_flow_streams": len(
                {coordinate.stream_index for coordinate in coordinates}
            ),
            "pre_materialized": True,
            "batch_size": 1,
            "context_packets": config.context_packets,
        },
        "latency": {
            "scope": "model_forward_only_pre_materialized_float32_tensor",
            "first_forward_ms": first_forward_ms,
            "warm_p50_ms": _percentile(latencies, 0.50),
            "warm_p95_ms": _percentile(latencies, 0.95),
            "warm_p99_ms": _percentile(latencies, 0.99),
            "warm_mean_ms": mean_ms,
            "predictions_per_second": 1_000.0 / mean_ms,
            "raw_ms": latencies,
        },
        "memory": {
            "process_before_model": before_model,
            "process_after_model": after_model,
            "process_after_benchmark": after_benchmark,
            "process_wide_measurement": True,
            "state_contract": _state_contract(model, spec.input_context),
        },
        "limitations": [
            "excludes packet parsing, queueing, and feature extraction",
            "process RSS includes Python and PyTorch runtime allocations",
            "energy is not measured",
            "streaming recurrent-state latency requires a deployment-specific benchmark",
        ],
    }
    write_manifest(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.windows, args.rounds, args.threads) <= 0:
        parser.error("runtime benchmark counts must be positive")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

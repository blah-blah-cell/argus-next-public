"""Latency-only model benchmark over real ordered PCAP feature windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .ahem_pretrain import (
    AhemNextPacket,
    CausalCnnNextPacket,
    LongTcnNextPacket,
    LstmNextPacket,
)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values), percentile))


def _load(
    model_name: str, checkpoint: Path, feature_dim: int, device: torch.device
) -> torch.nn.Module:
    constructors = {
        "ahem_fixed": AhemNextPacket,
        "lstm": LstmNextPacket,
        "cnn_local": CausalCnnNextPacket,
        "tcn_long": LongTcnNextPacket,
    }
    model = constructors[model_name](feature_dim).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"]
    )
    model.eval()
    return model


def _measure(
    model: torch.nn.Module, window: torch.Tensor, device: torch.device, iterations: int
) -> dict[str, float]:
    with torch.no_grad():
        for _ in range(10):
            model(window)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        values = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            model(window)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            values.append((time.perf_counter_ns() - started) / 1_000_000)
    return {
        "p50_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
        "p99_ms": _percentile(values, 99),
    }


def run(
    source: Path,
    checkpoints: dict[str, Path],
    output: Path,
    device: torch.device,
    iterations: int,
) -> dict[str, object]:
    stream = np.load(source)["features"].astype(np.float32)
    context = 30_000
    if len(stream) < context:
        raise ValueError("real PCAP source lacks a 30k-packet window")
    window = torch.from_numpy(stream[-context:]).unsqueeze(0).to(device)
    results = {}
    for name, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model = _load(name, checkpoint, window.size(-1), device)
        latency = _measure(model, window, device, iterations)
        peak = (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        )
        results[name] = {**latency, "peak_inference_memory_mib": peak}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "benchmark": "real_pcap_30k_latency_only",
        "source": str(source),
        "context_packets": context,
        "iterations": iterations,
        "device": str(device),
        "timing_scope": "single-decision forward pass after warmup; GPU timings synchronized",
        "results": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ahem", type=Path, required=True)
    parser.add_argument("--lstm", type=Path, required=True)
    parser.add_argument("--cnn", type=Path, required=True)
    parser.add_argument("--tcn", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    result = run(
        args.source,
        {
            "ahem_fixed": args.ahem,
            "lstm": args.lstm,
            "cnn_local": args.cnn,
            "tcn_long": args.tcn,
        },
        args.output,
        torch.device(args.device),
        args.iterations,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

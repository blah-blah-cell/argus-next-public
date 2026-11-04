"""Fair fresh-process runtime audit for AHEM-v5 and a stateful LSTM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from torch import Tensor

from .ahem_pretrain import LstmNextPacket
from .ahem_v4 import AhemV5SourceAlignedStreamNextPacket, LocalState
from .flow_v2_protocol import write_manifest
from .flow_v2_train import _source_provenance
from .flow_v3_train import V4_CHECKPOINT_SCHEMA, V4_PREREGISTRATION_SHA256

BENCHMARK_SCHEMA = "argus.ahem-v5-runtime/v2"
# Created at module import, not per call: two calls in one process are not
# independent fresh-process replicates.
PROCESS_INSTANCE_ID = uuid.uuid4().hex


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "mean_ms": fmean(values),
        "samples_ms": values,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_alternating(
    functions: dict[str, Callable[[], Tensor]],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, dict[str, Any]]:
    names = list(functions)
    samples = {name: [] for name in names}
    with torch.inference_mode():
        for _ in range(warmup):
            for function in functions.values():
                function()
        _synchronize(device)
        for iteration in range(iterations):
            order = names if iteration % 2 == 0 else list(reversed(names))
            for name in order:
                started = time.perf_counter_ns()
                functions[name]()
                _synchronize(device)
                samples[name].append((time.perf_counter_ns() - started) / 1_000_000)
    return {name: _summary(values) for name, values in samples.items()}


def _artifact(path: Path, expected_model: str) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("model") != expected_model or not isinstance(
        value.get("state_dict"), dict
    ):
        raise ValueError(f"{path.name} is not the expected {expected_model} artifact")
    return value


def _contract(value: dict[str, Any]) -> dict[str, Any]:
    """Extract an explicit runtime provenance contract from a checkpoint."""
    if value.get("schema") == V4_CHECKPOINT_SCHEMA:
        publication_evidence = bool(
            value.get("study_role") == "confirmatory"
            and value.get("publication_eligible") is True
            and value.get("preregistration_sha256") == V4_PREREGISTRATION_SHA256
            and value.get("source_provenance", {}).get("available") is True
            and value.get("source_provenance", {}).get("git_dirty") is False
        )
        contract = {
            "publication_evidence": publication_evidence,
            "preregistration_sha256": value.get("preregistration_sha256"),
            "protocol_sha256": value.get("protocol_sha256"),
            "coordinate_sha256": value.get("coordinate_sha256"),
            "training_seed": value.get("training_seed"),
        }
        if not all(
            isinstance(contract[field], str) and contract[field]
            for field in ("protocol_sha256", "coordinate_sha256")
        ) or not isinstance(contract["training_seed"], int):
            raise ValueError("v5 runtime checkpoint lacks binding metadata")
        return contract

    coordinate = value.get("protocol", {}).get("coordinate_sha256")
    if not isinstance(coordinate, str) or not coordinate:
        raise ValueError("exploratory runtime checkpoint lacks a coordinate binding")
    return {
        "publication_evidence": False,
        "preregistration_sha256": None,
        "protocol_sha256": None,
        "coordinate_sha256": coordinate,
        "training_seed": None,
    }


def _structural_context(length: int, device: torch.device) -> Tensor:
    if length < 128:
        raise ValueError("v5 timing requires at least 128 packets")
    generator = torch.Generator(device="cpu").manual_seed(20260830 + length)
    packets = torch.randn((1, length, 13), generator=generator)
    packets[..., 1] = packets[..., 1].abs()
    packets[..., -1] = 0.0
    positions = torch.linspace(0, length - 1, 17).round().to(torch.int64).unique()
    packets[:, positions, -1] = 1.0
    packets[:, -1, -1] = 1.0
    return packets.to(device)


def _projected_inputs(
    context: Tensor, model: AhemV5SourceAlignedStreamNextPacket
) -> tuple[Tensor, ...]:
    batch, length, _ = context.shape
    positions = torch.arange(length, device=context.device)[None].expand(batch, -1)
    selected = (
        positions.masked_fill(context[..., -1] <= 0.5, -1)
        .topk(model.max_transitions + 1, dim=1)
        .values
    )
    lane = context.gather(1, selected.clamp_min(0)[..., None].expand(-1, -1, 13))[
        ..., :12
    ]
    source = lane[:, 1:]
    successor = lane[:, :-1]
    span = context.new_full((batch,), float(length))
    gap = torch.log1p(
        (selected[:, :-1] - selected[:, 1:]).clamp_min(0).to(context.dtype)
    ) / torch.log1p(span[:, None])
    keys, values = model.project_transition(
        source.reshape(-1, 12), successor.reshape(-1, 12), gap.reshape(-1)
    )
    return (
        context[:, -model.local_packets :, :12],
        context[:, -1, :12],
        keys.reshape(batch, model.max_transitions, model.key_dim),
        values.reshape(batch, model.max_transitions, model.width),
        successor[..., :2],
        selected[:, 1:],
        (selected[:, 1:] >= 0) & (selected[:, :-1] >= 0),
        positions[:, -1],
        span,
    )


def _local_prefix_state(
    model: AhemV5SourceAlignedStreamNextPacket, local_packets: Tensor
) -> LocalState:
    state = None
    for index in range(local_packets.size(1) - 1):
        _, state = model.local.forward_step(local_packets[:, index], state)
    if state is None:
        raise ValueError("local prefix did not produce streaming state")
    return state


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
) -> dict[str, Any]:
    if len(lengths) != 1 or min(warmup, iterations, threads) <= 0:
        raise ValueError("lengths, warmup, iterations, and threads must be positive")
    torch.set_num_threads(threads)
    ahem_artifact = _artifact(ahem_checkpoint, "ahem_v5_source_aligned")
    lstm_artifact = _artifact(lstm_checkpoint, "lstm")
    ahem_contract = _contract(ahem_artifact)
    lstm_contract = _contract(lstm_artifact)
    if ahem_contract != lstm_contract:
        raise ValueError("AHEM-v5 and LSTM checkpoints use different provenance")
    ahem = AhemV5SourceAlignedStreamNextPacket().to(device).eval()
    lstm = LstmNextPacket(13, output_dim=2).to(device).eval()
    ahem.load_state_dict(ahem_artifact["state_dict"], strict=True)
    lstm.load_state_dict(lstm_artifact["state_dict"], strict=True)

    measurements: dict[str, Any] = {}
    with torch.inference_mode():
        for length in lengths:
            context = _structural_context(length, device)
            projected = _projected_inputs(context, ahem)
            local_state = _local_prefix_state(ahem, projected[0])
            if length > 1:
                _, lstm_state = lstm.encoder(context[:, :-1])
            else:
                lstm_state = None
            reference = ahem.forward_from_projected_cache(*projected)
            streamed, _ = ahem.forward_stream_step(
                projected[1],
                local_state,
                keys=projected[2],
                values=projected[3],
                successor_dynamic=projected[4],
                source_ordinals=projected[5],
                valid=projected[6],
                anchor_ordinals=projected[7],
                context_spans=projected[8],
            )
            error = float((streamed.prediction - reference.prediction).abs().max())
            if error > 1e-6:
                raise RuntimeError("v5 streaming and bounded reference paths differ")

            def stream_call(projected=projected, local_state=local_state):
                return ahem.forward_stream_step(
                    projected[1],
                    local_state,
                    keys=projected[2],
                    values=projected[3],
                    successor_dynamic=projected[4],
                    source_ordinals=projected[5],
                    valid=projected[6],
                    anchor_ordinals=projected[7],
                    context_spans=projected[8],
                )[0].prediction

            def bounded_call(projected=projected):
                return ahem.forward_from_projected_cache(*projected).prediction

            def lstm_call(context=context, lstm_state=lstm_state):
                return lstm.forward_step(context[:, -1], lstm_state)[0]

            source = context[:, -2, :12]
            successor = context[:, -1, :12]
            gap = context.new_zeros((1,))

            def write_call(source=source, successor=successor, gap=gap):
                return ahem.project_transition(source, successor, gap)[1]

            timing = _measure_alternating(
                {
                    "ahem_v5_stream_step": stream_call,
                    "ahem_v5_bounded_reference": bounded_call,
                    "ahem_v5_transition_write": write_call,
                    "lstm_stateful_step": lstm_call,
                },
                device=device,
                warmup=warmup,
                iterations=iterations,
            )
            timing["stream_reference_max_abs_error"] = error
            timing["ahem_p95_divided_by_lstm_p95"] = (
                timing["ahem_v5_stream_step"]["p95_ms"]
                / timing["lstm_stateful_step"]["p95_ms"]
            )
            measurements[str(length)] = timing

    result = {
        "schema": BENCHMARK_SCHEMA,
        "measurement_scope": "conditional_neural_step_with_prebuilt_query_state",
        "end_to_end_streaming_evidence": False,
        "created_utc": datetime.now(UTC).isoformat(),
        "process_instance_id": PROCESS_INSTANCE_ID,
        "process_id": os.getpid(),
        "source_provenance": _source_provenance(),
        "implementation_sha256": {
            name: _sha256(Path(__file__).with_name(name))
            for name in (
                "ahem_v3.py",
                "ahem_v4.py",
                "ahem_pretrain.py",
                "flow_v4_runtime_benchmark.py",
            )
        },
        "study_role": (
            "confirmatory_runtime_audit"
            if ahem_contract["publication_evidence"]
            else "exploratory_runtime_audit"
        ),
        **ahem_contract,
        "checkpoints": {
            "ahem_v5": {
                "path": str(ahem_checkpoint),
                "sha256": _sha256(ahem_checkpoint),
            },
            "lstm": {
                "path": str(lstm_checkpoint),
                "sha256": _sha256(lstm_checkpoint),
            },
        },
        "parameters": {
            "ahem_v5": sum(value.numel() for value in ahem.parameters()),
            "lstm": sum(value.numel() for value in lstm.parameters()),
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
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
            "measurement_order": "alternating_forward_reverse",
        },
        "measurements": measurements,
        "feasibility_gate": {
            "metric": "ahem_v5_stream_step_p95_ms",
            "threshold_ms": 2.0,
            "all_lengths_pass": all(
                value["ahem_v5_stream_step"]["p95_ms"] <= 2.0
                for value in measurements.values()
            ),
        },
        "limitations": [
            "LSTM relation bits depend on the current query flow; one prefix state cannot be reused across arbitrary flow switches",
            "query-dependent state construction and rolling-window resets are not measured",
            "structural deterministic tensors measure compute, not forecast quality",
            "prefix-state and projected-cache construction are excluded and separately timed where applicable",
            "single-machine exploratory audit; confirmatory timing needs fresh selected checkpoints",
        ],
    }
    write_manifest(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ahem-checkpoint", type=Path, required=True)
    parser.add_argument("--lstm-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[10_000],
        help="exactly one length; invoke a fresh process for every context size",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run(
        args.ahem_checkpoint,
        args.lstm_checkpoint,
        args.output,
        lengths=args.lengths,
        device=torch.device(args.device),
        warmup=args.warmup,
        iterations=args.iterations,
        threads=args.threads,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

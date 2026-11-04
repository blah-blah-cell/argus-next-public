"""Structural counterexamples for frozen v5; never scientific performance data.

This module does not change the trained architecture or historical statistics.
The compact-cache check deliberately exposes an existing contract violation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .ahem_v4 import AhemV5SourceAlignedStreamNextPacket
from .flow_v2_protocol import write_manifest


def structural_context(model: AhemV5SourceAlignedStreamNextPacket) -> Tensor:
    generator = torch.Generator().manual_seed(177)
    context = torch.randn(2, 257, 13, generator=generator)
    context[..., 1] = context[..., 1].abs()
    context[..., -1] = 0
    context[:, (0, 64, 130, 256), -1] = 1
    return context.to(next(model.parameters()))


def compact_inputs(
    model: AhemV5SourceAlignedStreamNextPacket, context: Tensor
) -> tuple[Tensor, ...]:
    """Independently construct the three actual completed transitions."""
    positions = torch.tensor([256, 130, 64, 0], device=context.device)
    lane = context[:, positions, :12]
    source, successor = lane[:, 1:], lane[:, :-1]
    span = context.new_full((2,), 257)
    gap = torch.log1p((positions[:-1] - positions[1:]).to(context.dtype))
    gap = gap[None].expand(2, -1) / torch.log1p(span[:, None])
    keys, values = model.project_transition(
        source.reshape(-1, 12), successor.reshape(-1, 12), gap.reshape(-1)
    )
    return (
        context[:, -128:, :12],
        context[:, -1, :12],
        keys.reshape(2, 3, 32),
        values.reshape(2, 3, 60),
        successor[..., :2],
        positions[None, 1:].expand(2, -1),
        torch.ones(2, 3, dtype=torch.bool, device=context.device),
        positions[0].expand(2),
        span,
    )


def pad_inputs(inputs: tuple[Tensor, ...], slots: int = 16) -> tuple[Tensor, ...]:
    padded = list(inputs)
    for index in (2, 3, 4, 5, 6):
        value = inputs[index]
        padding = value.new_zeros(
            value.size(0), slots - value.size(1), *value.shape[2:]
        )
        padded[index] = torch.cat((value, padding), dim=1)
    return tuple(padded)


@torch.inference_mode()
def cache_audit(model: AhemV5SourceAlignedStreamNextPacket) -> dict[str, Any]:
    model.eval()
    context = structural_context(model)
    compact = compact_inputs(model, context)
    dense = model.forward_diagnostics(context)
    short = model.forward_from_projected_cache(*compact)
    padded = model.forward_from_projected_cache(*pad_inputs(compact))
    state = None
    for packet in context[:, -128:, :12].unbind(1):
        local, state = model.local.forward_step(packet, state)
    assert state is not None
    local_error = float((local - model.local(compact[0])).abs().max())

    positions = (256, 130, 64, 0)
    lane = context[:, positions, :12]
    source, successor = lane[:, 1:], lane[:, :-1]
    gaps = context.new_tensor((126, 66, 64)).repeat(2)
    key_old, value_old = model.project_transition(
        source.reshape(-1, 12),
        successor.reshape(-1, 12),
        torch.log1p(gaps) / context.new_tensor(258).log(),
    )
    key_new, value_new = model.project_transition(
        source.reshape(-1, 12),
        successor.reshape(-1, 12),
        torch.log1p(gaps) / context.new_tensor(30_001).log(),
    )
    compact_error = float((short.prediction - dense.prediction).abs().max())
    padded_error = float((padded.prediction - dense.prediction).abs().max())
    return {
        "input_kind": "deterministic_structural_fixture_not_packet_data",
        "valid_transitions": 3,
        "dense_slots": 16,
        "compact_prediction_max_abs_error": compact_error,
        "compact_gate_max_abs_error": float(
            (short.memory_gate - dense.memory_gate).abs().max()
        ),
        "compact_attention_max_abs_error": float(
            (short.selected_weights - dense.selected_weights[:, :3]).abs().max()
        ),
        "padded_prediction_max_abs_error": padded_error,
        "local_dense_stream_max_abs_error": local_error,
        "compact_equivalence_at_1e_minus_6": compact_error <= 1e-6,
        "fixed_layout_equivalence_at_1e_minus_6": padded_error <= 1e-6,
        "changed_normalization_key_max_abs_error": float(
            (key_old - key_new).abs().max()
        ),
        "changed_normalization_value_max_abs_error": float(
            (value_old - value_new).abs().max()
        ),
    }


def shared_flow_variance_example(seed_count: int) -> dict[str, float | int]:
    """Exact enumeration: identical seed rows, four singleton flow effects.

    Flow values are a mathematical counterexample, not a data experiment.
    Independent within-seed resampling divides shared-flow variance by S.
    """
    if type(seed_count) is not int or seed_count < 1:
        raise ValueError("seed_count must be a positive integer")
    means = [Fraction(sum(draw), 4) for draw in product((-3, -1, 1, 3), repeat=4)]
    mean = sum(means) / len(means)
    variance = sum((value - mean) ** 2 for value in means) / len(means)
    return {
        "seed_count": seed_count,
        "enumerated_shared_flow_draws": len(means),
        "shared_flow_variance": float(variance),
        "independent_within_seed_variance": float(variance / seed_count),
        "variance_ratio": float(Fraction(1, seed_count)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compatible", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    reports = []
    for path in args.checkpoints:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if artifact.get("model") != "ahem_v5_source_aligned":
            raise ValueError("expected an actual v5 source-aligned checkpoint")
        if args.compatible:
            from .ahem_v5_compat import CompatibleV5

            model = CompatibleV5().eval()
        else:
            model = AhemV5SourceAlignedStreamNextPacket().eval()
        model.load_state_dict(artifact["state_dict"], strict=True)
        if not all(
            torch.isfinite(value).all() for value in model.state_dict().values()
        ):
            raise ValueError("checkpoint contains non-finite parameters")
        reports.append(
            {
                "checkpoint": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "audit": cache_audit(model),
            }
        )
    result = {
        "schema": "argus.ahem-v5-math-audit/v1",
        "publication_evidence": False,
        "inference_contract": "argus.v5-fixed-layout/v1"
        if args.compatible
        else "frozen_v5",
        "checkpoints": reports,
        "shared_flow_variance_counterexample": shared_flow_variance_example(3),
        "limitations": [
            "structural fixtures establish counterexamples, not accuracy or prevalence",
            "no checkpoint, dataset, historical report, or frozen model is modified",
            "padded equivalence does not prove complete online pipeline equivalence",
        ],
    }
    write_manifest(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

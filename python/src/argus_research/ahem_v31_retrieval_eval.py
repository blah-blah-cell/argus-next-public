"""Counterfactual cache-retrieval quality evaluation for frozen AHEM-v3.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor

from .ahem_v3 import AhemV31NextPacket, AhemV32NextPacket
from .ahem_v3_benchmark import _batch
from .experimental_cnn import adapt_runtime_rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _per_sample_loss(prediction: Tensor, target: Tensor) -> Tensor:
    if target.shape != prediction.shape:
        target = torch.broadcast_to(target, prediction.shape)
    return torch.nn.functional.smooth_l1_loss(
        prediction, target, reduction="none"
    ).mean(dim=-1)


def _candidate_from_retrieved(
    model: AhemV31NextPacket,
    local_state: Tensor,
    local_prediction: Tensor,
    retrieved: Tensor,
) -> Tensor:
    mixed = torch.cat((local_state, retrieved), dim=-1)
    prediction = local_prediction + model.memory_delta(mixed)
    return model._replace_time(prediction, model.memory_time(mixed))


def _batch_metrics(
    model: AhemV31NextPacket,
    packets: Tensor,
    target: Tensor,
    generator: torch.Generator,
) -> dict[str, Tensor]:
    memory = model.encode_completed_chunks(packets[:, : -model.local_packets])
    local_state = model.local(packets[:, -model.local_packets :])
    local_prediction = model._replace_time(
        model.local_predict(local_state), model.local_time(local_state)
    )
    output = model.forward_from_memory(packets[:, -model.local_packets :], memory)
    batch, chunks, _ = memory.shape
    expanded_local = local_state[:, None].expand(-1, chunks, -1)
    expanded_prediction = local_prediction[:, None].expand(-1, chunks, -1)
    if isinstance(model, AhemV32NextPacket):
        keys = memory[..., : model.width]
        successors = memory[..., model.width :]
        individual = model.memory_candidate(
            expanded_local.reshape(batch * chunks, -1),
            expanded_prediction.reshape(batch * chunks, -1),
            keys.reshape(batch * chunks, -1),
            successors.reshape(batch * chunks, -1),
        ).reshape(batch, chunks, -1)
    else:
        values = model.value(memory)
        individual = _candidate_from_retrieved(
            model,
            expanded_local.reshape(batch * chunks, -1),
            expanded_prediction.reshape(batch * chunks, -1),
            values.reshape(batch * chunks, -1),
        ).reshape(batch, chunks, -1)
    individual_loss = _per_sample_loss(individual, target[:, None])
    oracle_distribution = (-individual_loss / 0.04).softmax(dim=1)
    oracle_distribution_entropy = -(
        oracle_distribution * oracle_distribution.clamp_min(1e-8).log()
    ).sum(dim=1) / math.log(chunks)
    ordered_loss = individual_loss.sort(dim=1).values
    oracle_top_margin = ordered_loss[:, 1] - ordered_loss[:, 0]
    oracle_order = individual_loss.argsort(dim=1)
    retrieval_k = min(model.fine_top_k, chunks)
    oracle_top = oracle_order[:, :retrieval_k]
    selected = output.selected_chunks
    overlap = (
        (selected[:, :, None] == oracle_top[:, None, :]).any(dim=2).float().mean(dim=1)
    )
    best_recalled = (selected == oracle_order[:, :1]).any(dim=1).float()
    ranks = individual_loss.argsort(dim=1).argsort(dim=1).gather(1, selected)
    if batch > 1:
        shuffled_local = packets[:, -model.local_packets :].roll(1, dims=0)
        shuffled = model.forward_from_memory(shuffled_local, memory).selected_chunks
        shuffled_overlap = (
            (shuffled[:, :, None] == oracle_top[:, None, :])
            .any(dim=2)
            .float()
            .mean(dim=1)
        )
    else:
        shuffled_overlap = overlap.new_full(overlap.shape, float("nan"))

    random_indices = torch.stack(
        [
            torch.randperm(chunks, generator=generator)[:retrieval_k]
            for _ in range(batch)
        ]
    )
    if isinstance(model, AhemV32NextPacket):
        random_states = memory.gather(
            1,
            random_indices[:, :, None].expand(-1, -1, model.memory_width),
        )
        random_prediction = model.memory_candidate(
            local_state,
            local_prediction,
            random_states[..., : model.width].mean(dim=1),
            random_states[..., model.width :].mean(dim=1),
        )
    else:
        random_values = values.gather(
            1, random_indices[:, :, None].expand(-1, -1, model.width)
        ).mean(dim=1)
        random_prediction = _candidate_from_retrieved(
            model, local_state, local_prediction, random_values
        )
    weight_entropy = -(
        output.selected_weights * output.selected_weights.clamp_min(1e-8).log()
    ).sum(dim=1) / math.log(output.selected_weights.size(1))
    fractions = output.prediction.new_tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    blends = local_prediction[:, None] + fractions[None, :, None] * (
        output.memory_candidate[:, None] - local_prediction[:, None]
    )
    blend_losses = _per_sample_loss(blends, target[:, None].expand_as(blends))
    oracle_memory_fraction = fractions[blend_losses.argmin(dim=1)]
    return {
        "full_loss": _per_sample_loss(
            output.memory_candidate
            if isinstance(model, AhemV32NextPacket)
            else output.prediction,
            target,
        ),
        "local_loss": _per_sample_loss(local_prediction, target),
        "gated_loss": _per_sample_loss(output.prediction, target),
        "random_loss": _per_sample_loss(random_prediction, target),
        "oracle_single_loss": individual_loss.min(dim=1).values,
        "oracle_top_k_overlap": overlap,
        "oracle_best_recalled": best_recalled,
        "selected_normalized_rank": ranks.float().mean(dim=1) / max(chunks - 1, 1),
        "selected_weight_entropy": weight_entropy,
        "selected_weight_spread": (
            output.selected_weights.max(dim=1).values
            - output.selected_weights.min(dim=1).values
        ),
        "query_shuffled_oracle_top_k_overlap": shuffled_overlap,
        "oracle_target_entropy": oracle_distribution_entropy,
        "oracle_top_margin": oracle_top_margin,
        "oracle_loss_standard_deviation": individual_loss.std(dim=1, unbiased=False),
        "candidate_count": individual_loss.new_full((batch,), float(chunks)),
        "memory_gate": output.memory_gate.squeeze(1),
        "oracle_memory_fraction": oracle_memory_fraction,
    }


def run(
    features: list[Path],
    checkpoint: Path,
    output: Path,
    context: int,
    samples: int,
) -> dict[str, object]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("architecture") != "ahem-v3.1-adaptive-chunk-memory":
        raise ValueError("checkpoint is not packaged AHEM-v3.1")
    model = AhemV31NextPacket(12)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    streams = [adapt_runtime_rows(path)[0] for path in features]
    sampling = random.Random(20260825)
    random_retrieval = torch.Generator().manual_seed(20260825)
    values: dict[str, list[Tensor]] = {}
    with torch.inference_mode():
        for _ in range(math.ceil(samples / 4)):
            packets, target = _batch(streams, context, min(4, samples), sampling)
            for name, batch_value in _batch_metrics(
                model, packets, target, random_retrieval
            ).items():
                values.setdefault(name, []).append(batch_value)
    joined = {name: torch.cat(parts)[:samples] for name, parts in values.items()}
    full = joined["full_loss"]
    local = joined["local_loss"]
    random_loss = joined["random_loss"]
    result = {
        "evaluation": "ahem_v3.1_counterfactual_cache_retrieval_quality",
        "created_utc": datetime.now(UTC).isoformat(),
        "real_data": True,
        "not_intrusion_detection_efficacy": True,
        "context_packets": context,
        "samples": samples,
        "checkpoint_sha256": _sha256(checkpoint),
        "source_sha256": [_sha256(path) for path in features],
        "forecast_loss": {
            "retrieved_memory": float(full.mean()),
            "local_only": float(local.mean()),
            "random_memory": float(random_loss.mean()),
            "oracle_best_single_chunk": float(joined["oracle_single_loss"].mean()),
        },
        "cache_contribution": {
            "mean_loss_reduction_vs_local_fraction": float(
                (local.mean() - full.mean()) / local.mean().clamp_min(1e-8)
            ),
            "query_help_rate_vs_local": float((full < local).float().mean()),
            "query_win_rate_vs_random": float((full < random_loss).float().mean()),
        },
        "retrieval_quality": {
            "retrieval_k": min(model.fine_top_k, int(joined["candidate_count"][0])),
            "oracle_top_k_precision": float(joined["oracle_top_k_overlap"].mean()),
            "oracle_best_chunk_recall_at_k": float(
                joined["oracle_best_recalled"].mean()
            ),
            "mean_selected_normalized_oracle_rank": float(
                joined["selected_normalized_rank"].mean()
            ),
            "normalized_selected_weight_entropy": float(
                joined["selected_weight_entropy"].mean()
            ),
            "mean_selected_weight_spread": float(
                joined["selected_weight_spread"].mean()
            ),
        },
        "interpretation": {
            "oracle_top_k_precision_random_expectation": min(
                model.fine_top_k,
                (context - model.local_packets) // model.chunk_size,
            )
            / ((context - model.local_packets) // model.chunk_size),
            "normalized_rank_random_expectation": 0.5,
            "weight_entropy_uniform": 1.0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=10_000)
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.features,
                args.checkpoint,
                args.output,
                args.context,
                args.samples,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

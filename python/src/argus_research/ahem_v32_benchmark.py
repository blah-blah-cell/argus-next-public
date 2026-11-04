"""Train and gate AHEM-v3.2 temporal utility-ranked retrieval on real journals."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor, nn

from .ahem_pretrain import LstmNextPacket
from .ahem_v3 import AhemV31NextPacket, AhemV32NextPacket
from .ahem_v3_benchmark import _batch, _evaluate, _latency
from .ahem_v31_retrieval_eval import _batch_metrics
from .continual_learning import iter_segments


def _weighted_loss(
    prediction: Tensor, target: Tensor, reduction: str = "mean"
) -> Tensor:
    weights = prediction.new_tensor([1.0, 4.0] + [0.5] * 10)
    values = (
        nn.functional.smooth_l1_loss(prediction, target, reduction="none") * weights
    )
    per_sample = values.sum(dim=-1) / weights.sum()
    return per_sample.mean() if reduction == "mean" else per_sample


def _candidate_predictions(
    model: AhemV32NextPacket,
    local_state: Tensor,
    local_prediction: Tensor,
    memory: Tensor,
) -> Tensor:
    batch, chunks, _ = memory.shape
    keys = memory[..., : model.width]
    successors = memory[..., model.width :]
    local = local_state[:, None].expand(-1, chunks, -1).reshape(batch * chunks, -1)
    prediction = (
        local_prediction[:, None].expand(-1, chunks, -1).reshape(batch * chunks, -1)
    )
    return model.memory_candidate(
        local,
        prediction,
        keys.reshape(batch * chunks, -1),
        successors.reshape(batch * chunks, -1),
    ).reshape(batch, chunks, -1)


def _training_loss(
    model: AhemV32NextPacket, packets: Tensor, target: Tensor
) -> tuple[Tensor, dict[str, float]]:
    memory = model.encode_completed_chunks(packets[:, : -model.local_packets])
    local_packets = packets[:, -model.local_packets :]
    local_state = model.local(local_packets)
    local_prediction = model._replace_time(
        model.local_predict(local_state), model.local_time(local_state)
    )
    output = model.forward_from_memory(local_packets, memory)
    candidates = _candidate_predictions(model, local_state, local_prediction, memory)
    oracle_loss = _weighted_loss(
        candidates, target[:, None].expand_as(candidates), reduction="none"
    ).detach()
    target_distribution = (-oracle_loss / 0.04).softmax(dim=1)
    retrieval_logits = model.retrieval_scores(local_state, memory, local_packets)
    ranking = -(target_distribution * retrieval_logits.log_softmax(dim=1)).sum(
        dim=1
    ).mean() / math.log(memory.size(1))
    forecast = _weighted_loss(output.prediction, target)
    local = _weighted_loss(local_prediction, target)
    time = nn.functional.smooth_l1_loss(output.prediction[:, 1], target[:, 1])
    total = forecast + 0.08 * local + 0.20 * ranking + 0.50 * time
    return total, {
        "forecast": float(forecast.detach()),
        "ranking": float(ranking.detach()),
        "time": float(time.detach()),
    }


def _train(
    model: AhemV32NextPacket,
    train: list[Tensor],
    validation: list[Tensor],
    context: int,
    epochs: int,
    steps: int,
) -> tuple[AhemV32NextPacket, list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    rng = random.Random(20260826)
    best = deepcopy(model.state_dict())
    best_validation = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        totals = []
        rankings = []
        for _ in range(steps):
            packets, target = _batch(train, context, 2, rng)
            loss, components = _training_loss(model, packets, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite AHEM-v3.2 training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals.append(float(loss.detach()))
            rankings.append(components["ranking"])
        quality = _evaluate(model, validation, context, 32, 20260827)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": sum(totals) / len(totals),
                "ranking_loss": sum(rankings) / len(rankings),
                "validation_smooth_l1": quality.smooth_l1,
                "validation_time_mae": quality.log_interarrival_mae,
            }
        )
        if quality.smooth_l1 < best_validation:
            best_validation = quality.smooth_l1
            best = deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model.eval(), history


def _calibrate_retrieval_scale(
    model: AhemV32NextPacket,
    validation: list[Tensor],
    context: int,
) -> list[dict[str, float]]:
    candidates = (4.0, 8.0, 12.0, 16.0, 24.0, 30.0)
    results = []
    for scale in candidates:
        model.retrieval_log_scale.data.fill_(math.log(scale))
        quality = _evaluate(model, validation, context, 64, 20260832)
        results.append(
            {
                "scale": scale,
                "validation_smooth_l1": quality.smooth_l1,
                "validation_time_mae": quality.log_interarrival_mae,
            }
        )
    selected = min(results, key=lambda value: value["validation_smooth_l1"])
    model.retrieval_log_scale.data.fill_(math.log(selected["scale"]))
    return results


def _load(model: nn.Module, path: Path) -> nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def _retrieval_quality(
    model: AhemV31NextPacket,
    streams: list[Tensor],
    context: int,
    samples: int,
    seed: int,
) -> dict[str, float | bool | None | str]:
    rng = random.Random(seed)
    random_retrieval = torch.Generator().manual_seed(seed)
    parts: dict[str, list[Tensor]] = {}
    with torch.inference_mode():
        for _ in range(math.ceil(samples / 4)):
            packets, target = _batch(streams, context, min(4, samples), rng)
            for name, value in _batch_metrics(
                model, packets, target, random_retrieval
            ).items():
                parts.setdefault(name, []).append(value)
    values = {name: torch.cat(items)[:samples] for name, items in parts.items()}
    selected_precision = float(values["oracle_top_k_overlap"].mean())
    shuffled_precision = float(values["query_shuffled_oracle_top_k_overlap"].nanmean())
    candidate_count = float(values["candidate_count"].mean())
    retrieval_k = min(model.fine_top_k, int(candidate_count))
    random_precision = retrieval_k / candidate_count
    learned_gate = model.use_utility_gate and model.utility_gate is not None
    result: dict[str, float | bool | None | str] = {
        "retrieved_loss": float(values["full_loss"].mean()),
        "local_loss": float(values["local_loss"].mean()),
        "gated_loss": float(values["gated_loss"].mean()),
        "random_loss": float(values["random_loss"].mean()),
        "help_rate_vs_local": float(
            (values["full_loss"] < values["local_loss"]).float().mean()
        ),
        "win_rate_vs_random": float(
            (values["full_loss"] < values["random_loss"]).float().mean()
        ),
        "retrieval_k": retrieval_k,
        "oracle_top_k_precision": selected_precision,
        "oracle_best_recall_at_k": float(values["oracle_best_recalled"].mean()),
        "candidate_chunks": candidate_count,
        "random_expected_precision_at_k": random_precision,
        "oracle_precision_lift_over_random": selected_precision
        / max(random_precision, 1e-12),
        "query_shuffled_oracle_top_k_precision": shuffled_precision,
        "query_specific_precision_gain": selected_precision - shuffled_precision,
        "oracle_target_normalized_entropy": float(
            values["oracle_target_entropy"].mean()
        ),
        "oracle_top_margin": float(values["oracle_top_margin"].mean()),
        "oracle_loss_standard_deviation": float(
            values["oracle_loss_standard_deviation"].mean()
        ),
        "mean_selected_normalized_oracle_rank": float(
            values["selected_normalized_rank"].mean()
        ),
        "normalized_weight_entropy": float(values["selected_weight_entropy"].mean()),
        "mean_memory_gate": float(values["memory_gate"].mean()),
        "memory_blend_is_learned": learned_gate,
        "hard_memory_usage_rate": (
            float((values["memory_gate"] >= 0.5).float().mean())
            if learned_gate
            else None
        ),
        "mean_oracle_memory_fraction": float(values["oracle_memory_fraction"].mean()),
        "memory_fraction_mae": float(
            (values["memory_gate"] - values["oracle_memory_fraction"]).abs().mean()
        ),
        "cache_use_detection_accuracy": (
            float(
                (
                    (values["memory_gate"] >= 0.5)
                    == (values["full_loss"] < values["local_loss"])
                )
                .float()
                .mean()
            )
            if learned_gate
            else None
        ),
    }
    result["diagnosis"] = _diagnose_retrieval(result)
    return result


def _diagnose_retrieval(
    metrics: dict[str, float | bool | None | str],
) -> str:
    target_is_ambiguous = float(metrics["oracle_target_normalized_entropy"]) >= 0.90
    learned_is_random_like = (
        float(metrics["win_rate_vs_random"]) <= 0.55
        and float(metrics["mean_selected_normalized_oracle_rank"]) >= 0.45
        and float(metrics["query_specific_precision_gain"]) <= 0.02
    )
    if target_is_ambiguous and learned_is_random_like:
        return "both_ambiguous_retrieval_target_and_random_like_model"
    if target_is_ambiguous:
        return "evaluation_target_is_ambiguous"
    if learned_is_random_like:
        return "model_retrieval_is_random_like"
    return "retrieval_has_measurable_query_specific_signal"


def run(
    journal: Path,
    v31_checkpoint: Path,
    lstm_checkpoint: Path,
    output: Path,
    context: int,
    epochs: int,
    steps: int,
) -> dict[str, object]:
    torch.manual_seed(20260826)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    streams = [values for _, values in iter_segments(journal)]
    train_end = int(len(streams) * 0.70)
    validation_end = int(len(streams) * 0.85)
    train = streams[:train_end]
    validation = streams[train_end:validation_end]
    test = streams[validation_end:]
    v32, history = _train(
        AhemV32NextPacket(12), train, validation, context, epochs, steps
    )
    scale_calibration = _calibrate_retrieval_scale(v32, validation, context)
    models: dict[str, nn.Module] = {
        "ahem_v3.2": v32,
        "ahem_v3.1": _load(AhemV31NextPacket(12), v31_checkpoint),
        "lstm": _load(LstmNextPacket(12), lstm_checkpoint),
    }
    quality = {
        name: asdict(_evaluate(model, test, context, 128, 20260828))
        for name, model in models.items()
    }
    retrieval = {
        name: _retrieval_quality(model, test, context, 128, 20260829)
        for name, model in models.items()
        if name.startswith("ahem")
    }
    sample, _ = _batch(test, context, 1, random.Random(20260830))
    latency = {name: _latency(model, sample, 30) for name, model in models.items()}
    candidate_chunks = (context - v32.local_packets) // v32.chunk_size
    random_precision = min(v32.fine_top_k, candidate_chunks) / candidate_chunks
    gates = {
        "aggregate_loss_better_than_lstm": quality["ahem_v3.2"]["smooth_l1"]
        < quality["lstm"]["smooth_l1"],
        "time_mae_better_than_lstm": quality["ahem_v3.2"]["log_interarrival_mae"]
        < quality["lstm"]["log_interarrival_mae"],
        "no_more_than_2pct_loss_regression_vs_v31": quality["ahem_v3.2"]["smooth_l1"]
        <= quality["ahem_v3.1"]["smooth_l1"] * 1.02,
        "oracle_precision_beats_random": retrieval["ahem_v3.2"][
            "oracle_top_k_precision"
        ]
        > random_precision,
        "oracle_best_recall_beats_random": retrieval["ahem_v3.2"][
            "oracle_best_recall_at_k"
        ]
        > random_precision,
        "retrieval_beats_random_on_70pct_queries": retrieval["ahem_v3.2"][
            "win_rate_vs_random"
        ]
        >= 0.70,
        "p95_latency_better_than_lstm": latency["ahem_v3.2"]["p95_ms"]
        < latency["lstm"]["p95_ms"],
    }
    result = {
        "benchmark": "ahem_v3.2_temporal_utility_ranked_retrieval",
        "created_utc": datetime.now(UTC).isoformat(),
        "real_data": True,
        "not_intrusion_detection_efficacy": True,
        "split": "whole_segment_chronological_70_15_15",
        "context_packets": context,
        "epochs": epochs,
        "steps_per_epoch": steps,
        "parameters": {
            name: sum(parameter.numel() for parameter in model.parameters())
            for name, model in models.items()
        },
        "quality": quality,
        "retrieval": retrieval,
        "latency": latency,
        "random_oracle_top_k_precision_expectation": random_precision,
        "gates": gates,
        "admit_as_shadow_candidate": all(gates.values()),
        "training_history": history,
        "retrieval_scale_calibration": scale_calibration,
        "selected_retrieval_scale": float(v32.retrieval_log_scale.detach().exp()),
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": v32.state_dict(),
            "artifact_type": "causal_temporal_forecaster",
            "architecture": "ahem-v3.2-temporal-utility-ranked-memory",
            "feature_schema": "packet-causal-v1",
            "supervision": "next_packet_only",
            "security_detector": False,
            "parameters": result["parameters"]["ahem_v3.2"],
        },
        output / "ahem_v32.pt",
    )
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--v31-checkpoint", type=Path, required=True)
    parser.add_argument("--lstm-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.journal,
                args.v31_checkpoint,
                args.lstm_checkpoint,
                args.output,
                args.context,
                args.epochs,
                args.steps,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

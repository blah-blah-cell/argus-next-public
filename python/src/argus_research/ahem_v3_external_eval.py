"""One-shot evaluation of frozen AHEM-v3/LSTM artifacts on external captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from .ahem_pretrain import LstmNextPacket
from .ahem_v3 import AhemV3NextPacket, AhemV31NextPacket, AhemV32NextPacket
from .ahem_v3_benchmark import (
    _batch,
    _evaluate,
    _latency,
    _paired_quality_gain,
)
from .experimental_cnn import adapt_runtime_rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(model: torch.nn.Module, path: Path) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("feature_schema") != "packet-causal-v1"
        or payload.get("supervision") != "next_packet_only"
    ):
        raise ValueError(f"{path.name} has an incompatible forecast contract")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def run(
    features: list[Path],
    checkpoint_directory: Path | None,
    output: Path,
    context: int,
    candidate: str = "ahem_v3",
    candidate_checkpoint: Path | None = None,
    lstm_checkpoint: Path | None = None,
) -> dict[str, object]:
    streams = [adapt_runtime_rows(path)[0] for path in features]
    constructors = {
        "ahem_v3": AhemV3NextPacket,
        "ahem_v3.1": AhemV31NextPacket,
        "ahem_v3.2": AhemV32NextPacket,
    }
    if candidate not in constructors:
        raise ValueError(f"unsupported candidate architecture: {candidate}")
    if candidate_checkpoint is None:
        if checkpoint_directory is None:
            raise ValueError("candidate checkpoint or checkpoint directory is required")
        candidate_checkpoint = checkpoint_directory / "ahem_v3.pt"
    if lstm_checkpoint is None:
        if checkpoint_directory is None:
            raise ValueError("LSTM checkpoint or checkpoint directory is required")
        lstm_checkpoint = checkpoint_directory / "lstm.pt"
    ahem = _load(constructors[candidate](12), candidate_checkpoint)
    lstm = _load(LstmNextPacket(12), lstm_checkpoint)
    quality = {
        candidate: asdict(_evaluate(ahem, streams, context, 128, 20260821)),
        "lstm": asdict(_evaluate(lstm, streams, context, 128, 20260821)),
    }
    paired = _paired_quality_gain(ahem, lstm, streams, context, samples=256)
    sample, _ = _batch(streams, context, 1, random.Random(20260822))
    latency = {
        candidate: _latency(ahem, sample, 30),
        "lstm": _latency(lstm, sample, 30),
    }
    checks = {
        "smooth_l1": quality[candidate]["smooth_l1"] < quality["lstm"]["smooth_l1"],
        "log_length_mae": quality[candidate]["log_length_mae"]
        < quality["lstm"]["log_length_mae"],
        "log_interarrival_mae": quality[candidate]["log_interarrival_mae"]
        < quality["lstm"]["log_interarrival_mae"],
        "binary_feature_accuracy": quality[candidate]["binary_feature_accuracy"]
        > quality["lstm"]["binary_feature_accuracy"],
        "p95_latency": latency[candidate]["p95_ms"] < latency["lstm"]["p95_ms"],
        "paired_95pct_low_positive": paired["bootstrap_95pct_low"] > 0,
    }
    result = {
        "evaluation": f"{candidate}_frozen_external_real_capture_vs_lstm",
        "created_utc": datetime.now(UTC).isoformat(),
        "task": "causal_next_packet_forecast_only",
        "not_intrusion_detection_efficacy": True,
        "artifacts_frozen_before_external_evaluation": True,
        "pristine_architecture_selection_holdout": candidate == "ahem_v3",
        "candidate_artifacts": {
            candidate: {
                "path": str(candidate_checkpoint),
                "sha256": _sha256(candidate_checkpoint),
            },
            "lstm": {
                "path": str(lstm_checkpoint),
                "sha256": _sha256(lstm_checkpoint),
            },
        },
        "context_packets": context,
        "sources": [{"path": str(path), "sha256": _sha256(path)} for path in features],
        "quality": quality,
        "paired_quality": paired,
        "latency": latency,
        "checks": checks,
        "strict_external_win": all(checks.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, action="append", required=True)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument(
        "--candidate",
        choices=("ahem_v3", "ahem_v3.1", "ahem_v3.2"),
        default="ahem_v3",
    )
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--lstm-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=10_000)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.features,
                args.checkpoints,
                args.output,
                args.context,
                args.candidate,
                args.candidate_checkpoint,
                args.lstm_checkpoint,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

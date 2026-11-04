"""Data-only diagnostics for the exact exploratory Pro pilot coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .ahem_pro_pilot import (
    SCREEN_SOURCE,
    SEED,
    VALIDATION_SOURCE,
    PilotConfig,
    _inventory,
)
from .flow_v3_protocol import ProtocolConfig, compile_source_disjoint, materialize_batch
from .flow_v6_objective import dynamic_loss


def _summary(
    contexts: torch.Tensor, targets: torch.Tensor, train_mean: torch.Tensor
) -> dict[str, object]:
    relation = contexts[..., -1].sum(-1).float()
    persistence = contexts[:, -1, :2]
    quantiles = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "samples": len(targets),
        "target_mean": targets.mean(0).tolist(),
        "target_std": targets.std(0).tolist(),
        "target_quantiles": torch.quantile(targets, quantiles, dim=0).tolist(),
        "flow_lane_events_mean": float(relation.mean()),
        "flow_lane_events_min": int(relation.min()),
        "flow_lane_events_max": int(relation.max()),
        "baselines": {
            "current_packet_persistence": float(dynamic_loss(persistence, targets)),
            "zero": float(dynamic_loss(torch.zeros_like(targets), targets)),
            "training_target_mean": float(
                dynamic_loss(train_mean.expand_as(targets), targets)
            ),
        },
    }


def audit(journal: Path) -> dict[str, object]:
    _inventory(journal)
    config = PilotConfig()
    protocol = compile_source_disjoint(
        journal,
        VALIDATION_SOURCE,
        SCREEN_SOURCE,
        ProtocolConfig(
            context_packets=config.context_packets,
            local_packets=128,
            train_samples=config.train_samples,
            validation_samples=config.validation_samples,
            test_samples=config.screen_samples,
            seed=SEED,
            require_memory_match=True,
            sampling_policy="natural",
            minimum_completed_transitions=2,
        ),
    )
    data = {
        name: materialize_batch(
            protocol.partitions[name],
            protocol.coordinates[name],
            config.context_packets,
        )
        for name in ("train", "validation", "test")
    }
    train_mean = data["train"][1].mean(0)
    return {
        "schema": "argus.ahem-pro-pilot-data-audit/v1",
        "data_only": True,
        "publication_eligible": False,
        "coordinate_sha256": protocol.manifest["coordinate_sha256"],
        "partitions": {
            name: _summary(*values, train_mean) for name, values in data.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.journal)
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise ValueError("refusing to overwrite data audit")
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()

"""Source-balanced cross-day development screen for AHEM-v6.2.

This is a new development experiment. It reuses no result as a holdout and
cannot open the frozen Thursday-22, Friday-23, or CIC-IDS2017 sources.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .flow_v6_development import (
    ExperimentSpec,
    RunConfig,
    run_experiment,
)

SCHEMA = "argus.ahem-v6.2-training-development/v1"
SPEC = ExperimentSpec(
    schema=SCHEMA,
    candidate_name="ahem_v62_state_transition",
    candidate_description=(
        "source_balanced_scale_normalized_causal_state_transition_memory"
    ),
    protocol_revision="v6.2_cross_day_retrieval_regularization",
    models=(
        "ahem_v62_state_transition",
        "ahem_v62_local_only",
        "ahem_v62_recency",
        "flow_lane_lstm",
    ),
    references=(
        "ahem_v62_local_only",
        "ahem_v62_recency",
        "flow_lane_lstm",
    ),
    minimum_validation_top1_gain=0.03,
    maximum_train_to_screen_top1_drop=0.25,
)
CONFIG = RunConfig(
    train_samples=16_384,
    validation_samples=2_048,
    screen_samples=2_048,
    epochs=50,
    batch_size=256,
    learning_rate=2e-4,
    weight_decay=5e-4,
    patience=8,
    utility_temperature=0.04,
    minimum_completed_transitions=2,
    sampling_policy="source_balanced_natural",
    retrieval_ranking_weight=0.25,
    retrieval_ranking_margin=0.10,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    contract_path = (
        Path(__file__).resolve().parents[3] / "research/ahem-v62-vm-contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("run_authorized") is not True or contract.get("status") != "frozen":
        raise RuntimeError(
            "v6.2 is paused after the completion audit; finish the v6.1 study first"
        )
    return run_experiment(args, config=CONFIG, spec=SPEC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()

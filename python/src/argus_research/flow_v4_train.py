"""Fail-closed one-model runner for the frozen AHEM-v5 study."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .flow_v3_train import (
    PRIMARY_TEST_SOURCE,
    PRIMARY_VALIDATION_SOURCE,
    V4_MODEL_NAMES,
    run,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=V4_MODEL_NAMES,
        required=True,
    )
    parser.add_argument("--validation-source", default=PRIMARY_VALIDATION_SOURCE)
    parser.add_argument("--test-source", default=PRIMARY_TEST_SOURCE)
    parser.add_argument("--context", type=int, default=10_000)
    parser.add_argument("--train-samples", type=int, default=2_048)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--protocol-seed", type=int, default=20260830)
    parser.add_argument(
        "--sampling-policy", choices=("balanced", "natural"), default="balanced"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.study = "v5"
    run(args)


if __name__ == "__main__":
    main()

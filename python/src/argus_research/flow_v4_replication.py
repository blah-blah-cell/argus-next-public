"""Untouched CIC-IDS2017 replication entrypoint for frozen AHEM-v5 weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .flow_v3_replication import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ahem-run", type=Path, nargs=5, required=True)
    parser.add_argument("--ahem-checkpoint", type=Path, nargs=5, required=True)
    parser.add_argument("--lstm-run", type=Path, nargs=5, required=True)
    parser.add_argument("--lstm-checkpoint", type=Path, nargs=5, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    run(
        journal=args.journal,
        output=args.output,
        ahem_runs=args.ahem_run,
        ahem_checkpoints=args.ahem_checkpoint,
        lstm_runs=args.lstm_run,
        lstm_checkpoints=args.lstm_checkpoint,
        device=torch.device(args.device),
        samples=args.samples,
        batch_size=args.batch_size,
        resamples=args.resamples,
        study="v5",
    )


if __name__ == "__main__":
    main()

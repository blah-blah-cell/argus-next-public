"""One-attempt untouched-source replication for selected AHEM/LSTM checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

from .ahem_pretrain import LstmNextPacket
from .ahem_v3 import AhemV34FlowIndexedNextPacket
from .ahem_v4 import AhemV5SourceAlignedStreamNextPacket
from .continual_learning import iter_sequence_records_v2
from .flow_v2_multiseed import hierarchical_bootstrap
from .flow_v2_protocol import _journal_manifest, write_manifest
from .flow_v2_train import _canonical_sha256, _file_sha256
from .flow_v3_multiseed import (
    CANDIDATE,
    MINIMUM_RELATIVE_IMPROVEMENT,
    MULTISEED_SCHEMA,
)
from .flow_v3_protocol import (
    PREREGISTRATION_SHA256,
    CompiledProtocol,
    ProtocolConfig,
    condition_record,
    deterministic_coordinates,
)
from .flow_v3_smoke import _evaluate
from .flow_v3_train import (
    CHECKPOINT_SCHEMA,
    RUN_SCHEMA,
    V4_CHECKPOINT_SCHEMA,
    V4_PREREGISTRATION_SHA256,
    V4_RUN_SCHEMA,
)
from .flow_v4_multiseed import (
    CANDIDATE as V4_CANDIDATE,
)
from .flow_v4_multiseed import (
    MULTISEED_SCHEMA as V4_MULTISEED_SCHEMA,
)
from .flow_v4_multiseed import (
    TRAINING_SEEDS as V4_TRAINING_SEEDS,
)

REPLICATION_SOURCE = "Friday-WorkingHours"
REPLICATION_DATASET = "CIC-IDS2017"
REPLICATION_SEED = 20260829
REPLICATION_ACCESS_SCHEMA = "argus.flow-v3-replication-access/v1"
V4_REPLICATION_ACCESS_SCHEMA = "argus.ahem-v5-replication-access/v1"


def _load_selected(
    run_path: Path,
    checkpoint_path: Path,
    expected_model: str,
    *,
    run_schema: str = RUN_SCHEMA,
    checkpoint_schema: str = CHECKPOINT_SCHEMA,
    preregistration_sha256: str = PREREGISTRATION_SHA256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("schema") != run_schema or run.get("study_role") != "confirmatory":
        raise ValueError(f"{run_path.name} is not a confirmatory flow-v3 run")
    if run.get("publication_eligible") is not True:
        raise ValueError(f"{run_path.name} is not publication-eligible")
    if run.get("model", {}).get("name") != expected_model:
        raise ValueError(f"{run_path.name} belongs to another model")
    if run.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError(f"{run_path.name} belongs to another preregistration")
    access = run.get("test_access", {})
    if access.get("status") != "completed" or access.get(
        "best_checkpoint_sha256"
    ) != _file_sha256(checkpoint_path):
        raise ValueError(f"{run_path.name} does not bind the selected checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected = {
        "schema": checkpoint_schema,
        "preregistration_sha256": preregistration_sha256,
        "protocol_sha256": run["protocol_sha256"],
        "coordinate_sha256": run["coordinate_sha256"],
        "model": expected_model,
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{checkpoint_path.name} fingerprint differs from its run")
    return run, checkpoint


def _replication_protocol(
    journal: Path,
    samples: int,
    *,
    preregistration_sha256: str = PREREGISTRATION_SHA256,
    seed: int = REPLICATION_SEED,
    schema: str = "argus.flow-v3-replication-protocol/v1",
) -> CompiledProtocol:
    raw_records = [
        record
        for record in iter_sequence_records_v2(journal)
        if record.source_file == REPLICATION_SOURCE
    ]
    if not raw_records:
        raise ValueError("the preregistered CIC-IDS2017 replication source is absent")
    journal_manifest = _journal_manifest(journal, {REPLICATION_SOURCE})
    if journal_manifest.get("provenance_complete") is not True or journal_manifest.get(
        "datasets"
    ) != [REPLICATION_DATASET]:
        raise ValueError("replication journal lacks complete CIC-IDS2017 provenance")
    records = [condition_record(record) for record in raw_records]
    config = ProtocolConfig(
        context_packets=10_000,
        train_samples=1,
        validation_samples=1,
        test_samples=samples,
        seed=seed,
        require_memory_match=True,
    )
    coordinates = deterministic_coordinates(
        records,
        context_packets=config.context_packets,
        local_packets=config.local_packets,
        samples=samples,
        seed=config.seed,
        require_memory_match=True,
        changed_event_fraction=config.changed_event_fraction,
    )
    payload = [asdict(value) for value in coordinates]
    manifest = {
        "schema": schema,
        "preregistration_sha256": preregistration_sha256,
        "source": REPLICATION_SOURCE,
        "dataset": REPLICATION_DATASET,
        "config": {
            "context_packets": config.context_packets,
            "local_packets": config.local_packets,
            "samples": samples,
            "seed": config.seed,
            "require_memory_match": True,
            "changed_event_fraction": config.changed_event_fraction,
        },
        "journal": journal_manifest,
        "segments": sorted(record.segment_id for record in records),
        "coordinate_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return CompiledProtocol(
        partitions={"replication": records},
        coordinates={"replication": coordinates},
        manifest=manifest,
    )


def _reserve(
    output: Path,
    protocol: dict[str, Any],
    artifacts: list[tuple[Path, Path]],
    access_schema: str = REPLICATION_ACCESS_SCHEMA,
) -> tuple[Path, dict[str, Any]]:
    access_path = output.with_suffix(output.suffix + ".access.json")
    value = {
        "schema": access_schema,
        "status": "reserved",
        "attempt": 1,
        "reserved_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": _canonical_sha256(protocol),
        "coordinate_sha256": protocol["coordinate_sha256"],
        "selected_artifacts": [
            {
                "run": str(run),
                "run_sha256": _file_sha256(run),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _file_sha256(checkpoint),
            }
            for run, checkpoint in artifacts
        ],
    }
    access_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with access_path.open("x", encoding="utf-8") as destination:
            destination.write(json.dumps(value, indent=2) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
    except FileExistsError as error:
        raise ValueError("untouched replication access was already reserved") from error
    return access_path, value


def run(
    *,
    journal: Path,
    output: Path,
    ahem_runs: list[Path],
    ahem_checkpoints: list[Path],
    lstm_runs: list[Path],
    lstm_checkpoints: list[Path],
    device: torch.device,
    samples: int = 512,
    batch_size: int = 4,
    resamples: int = 10_000,
    study: str = "v3",
) -> dict[str, Any]:
    collections = (ahem_runs, ahem_checkpoints, lstm_runs, lstm_checkpoints)
    if any(len(values) != 5 for values in collections):
        raise ValueError("replication requires five AHEM and five LSTM selected runs")
    if min(samples, batch_size) <= 0:
        raise ValueError("replication dimensions must be positive")
    if study not in {"v3", "v5"}:
        raise ValueError("unknown replication study")
    candidate_name = V4_CANDIDATE if study == "v5" else CANDIDATE
    candidate_constructor = (
        AhemV5SourceAlignedStreamNextPacket
        if study == "v5"
        else AhemV34FlowIndexedNextPacket
    )
    run_schema = V4_RUN_SCHEMA if study == "v5" else RUN_SCHEMA
    checkpoint_schema = V4_CHECKPOINT_SCHEMA if study == "v5" else CHECKPOINT_SCHEMA
    preregistration_sha256 = (
        V4_PREREGISTRATION_SHA256 if study == "v5" else PREREGISTRATION_SHA256
    )
    replication_seed = 20260901 if study == "v5" else REPLICATION_SEED
    replication_schema = (
        "argus.ahem-v5-replication-protocol/v1"
        if study == "v5"
        else "argus.flow-v3-replication-protocol/v1"
    )
    access_schema = (
        V4_REPLICATION_ACCESS_SCHEMA if study == "v5" else REPLICATION_ACCESS_SCHEMA
    )
    result_schema = V4_MULTISEED_SCHEMA if study == "v5" else MULTISEED_SCHEMA
    selected: dict[str, dict[int, tuple[dict[str, Any], dict[str, Any]]]] = {
        candidate_name: {},
        "lstm": {},
    }
    artifacts = []
    for model, run_paths, checkpoint_paths in (
        (candidate_name, ahem_runs, ahem_checkpoints),
        ("lstm", lstm_runs, lstm_checkpoints),
    ):
        for run_path, checkpoint_path in zip(run_paths, checkpoint_paths, strict=True):
            run_value, checkpoint = _load_selected(
                run_path,
                checkpoint_path,
                model,
                run_schema=run_schema,
                checkpoint_schema=checkpoint_schema,
                preregistration_sha256=preregistration_sha256,
            )
            training_seed = int(run_value["training"]["seed"])
            if training_seed in selected[model]:
                raise ValueError(f"duplicate selected {model} seed")
            selected[model][training_seed] = (run_value, checkpoint)
            artifacts.append((run_path, checkpoint_path))
    seeds = (
        list(V4_TRAINING_SEEDS)
        if study == "v5"
        else [
            20260829,
            20260830,
            20260831,
            20260901,
            20260902,
        ]
    )
    if any(sorted(values) != seeds for values in selected.values()):
        raise ValueError("selected checkpoints do not cover preregistered seeds")
    primary_protocols = {
        run_value["protocol_sha256"]
        for values in selected.values()
        for run_value, _ in values.values()
    }
    if len(primary_protocols) != 1:
        raise ValueError("selected primary checkpoints use different protocols")

    protocol = (
        _replication_protocol(
            journal,
            samples,
            preregistration_sha256=preregistration_sha256,
            seed=replication_seed,
            schema=replication_schema,
        )
        if study == "v5"
        else _replication_protocol(journal, samples)
    )
    access_path, access = _reserve(
        output, protocol.manifest, artifacts, access_schema=access_schema
    )
    losses: dict[str, dict[int, list[float]]] = {candidate_name: {}, "lstm": {}}
    coordinates: list[dict[str, Any]] | None = None
    for model_name, constructor in (
        (candidate_name, candidate_constructor),
        ("lstm", lambda: LstmNextPacket(13, output_dim=2)),
    ):
        for training_seed in seeds:
            _, checkpoint = selected[model_name][training_seed]
            model = constructor().to(device).eval()
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            evaluation = _evaluate(
                model,
                protocol,
                "replication",
                10_000,
                batch_size,
                device,
            )
            if coordinates is None:
                coordinates = evaluation["sample_coordinates"]
            elif coordinates != evaluation["sample_coordinates"]:
                raise RuntimeError("replication sample order changed across models")
            losses[model_name][training_seed] = evaluation[
                "per_sample_weighted_smooth_l1"
            ]

    differences = [
        [
            float(lstm) - float(ahem)
            for lstm, ahem in zip(
                losses["lstm"][training_seed],
                losses[candidate_name][training_seed],
                strict=True,
            )
        ]
        for training_seed in seeds
    ]
    assert coordinates is not None
    identities = [
        (
            value["source_file"],
            value["segment_id"],
            value["flow_generation"],
        )
        for value in coordinates
    ]
    mapping = {
        identity: index for index, identity in enumerate(sorted(set(identities)))
    }
    cluster_ids = [[mapping[identity] for identity in identities] for _ in seeds]
    paired = hierarchical_bootstrap(
        differences,
        cluster_ids_by_seed=cluster_ids,
        resamples=resamples,
        seed=replication_seed,
    )
    baseline_mean = fmean(
        fmean(float(value) for value in losses["lstm"][training_seed])
        for training_seed in seeds
    )
    candidate_mean = fmean(
        fmean(float(value) for value in losses[candidate_name][training_seed])
        for training_seed in seeds
    )
    relative = paired["mean_baseline_minus_candidate"] / max(baseline_mean, 1e-12)
    gates = {
        "paired_interval_strictly_positive": paired["bootstrap_95pct_low"] > 0,
        "relative_improvement_at_least_five_percent": (
            relative >= MINIMUM_RELATIVE_IMPROVEMENT
        ),
    }
    gates["comparison_passed"] = all(gates.values())
    protocol_sha256 = _canonical_sha256(protocol.manifest)
    result = {
        "schema": result_schema,
        "study_role": "confirmatory_untouched_replication",
        "publication_evidence": True,
        "preregistration_sha256": preregistration_sha256,
        "protocol_sha256": protocol_sha256,
        "coordinate_sha256": protocol.manifest["coordinate_sha256"],
        "primary_protocol_sha256": next(iter(primary_protocols)),
        "candidate": candidate_name,
        "reference": "lstm",
        "training_seeds": seeds,
        "hierarchical_bootstrap_resamples": resamples,
        "bootstrap_contract": "seeds_then_flow_generations_then_coordinates",
        "paired_weighted_forecast_loss": paired,
        "mean_candidate_loss": candidate_mean,
        "mean_reference_loss": baseline_mean,
        "mean_relative_improvement": relative,
        "gates": gates,
        "protocol": protocol.manifest,
        "sample_coordinates_sha256": _canonical_sha256(coordinates),
        "per_seed_losses": losses,
    }
    write_manifest(output, result)
    access = {
        **access,
        "status": "completed",
        "completed_utc": datetime.now(UTC).isoformat(),
        "result_sha256": _file_sha256(output),
    }
    write_manifest(access_path, access)
    return result


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
    )


if __name__ == "__main__":
    main()

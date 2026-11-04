"""Prepare train/calibration decisions for the causal episode router v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import tempfile
import time
from bisect import bisect_left
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .ahem_vm_train import _journal_manifest
from .continual_learning import SequenceRecordV2, iter_sequence_records_v2
from .episode_router_samples import EpisodeRoutingSample, build_episode_samples
from .episode_router_supervision import (
    build_supervision,
    compact_episode_features,
    counterfactual_subset_outcomes,
    episode_evidence,
    expert_subsets,
    routing_questions_v2,
    subset_id,
    supervision_json,
)
from .laya_router_prepare import FrozenExpert, _sha256, load_experts
from .packet_episode_graph import BundleConfig, EpisodeConfig, records_from_tensor

SCHEMA = "argus.laya-episode-router-preparation/v2"
WORKFLOW = "argus_causal_episode_subset_routing"


def _combined_digest(*values: str) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _validate_provenance(
    train: dict[str, Any],
    calibration: dict[str, Any],
    run_path: Path,
    experts: list[FrozenExpert],
) -> dict[str, Any]:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("schema") != "argus.temporal-expert-run/v1":
        raise ValueError("expert run manifest schema differs")
    training = run.get("journal")
    if not isinstance(training, dict):
        raise ValueError("expert training journal provenance is absent")
    expected = {expert.training_journal_sha256 for expert in experts}
    if expected != {training.get("sha256")}:
        raise ValueError("expert training provenance differs from checkpoints")

    def hashes(manifest: dict[str, Any]) -> set[str]:
        members = manifest.get("members")
        if not isinstance(members, list):
            raise ValueError("journal member provenance is absent")
        output = {value.get("sha256") for value in members if isinstance(value, dict)}
        if None in output or len(output) != len(members):
            raise ValueError("journal member hash is absent or duplicated")
        return output

    training_hashes = hashes(training)
    train_hashes = hashes(train)
    calibration_hashes = hashes(calibration)
    if training_hashes & (train_hashes | calibration_hashes):
        raise ValueError("router journal overlaps frozen-expert training data")
    if train_hashes & calibration_hashes:
        raise ValueError("router train and calibration journals overlap")
    split = run.get("split")
    if not isinstance(split, dict) or split.get("method") != "journal_source_file_disjoint":
        raise ValueError("frozen experts were not source-file-disjoint")
    return {
        "run_manifest_sha256": _sha256(run_path),
        "training_journal_sha256": training["sha256"],
        "training_member_hashes": sorted(training_hashes),
        "train_member_hashes": sorted(train_hashes),
        "calibration_member_hashes": sorted(calibration_hashes),
        "all_intersections_empty": True,
    }


def _records(path: Path) -> list[SequenceRecordV2]:
    values = list(iter_sequence_records_v2(path))
    if not values:
        raise ValueError(f"journal contains no checksum-valid ARGUSLR2 records: {path}")
    return values


def _subset_latency(
    experts: list[FrozenExpert], context: int, device: torch.device
) -> dict[str, float]:
    sample = torch.zeros((1, context, 12), device=device)
    values: dict[str, float] = {}
    subsets = expert_subsets([expert.expert_id for expert in experts])
    with torch.inference_mode():
        for subset in subsets:
            selected = [expert for expert in experts if expert.expert_id in subset]
            for _ in range(3):
                predictions = [
                    expert.model(sample[:, -min(context, expert.input_context) :])
                    for expert in selected
                ]
                torch.stack(predictions).mean(0)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings = []
            for _ in range(12):
                started = time.perf_counter_ns()
                predictions = [
                    expert.model(sample[:, -min(context, expert.input_context) :])
                    for expert in selected
                ]
                torch.stack(predictions).mean(0)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
            values[subset_id(subset)] = statistics.median(timings)
    return values


def _score_sample(
    record: SequenceRecordV2,
    sample: EpisodeRoutingSample,
    experts: list[FrozenExpert],
    latency: dict[str, float],
    ordinal_to_index: dict[int, int],
    by_flow: dict[int, list[int]],
    context: int,
    compute_weight: float,
    device: torch.device,
) -> dict[str, Any] | None:
    windows: list[Tensor] = []
    targets: list[Tensor] = []
    for packet in sample.target_packets:
        index = ordinal_to_index.get(packet.source_ordinal)
        if index is None:
            raise RuntimeError("target ordinal is absent from its source record")
        flow = int(record.flow_ids[index])
        positions = by_flow[flow]
        location = bisect_left(positions, index)
        prior = positions[max(0, location - context) : location]
        if len(prior) != context:
            continue
        windows.append(record.features[prior])
        targets.append(record.features[index])
    if not windows:
        return None
    inputs = torch.stack(windows).to(device)
    target = torch.stack(targets).to(device)
    predictions: dict[str, Tensor] = {}
    with torch.inference_mode():
        for expert in experts:
            expert_context = inputs[:, -min(context, expert.input_context) :]
            predictions[expert.expert_id] = expert.model(expert_context)
    outcomes = counterfactual_subset_outcomes(
        predictions, target, latency, compute_weight=compute_weight
    )
    evidence = episode_evidence(
        sample.anchor,
        sample.bundle,
        {episode.episode_id: episode for episode in sample.bundle_episodes},
    )
    identifier = hashlib.sha256(
        f"{record.source_file}:{record.identifier}:{sample.anchor.episode_id}".encode()
    ).hexdigest()
    return {
        "id": identifier,
        "source_file": record.source_file,
        "segment_id": record.identifier,
        "decision_ordinal": sample.bundle.decision_source_ordinal,
        "target_start_ordinal": sample.target_packets[0].source_ordinal,
        "target_end_ordinal": sample.target_packets[-1].source_ordinal,
        "target_packets_scored": len(targets),
        "state": evidence,
        "compact_router_features": list(compact_episode_features(evidence)),
        "outcomes": outcomes,
    }


def collect_partition(
    records: list[SequenceRecordV2],
    experts: list[FrozenExpert],
    latency: dict[str, float],
    *,
    cap: int,
    context: int,
    compute_weight: float,
    seed: int,
    device: torch.device,
    partition_name: str,
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    completed_records: set[str] = set()
    source_rows_by_source: dict[str, int] = {}
    if checkpoint_path is not None and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected = {
            "schema": "argus.episode-router-progress/v1",
            "partition": partition_name,
            "cap": cap,
            "context": context,
            "compute_weight": compute_weight,
            "seed": seed,
        }
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError(f"incompatible preparation checkpoint: {checkpoint_path}")
        output = saved["rows"]
        completed_records = set(saved["completed_records"])
        source_rows_by_source = {
            str(key): int(value)
            for key, value in saved["source_rows_by_source"].items()
        }

    def save_progress() -> None:
        if checkpoint_path is None:
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save(
            {
                "schema": "argus.episode-router-progress/v1",
                "partition": partition_name,
                "cap": cap,
                "context": context,
                "compute_weight": compute_weight,
                "seed": seed,
                "rows": output,
                "completed_records": sorted(completed_records),
                "source_rows_by_source": source_rows_by_source,
            },
            temporary,
        )
        temporary.replace(checkpoint_path)

    episode_config = EpisodeConfig(min_packets=16, max_packets=256)
    bundle_config = BundleConfig(maximum_neighbors=8)
    by_source: dict[str, list[SequenceRecordV2]] = defaultdict(list)
    for record in records:
        by_source[record.source_file].append(record)
    per_source = math.ceil(cap / len(by_source))
    for source_name in sorted(by_source):
        source_rows = source_rows_by_source.get(source_name, 0)
        source_records = by_source[source_name]
        rng.shuffle(source_records)
        for record in source_records:
            record_key = f"{record.source_file}:{record.identifier}"
            if record_key in completed_records:
                continue
            capture_id = hashlib.sha256(
                f"{record.source_file}:{record.identifier}".encode()
            ).hexdigest()
            packets = records_from_tensor(
                capture_id=capture_id,
                interface_id="journal",
                flow_ids=record.flow_ids,
                timestamps_ns=record.timestamps_ns,
                source_ordinals=record.source_ordinals,
                features=record.features,
            )
            samples = list(
                build_episode_samples(
                    packets,
                    episode_config=episode_config,
                    bundle_config=bundle_config,
                )
            )
            rng.shuffle(samples)
            ordinal_to_index = {
                int(value): index
                for index, value in enumerate(record.source_ordinals.tolist())
            }
            by_flow: dict[int, list[int]] = {}
            for index, flow in enumerate(record.flow_ids.tolist()):
                by_flow.setdefault(int(flow), []).append(index)
            for sample in samples:
                row = _score_sample(
                    record,
                    sample,
                    experts,
                    latency,
                    ordinal_to_index,
                    by_flow,
                    context,
                    compute_weight,
                    device,
                )
                if row is not None:
                    output.append(row)
                    source_rows += 1
                if source_rows >= per_source:
                    break
            completed_records.add(record_key)
            source_rows_by_source[source_name] = source_rows
            save_progress()
            print(
                json.dumps(
                    {
                        "stage": "episode_preparation_progress",
                        "partition": partition_name,
                        "source": source_name,
                        "record": record.identifier,
                        "rows": source_rows,
                        "total_rows": len(output),
                    }
                ),
                flush=True,
            )
            if source_rows >= per_source:
                break
        print(
            json.dumps(
                {
                    "stage": "episode_preparation",
                    "partition": partition_name,
                    "source": source_name,
                    "rows": source_rows,
                    "total_rows": len(output),
                }
            ),
            flush=True,
        )
    if not output:
        raise ValueError("partition produced no context-eligible episode transitions")
    rng.shuffle(output)
    return output[:cap]


def _temperature(rows: list[dict[str, Any]]) -> float:
    gaps = []
    for row in rows:
        ranked = sorted(value.utility for value in row["outcomes"])
        if len(ranked) > 1 and math.isfinite(ranked[1] - ranked[0]) and ranked[1] > ranked[0]:
            gaps.append(ranked[1] - ranked[0])
    return max(1e-4, statistics.median(gaps) if gaps else 0.01)


def _finish_rows(rows: list[dict[str, Any]], temperature: float) -> None:
    for row in rows:
        supervision = build_supervision(
            row["outcomes"],
            temperature=temperature,
            identity_health=str(row["state"]["identity_health"]),
        )
        probabilities = dict(zip(supervision.subset_order, supervision.soft_targets, strict=True))
        cardinality = len(next(
            value.experts for value in supervision.outcomes
            if value.subset_id == supervision.oracle_subset
        ))
        budget = "fast" if cardinality == 1 else "balanced" if cardinality == 2 else "deep"
        row["supervision"] = supervision
        row["gold"] = {
            "route_subset": {"probabilities": probabilities},
            "compute_budget": {
                "probabilities": {
                    name: 0.98 if name == budget else 0.01
                    for name in ("fast", "balanced", "deep")
                }
            },
            "abstain": {
                "probabilities": {
                    "yes": 0.99 if supervision.abstain_target else 0.01,
                    "no": 0.01 if supervision.abstain_target else 0.99,
                }
            },
        }


def _write(path: Path, rows: list[dict[str, Any]], experts: list[str]) -> tuple[str, str]:
    questions = routing_questions_v2(experts)
    public_hash = hashlib.sha256()
    counter_hash = hashlib.sha256()
    counter_path = path.with_name(path.stem + ".counterfactual.jsonl")
    with path.open("wb") as public, counter_path.open("wb") as counter:
        for row in rows:
            public_row = {
                "id": row["id"],
                "workflow": WORKFLOW,
                "state": json.dumps(row["state"], sort_keys=True, separators=(",", ":")),
                "questions": json.dumps(questions, sort_keys=True, separators=(",", ":")),
                "gold": json.dumps(row["gold"], sort_keys=True, separators=(",", ":")),
                "provenance": {
                    key: row[key]
                    for key in (
                        "source_file", "segment_id", "decision_ordinal",
                        "target_start_ordinal", "target_end_ordinal", "target_packets_scored",
                    )
                },
            }
            private_row = {
                "id": row["id"],
                "compact_router_features": row["compact_router_features"],
                "supervision": supervision_json(row["supervision"]),
            }
            for destination, value, digest in (
                (public, public_row, public_hash),
                (counter, private_row, counter_hash),
            ):
                encoded = (json.dumps(value, sort_keys=True) + "\n").encode()
                destination.write(encoded)
                digest.update(encoded)
    return public_hash.hexdigest(), counter_hash.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("episode-router preparation output already exists")
    if min(args.context, args.train_samples, args.calibration_samples) <= 0:
        raise ValueError("context and sample counts must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    train_manifest = _journal_manifest(args.train_journal)
    calibration_manifest = _journal_manifest(args.calibration_journal)
    scoring_hash = _combined_digest(
        train_manifest["sha256"], calibration_manifest["sha256"]
    )
    experts = load_experts(args.models, scoring_hash, device)
    provenance = _validate_provenance(
        train_manifest, calibration_manifest, args.expert_run_manifest, experts
    )
    names = [expert.expert_id for expert in experts]
    latency = _subset_latency(experts, args.context, device)
    partitions = {
        "train": collect_partition(
            _records(args.train_journal), experts, latency,
            cap=args.train_samples, context=args.context,
            compute_weight=args.compute_weight, seed=args.seed, device=device,
            partition_name="train",
            checkpoint_path=(args.progress_dir / "train.pt") if args.progress_dir else None,
        ),
        "calibration": collect_partition(
            _records(args.calibration_journal), experts, latency,
            cap=args.calibration_samples, context=args.context,
            compute_weight=args.compute_weight, seed=args.seed + 1, device=device,
            partition_name="calibration",
            checkpoint_path=(args.progress_dir / "calibration.pt") if args.progress_dir else None,
        ),
    }
    temperature = _temperature(partitions["train"])
    for rows in partitions.values():
        _finish_rows(rows, temperature)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="argus-episode-router-", dir=args.output.parent) as temporary:
        stage = Path(temporary) / "prepared"
        stage.mkdir()
        files: dict[str, str] = {}
        for name, rows in partitions.items():
            public_hash, counter_hash = _write(stage / f"{name}.jsonl", rows, names)
            files[f"{name}.jsonl"] = public_hash
            files[f"{name}.counterfactual.jsonl"] = counter_hash
        manifest = {
            "schema": SCHEMA,
            "created_utc": datetime.now(UTC).isoformat(),
            "task": "causal_episode_cost_aware_subset_routing",
            "security_detector": False,
            "attack_labels_used": False,
            "future_packets_in_router_input": False,
            "expert_predictions_in_router_input": False,
            "locked_test_opened": False,
            "context_packets": args.context,
            "seed": args.seed,
            "compute_weight": args.compute_weight,
            "utility_soft_target_temperature": temperature,
            "counts": {name: len(rows) for name, rows in partitions.items()},
            "train_journal": train_manifest,
            "calibration_journal": calibration_manifest,
            "expert_training_provenance": provenance,
            "experts": [
                {
                    "expert_id": expert.expert_id,
                    "checkpoint_sha256": expert.sha256,
                    "input_context": min(args.context, expert.input_context),
                }
                for expert in experts
            ],
            "profiled_sequential_subset_latency_ms": latency,
            "files": files,
            "locked_test_must_not_be_used_for_training_or_calibration": True,
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (stage / "manifest.sha256").write_text(
            f"{_sha256(manifest_path)}  manifest.json\n", encoding="ascii"
        )
        stage.replace(args.output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-journal", type=Path, required=True)
    parser.add_argument("--calibration-journal", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--expert-run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=4_000)
    parser.add_argument("--calibration-samples", type=int, default=1_000)
    parser.add_argument("--compute-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--progress-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

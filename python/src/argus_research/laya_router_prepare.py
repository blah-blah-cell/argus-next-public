"""Build source-disjoint typed decisions for the optional Laya compute router.

The builder runs frozen next-packet experts counterfactually on checksum-verified
ARGUSLR2 records.  It writes separate train, calibration, and locked-test files
and never trains Laya or consumes intrusion labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .ahem_v32_benchmark import _weighted_loss
from .ahem_vm_train import _journal_manifest
from .continual_learning import SequenceRecordV2, iter_sequence_records_v2
from .laya_window_router import causal_window_evidence, routing_questions
from .memory_models import causal_router_features
from .temporal_expert_vm_train import EXPERTS

SCHEMA = "argus.laya-router-preparation/v1"
WORKFLOW = "argus_cost_aware_forecast_routing"
EXPERT_FILES = {
    "cnn_cse": ("cnn", "cnn_cse_cic_ids2018_best.pt"),
    "lstm_cse": ("lstm", "lstm_cse_cic_ids2018_best.pt"),
    "gru_cse": ("gru", "gru_cse_cic_ids2018_best.pt"),
    "tcn_cse": ("tcn", "tcn_cse_cic_ids2018_best.pt"),
}


@dataclass(frozen=True)
class FrozenExpert:
    expert_id: str
    checkpoint: Path
    sha256: str
    training_journal_sha256: str
    input_context: int
    model: nn.Module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_group(source_file: str) -> str:
    """Keep hourly files from the same UTC day in one split."""
    match = re.match(r"^(\d{4}-\d{2}-\d{2})(?:T|$)", source_file)
    return match.group(1) if match else source_file


def inventory_records(journal: Path, context: int) -> tuple[list[SequenceRecordV2], dict[str, Any]]:
    records = [record for record in iter_sequence_records_v2(journal) if len(record.features) > context]
    if not records:
        raise ValueError("journal contains no checksum-valid ARGUSLR2 record longer than context")
    groups: dict[str, dict[str, int]] = {}
    for record in records:
        group = _source_group(record.source_file)
        entry = groups.setdefault(group, {"records": 0, "packets": 0})
        entry["records"] += 1
        entry["packets"] += len(record.features)
    if len(groups) < 8:
        raise ValueError("at least eight source-day groups are required")
    return records, {"source_groups": groups, "record_count": len(records)}


def source_disjoint_split(records: list[SequenceRecordV2]) -> tuple[dict[str, list[SequenceRecordV2]], dict[str, list[str]]]:
    groups = sorted({_source_group(record.source_file) for record in records})
    if len(groups) < 8:
        raise ValueError("source-disjoint routing requires at least eight source-day groups")
    split_groups = {
        "train": groups[:-5],
        "calibration": groups[-5:-3],
        "test": groups[-3:],
    }
    split_records = {
        name: [record for record in records if _source_group(record.source_file) in selected]
        for name, selected in split_groups.items()
    }
    if any(not values for values in split_records.values()):
        raise ValueError("source-disjoint split contains an empty partition")
    return split_records, split_groups


def load_experts(model_dir: Path, scoring_journal_sha256: str, device: torch.device) -> list[FrozenExpert]:
    loaded: list[FrozenExpert] = []
    training_hashes: set[str] = set()
    for expert_id, (expert_name, file_name) in EXPERT_FILES.items():
        checkpoint = model_dir / file_name
        if not checkpoint.is_file():
            raise FileNotFoundError(f"required frozen expert is absent: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        checks = {
            "schema": "argus.temporal-expert-checkpoint/v1",
            "expert": expert_name,
            "feature_schema": "packet-causal-v1",
            "supervision": "next_packet_only",
            "security_detector": False,
        }
        for key, expected in checks.items():
            if payload.get(key) != expected:
                raise ValueError(f"{expert_id} checkpoint {key} differs")
        training_hash = payload.get("journal_sha256")
        if not isinstance(training_hash, str) or len(training_hash) != 64:
            raise ValueError(f"{expert_id} lacks a training-journal hash")
        if training_hash == scoring_journal_sha256:
            raise ValueError(f"{expert_id} was trained on the scoring journal")
        training_hashes.add(training_hash)
        spec = EXPERTS[expert_name]
        model = spec.constructor()
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(device).eval()
        loaded.append(
            FrozenExpert(
                expert_id=expert_id,
                checkpoint=checkpoint.resolve(),
                sha256=_sha256(checkpoint),
                training_journal_sha256=training_hash,
                input_context=spec.input_context,
                model=model,
            )
        )
    if len(training_hashes) != 1:
        raise ValueError("frozen experts were not trained against one shared journal")
    return loaded


def verify_expert_training_provenance(
    run_manifest_path: Path,
    experts: list[FrozenExpert],
    scoring_journal: dict[str, Any],
    scored_source_files: set[str],
) -> dict[str, Any]:
    if not run_manifest_path.is_file():
        raise FileNotFoundError("expert training run manifest is absent")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("schema") != "argus.temporal-expert-run/v1":
        raise ValueError("expert training run manifest schema differs")
    training_journal = run_manifest.get("journal")
    if not isinstance(training_journal, dict):
        raise ValueError("expert training journal provenance is absent")
    expected_hashes = {expert.training_journal_sha256 for expert in experts}
    if expected_hashes != {training_journal.get("sha256")}:
        raise ValueError("expert run manifest journal differs from frozen checkpoints")
    training_members = training_journal.get("members")
    scoring_members = scoring_journal.get("members")
    if not isinstance(training_members, list) or not isinstance(scoring_members, list):
        raise ValueError("journal member provenance is absent")
    training_hashes = {
        member.get("sha256") for member in training_members if isinstance(member, dict)
    }
    scored_members = {
        member.get("name"): member.get("sha256")
        for member in scoring_members
        if isinstance(member, dict) and member.get("name") in scored_source_files
    }
    missing = scored_source_files - set(scored_members)
    if missing:
        raise ValueError(f"scored journal sources lack hashes: {sorted(missing)}")
    overlap = training_hashes.intersection(scored_members.values())
    if overlap:
        raise ValueError("scored source file overlaps the experts' training journal")
    split = run_manifest.get("split")
    if not isinstance(split, dict) or split.get("method") != "journal_source_file_disjoint":
        raise ValueError("expert training run was not source-file-disjoint")
    return {
        "run_manifest": run_manifest_path.name,
        "run_manifest_sha256": _sha256(run_manifest_path),
        "training_journal_sha256": training_journal["sha256"],
        "training_member_sha256": sorted(training_hashes),
        "scored_member_sha256": dict(sorted(scored_members.items())),
        "member_hash_intersection": [],
        "expert_split_method": split["method"],
    }


def _rows(record: SequenceRecordV2, start: int, end: int) -> list[dict[str, str]]:
    return [
        {
            "opaque_flow_id": str(int(record.flow_ids[index])),
            "monotonic_timestamp_ns": str(int(record.timestamps_ns[index])),
            "source_capture_ordinal": str(int(record.source_ordinals[index])),
        }
        for index in range(start, end)
    ]


def _candidates(
    records: list[SequenceRecordV2], context: int, stride: int, cap: int, seed: int
) -> list[tuple[SequenceRecordV2, int]]:
    choices = [
        (record, target)
        for record in records
        for target in range(context, len(record.features), stride)
    ]
    random.Random(seed).shuffle(choices)
    return choices[:cap]


def _profile_latency(experts: list[FrozenExpert], sample: Tensor, device: torch.device) -> dict[str, float]:
    values: dict[str, float] = {}
    with torch.inference_mode():
        for expert in experts:
            context = sample[:, -min(expert.input_context, sample.size(1)) :].to(device)
            for _ in range(5):
                expert.model(context)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings = []
            for _ in range(20):
                started = time.perf_counter_ns()
                expert.model(context)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
            values[expert.expert_id] = statistics.median(timings)
    return values


def collect_partition(
    records: list[SequenceRecordV2],
    experts: list[FrozenExpert],
    context: int,
    stride: int,
    cap: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    choices = _candidates(records, context, stride, cap, seed)
    output: list[dict[str, Any]] = []
    names = [expert.expert_id for expert in experts]
    with torch.inference_mode():
        for offset in range(0, len(choices), batch_size):
            batch_choices = choices[offset : offset + batch_size]
            windows = torch.stack(
                [record.features[target - context : target] for record, target in batch_choices]
            )
            targets = torch.stack([record.features[target] for record, target in batch_choices]).to(device)
            losses: dict[str, list[float]] = {}
            for expert in experts:
                expert_context = windows[:, -min(expert.input_context, context) :].to(device)
                prediction = expert.model(expert_context)
                losses[expert.expert_id] = [
                    float(value)
                    for value in _weighted_loss(prediction, targets, reduction="none").cpu()
                ]
            for local, (record, target) in enumerate(batch_choices):
                start = target - context
                identifier = hashlib.sha256(
                    f"{record.source_file}:{record.identifier}:{target}".encode()
                ).hexdigest()
                output.append(
                    {
                        "id": identifier,
                        "source_file": record.source_file,
                        "source_group": _source_group(record.source_file),
                        "segment_id": record.identifier,
                        "target_ordinal": int(record.source_ordinals[target]),
                        "state": causal_window_evidence(
                            windows[local],
                            _rows(record, start, target),
                            f"arguslr2:{record.source_file}:{record.identifier}",
                        ),
                        "compact_router_features": [
                            float(value)
                            for value in causal_router_features(
                                windows[local].unsqueeze(0)
                            )[0]
                        ],
                        "forecast_loss": {name: losses[name][local] for name in names},
                    }
                )
    if not output:
        raise ValueError("partition produced no causal windows")
    return output


def _soft_targets(
    records: dict[str, list[dict[str, Any]]],
    names: list[str],
    latency_ms: dict[str, float],
    compute_weight: float,
) -> float:
    maximum_latency = max(latency_ms.values())
    for rows in records.values():
        for row in rows:
            row["utility"] = {
                name: row["forecast_loss"][name]
                + compute_weight * latency_ms[name] / maximum_latency
                for name in names
            }
    gaps = []
    for row in records["train"]:
        ranked = sorted(row["utility"].values())
        gap = ranked[1] - ranked[0]
        if math.isfinite(gap) and gap > 0:
            gaps.append(gap)
    scale = max(1e-4, statistics.median(gaps) if gaps else 0.01)
    for rows in records.values():
        for row in rows:
            utility = torch.tensor([row["utility"][name] for name in names])
            probabilities = torch.softmax(-(utility - utility.min()) / scale, dim=0)
            primary = {name: float(probabilities[index]) for index, name in enumerate(names)}
            confidence = max(primary.values())
            if confidence >= 0.75:
                budget = "fast"
            elif confidence >= 0.50:
                budget = "balanced"
            else:
                budget = "deep"
            row["gold"] = {
                "route_primary": {"probabilities": primary},
                "compute_budget": {
                    "probabilities": {
                        name: 0.96 if name == budget else 0.02
                        for name in ("fast", "balanced", "deep")
                    }
                },
                "abstain": {
                    "probabilities": {"false": confidence, "true": 1.0 - confidence}
                },
            }
            best = min(row["utility"].values())
            for index, name in enumerate(names):
                admission = math.exp(-(row["utility"][name] - best) / scale)
                row["gold"][f"use_expert_{index}"] = {
                    "probabilities": {"false": 1.0 - admission, "true": admission}
                }
    return scale


def _jsonl(path: Path, rows: list[dict[str, Any]], names: list[str]) -> str:
    questions = routing_questions(names)
    digest = hashlib.sha256()
    with path.open("wb") as output:
        for row in rows:
            public = {
                "id": row["id"],
                "workflow": WORKFLOW,
                "state": json.dumps(row["state"], sort_keys=True, separators=(",", ":")),
                "questions": json.dumps(questions, sort_keys=True, separators=(",", ":")),
                "gold": json.dumps(row["gold"], sort_keys=True, separators=(",", ":")),
                "provenance": {
                    "source_group": row["source_group"],
                    "source_file": row["source_file"],
                    "segment_id": row["segment_id"],
                    "target_ordinal": row["target_ordinal"],
                },
            }
            encoded = (json.dumps(public, sort_keys=True) + "\n").encode()
            output.write(encoded)
            digest.update(encoded)
    return digest.hexdigest()


def _counterfactual_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as output:
        for row in rows:
            private = {key: value for key, value in row.items() if key != "state"}
            encoded = (json.dumps(private, sort_keys=True) + "\n").encode()
            output.write(encoded)
            digest.update(encoded)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("output directory already exists; preparation is immutable")
    if min(args.context, args.stride, args.batch_size, args.train_samples, args.calibration_samples, args.test_samples) <= 0:
        raise ValueError("contexts, counts, stride, and batch size must be positive")
    if args.compute_weight < 0:
        raise ValueError("compute weight must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    records, inventory = inventory_records(args.journal, args.context)
    journal = _journal_manifest(args.journal)
    split_records, split_groups = source_disjoint_split(records)
    experts = load_experts(args.models, journal["sha256"], device)
    scored_source_files = {f"{record.source_file}.arguslr" for record in records}
    expert_training_provenance = verify_expert_training_provenance(
        args.expert_run_manifest, experts, journal, scored_source_files
    )
    names = [expert.expert_id for expert in experts]
    sample_record = split_records["train"][0]
    sample = sample_record.features[: args.context].unsqueeze(0)
    latency_ms = _profile_latency(experts, sample, device)
    counts = {
        "train": args.train_samples,
        "calibration": args.calibration_samples,
        "test": args.test_samples,
    }
    partitions = {
        name: collect_partition(
            split_records[name], experts, args.context, args.stride, counts[name],
            args.batch_size, args.seed + index, device,
        )
        for index, name in enumerate(("train", "calibration", "test"))
    }
    utility_scale = _soft_targets(partitions, names, latency_ms, args.compute_weight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="argus-laya-", dir=args.output.parent) as temporary:
        stage = Path(temporary) / "prepared"
        stage.mkdir()
        file_hashes: dict[str, str] = {}
        for name, rows in partitions.items():
            file_hashes[f"{name}.jsonl"] = _jsonl(stage / f"{name}.jsonl", rows, names)
            file_hashes[f"{name}.counterfactual.jsonl"] = _counterfactual_jsonl(
                stage / f"{name}.counterfactual.jsonl", rows
            )
        manifest = {
            "schema": SCHEMA,
            "created_utc": datetime.now(UTC).isoformat(),
            "task": "cost_aware_forecast_expert_routing",
            "security_detector": False,
            "real_data": True,
            "attack_labels_used": False,
            "future_packets_in_router_input": False,
            "context_packets": args.context,
            "stride_packets": args.stride,
            "seed": args.seed,
            "compute_weight": args.compute_weight,
            "utility_soft_target_scale": utility_scale,
            "device": str(device),
            "journal": journal,
            "inventory": inventory,
            "split_policy": "whole_utc_day_source_disjoint_3plus_train_2_calibration_3_test",
            "split_groups": split_groups,
            "counts": {name: len(rows) for name, rows in partitions.items()},
            "profiled_isolated_median_latency_ms": latency_ms,
            "experts": [
                {
                    "expert_id": expert.expert_id,
                    "checkpoint": expert.checkpoint.name,
                    "checkpoint_sha256": expert.sha256,
                    "training_journal_sha256": expert.training_journal_sha256,
                    "input_context": min(expert.input_context, args.context),
                }
                for expert in experts
            ],
            "expert_training_provenance": expert_training_provenance,
            "files": file_hashes,
            "locked_test_must_not_be_used_for_training_or_calibration": True,
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (stage / "manifest.sha256").write_text(
            f"{_sha256(manifest_path)}  manifest.json\n", encoding="ascii"
        )
        shutil.move(str(stage), str(args.output))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--expert-run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=4_000)
    parser.add_argument("--calibration-samples", type=int, default=1_000)
    parser.add_argument("--test-samples", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--compute-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

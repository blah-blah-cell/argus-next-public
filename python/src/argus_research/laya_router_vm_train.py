"""DDP fine-tuning entry point for an Argus-prepared Laya router dataset.

Run this module under ``torchrun``.  It reads only ``train.jsonl`` and
``calibration.jsonl``; the locked test file is deliberately never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .laya_router_prepare import SCHEMA as PREPARATION_SCHEMA

LAYA_VERSION = "0.3.4"
EPISODE_PREPARATION_SCHEMA = "argus.laya-episode-router-preparation/v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optimizer_step_succeeded(scale_before: float, scale_after: float) -> bool:
    """Return false when GradScaler skipped the optimizer step after overflow."""
    return scale_after >= scale_before


def _load_manifest(prepared: Path) -> dict[str, Any]:
    manifest_path = prepared / "manifest.json"
    expected_path = prepared / "manifest.sha256"
    if not manifest_path.is_file() or not expected_path.is_file():
        raise FileNotFoundError("prepared manifest or detached hash is absent")
    expected = expected_path.read_text(encoding="ascii").split()[0]
    if _sha256(manifest_path) != expected:
        raise ValueError("prepared manifest SHA-256 differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") not in {PREPARATION_SCHEMA, EPISODE_PREPARATION_SCHEMA}:
        raise ValueError("prepared dataset schema differs")
    if manifest.get("locked_test_must_not_be_used_for_training_or_calibration") is not True:
        raise ValueError("prepared dataset does not lock its test partition")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("prepared dataset file hashes are absent")
    for name in ("train.jsonl", "calibration.jsonl"):
        path = prepared / name
        if not path.is_file() or _sha256(path) != files.get(name):
            raise ValueError(f"prepared {name} SHA-256 differs")
    return manifest


def _typed_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
                state = json.loads(row["state"])
                questions = json.loads(row["questions"])
                gold = json.loads(row["gold"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"invalid typed-decision row {line_number}") from error
            if set(questions) != set(gold):
                raise ValueError(f"typed-decision question/gold mismatch at row {line_number}")
            rows.append({"id": row["id"], "state": state, "questions": questions, "gold": gold})
    if not rows:
        raise ValueError("typed-decision file is empty")
    return rows


def _target(question: dict[str, Any], gold: dict[str, Any]) -> list[float]:
    question_type = question.get("type")
    criteria = question.get("criteria", {})
    probabilities = gold.get("probabilities")
    if not isinstance(probabilities, dict):
        raise ValueError("gold probabilities are absent")
    if question_type == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice criteria are invalid")
        keys = list(criteria)
    elif question_type == "noul":
        keys = ["false", "true"]
    elif question_type == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score criteria are invalid")
        keys = [str(index) for index in range(len(criteria))]
    else:
        raise ValueError("unknown Laya question type")
    values = [float(probabilities.get(key, 0.0)) for key in keys]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("gold probability is invalid")
    total = sum(values)
    if total <= 0:
        raise ValueError("gold probabilities have no positive mass")
    return [value / total for value in values]


def _build_items(rows: list[dict[str, Any]], tokenizer: Any, config: dict[str, Any], common: Any) -> list[dict[str, Any]]:
    items = []
    for row in rows:
        for question_id, question in row["questions"].items():
            question_type = question["type"]
            criteria = question.get("criteria", {})
            target = _target(question, row["gold"][question_id])
            sequence, markers = common.build_sequence(
                tokenizer,
                row["state"],
                {"t": question_type, "ins": question["instructions"], "crit": criteria},
                config["max_len"],
                config["head_max_len"],
            )
            rendered = common.render_options({"t": question_type, "crit": criteria})
            if len(markers) != len(rendered) or len(target) != len(markers):
                raise ValueError(f"Laya marker count differs for {row['id']}:{question_id}")
            items.append(
                {
                    "ids": sequence,
                    "markers": markers,
                    "qtype": common.QTYPES[question_type],
                    "target": target,
                    "label": target.index(max(target)),
                }
            )
    if not items:
        raise ValueError("no Laya training items were built")
    return items


def _collate(items: list[dict[str, Any]], pad_id: int) -> dict[str, Tensor]:
    count = len(items)
    length = max(len(item["ids"]) for item in items)
    options = max(len(item["markers"]) for item in items)
    ids = torch.full((count, length), pad_id, dtype=torch.long)
    attention = torch.zeros((count, length), dtype=torch.long)
    marker_positions = torch.zeros((count, options), dtype=torch.long)
    marker_mask = torch.zeros((count, options), dtype=torch.bool)
    target = torch.zeros((count, options), dtype=torch.float32)
    for index, item in enumerate(items):
        ids[index, : len(item["ids"])] = torch.tensor(item["ids"])
        attention[index, : len(item["ids"])] = 1
        markers = len(item["markers"])
        marker_positions[index, :markers] = torch.tensor(item["markers"])
        marker_mask[index, :markers] = True
        target[index, : len(item["target"])] = torch.tensor(item["target"])
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "marker_pos": marker_positions,
        "marker_mask": marker_mask,
        "target": target,
        "qtype": torch.tensor([item["qtype"] for item in items]),
    }


def _fit_temperature(samples: list[tuple[list[float], list[float]]]) -> float:
    if len(samples) < 10:
        raise ValueError("calibration question type has fewer than ten samples")
    width = max(len(logits) for logits, _ in samples)
    logits_tensor = torch.full((len(samples), width), -1e4)
    target_tensor = torch.zeros((len(samples), width))
    for index, (logits, target) in enumerate(samples):
        logits_tensor[index, : len(logits)] = torch.tensor(logits)
        target_tensor[index, : len(target)] = torch.tensor(target)
    log_temperature = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = -(
            target_tensor * torch.log_softmax(logits_tensor / log_temperature.exp(), -1)
        ).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().detach().clamp(0.1, 10.0))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Laya router fine-tuning requires CUDA")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    stage_output = args.output.with_name(f"{args.output.name}.partial")
    installed = importlib.metadata.version("laya")
    if installed != LAYA_VERSION:
        raise RuntimeError(f"Laya {LAYA_VERSION} is required, found {installed}")
    manifest = _load_manifest(args.prepared)
    episode_v2 = manifest["schema"] == EPISODE_PREPARATION_SCHEMA
    train_rows = _typed_rows(args.prepared / "train.jsonl")
    calibration_rows = _typed_rows(args.prepared / "calibration.jsonl")

    from laya import common  # type: ignore[import-not-found]
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    required = [
        args.base_model / "model.safetensors",
        args.base_model / "rl_agent_config.json",
        args.base_model / "encoder",
        args.base_model / "tokenizer",
    ]
    if any(not path.exists() for path in required):
        raise FileNotFoundError("local base Laya checkpoint is incomplete")
    config = json.loads((args.base_model / "rl_agent_config.json").read_text(encoding="utf-8"))
    gradient_checkpointing = not args.disable_gradient_checkpointing
    config["gradient_checkpointing"] = gradient_checkpointing
    config["max_tokens_per_batch"] = args.max_tokens_per_batch
    config["max_len"] = args.max_len
    config["head_max_len"] = args.head_max_len
    tokenizer = AutoTokenizer.from_pretrained(args.base_model / "tokenizer", local_files_only=True)
    train_items = _build_items(train_rows, tokenizer, config, common)
    calibration_items = _build_items(calibration_rows, tokenizer, config, common)

    if distributed:
        torch.distributed.init_process_group("nccl")
        output_state = [
            (args.output.exists() or stage_output.exists()) if rank == 0 else False
        ]
        torch.distributed.broadcast_object_list(output_state, src=0)
        if output_state[0]:
            raise FileExistsError(
                "output directory already exists; training artifacts are immutable"
            )
    elif args.output.exists() or stage_output.exists():
        raise FileExistsError(
            "output directory already exists; training artifacts are immutable"
        )
    if rank == 0:
        stage_output.mkdir(parents=True)
    if distributed:
        torch.distributed.barrier()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    model = common.build_model(config, encoder_dir=str(args.base_model / "encoder"))
    model.load_state_dict(load_file(args.base_model / "model.safetensors"), strict=True)
    if gradient_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.head_checkpointing = gradient_checkpointing
    model.to(device).train()
    wrapped: nn.Module = model
    if distributed:
        wrapped = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )
    encoder_parameters = [parameter for name, parameter in wrapped.named_parameters() if "encoder." in name]
    head_parameters = [parameter for name, parameter in wrapped.named_parameters() if "encoder." not in name]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_learning_rate},
            {"params": head_parameters, "lr": args.head_learning_rate},
        ],
        weight_decay=0.01,
    )
    per_rank = len(train_items) // world_size
    usable = per_rank * world_size
    if per_rank < args.micro_batch:
        raise ValueError("training set is too small for the requested world size and batch")
    local_items = train_items[:usable][rank::world_size]
    updates_per_epoch = math.ceil(len(local_items) / args.micro_batch / args.gradient_accumulation)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, updates_per_epoch * args.epochs), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda")
    progress_path = stage_output / "progress.jsonl"
    started = time.time()
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch + rank).shuffle(local_items)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * (
            epoch / max(1, args.epochs - 1)
        )
        for batch_index in range(0, len(local_items), args.micro_batch):
            chunk = local_items[batch_index : batch_index + args.micro_batch]
            batch = _collate(chunk, tokenizer.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, activation = wrapped(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            target = batch["target"].to(device)
            option_count = mask.sum(-1, keepdim=True).float()
            noise = torch.randn((args.group_size,) + logits.shape, device=device) * sigma * mask
            noise = (noise - noise.sum(-1, keepdim=True) / option_count) * mask
            sampled_logits = logits.detach().unsqueeze(0) + noise
            probabilities = torch.softmax(sampled_logits.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                rewards = common.proper_reward(
                    probabilities,
                    target.unsqueeze(0),
                    batch["qtype"].to(device),
                    mask,
                    w_sph=0.75,
                    w_rps=1.0,
                )
                advantage = rewards - rewards.mean(0, keepdim=True)
                advantage = advantage / (advantage.std() + 1e-6)
            log_probability = -(
                ((sampled_logits - logits.unsqueeze(0)).square() * mask).sum(-1)
                / (2 * sigma**2)
            )
            policy_loss = -(advantage * log_probability).mean()
            cross_entropy = -(
                target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
            ).sum(-1).mean()
            loss = (policy_loss + cross_entropy) / args.gradient_accumulation + 0.0 * activation.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite Laya router loss")
            scaler.scale(loss).backward()
            accumulation_step = batch_index // args.micro_batch + 1
            final_batch = batch_index + args.micro_batch >= len(local_items)
            if accumulation_step % args.gradient_accumulation == 0 or final_batch:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0)
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # GradScaler may suppress an optimizer update while it learns a
                # safe scale. Advancing the schedule on that skipped update both
                # drops an LR value and makes the result environment-dependent.
                if _optimizer_step_succeeded(scale_before_step, scaler.get_scale()):
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()) * args.gradient_accumulation)
        if rank == 0:
            record = {
                "epoch": epoch + 1,
                "mean_loss": sum(losses) / len(losses),
                "sigma": sigma,
                "elapsed_seconds": time.time() - started,
                "gpu_peak_bytes": torch.cuda.max_memory_allocated(device),
            }
            with progress_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")
        if distributed:
            torch.distributed.barrier()

    result: dict[str, Any] = {"state": "worker_complete", "rank": rank}
    if rank == 0:
        model.eval()
        predictions: dict[int, list[tuple[list[float], list[float]]]] = {0: [], 1: [], 2: []}
        with torch.inference_mode():
            for offset in range(0, len(calibration_items), args.calibration_batch):
                chunk = calibration_items[offset : offset + args.calibration_batch]
                batch = _collate(chunk, tokenizer.pad_token_id)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits, _ = model(
                        batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                        batch["marker_pos"].to(device),
                        batch["marker_mask"].to(device),
                        batch["qtype"].to(device),
                    )
                for index, item in enumerate(chunk):
                    width = len(item["markers"])
                    predictions[item["qtype"]].append(
                        (logits[index, :width].float().cpu().tolist(), item["target"])
                    )
        configured_temperatures = config.get("temperature", [1.0, 1.0, 1.0])
        if not isinstance(configured_temperatures, list) or len(configured_temperatures) != 3:
            configured_temperatures = [1.0, 1.0, 1.0]
        temperatures = [
            _fit_temperature(predictions[index])
            if predictions[index]
            else float(configured_temperatures[index])
            for index in range(3)
        ]
        weights = {name: value.half().contiguous().cpu() for name, value in model.state_dict().items()}
        save_file(weights, stage_output / "model.safetensors")
        model.encoder.config.save_pretrained(stage_output / "encoder")
        tokenizer.save_pretrained(stage_output / "tokenizer")
        config["fine_tuned"] = True
        config["model_name"] = (
            "argus-laya-episode-router-v2" if episode_v2 else "argus-laya-router-v1"
        )
        config["temperature"] = temperatures
        (stage_output / "rl_agent_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        provenance = {
            "schema": "argus.laya-router-training-provenance/v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "preparation_manifest_sha256": _sha256(args.prepared / "manifest.json"),
            "train_jsonl_sha256": manifest["files"]["train.jsonl"],
            "calibration_jsonl_sha256": manifest["files"]["calibration.jsonl"],
            "locked_test_jsonl_sha256": manifest["files"].get("test.jsonl"),
            "locked_test_opened_during_training": False,
            "expert_sha256": {
                expert["expert_id"]: expert["checkpoint_sha256"] for expert in manifest["experts"]
            },
            "split_groups": manifest.get("split_groups"),
            "train_journal": manifest.get("train_journal"),
            "calibration_journal": manifest.get("calibration_journal"),
            "epochs": args.epochs,
            "seed": args.seed,
            "world_size": world_size,
            "gradient_checkpointing": gradient_checkpointing,
            "temperatures": temperatures,
        }
        reports = stage_output / "reports"
        reports.mkdir()
        provenance_path = reports / "laya-router-source-disjoint.json"
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        admission = {
            "schema": "argus.laya-episode-router/v2" if episode_v2 else "argus.laya-window-router/v1",
            "feature_schema": "argus.causal-episode-evidence/v2" if episode_v2 else "argus.causal-window-evidence/v1",
            "question_schema": "argus.episode-router-questions/v2" if episode_v2 else "argus.laya-expert-routing-questions/v1",
            "task": "causal_episode_cost_aware_subset_routing" if episode_v2 else "cost_aware_forecast_expert_routing",
            "model_id": "argus.laya-episode-router-v2" if episode_v2 else "argus.laya-router-v1",
            "laya_version": LAYA_VERSION,
            "security_detector": False,
            "calibrated": True,
            "trained_out_of_fold": True,
            "calibration_source_disjoint": True,
            "expert_ids": [expert["expert_id"] for expert in manifest["experts"]],
            "top_k": 4 if episode_v2 else 2,
            "routes_expert_subsets": episode_v2,
            "minimum_selection_probability": 0.7,
            "abstain_probability": 0.6,
            "model_safetensors_sha256": _sha256(stage_output / "model.safetensors"),
            "training_provenance": "reports/laya-router-source-disjoint.json",
            "training_provenance_sha256": _sha256(provenance_path),
        }
        (stage_output / "argus-laya-router.json").write_text(
            json.dumps(admission, indent=2) + "\n", encoding="utf-8"
        )
        result = {"state": "complete", "admission": admission, "provenance": provenance}
    if distributed:
        torch.distributed.barrier()
    if rank == 0:
        os.replace(stage_output, args.output)
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--calibration-batch", type=int, default=16)
    parser.add_argument("--encoder-learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--sigma-start", type=float, default=0.4)
    parser.add_argument("--sigma-end", type=float, default=0.1)
    parser.add_argument("--max-tokens-per-batch", type=int, default=4096)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--head-max-len", type=int, default=256)
    parser.add_argument(
        "--disable-gradient-checkpointing",
        action="store_true",
        help="retain activations instead of recomputing them; faster but requires more VRAM",
    )
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    if min(
        args.epochs,
        args.micro_batch,
        args.gradient_accumulation,
        args.group_size,
        args.calibration_batch,
        args.max_tokens_per_batch,
        args.max_len,
        args.head_max_len,
    ) <= 0:
        parser.error("training counts and lengths must be positive")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

"""Bounded, source-disjoint zero-shot evaluation of a frozen AHEM bundle.

Input is an ARGUSLR2 journal produced from a separately audited capture. A
source contract and a frozen protocol bind the capture, journal, code, model,
and sampling rule. This module never downloads or trains on final data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import hashlib
import importlib.util
from pathlib import Path
import platform
import random

import numpy as np
import torch

from .ahem_consolidated import ConsolidatedAhem
from .ahem_final_access import begin_final_stage, claim_final_access, verify_final_access_claim
from .ahem_forecast_audit import causal_ewma, forecast_quality
from .ahem_inference_fast import FastConsolidatedAhem
from .ahem_phase_reader import project_phase_queries
from .ahem_population_audit import sha256, strict_records
from .ahem_reader_experiment import INPUTS, atomic_json
from .ahem_retrieval_audit import retrieval_quality
from .ahem_source_contract import authorize_role, validate_sources
from .ahem_startup_inline import InlineStartupAhem
from .flow_v3_protocol import condition_record
from .ahem_v6 import FlowLaneLstmNextPacket
from .flow_v2_features import FEATURE_CONTRACT_SHA256


def _sha(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_protocol(protocol: dict, contract: dict, *, code_sha256: str) -> None:
    required = {"schema", "source_manifest_sha256", "pack_manifest_sha256",
                "evaluator_sha256", "converter_sha256", "bundle_sha256",
                "calibration_sha256", "sample_count_per_source", "sampling_seed",
                "horizon_packets", "segment_packets", "projection_batch", "inference_batch", "source_ids",
                "capture_reader_sha256", "capture_policy", "readout_controls"}
    if (set(protocol) != required or protocol["schema"] != "argus.ahem-zero-shot/v2"
            or protocol["source_manifest_sha256"] != validate_sources(contract)["manifest_sha256"]
            or protocol["evaluator_sha256"] != code_sha256
            or protocol["converter_sha256"] != sha256(Path(__file__).with_name("ahem_external_pcap.py"))
            or protocol["capture_reader_sha256"] != sha256(Path(__file__).with_name("ahem_external_capture.py"))
            or protocol["capture_policy"] != "complete_ethernet_frames_only"
            or protocol["readout_controls"] != ["recency_same_readout", "uniform_same_readout"]
            or any(not _sha(protocol[key]) for key in ("pack_manifest_sha256", "converter_sha256",
                                                        "bundle_sha256", "calibration_sha256"))
            or type(protocol["sample_count_per_source"]) is not int
            or protocol["sample_count_per_source"] <= 0
            or type(protocol["sampling_seed"]) is not int
            or protocol["horizon_packets"] != 10_000
            or type(protocol["segment_packets"]) is not int
            or not 30_001 <= protocol["segment_packets"] <= 1_000_000
            or any(type(protocol[key]) is not int or not 1 <= protocol[key] <= 256
                   for key in ("projection_batch", "inference_batch"))):
        raise ValueError("unfrozen or invalid zero-shot protocol")
    final_ids = [entry["source_id"] for entry in contract["sources"] if entry["role"] == "final"]
    if protocol["source_ids"] != final_ids or len(final_ids) != len(set(final_ids)):
        raise ValueError("final source identities differ from frozen contract")
    for entry in contract["sources"]:
        capture = entry.get("capture_name")
        if (not isinstance(capture, str) or not capture or capture in (".", "..")
                or "/" in capture or "\\" in capture or ":" in capture
                or Path(capture).name != capture):
            raise ValueError("single capture filename required in source contract")
        if (not isinstance(entry.get("journal_name"), str)
                or Path(entry["journal_name"]).name != entry["journal_name"]
                or "/" in entry["journal_name"] or "\\" in entry["journal_name"]
                or not entry["journal_name"].endswith(".arguslr")):
            raise ValueError("journal binding missing from source contract")
        if "journal_sha256" in entry and not _sha(entry["journal_sha256"]):
            raise ValueError("invalid optional predeclared journal hash")
    finals = [entry for entry in contract["sources"] if entry["role"] == "final"]
    for field in ("journal_name", "sha256", "parent_capture_id"):
        values = [entry[field] for entry in finals]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate final capture declaration: {field}")


def load_controls(pack_root: Path, device: str) -> tuple[dict, dict]:
    spec = importlib.util.spec_from_file_location("ahem_frozen_dual_control",
                                                 pack_root / "controls/dual_lane_lstm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    controls, identities = {}, {}
    for seed in (20260906, 20260907, 20260908):
        for family, constructor in (("flow", FlowLaneLstmNextPacket),
                                    ("dual", module.DualLaneLstm)):
            path = pack_root / f"controls/{family}-{seed}.pt"
            saved = torch.load(path, weights_only=True, map_location="cpu")
            control = constructor().to(device).eval().requires_grad_(False)
            control.load_state_dict(saved["state_dict"], strict=True)
            if any(not torch.isfinite(value).all() for value in control.state_dict().values()):
                raise ValueError("nonfinite frozen comparator")
            name = f"{family}_lstm_{seed}"
            controls[name] = control
            identities[name] = {"sha256": sha256(path), "seed": seed,
                                "parameters": sum(p.numel() for p in control.parameters()),
                                "training": "saved CSE-CIC-IDS2018 checkpoint; zero-shot on new source"}
    return controls, identities


@torch.no_grad()
def predict_controls(controls: dict, record, anchors: list[int], device: str) -> dict:
    lanes, global_lanes = [], []
    for anchor in anchors:
        start = max(0, anchor - 9999)
        window = record.features[start:anchor + 1]
        ids = record.flow_ids[start:anchor + 1]
        lanes.append(window[ids == record.flow_ids[anchor]])
        global_lane = record.features[max(0, anchor - 127):anchor + 1]
        global_lanes.append(torch.nn.functional.pad(global_lane, (0, 0, 128 - len(global_lane), 0)))
    lengths = torch.tensor([len(lane) for lane in lanes])
    flow = torch.nn.utils.rnn.pad_sequence(lanes, batch_first=True).to(device)
    globals_ = torch.stack(global_lanes).to(device)
    result = {}
    for name, control in controls.items():
        if name.startswith("dual_"):
            prediction = control.forward_lanes(globals_, flow, lengths)
        else:
            packed = torch.nn.utils.rnn.pack_padded_sequence(flow, lengths, batch_first=True,
                                                            enforce_sorted=False)
            _, (hidden, _) = control.encoder(packed)
            prediction = control.predict(hidden[-1])
        if not torch.isfinite(prediction).all():
            raise ValueError("nonfinite LSTM forecast")
        result[name] = prediction.cpu()
    return result


@torch.no_grad()
def predict_fixed_read(model: InlineStartupAhem, projected: dict, current: torch.Tensor,
                       first: int, last: int, *, policy: str) -> torch.Tensor:
    """Fixed weights with the identical frozen head/calibration, without refitting."""
    valid = projected["phase_valid"][first:last]
    logits = torch.zeros_like(valid, dtype=current.dtype)
    if policy == "recency":
        logits.scatter_(1, valid.long().argmax(-1, keepdim=True), 10_000)
    elif policy != "uniform":
        raise ValueError("unknown fixed-read policy")
    _, _, _, prediction = model.readout(logits, valid,
        projected["successor_dynamic"][first:last], current,
        projected["query_phase"][first:last])
    anchor = projected["anchor_ordinals"][first:last]
    if model.enabled:
        corrected = prediction + model.complement * (current - prediction)
        prediction = torch.where((anchor >= model.window_packets - 1)[:, None], prediction, corrected)
    return prediction


def select_coordinates(path: Path, count: int, seed: int) -> tuple[list[tuple[int, int]], dict]:
    """Uniform reservoir over every observed-successor anchor, without memory filtering."""
    rng = random.Random(seed)
    selected: list[tuple[int, int]] = []
    observed = population = records = startup_population = first_flow_population = 0
    for record_index, raw in enumerate(strict_records(path)):
        record = condition_record(raw)
        observed += len(record.features)
        eligible = torch.nonzero(record.next_same_flow >= 0).flatten().tolist()
        startup_population += sum(anchor < 127 for anchor in eligible)
        first_flow_population += sum(int(record.previous_same_flow[anchor]) < 0 for anchor in eligible)
        for anchor in eligible:
            population += 1
            coordinate = (record_index, anchor)
            if len(selected) < count:
                selected.append(coordinate)
            else:
                replacement = rng.randrange(population)
                if replacement < count:
                    selected[replacement] = coordinate
        records += 1
        if records % 20 == 0:
            print(json.dumps({"stage": "sampling", "records": records,
                              "observed_packets": observed, "successor_queries": population}), flush=True)
    if not population:
        raise ValueError("source has no observed-successor forecast coordinates")
    return sorted(selected), {"records": records, "observed_packets": observed,
                              "observed_successor_population": population,
                              "right_censored_packets": observed - population,
                              "short_global_history_population": startup_population,
                              "first_flow_query_population": first_flow_population,
                              "startup_coverage_warning": "Uniform samples may contain no startup queries; absence is not evidence of startup safety",
                              "sampled_queries": len(selected),
                              "sampling": "uniform_reservoir_over_observed_successor_anchors"}


@torch.no_grad()
def evaluate_journal(path: Path, model: InlineStartupAhem, *, samples: int, seed: int,
                     projection_batch: int, inference_batch: int, device: str,
                     controls: dict | None = None) -> dict:
    coordinates, population = select_coordinates(path, samples, seed)
    by_record: dict[int, list[int]] = defaultdict(list)
    for record_index, anchor in coordinates:
        by_record[record_index].append(anchor)
    collected: dict[str, list] = defaultdict(list)
    controls = controls or {}
    query_identities = []
    for record_index, raw in enumerate(strict_records(path)):
        anchors = by_record.get(record_index)
        if not anchors:
            continue
        record = condition_record(raw)
        features = record.features.numpy()
        ewma = causal_ewma(features, record.previous_same_flow.numpy())
        phase_cache: dict = {}
        for start in range(0, len(anchors), projection_batch):
            chosen = anchors[start:start + projection_batch]
            projected = project_phase_queries(model.reader, record, chosen,
                                              horizon_packets=10_000, phase_cache=phase_cache)
            for first in range(0, len(chosen), inference_batch):
                last = first + inference_batch
                part = chosen[first:last]
                tensors = [projected[name][first:last].to(device) for name in (*INPUTS,
                    "query_phase", "key_phase", "phase_valid")]
                current = record.features[part, :2].to(device)
                prediction, weights, _, _ = model(*tensors, current)
                target = record.features[record.next_same_flow[part], :2]
                collected["prediction"].append(prediction.cpu())
                for policy in ("recency", "uniform"):
                    collected[policy + "_same_readout"].append(
                        predict_fixed_read(model, projected, current, first, last, policy=policy).cpu())
                if controls:
                    for name, forecast in predict_controls(controls, record, part, device).items():
                        collected[name].append(forecast)
                collected["weights"].append(weights.cpu())
                collected["successor"].append(projected["successor_dynamic"][first:last].cpu())
                collected["valid"].append(projected["phase_valid"][first:last].cpu())
                collected["target"].append(target.cpu())
                collected["persistence"].append(record.features[part, :2].cpu())
                collected["ewma"].append(torch.as_tensor(ewma[part]).cpu())
                collected["startup"].extend(anchor < model.window_packets - 1 for anchor in part)
                collected["protocol"].extend("tcp" if features[anchor, 4] > .5 else
                                             "udp" if features[anchor, 5] > .5 else "other"
                                             for anchor in part)
                query_identities.extend({"record": record_index, "segment_id": record.segment_id,
                    "anchor": anchor, "target": int(record.next_same_flow[anchor]),
                    "flow_id": int(record.flow_ids[anchor])} for anchor in part)
        print(json.dumps({"stage": "record_scored", "record": record_index,
                          "sampled_queries": len(anchors)}), flush=True)
    if sum(len(chunk) for chunk in collected["prediction"]) != len(coordinates):
        raise ValueError("selected coordinates were not all evaluated")
    tensors = {name: torch.cat(collected[name]) for name in
               ("prediction", "weights", "successor", "valid", "target", "persistence", "ewma")}
    targets = tensors["target"].numpy()
    predictions = {name: tensors[name].numpy() for name in ("prediction", "persistence", "ewma")}
    predictions.update({name: torch.cat(collected[name]).numpy()
                        for name in ("recency_same_readout", "uniform_same_readout", *controls)})
    masks = {"startup": np.asarray(collected["startup"], dtype=bool),
             "no_memory": ~tensors["valid"].any(-1).numpy(),
             "one_slot": (tensors["valid"].sum(-1) == 1).numpy(),
             "multiple_slots": (tensors["valid"].sum(-1) >= 2).numpy()}
    masks.update({f"protocol_{name}": np.asarray(collected["protocol"]) == name
                  for name in ("tcp", "udp", "other")})
    quality = {name: forecast_quality(value, targets) for name, value in predictions.items()}
    return {"population": population, "forecast": quality,
            "comparisons": {name: {"relative_forecast_loss_reduction":
                1 - quality["prediction"]["weighted_smooth_l1"] / metrics["weighted_smooth_l1"]
                if metrics["weighted_smooth_l1"] > 0 else None}
                for name, metrics in quality.items() if name != "prediction"},
            "paired_evidence": {"coordinates": query_identities,
                "coordinate_sha256": hashlib.sha256(json.dumps(query_identities, sort_keys=True,
                                                               separators=(",", ":")).encode()).hexdigest(),
                "targets_log1p": targets.tolist(),
                "predictions_log1p": {name: value.tolist() for name, value in predictions.items()}},
            "retrieval": retrieval_quality(tensors["successor"], tensors["target"],
                                           tensors["valid"], tensors["weights"]),
            "strata": {name: {"queries": int(mask.sum()), "forecast":
                       ({model_name: forecast_quality(value[mask], targets[mask])
                         for model_name, value in predictions.items()} if mask.any() else None)}
                       for name, mask in masks.items()},
            "limitations": ["Observed-successor population excludes right-censored last flow packets",
                            "Journal record boundaries reset flow history",
                            "Dual-lane warmup uses zero padding, an inference extension outside full-window training",
                            "Recency returns the current packet outcome and is a persistence-equivalence check",
                            "Uniform/recency controls preserve the learned-read-trained head, without refitting",
                            "No inference latency or intrusion-detection accuracy claim"]}


def _journal_path(root: Path, name: str) -> Path:
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("journal escapes declared root")
    return path


def verify_pack(root: Path, expected_manifest_hash: str) -> None:
    manifest_path = root / "MANIFEST.json"
    if sha256(manifest_path) != expected_manifest_hash:
        raise ValueError("frozen pack manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
            raise ValueError(f"pack member hash mismatch: {name}")
    # Prevent an editable local checkout from silently executing another
    # evaluator/converter/access guard alongside an otherwise intact archive.
    for filename in ("ahem_zero_shot.py", "ahem_external_pcap.py", "ahem_final_access.py",
                     "ahem_external_capture.py"):
        name = "python/src/argus_research/" + filename
        if manifest["files"].get(name) != sha256(Path(__file__).with_name(filename)):
            raise ValueError(f"executing code differs from frozen pack: {filename}")


def freeze_protocol(contract_path: Path, pack_root: Path, output: Path, *,
                    samples: int, seed: int, projection_batch: int = 16,
                    inference_batch: int = 16) -> dict:
    """Freeze code/model/source identities and the exact uniform sampling rule."""
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError("immutable protocol already exists")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    binding = validate_sources(contract)
    protocol = {"schema": "argus.ahem-zero-shot/v2",
                "source_manifest_sha256": binding["manifest_sha256"],
                "pack_manifest_sha256": sha256(pack_root / "MANIFEST.json"),
                "evaluator_sha256": sha256(Path(__file__)),
                "converter_sha256": sha256(Path(__file__).with_name("ahem_external_pcap.py")),
                "capture_reader_sha256": sha256(Path(__file__).with_name("ahem_external_capture.py")),
                "capture_policy": "complete_ethernet_frames_only",
                "readout_controls": ["recency_same_readout", "uniform_same_readout"],
                "bundle_sha256": sha256(pack_root / "model/consolidated.pt"),
                "calibration_sha256": sha256(pack_root / "model/startup-calibration.pt"),
                "sample_count_per_source": samples, "sampling_seed": seed,
                "horizon_packets": 10_000, "segment_packets": 250_000,
                "projection_batch": projection_batch,
                "inference_batch": inference_batch,
                "source_ids": [entry["source_id"] for entry in contract["sources"]
                               if entry["role"] == "final"]}
    validate_protocol(protocol, contract, code_sha256=sha256(Path(__file__)))
    verify_pack(pack_root, protocol["pack_manifest_sha256"])
    atomic_json(protocol, output)
    return protocol


def claim_protocol(protocol_path: Path, contract_path: Path, pack_root: Path,
                   ledger: Path) -> dict:
    """Claim final access once, immediately before conversion opens raw captures."""
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol, contract, code_sha256=sha256(Path(__file__)))
    verify_pack(pack_root, protocol["pack_manifest_sha256"])
    if (sha256(pack_root / "model/consolidated.pt") != protocol["bundle_sha256"]
            or sha256(pack_root / "model/startup-calibration.pt") != protocol["calibration_sha256"]):
        raise ValueError("model hashes changed after protocol freeze")
    return claim_final_access(ledger, checkpoint_sha256=protocol["bundle_sha256"],
                              protocol_sha256=sha256(protocol_path),
                              source_manifest_sha256=sha256(contract_path),
                              source_ids=protocol["source_ids"])


def run(protocol_path: Path, contract_path: Path, pack_root: Path, journal_root: Path,
        output: Path, ledger: Path, *, device: str = "cpu") -> dict:
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError("immutable zero-shot result already exists")
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("requested execution device unavailable")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol_hash, contract_hash = sha256(protocol_path), sha256(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol, contract, code_sha256=sha256(Path(__file__)))
    verify_pack(pack_root, protocol["pack_manifest_sha256"])
    if (sha256(pack_root / "model/consolidated.pt") != protocol["bundle_sha256"]
            or sha256(pack_root / "model/startup-calibration.pt") != protocol["calibration_sha256"]):
        raise ValueError("frozen model or calibration hash mismatch")
    source_ids = protocol["source_ids"]
    entries = {entry["source_id"]: entry for entry in authorize_role(
        contract, "final", frozen_checkpoint_sha256=protocol["bundle_sha256"])}
    if set(entries) != set(source_ids):
        raise ValueError("final source set mismatch")
    model = ConsolidatedAhem.from_bundle(pack_root / "model/consolidated.pt", device)
    saved = torch.load(pack_root / "model/consolidated.pt", weights_only=True, map_location="cpu")
    calibration = torch.load(pack_root / "model/startup-calibration.pt", weights_only=True,
                             map_location="cpu")
    if calibration["binding"]["original_bundle_sha256"] != protocol["bundle_sha256"]:
        raise ValueError("startup calibration does not bind frozen model")
    model = InlineStartupAhem(FastConsolidatedAhem(model, saved["rho"].to(device)),
                              calibration["scale"])
    controls, control_identities = load_controls(pack_root, device)
    # Conversion must have been claimed *before* the raw final capture was hashed.
    verify_final_access_claim(ledger, checkpoint_sha256=protocol["bundle_sha256"],
                              protocol_sha256=protocol_hash,
                              source_manifest_sha256=contract_hash, source_ids=source_ids)
    begin_final_stage(ledger, stage="evaluation", destination=output)
    report = {"schema": "argus.ahem-zero-shot-result/v1", "study_role": "new_source_zero_shot",
              "protocol_sha256": protocol_hash, "source_manifest_sha256": contract_hash,
              "bundle_sha256": protocol["bundle_sha256"], "evaluator_sha256": protocol["evaluator_sha256"],
              "final_access_ledger_sha256": sha256(ledger), "controls": control_identities,
              "execution": {"device": device, "torch": str(torch.__version__),
                            "python": platform.python_version(), "platform": platform.platform(),
                            "threads": torch.get_num_threads(), "precision": "FP32; TF32 disabled",
                            "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name() if device == "cuda" else None},
              "sources": {}}
    try:
        for source_id in source_ids:
            entry = entries[source_id]
            journal = _journal_path(journal_root, entry["journal_name"])
            journal_before = journal.stat()
            journal_hash = sha256(journal)
            if "journal_sha256" in entry and journal_hash != entry["journal_sha256"]:
                raise ValueError(f"final journal hash mismatch: {source_id}")
            sidecar_path = journal.with_suffix(journal.suffix + ".manifest.json")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if (sidecar.get("schema") != "argus.real-pcap-journal/v4"
                    or sidecar.get("synthetic_data") is not False
                    or sidecar.get("archive", {}).get("sha256") != entry["sha256"]
                    or sidecar.get("archive", {}).get("source_uri") != entry["url"]
                    or sidecar.get("journal", {}).get("sha256") != journal_hash
                    or sidecar.get("journal", {}).get("feature_dimension") != 12
                    or sidecar.get("journal", {}).get("feature_contract_sha256") != FEATURE_CONTRACT_SHA256
                    or sidecar.get("conversion_code_sha256") != protocol["converter_sha256"]
                    or sidecar.get("capture_reader_sha256") != protocol["capture_reader_sha256"]
                    or sidecar.get("capture_policy") != protocol["capture_policy"]
                    or sidecar.get("protocol_sha256") != sha256(protocol_path)
                    or sidecar.get("ledger_sha256") != sha256(ledger)):
                raise ValueError(f"conversion provenance mismatch: {source_id}")
            report["sources"][source_id] = {"journal_sha256": journal_hash,
                "conversion_manifest_sha256": sha256(sidecar_path),
                **evaluate_journal(journal, model, samples=protocol["sample_count_per_source"],
                                   seed=protocol["sampling_seed"],
                                   projection_batch=protocol["projection_batch"],
                                   inference_batch=protocol["inference_batch"], device=device,
                                   controls=controls)}
            print(json.dumps({"stage": "source_complete", "source_id": source_id,
                              "queries": report["sources"][source_id]["population"]["sampled_queries"]}), flush=True)
            journal_after = journal.stat()
            if (journal_before.st_size, journal_before.st_mtime_ns) != (
                    journal_after.st_size, journal_after.st_mtime_ns):
                raise ValueError("final journal changed during evaluation")
            atomic_json(report, output.with_name(output.name + ".progress.json"))
        if sha256(protocol_path) != protocol_hash or sha256(contract_path) != contract_hash:
            raise ValueError("protocol or source contract changed during evaluation")
        verify_pack(pack_root, protocol["pack_manifest_sha256"])
    except Exception as error:
        # Keep the one-use access claim and any progress; never silently retry.
        atomic_json({"error_type": type(error).__name__, "error": str(error),
                     "protocol_sha256": protocol_hash,
                     "ledger_sha256": sha256(ledger)},
                    output.with_name(output.name + ".failure.json"))
        raise
    atomic_json(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("freeze", "claim", "evaluate"):
        command = commands.add_parser(action)
        for name in (("contract", "pack_root", "output") if action == "freeze" else
                     ("protocol", "contract", "pack_root", "ledger") if action == "claim" else
                     ("protocol", "contract", "pack_root", "journal_root", "output", "ledger")):
            command.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
        if action == "freeze":
            command.add_argument("--samples", type=int, required=True)
            command.add_argument("--seed", type=int, required=True)
            command.add_argument("--projection-batch", type=int, default=16)
            command.add_argument("--inference-batch", type=int, default=16)
        if action == "evaluate":
            command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args()
    if arguments.action == "freeze":
        result = freeze_protocol(arguments.contract, arguments.pack_root, arguments.output,
                                 samples=arguments.samples, seed=arguments.seed,
                                 projection_batch=arguments.projection_batch,
                                 inference_batch=arguments.inference_batch)
        print(json.dumps({"protocol": str(arguments.output), "source_ids": result["source_ids"]}))
    elif arguments.action == "claim":
        print(json.dumps(claim_protocol(arguments.protocol, arguments.contract,
                                        arguments.pack_root, arguments.ledger)))
    else:
        result = run(arguments.protocol, arguments.contract, arguments.pack_root,
                     arguments.journal_root, arguments.output, arguments.ledger,
                     device=arguments.device)
        print(json.dumps({"sources": list(result["sources"]), "output": str(arguments.output)}))


if __name__ == "__main__":
    main()

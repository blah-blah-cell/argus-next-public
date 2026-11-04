"""Development/engineering replay audit with explicit source selection and coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .ahem_bridge_events import AHEM_NAME, LSTM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_forecast_audit import causal_ewma, forecast_quality
from .ahem_retrieval_audit import query_coverage, retrieval_quality
from .ahem_v6_stream import AhemV6Stream, FlowLaneLstmStream
from .continual_learning import MAGIC_V2, FEATURE_DIM, V2_PACKET_BYTES, SequenceRecordV2
from .flow_v3_protocol import condition_record
from .flow_v6_final_evaluate import _load_frozen_checkpoint
from .flow_v6_objective import dynamic_loss


def aggregate_retrieval(rows):
    report = {name: sum(row[name] for row in rows)
              for name in ("queries", "without_memory", "single_candidate", "multiple_candidates", "consequential_queries")}
    report["memory_coverage"] = 1 - report["without_memory"] / report["queries"]
    for name in rows[0]:
        if name in report or name in {"schema", "tie_tolerance", "consequential_regret_threshold"}:
            continue
        values = [row[name] for row in rows if row[name] is not None]
        report[name] = sum(values) / len(values) if values else None
    return report


def retention_headroom(record, anchor: int, target, *, horizon: int = 10_000, capacity: int = 16):
    """Compare retained FIFO oracle to all completed history, offline only."""
    oldest = max(0, anchor - horizon + 1)
    positions = [anchor]
    cursor = anchor
    while int(record.previous_same_flow[cursor]) >= oldest:
        cursor = int(record.previous_same_flow[cursor])
        positions.append(cursor)
    successors = positions[:-2]
    if not successors:
        return {"complete_candidates": 0, "retained_candidates": 0,
                "full_history_oracle_loss": None, "retained_oracle_loss": None,
                "retention_regret": None}
    outcomes = record.features[successors, :2]
    losses = dynamic_loss(outcomes, target.expand_as(outcomes), reduction="none")
    full, retained = float(losses.min()), float(losses[:capacity].min())
    return {"complete_candidates": len(successors), "retained_candidates": min(capacity, len(successors)),
            "full_history_oracle_loss": full, "retained_oracle_loss": retained,
            "retention_regret": retained - full}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_records(path: Path) -> Iterator[SequenceRecordV2]:
    """Reject damaged tails rather than silently treating a prefix as a corpus."""
    with path.open("rb") as source:
        if source.read(8) != MAGIC_V2:
            raise ValueError(f"{path}: expected ARGUSLR2 journal")
        while True:
            count_bytes = source.read(4)
            if not count_bytes:
                return
            identity = source.read(32)
            digest = source.read(32)
            if len(count_bytes) != 4 or len(identity) != 32 or len(digest) != 32:
                raise ValueError(f"{path}: truncated record header")
            count = struct.unpack("<I", count_bytes)[0]
            if not 0 < count <= 1_000_000:
                raise ValueError(f"{path}: invalid record size")
            payload = source.read(count * V2_PACKET_BYTES)
            if len(payload) != count * V2_PACKET_BYTES or hashlib.sha256(payload).digest() != digest:
                raise ValueError(f"{path}: truncated or checksum-invalid record payload")
            feature_end = count * FEATURE_DIM * 4
            int_bytes = count * 8
            values = np.frombuffer(payload[:feature_end], dtype="<f4").copy().reshape(count, FEATURE_DIM)
            integers = [np.frombuffer(payload[feature_end + offset * int_bytes:
                                               feature_end + (offset + 1) * int_bytes], dtype="<u8")
                        for offset in range(3)]
            if not np.isfinite(values).all() or any(np.any(column > np.iinfo(np.int64).max) for column in integers):
                raise ValueError(f"{path}: invalid features or unsigned metadata overflow")
            flows, timestamps, ordinals = [column.astype(np.int64) for column in integers]
            if np.any(timestamps[1:] < timestamps[:-1]) or np.any(ordinals[1:] <= ordinals[:-1]):
                raise ValueError(f"{path}: nonmonotonic timestamps or ordinals")
            yield SequenceRecordV2(path.stem, identity.hex(), torch.from_numpy(values),
                                   torch.from_numpy(flows), torch.from_numpy(timestamps),
                                   torch.from_numpy(ordinals))


def select_queries(records, count: int, seed: int):
    """Uniform sample across observed-successor anchors, without memory filtering."""
    pools = [np.flatnonzero(record.next_same_flow.numpy() >= 0) for record in records]
    totals = np.cumsum([len(pool) for pool in pools])
    population = int(totals[-1]) if len(totals) else 0
    selected = random.Random(seed).sample(range(population), min(count, population))
    coordinates = []
    for index in selected:
        record_index = int(np.searchsorted(totals, index, side="right"))
        previous = int(totals[record_index - 1]) if record_index else 0
        coordinates.append((record_index, int(pools[record_index][index - previous])))
    return sorted(coordinates)


def run(args):
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".partial").exists():
        raise FileExistsError("population audit output already exists")
    if args.samples <= 0 or args.threads <= 0:
        raise ValueError("sample and CPU-thread counts must be positive")
    if len(set(args.sources)) != len(args.sources):
        raise ValueError("source selection contains duplicates")
    paths = []
    for name in args.sources:
        if Path(name).name != name or not name.endswith(".arguslr"):
            raise ValueError("sources must be explicit journal filenames")
        if any(token in name.lower() for token in ("thursday-22", "friday-23", "cic-ids2017")):
            raise ValueError("this development runner does not open original final-test journals")
        path = args.journal / name
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    result = {"schema": "argus.ahem-population-audit/v1", "role": args.role,
              "publication_eligible": False, "sampling": "uniform_observed_successor_per_source",
              "samples_per_source": args.samples, "sampling_seed": args.sampling_seed,
              "warmup_prediction_contract": "emit_from_first_packet_fixed_10000_gap_normalization",
              "timing_claim": False, "sources": [], "checkpoint_identities": {}}
    models = {}
    for seed in args.seeds:
        models[seed] = []
        for name in (AHEM_NAME, LSTM_NAME):
            model, digest, identity = _load_frozen_checkpoint(
                args.checkpoints, seed, name, EXPECTED_CHECKPOINT_SHA256[seed][name]
            )
            models[seed].append(model.to(device))
            result["checkpoint_identities"][f"{seed}/{name}"] = {"sha256": digest, **identity}
    seen_hashes = set()
    for path in paths:
        digest = sha256(path)
        if digest in seen_hashes:
            raise ValueError("duplicate journal contents")
        seen_hashes.add(digest)
        raw = list(strict_records(path))
        if not raw:
            raise ValueError(f"{path}: no complete records")
        records = [condition_record(record) for record in raw]
        coordinates = select_queries(records, args.samples, args.sampling_seed)
        if not coordinates:
            raise ValueError(f"{path}: no observed-successor queries")
        source_report = {"path": path.name, "sha256": digest, "records": len(raw),
                         "coverage_by_record": [], "models": [], "coordinates": []}
        for record in records:
            coverage = query_coverage(record.previous_same_flow.numpy(), record.next_same_flow.numpy())
            source_report["coverage_by_record"].append({"id": record.segment_id, **coverage})
        # Baselines and targets are calculated on exactly the same coordinates.
        targets, persistence, ewma, protocol, retention = [], [], [], [], []
        for record_index, anchor in coordinates:
            record = records[record_index]
            target = int(record.next_same_flow[anchor])
            targets.append(record.features[target, :2].numpy())
            persistence.append(record.features[anchor, :2].numpy())
            retention.append(retention_headroom(record, anchor, record.features[target, :2]))
            protocol.append("tcp" if record.features[anchor, 4] > 0.5 else "udp" if record.features[anchor, 5] > 0.5 else "other")
            source_report["coordinates"].append({"record": record_index, "anchor": anchor,
                "target": target, "flow": int(record.flow_ids[anchor])})
        cached_ewma = {index: causal_ewma(records[index].features.numpy(), records[index].previous_same_flow.numpy())
                       for index in {index for index, _ in coordinates}}
        ewma = [cached_ewma[index][anchor] for index, anchor in coordinates]
        targets = np.asarray(targets)
        source_report["baselines"] = {"persistence": forecast_quality(persistence, targets),
                                       "ewma_alpha_0_2": forecast_quality(ewma, targets)}
        source_report["retention_per_query"] = retention
        for seed, (ahem, lstm) in models.items():
            predictions, references, retrievals = [], [], []
            for record_index in sorted({index for index, _ in coordinates}):
                record = records[record_index]
                selected = {anchor for index, anchor in coordinates if index == record_index}
                capture = f"{path.name}:{record_index}"
                stream = AhemV6Stream(ahem, capture, emit_warmup=True)
                reference = FlowLaneLstmStream(lstm, capture)
                features = record.features.to(device)
                for ordinal in range(max(selected) + 1):
                    packet = features[ordinal]
                    flow = int(record.flow_ids[ordinal])
                    output = stream.step(packet, flow, ordinal, capture_id=capture)
                    reference_output = reference.step(packet, flow, ordinal, capture_id=capture)
                    if ordinal not in selected:
                        continue
                    target = record.features[int(record.next_same_flow[ordinal]), :2].to(device)[None]
                    predictions.append(output.prediction[0].cpu().numpy())
                    references.append(reference_output.cpu().numpy())
                    retrievals.append(retrieval_quality(output.successor_dynamic, target, output.valid,
                                                        output.retrieval_weights))
            source_report["models"].append({"seed": seed,
                "ahem": forecast_quality(predictions, targets), "flow_lstm": forecast_quality(references, targets),
                "retrieval_per_query": retrievals,
                "retrieval_summary": aggregate_retrieval(retrievals),
                "forecast_strata": {
                    name: {"queries": int(mask.sum()),
                           "ahem": forecast_quality(np.asarray(predictions)[mask], targets[mask]),
                           "flow_lstm": forecast_quality(np.asarray(references)[mask], targets[mask])}
                    for name, mask in (
                        [(f"protocol_{kind}", np.asarray(protocol) == kind) for kind in sorted(set(protocol))]
                        + [("no_memory", np.asarray([row["without_memory"] > 0 for row in retrievals])),
                           ("single_slot", np.asarray([row["single_candidate"] > 0 for row in retrievals])),
                           ("multiple_slots", np.asarray([row["multiple_candidates"] > 0 for row in retrievals])),
                           ("global_warmup", np.asarray([anchor < 9999 for _, anchor in coordinates]))]
                    )
                    if mask.any()
                }})
        result["sources"].append(source_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("xb") as destination:
        destination.write((json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, args.output)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("engineering", "development"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, choices=tuple(EXPECTED_CHECKPOINT_SHA256), default=[20260906])
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--sampling-seed", type=int, default=20261002)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    result = run(parser.parse_args())
    print(json.dumps({"output_schema": result["schema"], "sources": len(result["sources"])}))


if __name__ == "__main__":
    main()

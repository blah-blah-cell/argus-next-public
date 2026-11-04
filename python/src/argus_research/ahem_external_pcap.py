"""Convert one hash-bound external PCAP/PCAPNG(.gz) after final access is claimed."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path

from .ahem_final_access import begin_final_stage, verify_final_access_claim
from .ahem_external_capture import full_ethernet_packets
from .ahem_reader_experiment import atomic_json
from .ahem_population_audit import sha256
from .ahem_source_contract import validate_sources
from .cse_pcap_to_journal import (
    ConversionStats, FlowGenerationTracker, JournalWriter, SegmentBuffer,
    TimestampNormalizer, _features, _parse_packet,
)
from .continual_learning import FEATURE_DIM
from .flow_v2_features import FEATURE_CONTRACT_SHA256, FEATURE_SCHEMA


def convert_source(protocol_path: Path, contract_path: Path, ledger_path: Path,
                   capture_root: Path, journal_root: Path, source_id: str) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (validate_sources(contract)["manifest_sha256"] != protocol.get("source_manifest_sha256")
            or protocol.get("schema") != "argus.ahem-zero-shot/v2"
            or protocol.get("capture_policy") != "complete_ethernet_frames_only"
            or protocol.get("capture_reader_sha256") != sha256(Path(__file__).with_name("ahem_external_capture.py"))
            or protocol.get("converter_sha256") != sha256(Path(__file__))):
        raise ValueError("converter/source protocol binding mismatch")
    entry = next((row for row in contract["sources"] if row["source_id"] == source_id), None)
    if entry is None or entry["role"] != "final" or source_id not in protocol["source_ids"]:
        raise ValueError("source is not a frozen final capture")
    verify_final_access_claim(ledger_path, checkpoint_sha256=protocol["bundle_sha256"],
                              protocol_sha256=sha256(protocol_path),
                              source_manifest_sha256=sha256(contract_path),
                              source_ids=protocol["source_ids"])
    capture_name = entry.get("capture_name")
    journal_name = entry.get("journal_name")
    if (not isinstance(capture_name, str) or Path(capture_name).name != capture_name
            or not isinstance(journal_name, str) or Path(journal_name).name != journal_name
            or not journal_name.endswith(".arguslr")):
        raise ValueError("single capture/journal filename required")
    capture = (capture_root / capture_name).resolve(strict=True)
    if not capture.is_file() or not capture.is_relative_to(capture_root.resolve()):
        raise ValueError("capture must remain within declared staging root")
    journal_root.mkdir(parents=True, exist_ok=True)
    output = journal_root / journal_name
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or sidecar.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise FileExistsError("conversion output already exists; no implicit final-data retry")
    begin_final_stage(ledger_path, stage="conversion", source_id=source_id,
                      destination=output)
    before = capture.stat()
    # Hashing the final raw capture counts as access; the ledger already exists.
    capture_hash = sha256(capture)
    if capture_hash != entry["sha256"]:
        raise ValueError("frozen raw-capture SHA-256 mismatch")
    stats = ConversionStats()
    clock = TimestampNormalizer()
    flows = FlowGenerationTracker()
    chunk = SegmentBuffer()
    previous_timestamp_ns = None
    segment = 0
    writer = JournalWriter(output)
    try:
        opener = gzip.open if capture.name.lower().endswith(".gz") else open
        with opener(capture, "rb") as stream:
            for source_ordinal, (timestamp, frame) in enumerate(full_ethernet_packets(stream)):
                stats.source_records += 1
                try:
                    normalized_ns, rollback = clock.observe(float(timestamp))
                except ValueError as error:
                    stats.malformed_packets += 1
                    raise ValueError(f"invalid raw timestamp at record {source_ordinal}") from error
                if rollback:
                    stats.clock_rollbacks += 1
                    raise ValueError("raw capture timestamp rollback; preserve conversion evidence for review")
                try:
                    packet = _parse_packet(frame)
                except ValueError:
                    stats.malformed_packets += 1
                    continue
                if packet is None:
                    stats.unsupported_packets += 1
                    continue
                delta = (0.0 if previous_timestamp_ns is None else
                         (normalized_ns - previous_timestamp_ns) / 1_000_000_000)
                features = _features(packet, delta)
                if not all(math.isfinite(value) for value in features):
                    stats.malformed_packets += 1
                    continue
                flow = flows.observe(packet.flow_key, normalized_ns,
                                     closes_flow=packet.closes_flow)
                chunk.append(features, flow, normalized_ns, source_ordinal)
                previous_timestamp_ns = normalized_ns
                stats.accepted_packets += 1
                if chunk.count == protocol["segment_packets"]:
                    writer.append(f"{source_id}:{capture_hash}:segment-{segment}", chunk)
                    segment += 1
                    stats.journal_segments += 1
                    chunk = SegmentBuffer()
                    # Journal history already resets here; retired flow tuples
                    # must not accumulate for the remainder of a large capture.
                    flows = FlowGenerationTracker()
                    print(json.dumps({"stage": "conversion", "source_id": source_id,
                                      "segments": stats.journal_segments,
                                      "accepted_packets": stats.accepted_packets}), flush=True)
        if chunk.count:
            writer.append(f"{source_id}:{capture_hash}:segment-{segment}", chunk)
            stats.journal_segments += 1
        if not stats.accepted_packets:
            raise ValueError("no accepted packets in external capture")
        after = capture.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("raw capture changed during conversion")
        journal_hash = writer.finish()
    except Exception as error:
        writer.output.close()
        # Keep partial evidence on conversion failure; never silently restart.
        atomic_json({"source_id": source_id, "error_type": type(error).__name__,
                     "error": str(error), "stats": asdict(stats),
                     "ledger_sha256": sha256(ledger_path)},
                    output.with_name(output.name + ".failure.json"))
        raise
    result = {"schema": "argus.real-pcap-journal/v4", "dataset": source_id,
              "synthetic_data": False, "task": "causal_next_packet_forecast_only",
              "archive": {"name": capture.name, "source_uri": entry["url"],
                          "format": "pcap-or-pcapng-gzip" if capture.name.lower().endswith(".gz") else "pcap-or-pcapng",
                          "bytes": capture.stat().st_size, "sha256": capture_hash},
              "journal": {"name": output.name, "bytes": output.stat().st_size,
                          "sha256": journal_hash, "binary_format": "ARGUSLR2",
                          "feature_schema": FEATURE_SCHEMA,
                          "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
                          "feature_dimension": FEATURE_DIM,
                          "segment_packets": protocol["segment_packets"]},
              "stats": asdict(stats), "conversion_code_sha256": sha256(Path(__file__)),
              "capture_reader_sha256": sha256(Path(__file__).with_name("ahem_external_capture.py")),
              "capture_policy": "complete_ethernet_frames_only",
              "protocol_sha256": sha256(protocol_path),
              "ledger_sha256": sha256(ledger_path),
              "limitations": ["Flow continuity resets at journal segment boundaries",
                              "Only complete Ethernet frames; snapshot truncation is rejected",
                              "No payload content retained in ARGUSLR2 journal"]}
    atomic_json(result, sidecar)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "contract", "ledger", "capture_root", "journal_root"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    args = parser.parse_args()
    result = convert_source(args.protocol, args.contract, args.ledger,
                            args.capture_root, args.journal_root, args.source_id)
    print(json.dumps({"journal": result["journal"], "stats": result["stats"]}))


if __name__ == "__main__":
    main()

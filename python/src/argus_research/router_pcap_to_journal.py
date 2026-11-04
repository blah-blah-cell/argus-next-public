"""Convert one receipt-verified PCAP or PCAP.GZ into an ARGUSLR2 journal."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .cse_pcap_to_journal import (
    ConversionStats,
    FlowGenerationTracker,
    JournalWriter,
    SegmentBuffer,
    TimestampNormalizer,
    _features,
    _parse_packet,
    _reader,
    _sha256,
)
SCHEMA = "argus.router-pcap-journal/v1"
FEATURE_SCHEMA = "packet-causal-v1"
FEATURE_CONTRACT = {
    "schema": FEATURE_SCHEMA,
    "dimension": 12,
    "fields": [
        {"index": 0, "name": "frame_length", "unit": "log1p(bytes)"},
        {
            "index": 1,
            "name": "accepted_capture_interarrival",
            "unit": "log1p(seconds)",
            "clip_seconds": [0.0, 60.0],
            "first_packet": 0.0,
        },
        {"index": 2, "name": "is_ipv4", "unit": "boolean"},
        {"index": 3, "name": "is_ipv6", "unit": "boolean"},
        {"index": 4, "name": "is_tcp", "unit": "boolean"},
        {"index": 5, "name": "is_udp", "unit": "boolean"},
        {"index": 6, "name": "tcp_syn", "unit": "boolean"},
        {"index": 7, "name": "tcp_ack", "unit": "boolean"},
        {"index": 8, "name": "tcp_fin", "unit": "boolean"},
        {"index": 9, "name": "tcp_rst", "unit": "boolean"},
        {"index": 10, "name": "tcp_push", "unit": "boolean"},
        {"index": 11, "name": "tcp_urg", "unit": "boolean"},
    ],
    "availability": "current_and_causally_prior_packets_only",
    "endpoint_identifiers_in_features": False,
    "payload_bytes_in_features": False,
}
FEATURE_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(FEATURE_CONTRACT, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def convert_capture(
    capture: Path,
    output: Path,
    *,
    source_id: str,
    source_uri: str,
    expected_sha256: str | None = None,
    segment_packets: int = 250_000,
    minimum_record_packets: int = 32,
    max_packets: int | None = None,
) -> dict[str, object]:
    if not capture.is_file():
        raise FileNotFoundError(f"capture is absent: {capture}")
    if not source_id.strip() or not source_uri.strip():
        raise ValueError("source identity and URI must be non-empty")
    if not 32 <= minimum_record_packets <= segment_packets <= 1_000_000:
        raise ValueError("record packet bounds are invalid")
    if max_packets is not None and max_packets < minimum_record_packets:
        raise ValueError("max_packets is smaller than the minimum record")
    digest = _sha256(capture)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("capture SHA-256 differs from its receipt")

    stats = ConversionStats()
    writer = JournalWriter(output)
    clock = TimestampNormalizer()
    flows = FlowGenerationTracker()
    chunk = SegmentBuffer()
    segment = 0
    previous_timestamp_ns: int | None = None
    truncated_tail = False
    truncated_tail_error: str | None = None

    def publish() -> None:
        nonlocal chunk, segment
        writer.append(
            f"{source_id}:{digest}:segment-{segment}",
            chunk,
        )
        segment += 1
        stats.journal_segments += 1
        chunk = SegmentBuffer()

    compressed = capture.suffix.lower() == ".gz"
    opener = gzip.open(capture, "rb") if compressed else capture.open("rb")
    try:
        with opener as source:
            try:
                for source_ordinal, (timestamp, frame) in enumerate(_reader(source)):
                    stats.source_records += 1
                    if max_packets is not None and stats.accepted_packets >= max_packets:
                        break
                    try:
                        packet = _parse_packet(frame)
                    except ValueError:
                        stats.malformed_packets += 1
                        continue
                    if packet is None:
                        stats.unsupported_packets += 1
                        continue
                    try:
                        timestamp_ns, rollback = clock.observe(float(timestamp))
                    except ValueError:
                        stats.malformed_packets += 1
                        continue
                    stats.clock_rollbacks += int(rollback)
                    delta_seconds = (
                        0.0
                        if previous_timestamp_ns is None
                        else (timestamp_ns - previous_timestamp_ns) / 1_000_000_000.0
                    )
                    features = _features(packet, delta_seconds)
                    if not all(math.isfinite(value) for value in features):
                        stats.malformed_packets += 1
                        continue
                    flow = flows.observe(
                        packet.flow_key,
                        timestamp_ns,
                        closes_flow=packet.closes_flow,
                    )
                    chunk.append(features, flow, timestamp_ns, source_ordinal)
                    previous_timestamp_ns = timestamp_ns
                    stats.accepted_packets += 1
                    if chunk.count >= segment_packets:
                        publish()
            except (EOFError, ValueError) as error:
                truncated_tail = True
                truncated_tail_error = str(error)
                stats.truncated_capture_members += 1
        if chunk.count >= minimum_record_packets:
            publish()
        if stats.journal_segments == 0:
            raise ValueError("capture produced no complete router journal record")
        journal_sha256 = writer.finish()
    except Exception:
        writer.output.close()
        writer.temporary.unlink(missing_ok=True)
        raise

    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "converter_source_sha256": _sha256(Path(__file__)),
        "source_id": source_id,
        "source_uri": source_uri,
        "synthetic_data": False,
        "attack_labels_used": False,
        "raw_payload_retained": False,
        "capture": {
            "name": capture.name,
            "bytes": capture.stat().st_size,
            "sha256": digest,
            "compression": "gzip" if compressed else "none",
        },
        "journal": {
            "name": output.name,
            "bytes": output.stat().st_size,
            "sha256": journal_sha256,
            "binary_format": "ARGUSLR2",
            "feature_schema": FEATURE_SCHEMA,
            "feature_contract": FEATURE_CONTRACT,
            "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
            "flow_identity": "directional_5tuple_plus_generation",
            "timestamp_unit": "nanoseconds",
            "source_ordinal_scope": "capture",
            "segment_packets": segment_packets,
            "minimum_record_packets": minimum_record_packets,
        },
        "truncated_tail": truncated_tail,
        "truncated_tail_error": truncated_tail_error,
        "stats": asdict(stats),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--segment-packets", type=int, default=250_000)
    parser.add_argument("--minimum-record-packets", type=int, default=32)
    parser.add_argument("--max-packets", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            convert_capture(
                args.capture,
                args.output,
                source_id=args.source_id,
                source_uri=args.source_uri,
                expected_sha256=args.expected_sha256,
                segment_packets=args.segment_packets,
                minimum_record_packets=args.minimum_record_packets,
                max_packets=args.max_packets,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

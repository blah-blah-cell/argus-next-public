"""Strict, read-only original-order capture audit; no sorting or model access.

Run on the notebook holding the captures. This produces physical evidence, not
permission to use a source, causal-cache approval, or a claim of independence.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from .ahem_external_capture import full_ethernet_packets
from .cse_pcap_to_journal import MAX_I64, _parse_packet, _sha256


def audit_capture(
    path: str | Path,
    *,
    role: str,
    expected_sha256: str,
    record_limit: int | None = None,
) -> dict[str, Any]:
    """Reject incomplete frames/tails; measure, never fix, clock regressions.

    Ordinals are zero-based capture record indices. Float seconds are used for
    diagnostics only, not claimed to preserve nanosecond capture precision.
    """
    if role not in {"train", "development", "retention"}:
        raise ValueError("only explicit train/development/retention roles allowed")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_sha256)
    ):
        raise ValueError("expected SHA-256 must be 64 lowercase hex characters")
    if record_limit is not None and (type(record_limit) is not int or record_limit < 1):
        raise ValueError("record limit must be a positive integer")
    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise ValueError("capture must be a regular file")
    before = source.stat()
    if _sha256(source) != expected_sha256:
        raise ValueError("capture hash mismatch before packet access")
    counts = dict(
        records_scanned=0,
        supported_packets=0,
        unsupported_packets=0,
        malformed_packets=0,
        timestamp_rollbacks=0,
        records_below_prior_high_watermark=0,
        equal_adjacent_timestamps=0,
        frame_length_out_of_contract=0,
    )
    previous = first = last = high_watermark = None
    max_rollback = max_high_watermark_deficit = 0.0
    max_frame_bytes = 0
    rollback_examples: list[dict[str, float | int]] = []
    opener = gzip.open if source.name.lower().endswith(".gz") else open
    with opener(source, "rb") as stream:
        for ordinal, (timestamp, frame) in enumerate(full_ethernet_packets(stream)):
            if not math.isfinite(timestamp) or not 0 <= timestamp <= MAX_I64 / 1e9:
                raise ValueError("timestamp outside finite nonnegative journal range")
            counts["records_scanned"] += 1
            if first is None:
                first = timestamp
            if previous is not None:
                if timestamp < previous:
                    rollback = previous - timestamp
                    counts["timestamp_rollbacks"] += 1
                    max_rollback = max(max_rollback, rollback)
                    if len(rollback_examples) < 16:
                        rollback_examples.append(
                            dict(ordinal=ordinal, seconds=rollback)
                        )
                elif timestamp == previous:
                    counts["equal_adjacent_timestamps"] += 1
            if high_watermark is not None and timestamp < high_watermark:
                counts["records_below_prior_high_watermark"] += 1
                max_high_watermark_deficit = max(
                    max_high_watermark_deficit, high_watermark - timestamp
                )
            previous = last = timestamp
            high_watermark = (
                max(high_watermark, timestamp)
                if high_watermark is not None
                else timestamp
            )
            max_frame_bytes = max(max_frame_bytes, len(frame))
            try:
                packet = _parse_packet(frame)
            except ValueError:
                counts["malformed_packets"] += 1
            else:
                if packet is None:
                    counts["unsupported_packets"] += 1
                else:
                    counts["supported_packets"] += 1
                    counts["frame_length_out_of_contract"] += int(
                        packet.frame_length > 65535
                    )
            if record_limit is not None and counts["records_scanned"] >= record_limit:
                break
    after = source.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ) or _sha256(source) != expected_sha256:
        raise ValueError("capture changed during physical audit")
    complete = record_limit is None
    reasons = []
    if not complete:
        reasons.append("bounded scan; unseen tail is unaudited")
    if counts["timestamp_rollbacks"]:
        reasons.append("timestamp rollback; original-order conversion not admitted")
    if counts["frame_length_out_of_contract"]:
        reasons.append("supported frame length exceeds 65535-byte model contract")
    if not counts["supported_packets"]:
        reasons.append("no supported IPv4/IPv6 TCP/UDP packets")
    return {
        "schema": "argus.ahem-strict-source-audit/v1",
        "capture_name": source.name,
        "role": role,
        "capture_sha256": expected_sha256,
        "file_bytes": after.st_size,
        "scan_complete": complete,
        "record_limit": record_limit,
        "full_frame_validation": "strict captured/original lengths, Ethernet, structure and EOF within scanned region",
        "timestamp_first_seconds": first,
        "timestamp_last_seconds": last,
        "timestamp_high_watermark_seconds": high_watermark,
        "maximum_adjacent_rollback_seconds": max_rollback,
        "maximum_high_watermark_deficit_seconds": max_high_watermark_deficit,
        "rollback_examples_first_16": rollback_examples,
        "maximum_observed_frame_bytes": max_frame_bytes,
        **counts,
        "physical_original_order_eligible": not reasons,
        "blocking_reasons": reasons,
        "timestamp_policy": "original_capture_record_order; no repair or sorting",
        "training_launched": False,
        "protocol_approved": False,
        "limitations": "Physical audit only; no access-history, benign-label, near-duplicate, causal-cache, GPU or generalization verification",
    }


def save_report(path: Path, report: dict[str, Any]) -> str:
    """Exclusive report; a crash leaves only .partial, never a published prefix."""
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".partial")
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(path)
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument(
        "--role", choices=("train", "development", "retention"), required=True
    )
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--record-limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Refuse overwrite before doing a potentially long scan.
    if (
        args.output.exists()
        or args.output.with_name(args.output.name + ".partial").exists()
    ):
        raise FileExistsError(args.output)
    try:
        report = audit_capture(
            args.capture,
            role=args.role,
            expected_sha256=args.expected_sha256,
            record_limit=args.record_limit,
        )
    except (ValueError, OSError, EOFError) as error:
        report = {
            "schema": "argus.ahem-strict-source-audit-failure/v1",
            "capture_name": args.capture.name,
            "declared_capture_sha256": args.expected_sha256,
            "scan_complete": False,
            "physical_original_order_eligible": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "training_launched": False,
            "protocol_approved": False,
        }
    report_hash = save_report(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": report_hash,
                "physical_original_order_eligible": report[
                    "physical_original_order_eligible"
                ],
            }
        ),
        flush=True,
    )
    if not report["physical_original_order_eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

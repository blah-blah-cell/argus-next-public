"""Read-only physical audit for *authorized development* packet captures.

This deliberately does not open final-role data, infer near-duplicate families,
or claim that a valid manifest proves independent sources. Run on the notebook
where the captures live; never stage raw public datasets on the laptop merely
to audit them.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

from .cse_pcap_to_journal import _parse_packet, _reader, _sha256


def audit_development_capture(
    path: str | Path, *, role: str, expected_sha256: str,
    record_limit: int | None = None
) -> dict:
    """Hash and scan a PCAP/PCAPNG or gzip-compressed capture without editing it.

    A bounded scan is explicitly incomplete. No endpoint addresses, payloads or
    packet contents are persisted in the returned summary.
    """
    if role not in {"train", "development"}:
        raise ValueError("physical preflight cannot open a final-role capture")
    capture = Path(path).resolve(strict=True)
    if not capture.is_file():
        raise ValueError("capture path must be a regular file")
    stat_before = capture.stat()
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("expected SHA-256 must be 64 lowercase hex characters")
    if record_limit is not None and (type(record_limit) is not int or record_limit <= 0):
        raise ValueError("record limit must be positive")

    digest = _sha256(capture)
    if digest != expected_sha256:
        raise ValueError("capture byte hash does not match the frozen declaration")

    counts = {"records_scanned": 0, "supported_packets": 0,
              "unsupported_packets": 0, "malformed_packets": 0,
              "timestamp_rollbacks": 0}
    first_timestamp = last_timestamp = previous_timestamp = None
    opener = gzip.open if capture.name.lower().endswith(".gz") else open
    with opener(capture, "rb") as stream:
        for timestamp, frame in _reader(stream):
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError("capture contains invalid timestamp")
            counts["records_scanned"] += 1
            first_timestamp = timestamp if first_timestamp is None else first_timestamp
            if previous_timestamp is not None and timestamp < previous_timestamp:
                counts["timestamp_rollbacks"] += 1
            previous_timestamp = last_timestamp = timestamp
            try:
                packet = _parse_packet(frame)
            except ValueError:
                counts["malformed_packets"] += 1
            else:
                counts["supported_packets" if packet is not None else "unsupported_packets"] += 1
            if record_limit is not None and counts["records_scanned"] >= record_limit:
                break

    stat_after = capture.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns):
        raise ValueError("capture changed during physical audit")

    return {"schema": "argus.ahem-physical-source-audit/v1",
            "capture_name": capture.name, "file_bytes": capture.stat().st_size,
            "sha256": digest, "scan_complete": record_limit is None,
            "timestamp_first": first_timestamp, "timestamp_last": last_timestamp,
            "chronology_monotonic": counts["timestamp_rollbacks"] == 0,
            **counts,
            "limitations": "No prior-access proof, flow-family overlap test, final-role access or model evaluation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--role", choices=("train", "development"), required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--record-limit", type=int)
    args = parser.parse_args()
    result = audit_development_capture(
        args.capture, role=args.role, expected_sha256=args.expected_sha256,
        record_limit=args.record_limit)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

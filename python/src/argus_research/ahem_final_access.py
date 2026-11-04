"""One-use final-source access claim for a future frozen evaluator.

Claim *before* opening a final capture. A failed/interrupted run remains claimed;
recovery requires a separately documented scientific decision, never an implicit
retry that silently increases final-set access.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import hashlib
import os
from pathlib import Path


def _digest(value: str, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def claim_final_access(
    ledger_path: str | Path, *, checkpoint_sha256: str,
    protocol_sha256: str, source_manifest_sha256: str,
    source_ids: list[str],
) -> dict:
    """Exclusively create a durable claim; never replace an existing ledger."""
    hashes = {"checkpoint_sha256": _digest(checkpoint_sha256, "checkpoint"),
              "protocol_sha256": _digest(protocol_sha256, "protocol"),
              "source_manifest_sha256": _digest(source_manifest_sha256, "source manifest")}
    if (not isinstance(source_ids, list) or not source_ids
            or any(not isinstance(s, str) or not s.strip() for s in source_ids)
            or len(set(source_ids)) != len(source_ids)):
        raise ValueError("nonempty distinct source IDs required")
    path = Path(ledger_path)
    if path.suffix.lower() != ".json":
        raise ValueError("explicit JSON ledger path required")
    record = {"schema": "argus.ahem-final-access-claim/v1",
              "claimed_at_utc": datetime.now(UTC).isoformat(),
              "source_ids": source_ids, **hashes,
              "state": "claimed-before-final-open",
              "note": "No final bytes have been inspected by this function"}
    payload = json.dumps(record, sort_keys=True, allow_nan=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Claim creation is exclusive; stages have their own ledger-anchored guards.
    # fsync makes the claim durable before the caller opens final bytes.
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return record


def begin_final_stage(ledger_path: str | Path, *, stage: str,
                      destination: str | Path, source_id: str | None = None) -> Path:
    """One invocation per ledger/stage, regardless of destination filename.

    This protects accidental repeats under the same ledger; it is not a
    tamper-proof registry against deliberately copying/removing ledger files.
    Keep the marker on failure for an explicit recovery decision.
    """
    ledger = Path(ledger_path).resolve(strict=True)
    record = json.loads(ledger.read_text(encoding="utf-8"))
    if stage not in {"conversion", "evaluation"}:
        raise ValueError("unknown final-access stage")
    if stage == "conversion":
        if source_id not in record.get("source_ids", []):
            raise ValueError("conversion source is absent from access claim")
        suffix = ".conversion-" + hashlib.sha256(source_id.encode()).hexdigest()[:32]
    else:
        if source_id is not None:
            raise ValueError("evaluation covers the entire claimed source set")
        suffix = ".evaluation"
    marker = ledger.with_name(ledger.stem + suffix + ".started.json")
    with marker.open("xb") as stream:
        stream.write(json.dumps({"schema": "argus.final-stage/v1", "stage": stage,
            "source_id": source_id, "destination": str(Path(destination).resolve()),
            "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
            "started_at_utc": datetime.now(UTC).isoformat()}, allow_nan=False).encode())
        stream.flush()
        os.fsync(stream.fileno())
    return marker


def verify_final_access_claim(
    ledger_path: str | Path, *, checkpoint_sha256: str,
    protocol_sha256: str, source_manifest_sha256: str, source_ids: list[str],
) -> dict:
    """Require the exact durable pre-open claim before conversion/evaluation."""
    path = Path(ledger_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("schema") != "argus.ahem-final-access-claim/v1"
            or record.get("state") != "claimed-before-final-open"
            or record.get("checkpoint_sha256") != _digest(checkpoint_sha256, "checkpoint")
            or record.get("protocol_sha256") != _digest(protocol_sha256, "protocol")
            or record.get("source_manifest_sha256") != _digest(source_manifest_sha256, "source manifest")
            or record.get("source_ids") != source_ids):
        raise ValueError("final access claim does not match frozen experiment")
    return record

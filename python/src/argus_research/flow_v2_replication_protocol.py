"""Freeze untouched flow-v2 replication bytes and coordinates before training."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .continual_learning import flow_sequences_from_record, iter_sequence_records_v2
from .flow_v2_protocol import (
    FEATURE_SCHEMA,
    PREREGISTRATION_SHA256,
    PROTOCOL_SCHEMA,
    CompiledProtocol,
    _journal_manifest,
    deterministic_coordinates,
    write_manifest,
)
from .flow_v2_train import _canonical_sha256, _source_provenance

REPLICATION_PROTOCOL_SCHEMA = "argus.flow-v2-untouched-replication-protocol/v1"


def compile_replication_protocol(
    primary_protocol: dict[str, Any],
    journal: Path,
    source: str,
    *,
    context_packets: int = 10_000,
    samples: int = 512,
    seed: int = 20260827,
    source_provenance: dict[str, Any] | None = None,
) -> CompiledProtocol:
    if primary_protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("replication preflight requires a flow-v2 primary protocol")
    if primary_protocol.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("primary protocol is not bound to the pre-registration")
    if min(context_packets, samples) <= 0:
        raise ValueError("replication context and sample count must be positive")
    primary_hashes = {
        member["sha256"] for member in primary_protocol["journal"]["members"]
    }
    journal_manifest = _journal_manifest(journal, {source})
    if primary_hashes & {member["sha256"] for member in journal_manifest["members"]}:
        raise ValueError("replication source was already present in primary data")

    streams = []
    for record in iter_sequence_records_v2(journal):
        if record.source_file == source:
            streams.extend(flow_sequences_from_record(record))
    coordinates = deterministic_coordinates(streams, context_packets, samples, seed)
    coordinate_sha256 = hashlib.sha256(
        json.dumps(
            [asdict(value) for value in coordinates],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest = {
        "schema": REPLICATION_PROTOCOL_SCHEMA,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "feature_schema": FEATURE_SCHEMA,
        "primary_protocol_sha256": _canonical_sha256(primary_protocol),
        "source": source,
        "journal": journal_manifest,
        "context_packets": context_packets,
        "samples": samples,
        "seed": seed,
        "coordinate_sha256": coordinate_sha256,
        "source_absent_from_primary_journal": True,
        "source_provenance": source_provenance,
    }
    return CompiledProtocol(
        partitions={"replication": streams},
        coordinates={"replication": coordinates},
        manifest=manifest,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-protocol", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--source", default="Friday-WorkingHours")
    parser.add_argument("--context", type=int, default=10_000)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty-source", action="store_true")
    args = parser.parse_args()
    source_provenance = _source_provenance()
    if not args.allow_dirty_source and (
        source_provenance.get("available") is not True
        or source_provenance.get("git_dirty") is not False
    ):
        parser.error("publication preflight requires an available clean Git source")
    primary = json.loads(args.primary_protocol.read_text(encoding="utf-8"))
    compiled = compile_replication_protocol(
        primary,
        args.journal,
        args.source,
        context_packets=args.context,
        samples=args.samples,
        seed=args.seed,
        source_provenance=source_provenance,
    )
    write_manifest(args.output, compiled.manifest)
    print(json.dumps(compiled.manifest, indent=2))


if __name__ == "__main__":
    main()

"""Resumable, provenance-recorded downloader for approved router data sources.

This module deliberately does not extract archives or convert packets.  Raw
captures remain immutable inputs until a separate checksum-recorded conversion
step produces payload-free causal journals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "argus.router-data-sources/v2"
RECEIPT_SCHEMA = "argus.router-source-receipt/v1"


@dataclass(frozen=True)
class Artifact:
    source_id: str
    role: str
    license_class: str
    artifact_id: str
    url: str
    filename: str
    expected_bytes: int
    format: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("router source manifest schema differs")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("router source manifest has no sources")
    identifiers = [source.get("id") for source in sources if isinstance(source, dict)]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("router source identifiers are not unique")
    return manifest


def select_artifacts(
    manifest: dict[str, Any],
    source_ids: list[str],
    accepted_licenses: set[str],
) -> list[Artifact]:
    selected = []
    known = {source["id"]: source for source in manifest["sources"]}
    missing = set(source_ids) - set(known)
    if missing:
        raise ValueError(f"unknown router sources: {sorted(missing)}")
    for source_id in source_ids:
        source = known[source_id]
        license_class = source.get("license_class")
        if license_class not in accepted_licenses:
            raise PermissionError(
                f"source {source_id} requires explicit acceptance of {license_class}"
            )
        artifacts = source.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"source {source_id} has no automatically downloadable artifacts")
        for value in artifacts:
            expected_bytes = value.get("expected_bytes")
            if not isinstance(expected_bytes, int) or expected_bytes <= 0:
                raise ValueError(f"source {source_id} artifact lacks a positive byte count")
            selected.append(
                Artifact(
                    source_id=source_id,
                    role=source["role"],
                    license_class=license_class,
                    artifact_id=value["id"],
                    url=value["url"],
                    filename=value["filename"],
                    expected_bytes=expected_bytes,
                    format=value["format"],
                )
            )
    return selected


def required_bytes(artifacts: list[Artifact]) -> int:
    # Keep headroom for partial files and later extraction; extraction itself is
    # intentionally a different stage with its own disk preflight.
    return sum(artifact.expected_bytes for artifact in artifacts) * 2


def _write_receipt(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fetch(artifact: Artifact, root: Path) -> dict[str, Any]:
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for resumable dataset downloads")
    directory = root / "raw" / artifact.source_id
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / artifact.filename
    partial = destination.with_suffix(destination.suffix + ".partial")
    receipt_path = destination.with_suffix(destination.suffix + ".receipt.json")
    if destination.exists():
        size = destination.stat().st_size
        if size != artifact.expected_bytes:
            raise ValueError(f"existing artifact has wrong size: {destination}")
    else:
        subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--retry",
                "5",
                "--retry-all-errors",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                artifact.url,
            ],
            check=True,
        )
        size = partial.stat().st_size
        if size != artifact.expected_bytes:
            raise ValueError(
                f"downloaded byte count differs for {artifact.artifact_id}: "
                f"expected {artifact.expected_bytes}, found {size}"
            )
        os.replace(partial, destination)
    digest = _sha256(destination)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_id": artifact.source_id,
        "artifact_id": artifact.artifact_id,
        "role": artifact.role,
        "license_class": artifact.license_class,
        "url": artifact.url,
        "filename": artifact.filename,
        "bytes": artifact.expected_bytes,
        "sha256": digest,
        "format": artifact.format,
        "extracted": False,
    }
    _write_receipt(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--accept-license", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    artifacts = select_artifacts(manifest, args.source, set(args.accept_license))
    needed = required_bytes(artifacts)
    args.root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.root).free
    plan = {
        "artifacts": [artifact.__dict__ for artifact in artifacts],
        "download_bytes": sum(value.expected_bytes for value in artifacts),
        "required_free_bytes_with_staging_headroom": needed,
        "available_free_bytes": free,
    }
    if free < needed:
        raise RuntimeError("insufficient free space for downloads and staging headroom")
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    receipts = [fetch(artifact, args.root) for artifact in artifacts]
    print(json.dumps({**plan, "receipts": receipts}, indent=2))


if __name__ == "__main__":
    main()

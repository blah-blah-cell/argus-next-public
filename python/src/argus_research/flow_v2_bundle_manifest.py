"""Create a deterministic checksum inventory for a flow-v2 evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .flow_v2_protocol import PREREGISTRATION_SHA256, write_manifest
from .flow_v2_train import _source_provenance

BUNDLE_SCHEMA = "argus.flow-v2-evidence-bundle/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(root: Path, artifacts: list[Path]) -> dict[str, Any]:
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError("evidence bundle root is not a directory")
    members = []
    seen: set[Path] = set()
    for artifact in artifacts:
        resolved = artifact.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"evidence artifact is not a file: {artifact}")
        try:
            relative = resolved.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(
                f"evidence artifact escapes bundle root: {artifact}"
            ) from error
        if resolved in seen:
            raise ValueError(f"duplicate evidence artifact: {artifact}")
        seen.add(resolved)
        members.append(
            {
                "path": relative.as_posix(),
                "bytes": resolved.stat().st_size,
                "sha256": _sha256(resolved),
            }
        )
    if not members:
        raise ValueError("evidence bundle contains no artifacts")
    members.sort(key=lambda value: value["path"])
    source = _source_provenance()
    blockers = []
    if source.get("available") is not True:
        blockers.append("source revision is unavailable")
    elif source.get("git_dirty") is not False:
        blockers.append("source tree is dirty")
    return {
        "schema": BUNDLE_SCHEMA,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "source_provenance": source,
        "members": members,
        "member_count": len(members),
        "publication_eligible": not blockers,
        "publication_blockers": blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(args.root, args.artifact)
    write_manifest(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

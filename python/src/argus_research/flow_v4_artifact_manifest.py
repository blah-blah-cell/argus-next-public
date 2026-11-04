"""Hash and verify the complete frozen AHEM-v5 evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .flow_v2_protocol import write_manifest
from .flow_v2_train import _canonical_sha256
from .flow_v4_experiment_plan import PLAN_SCHEMA

MANIFEST_SCHEMA = "argus.ahem-v5-artifact-manifest/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tasks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    values = []
    for name in (
        "training_tasks",
        "mechanism_tasks",
        "runtime_tasks",
        "runtime_aggregate_tasks",
        "comparison_tasks",
    ):
        values.extend(plan[name])
    values.extend((plan["replication_task"], plan["claim_task"]))
    return values


def _expected_paths(task: dict[str, Any]) -> list[Path]:
    output = Path(task["output"])
    return [
        Path(value) if Path(value).is_absolute() else output / value
        for value in task.get("expected_artifacts", [str(output)])
    ]


def build(plan_path: Path, run_root: Path, output: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("artifact manifest requires an AHEM-v5 experiment plan")
    supplied_hash = plan.get("plan_sha256")
    unsigned_plan = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if supplied_hash != _canonical_sha256(unsigned_plan):
        raise ValueError("artifact manifest received an invalid plan hash")
    resolved_root = run_root.resolve(strict=True)
    if resolved_root != Path(plan["run_root"]).resolve():
        raise ValueError("artifact manifest run root differs from the plan")
    resolved_output = output.resolve()

    expected = []
    for task in _tasks(plan):
        for path in _expected_paths(task):
            resolved = path.resolve()
            if resolved_output == resolved:
                continue
            if resolved_root not in resolved.parents:
                raise ValueError("planned artifact escapes the frozen run root")
            if not resolved.is_file():
                raise ValueError(f"planned artifact is missing: {resolved}")
            expected.append(resolved)

    files = [
        path
        for path in sorted(resolved_root.rglob("*"))
        if path.is_file()
        and path.resolve() != resolved_output
        and not path.name.endswith((".tmp", ".partial"))
    ]
    expected_set = set(expected)
    file_set = {path.resolve() for path in files}
    if not expected_set <= file_set:
        raise ValueError("artifact scan omitted a planned file")
    artifacts = [
        {
            "path": path.relative_to(resolved_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "required_by_plan": path.resolve() in expected_set,
        }
        for path in files
    ]
    unsigned = {
        "schema": MANIFEST_SCHEMA,
        "plan": str(plan_path.resolve()),
        "plan_sha256": supplied_hash,
        "preregistration_sha256": plan["preregistration_sha256"],
        "source_provenance": plan["source_provenance"],
        "run_root": str(resolved_root),
        "required_artifact_count": len(expected_set),
        "file_count": len(artifacts),
        "artifacts": artifacts,
    }
    result = {**unsigned, "manifest_sha256": _canonical_sha256(unsigned)}
    write_manifest(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.plan, args.run_root, args.output), indent=2))


if __name__ == "__main__":
    main()

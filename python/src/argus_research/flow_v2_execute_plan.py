"""Execute a frozen flow-v2 pilot or confirmatory plan with durable state."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .flow_v2_experiment_plan import PLAN_SCHEMA
from .flow_v2_pilot_plan import PILOT_PLAN_SCHEMA
from .flow_v2_protocol import write_manifest
from .flow_v2_train import _canonical_sha256, _source_provenance
from .flow_v4_artifact_manifest import build as build_v4_artifact_manifest
from .flow_v4_experiment_plan import PLAN_SCHEMA as V4_PLAN_SCHEMA
from .flow_v4_screen_plan import SCREEN_PLAN_SCHEMA

EXECUTOR_SCHEMA = "argus.flow-v2-plan-execution/v1"


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") not in {
        PLAN_SCHEMA,
        PILOT_PLAN_SCHEMA,
        V4_PLAN_SCHEMA,
        SCREEN_PLAN_SCHEMA,
    }:
        raise ValueError("executor requires a flow-v2 pilot or experiment plan")
    supplied = plan.get("plan_sha256")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if supplied != _canonical_sha256(unsigned):
        raise ValueError("experiment plan self-hash is invalid")
    source = _source_provenance()
    planned_source = plan.get("source_provenance", {})
    if (
        source.get("available") is not True
        or source.get("git_dirty") is not False
        or source.get("git_revision") != planned_source.get("git_revision")
        or planned_source.get("git_dirty") is not False
    ):
        raise ValueError("executor source differs from the clean planned revision")


def _phases(plan: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]], bool]]:
    if plan["schema"] == PILOT_PLAN_SCHEMA:
        return [
            ("training", list(plan["training_tasks"]), True),
            ("comparison", list(plan["comparison_tasks"]), False),
            ("report", [plan["report_task"]], False),
        ]
    if plan["schema"] == V4_PLAN_SCHEMA:
        return [
            ("training", list(plan["training_tasks"]), True),
            ("mechanism", list(plan["mechanism_tasks"]), False),
            ("runtime", list(plan["runtime_tasks"]), False),
            ("runtime-aggregate", list(plan["runtime_aggregate_tasks"]), False),
            ("comparison", list(plan["comparison_tasks"]), False),
            ("replication", [plan["replication_task"]], False),
            ("claim", [plan["claim_task"]], False),
        ]
    if plan["schema"] == SCREEN_PLAN_SCHEMA:
        return [
            ("screen-training", list(plan["training_tasks"]), True),
            ("screen-mechanism", list(plan["mechanism_tasks"]), False),
            ("screen-runtime", list(plan["runtime_tasks"]), False),
            ("screen-report", [plan["report_task"]], False),
        ]
    return [
        ("training", list(plan["training_tasks"]), True),
        ("runtime", list(plan["runtime_tasks"]), False),
        ("comparison", list(plan["comparison_tasks"]), False),
        ("replication", list(plan["replication_tasks"]), True),
        ("replication-comparison", [plan["replication_comparison_task"]], False),
        ("claim", [plan["claim_task"]], False),
    ]


def _artifact_paths(task: dict[str, Any]) -> list[Path]:
    output = Path(task["output"])
    values = task.get("expected_artifacts")
    if not values:
        return [output]
    paths = []
    for value in values:
        path = Path(value)
        paths.append(path if path.is_absolute() else output / path)
    return paths


def _complete(task: dict[str, Any]) -> bool:
    return all(path.is_file() for path in _artifact_paths(task))


def _resume_argv(task: dict[str, Any]) -> list[str]:
    argv = list(task["argv"])
    if (
        not any(
            module in argv
            for module in (
                "argus_research.flow_v2_train",
                "argus_research.flow_v4_train",
            )
        )
        or "--resume" in argv
    ):
        return argv
    output = Path(task["output"])
    latest = output / "latest.pt"
    metrics = output / "metrics.json"
    access = output / "test-access.json"
    if metrics.is_file():
        return argv
    if access.is_file():
        raise RuntimeError(
            f"{task['id']} has reserved test access without complete metrics"
        )
    if latest.is_file():
        argv.extend(("--resume", str(latest)))
    return argv


def _copy_artifacts(task: dict[str, Any], backup_root: Path) -> None:
    task_root = backup_root / task["id"].replace(":", "_")
    task_root.mkdir(parents=True, exist_ok=True)
    for source in _artifact_paths(task):
        if not source.is_file():
            raise RuntimeError(f"task did not produce expected artifact: {source}")
        destination = task_root / source.name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)


def _run_task(
    task: dict[str, Any],
    *,
    log_root: Path,
    backup_root: Path | None,
) -> dict[str, Any]:
    if _complete(task):
        return {"id": task["id"], "status": "already_complete"}
    argv = _resume_argv(task)
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{task['id'].replace(':', '_')}.log"
    started = datetime.now(UTC).isoformat()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"started_utc": started, "argv": argv}) + "\n")
        log.flush()
        completed = subprocess.run(
            argv,
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"task {task['id']} failed with exit code {completed.returncode}; "
            f"see {log_path}"
        )
    if not _complete(task):
        raise RuntimeError(f"task {task['id']} exited without its artifacts")
    if backup_root is not None:
        _copy_artifacts(task, backup_root)
    return {
        "id": task["id"],
        "status": "completed",
        "started_utc": started,
        "completed_utc": datetime.now(UTC).isoformat(),
        "log": str(log_path),
        "resumed": "--resume" in argv,
    }


def execute(
    plan_path: Path,
    *,
    workers: int,
    backup_root: Path | None,
    state_path: Path,
    stop_before: str | None = None,
) -> dict[str, Any]:
    if workers <= 0 or workers > 4:
        raise ValueError("GPU workers must be between one and four")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _validate_plan(plan)
    phases = _phases(plan)
    if stop_before is not None and stop_before not in {name for name, _, _ in phases}:
        raise ValueError("requested pause phase does not exist in the plan")
    log_root = state_path.parent / "logs"
    state: dict[str, Any] = {
        "schema": EXECUTOR_SCHEMA,
        "plan": str(plan_path.resolve()),
        "plan_sha256": plan["plan_sha256"],
        "source_provenance": _source_provenance(),
        "started_utc": datetime.now(UTC).isoformat(),
        "workers": workers,
        "tasks": [],
        "status": "running",
    }
    write_manifest(state_path, state)
    try:
        for phase_name, tasks, parallel in phases:
            if phase_name == stop_before:
                state["status"] = "paused_before_phase"
                state["next_phase"] = phase_name
                state["paused_utc"] = datetime.now(UTC).isoformat()
                write_manifest(state_path, state)
                return state
            if parallel:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(
                            _run_task,
                            task,
                            log_root=log_root,
                            backup_root=backup_root,
                        ): task
                        for task in tasks
                    }
                    for future in as_completed(futures):
                        result = future.result()
                        state["tasks"].append({"phase": phase_name, **result})
                        write_manifest(state_path, state)
            else:
                for task in tasks:
                    result = _run_task(task, log_root=log_root, backup_root=backup_root)
                    state["tasks"].append({"phase": phase_name, **result})
                    write_manifest(state_path, state)
        if plan["schema"] == V4_PLAN_SCHEMA:
            manifest_path = Path(plan["run_root"]) / "artifact-manifest.json"
            build_v4_artifact_manifest(
                plan_path,
                Path(plan["run_root"]),
                manifest_path,
            )
            manifest_task = {
                "id": "artifact-manifest",
                "output": str(manifest_path),
                "expected_artifacts": [str(manifest_path)],
            }
            if backup_root is not None:
                _copy_artifacts(manifest_task, backup_root)
            state["tasks"].append(
                {
                    "phase": "artifact-manifest",
                    "id": "artifact-manifest",
                    "status": "completed",
                    "completed_utc": datetime.now(UTC).isoformat(),
                }
            )
            write_manifest(state_path, state)
    except Exception as error:
        state["status"] = "failed"
        state["failure"] = str(error)
        state["completed_utc"] = datetime.now(UTC).isoformat()
        write_manifest(state_path, state)
        raise
    state["status"] = "completed"
    state["completed_utc"] = datetime.now(UTC).isoformat()
    write_manifest(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument(
        "--stop-before", help="pause durably before this phase; do not mark complete"
    )
    args = parser.parse_args()
    result = execute(
        args.plan,
        workers=args.workers,
        backup_root=args.backup_root,
        state_path=args.state,
        stop_before=args.stop_before,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

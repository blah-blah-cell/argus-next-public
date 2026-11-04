"""Single-GPU training queue. No downloads, scientific retries or shell commands.

Workers must emit immutable epoch-recovery commits and SUCCESS.json. Resource
reservations are admission estimates, not enforced VRAM limits or speed claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".partial")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def inside(root, name):
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError("portable relative path required")
    candidate = (root / name).resolve()
    if not name or not candidate.is_relative_to(root) or candidate == root:
        raise ValueError("path escapes workspace")
    return candidate


def validate_plan(plan, root):
    if plan.get("schema") != "argus.parallel-training/v1":
        raise ValueError("unsupported queue schema")
    for key in ("max_parallel", "cpu_threads", "max_seconds", "poll_seconds"):
        if type(plan.get(key)) is not int or plan[key] < 1:
            raise ValueError(f"positive integer required: {key}")
    if plan["max_parallel"] > 8 or not 1 <= plan["poll_seconds"] <= 60:
        raise ValueError("unsafe concurrency/poll interval")
    if plan["cpu_threads"] > (os.cpu_count() or 1):
        raise ValueError("CPU budget exceeds available cores")
    if type(plan.get("gpu_index")) is not int or plan["gpu_index"] < 0:
        raise ValueError("GPU index required")
    if type(plan.get("reserve_mib")) is not int or plan["reserve_mib"] < 512:
        raise ValueError("reserve at least 512 MiB GPU headroom")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("nonempty frozen job queue required")
    names = set()
    for job in jobs:
        name = job.get("id", "")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name) or name in names:
            raise ValueError("unique safe job identity required")
        names.add(name)
        for key in ("vram_mib", "cpu_threads"):
            if type(job.get(key)) is not int or job[key] < 1:
                raise ValueError("positive per-job resource estimate required")
        if job["cpu_threads"] > plan["cpu_threads"]:
            raise ValueError("job cannot fit CPU budget")
        if not isinstance(job.get("argv"), list) or not job["argv"]:
            raise ValueError("Python script argv required")
        if any(not isinstance(arg, str) for arg in job["argv"]):
            raise ValueError("argv must contain only strings")
        script = inside(root, job["argv"][0])
        if script.suffix != ".py" or "{output}" not in job["argv"]:
            raise ValueError("bound Python script and explicit {output} argument required")
        bindings = job.get("bindings")
        if not isinstance(bindings, dict) or set(bindings) != {
                "protocol_sha256", "source_manifest_sha256", "implementation_sha256"}:
            raise ValueError("epoch protocol/source/implementation bindings required")
        if any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h)
               for h in bindings.values()):
            raise ValueError("invalid epoch SHA-256 binding")
        inputs = job.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("hash-bound train/dev/code/protocol inputs required")
        kinds = set()
        data_roles = {}
        for item in inputs:
            if item.get("role") not in ("train", "development", "code", "protocol", "source_manifest", "initialization"):
                raise ValueError("final/test inputs forbidden in training queue")
            path = inside(root, item["path"])
            if not path.is_file() or digest(path) != item["sha256"]:
                raise ValueError(f"input hash mismatch: {item['path']}")
            kinds.add(item["role"])
            if item["role"] in ("train", "development"):
                previous = data_roles.setdefault(item["sha256"], item["role"])
                if previous != item["role"]:
                    raise ValueError("same data bytes assigned to train and development")
        if not {"train", "development", "code", "protocol", "source_manifest"} <= kinds:
            raise ValueError("training authorization bindings incomplete")
        if not any(item["path"] == job["argv"][0] and item["role"] == "code"
                   for item in inputs):
            raise ValueError("entry-point code not bound")
        for key, role in (("protocol_sha256", "protocol"),
                          ("source_manifest_sha256", "source_manifest"),
                          ("implementation_sha256", "code")):
            if not any(item["role"] == role and item["sha256"] == bindings[key] for item in inputs):
                raise ValueError("epoch binding not present in verified inputs")
    return jobs


def gpu_snapshot(index):
    # Respect provider CUDA_VISIBLE_DEVICES, including a 20 GB MIG partition.
    # Parent nvidia-smi totals can incorrectly describe the entire physical GPU.
    import torch
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise RuntimeError("selected CUDA device unavailable; no CPU fallback")
    free, total = torch.cuda.mem_get_info(index)
    properties = torch.cuda.get_device_properties(index)
    return {"logical_index": index, "name": properties.name,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "total_mib": total // (1024 * 1024), "free_mib": free // (1024 * 1024)}


def configure_worker():
    """Call before worker model/data allocation. Does not alter batch/loss/AMP."""
    import torch
    torch.set_num_threads(int(os.environ["ARGUS_CPU_THREADS"]))
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(int(os.environ["ARGUS_GPU_INDEX"]))


def can_admit(job, running, plan, gpu):
    return (len(running) < plan["max_parallel"]
        and sum(row["job"]["cpu_threads"] for row in running) + job["cpu_threads"] <= plan["cpu_threads"]
        and sum(row["job"]["vram_mib"] for row in running) + job["vram_mib"] <= gpu["total_mib"] - plan["reserve_mib"]
        and job["vram_mib"] <= gpu["free_mib"] - plan["reserve_mib"])


def mirror_commits(output, mirror, job):
    from .ahem_epoch_recovery import mirror_epoch
    count = 0
    for marker in sorted(output.glob("epoch-*/COMMITTED.json")):
        if marker.is_symlink() or marker.parent.is_symlink() or (marker.parent / "checkpoint.pt").is_symlink():
            raise ValueError("symlinked epoch evidence forbidden")
        receipt_hash = digest(marker)
        target = mirror / marker.parent.name
        if target.exists():
            receipt = json.loads(marker.read_text())
            if (digest(target / "COMMITTED.json") != receipt_hash
                    or digest(target / "checkpoint.pt") != receipt["checkpoint_sha256"]):
                raise ValueError("mirrored epoch changed")
        else:
            mirror_epoch(marker.parent, mirror, receipt_sha256=receipt_hash,
                         binding=job["bindings"])
        count += 1
    return count


def stop_workers(running):
    # Stop only child processes created by this queue; never cloud instances.
    for row in running:
        if row["process"].poll() is None:
            row["process"].terminate()
    deadline = time.monotonic() + 10
    for row in running:
        process = row["process"]
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        row["log"].close()


def run(plan_path, root, output, mirror, *, dry_run=False):
    root = root.resolve(strict=True)
    plan = json.loads(plan_path.read_text())
    jobs = validate_plan(plan, root)
    gpu = gpu_snapshot(plan["gpu_index"])
    initial_device = {key: gpu.get(key) for key in ("logical_index", "name", "visible_devices", "total_mib")}
    if any(job["vram_mib"] > gpu["total_mib"] - plan["reserve_mib"] for job in jobs):
        raise ValueError("job cannot fit selected GPU; do not silently change batch/protocol")
    if dry_run:
        return {"validated": True, "launched": False, "gpu": gpu, "jobs": len(jobs)}
    output = output.resolve()
    mirror = mirror.resolve()
    if output == mirror or output.is_relative_to(mirror) or mirror.is_relative_to(output):
        raise ValueError("mirror must be separate from live output")
    if output.exists() or mirror.exists():
        raise FileExistsError("immutable queue output/mirror already exists")
    output.mkdir(parents=True)
    mirror.mkdir(parents=True)
    write_json(output / "QUEUE.json", {"plan": plan, "plan_sha256": digest(plan_path),
        "launcher_sha256": digest(Path(__file__)), "gpu": gpu,
        "mirror_is_durable": False, "started_unix": time.time()})
    pending, running, finished = list(jobs), [], []
    start = time.monotonic()
    try:
        with (output / "TELEMETRY.jsonl").open("x", encoding="utf-8") as telemetry:
            while pending or running:
                if time.monotonic() - start >= plan["max_seconds"]:
                    raise TimeoutError("queue wall-time budget reached; preserve completed commits")
                gpu = gpu_snapshot(plan["gpu_index"])
                if any(gpu.get(key) != value for key, value in initial_device.items()):
                    raise ValueError("visible GPU configuration changed during queue")
                telemetry.write(json.dumps({"elapsed_seconds": time.monotonic() - start,
                    "gpu": gpu, "running": [row["job"]["id"] for row in running]}) + "\n")
                telemetry.flush()
                for row in list(running):
                    row["epochs"] = mirror_commits(row["output"], mirror / row["job"]["id"], row["job"])
                    code = row["process"].poll()
                    if code is None:
                        continue
                    row["log"].close()
                    if code != 0:
                        raise RuntimeError(f"worker {row['job']['id']} exited {code}; see worker.log; no automatic retry")
                    # A worker can commit its last epoch between the earlier
                    # scan and poll(). Rescan after exit before checking count.
                    row["epochs"] = mirror_commits(row["output"], mirror / row["job"]["id"], row["job"])
                    validate_plan({**plan, "jobs": [row["job"]]}, root)
                    done = json.loads((row["output"] / "SUCCESS.json").read_text())
                    if done != {"complete": True, "committed_epochs": row["epochs"], "bindings": row["job"]["bindings"]} or row["epochs"] < 1:
                        raise ValueError("worker exit is not a verified trained-checkpoint completion")
                    shutil.copyfile(row["output"] / "SUCCESS.json", mirror / row["job"]["id"] / "SUCCESS.json")
                    finished.append({"id": row["job"]["id"], "epochs": row["epochs"],
                                     "seconds": time.monotonic() - row["started"]})
                    running.remove(row)
                for job in list(pending):
                    if not can_admit(job, running, plan, gpu):
                        continue
                    # Recheck before launch; prior CPU preparation must publish immutable inputs.
                    validate_plan({**plan, "jobs": [job]}, root)
                    destination = output / job["id"]
                    destination.mkdir()
                    log = (destination / "worker.log").open("x", encoding="utf-8")
                    env = dict(os.environ, ARGUS_GPU_INDEX=str(plan["gpu_index"]),
                        OMP_NUM_THREADS=str(job["cpu_threads"]), MKL_NUM_THREADS=str(job["cpu_threads"]),
                        OPENBLAS_NUM_THREADS=str(job["cpu_threads"]), PYTHONUNBUFFERED="1",
                        ARGUS_CPU_THREADS=str(job["cpu_threads"]))
                    argv = [sys.executable, str(inside(root, job["argv"][0])),
                            *(str(destination) if arg == "{output}" else arg for arg in job["argv"][1:])]
                    write_json(destination / "LAUNCH.json", {"argv": argv, "job": job})
                    try:
                        process = subprocess.Popen(argv, cwd=root, env=env, stdout=log,
                                                   stderr=subprocess.STDOUT, shell=False)
                    except BaseException:
                        log.close()
                        raise
                    running.append({"job": job, "output": destination, "process": process,
                                    "log": log, "started": time.monotonic()})
                    pending.remove(job)
                    gpu = {**gpu, "free_mib": gpu["free_mib"] - job["vram_mib"]}
                time.sleep(plan["poll_seconds"])
        result = {"complete": True, "jobs": finished, "seconds": time.monotonic() - start,
                  "mirror_is_durable": False, "no_final_data_evaluated_by_launcher": True}
        write_json(output / "RESULT.json", result)
        return result
    except BaseException as error:
        stop_workers(running)
        mirror_errors = []
        for row in running:
            try:
                mirror_commits(row["output"], mirror / row["job"]["id"], row["job"])
            except Exception as mirror_error:
                mirror_errors.append({"job": row["job"]["id"], "error": str(mirror_error)})
        write_json(output / "FAILURE.json", {"error_type": type(error).__name__,
            "error": str(error), "completed": finished, "mirror_errors": mirror_errors,
            "pending": [job["id"] for job in pending]})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "root", "output", "mirror"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.root, args.output, args.mirror, dry_run=args.dry_run)))


if __name__ == "__main__":
    main()

"""Record and enforce the Linux GPU environment before flow-v2 execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .flow_v2_protocol import write_manifest
from .flow_v2_train import _source_provenance

PREFLIGHT_SCHEMA = "argus.flow-v2-vm-preflight/v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_environment(
    *,
    data_root: Path,
    run_root: Path,
    minimum_free_gib: float,
    minimum_vram_gib: float,
) -> dict[str, Any]:
    if platform.system() != "Linux":
        raise RuntimeError("publication GPU preflight requires Linux")
    source = _source_provenance()
    if source.get("available") is not True or source.get("git_dirty") is not False:
        raise RuntimeError("GPU preflight requires a clean Git revision")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the frozen Python environment")
    for path in (data_root, run_root):
        path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(data_root)
    free_gib = usage.free / (1024**3)
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            f"data volume has {free_gib:.1f} GiB free; requires {minimum_free_gib:.1f}"
        )
    properties = torch.cuda.get_device_properties(0)
    vram_gib = properties.total_memory / (1024**3)
    if vram_gib < minimum_vram_gib:
        raise RuntimeError(
            f"GPU has {vram_gib:.1f} GiB VRAM; requires {minimum_vram_gib:.1f}"
        )
    repository = Path(__file__).resolve().parents[3]
    lock_path = repository / "python" / "uv.lock"
    nvidia_smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    memory_bytes: int | None = None
    try:
        memory_bytes = int(os.sysconf("SC_PAGE_SIZE")) * int(
            os.sysconf("SC_PHYS_PAGES")
        )
    except (AttributeError, OSError, ValueError):
        pass
    return {
        "schema": PREFLIGHT_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_provenance": source,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "uv_lock_sha256": _sha256(lock_path),
        "cpu_count": os.cpu_count(),
        "system_memory_bytes": memory_bytes,
        "gpu": {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "nvidia_smi": nvidia_smi,
        },
        "storage": {
            "data_root": str(data_root.resolve()),
            "run_root": str(run_root.resolve()),
            "total_bytes": usage.total,
            "free_bytes": usage.free,
        },
        "requirements": {
            "minimum_free_gib": minimum_free_gib,
            "minimum_vram_gib": minimum_vram_gib,
        },
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=200.0)
    parser.add_argument("--minimum-vram-gib", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.minimum_free_gib, args.minimum_vram_gib) <= 0:
        parser.error("minimum storage and VRAM must be positive")
    result = inspect_environment(
        data_root=args.data_root,
        run_root=args.run_root,
        minimum_free_gib=args.minimum_free_gib,
        minimum_vram_gib=args.minimum_vram_gib,
    )
    write_manifest(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Fetch and freeze the upstream typed-decisions base on a training VM."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .laya_router_vm_train import LAYA_VERSION, _sha256

REPO_ID = "convaiinnovations/laya"
SUBFOLDER = "typed-decisions"


def validate_base(directory: Path) -> dict[str, Any]:
    required_files = [
        directory / "model.safetensors",
        directory / "rl_agent_config.json",
        directory / "tokenizer" / "tokenizer_config.json",
    ]
    required_directories = [directory / "encoder", directory / "tokenizer"]
    missing = [str(path.relative_to(directory)) for path in required_files if not path.is_file()]
    missing.extend(
        str(path.relative_to(directory)) for path in required_directories if not path.is_dir()
    )
    if missing:
        raise FileNotFoundError(f"Laya base checkpoint is incomplete: {sorted(missing)}")
    config = json.loads((directory / "rl_agent_config.json").read_text(encoding="utf-8"))
    tokenizer = json.loads(
        (directory / "tokenizer" / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    # Laya's typed-decisions encoder builds BERT-style sequences with CLS/SEP;
    # BOS/EOS are generative-model tokens and are intentionally absent upstream.
    required_tokens = ("cls_token", "sep_token", "pad_token", "unk_token")
    missing_tokens = [name for name in required_tokens if tokenizer.get(name) is None]
    extra_special_tokens = tokenizer.get("extra_special_tokens")
    repair_needed = tokenizer.get("tokenizer_class") in (None, "TokenizersBackend") or isinstance(
        extra_special_tokens, list
    )
    if missing_tokens or repair_needed:
        details = []
        if missing_tokens:
            details.append(f"missing required tokens {missing_tokens}")
        if repair_needed:
            details.append("tokenizer metadata still needs compatibility repair")
        raise ValueError("Laya tokenizer is incompatible: " + "; ".join(details))
    return {
        "model_safetensors_sha256": _sha256(directory / "model.safetensors"),
        "config_sha256": _sha256(directory / "rl_agent_config.json"),
        "model_name": config.get("model_name"),
    }


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("base output already exists; fetched artifacts are immutable")
    from huggingface_hub import HfApi, snapshot_download
    from laya.agent import _fix_tokenizer_config  # type: ignore[import-not-found]

    info = HfApi().model_info(REPO_ID)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="argus-laya-base-", dir=output.parent) as temporary:
        root = Path(temporary)
        snapshot_download(
            REPO_ID,
            revision=info.sha,
            allow_patterns=[f"{SUBFOLDER}/**"],
            local_dir=root,
        )
        source = root / SUBFOLDER
        _fix_tokenizer_config(str(source))
        validation = validate_base(source)
        provenance = {
            "schema": "argus.laya-base-provenance/v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "repo_id": REPO_ID,
            "subfolder": SUBFOLDER,
            "revision": info.sha,
            "laya_version": LAYA_VERSION,
            **validation,
        }
        (source / "argus-base-provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        shutil.move(str(source), str(output))
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()

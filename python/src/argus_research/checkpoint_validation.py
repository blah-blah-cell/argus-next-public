"""Strict, CPU-safe validation of transferred research checkpoint contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .ahem_pretrain import (
    AhemNextPacket,
    CausalCnnNextPacket,
    LongTcnNextPacket,
    LstmNextPacket,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(directory: Path) -> dict[str, object]:
    models = {
        "ahem_fixed": AhemNextPacket(12),
        "lstm": LstmNextPacket(12),
        "cnn_local": CausalCnnNextPacket(12),
        "tcn_long": LongTcnNextPacket(12),
    }
    report: dict[str, object] = {
        "checkpoint_contract": "packet-causal-v1/next_packet_only",
        "models": {},
    }
    for name, model in models.items():
        path = directory / f"{name}_real_pcap_30k.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("feature_schema") != "packet-causal-v1"
            or payload.get("supervision") != "next_packet_only"
        ):
            raise ValueError(f"{path.name}: unsupported checkpoint contract")
        model.load_state_dict(payload["state_dict"], strict=True)
        with torch.inference_mode():
            output = model(torch.zeros((1, 30_000, 12)))
        if output.shape != (1, 12) or not torch.isfinite(output).all():
            raise ValueError(f"{path.name}: invalid forward result")
        report["models"][name] = {
            "sha256": _sha256(path),
            "parameters": sum(p.numel() for p in model.parameters()),
        }
    flow_path = directory / "flow_mlp_real_cic_ids2018.pt"
    flow = torch.load(flow_path, map_location="cpu", weights_only=True)
    if flow.get("feature_count") != 78 or flow.get("real_data") is not True:
        raise ValueError(
            "flow MLP checkpoint has an invalid retrospective-flow contract"
        )
    report["flow_mlp"] = {
        "sha256": _sha256(flow_path),
        "feature_count": 78,
        "retrospective_only": True,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.directory), indent=2))


if __name__ == "__main__":
    main()

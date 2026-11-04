"""Export a hash-bound, forecast-only AHEM v3.4 cache-native ONNX artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn

from .ahem_v3 import AhemV34FlowIndexedNextPacket

CHECKPOINT_SCHEMA = "argus.flow-conditioned-global-smoke/v1"
MODEL_NAME = "ahem_v3.4_flow_indexed"
FEATURE_SCHEMA = "argus.packet-causal-flow-v34"


class _CacheNativeGraph(nn.Module):
    def __init__(self, model: AhemV34FlowIndexedNextPacket) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        local_packets: Tensor,
        flow_packets: Tensor,
        flow_ordinals: Tensor,
        flow_valid: Tensor,
        context_spans: Tensor,
    ) -> Tensor:
        return self.model.forward_from_indexed_lane(
            local_packets,
            flow_packets,
            flow_ordinals,
            flow_valid,
            context_spans,
        ).prediction


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _inputs() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260829)
    local = torch.randn((1, 128, 13), generator=generator)
    local[..., 1] = local[..., 1].abs()
    local[..., -1] = 0.0
    local[:, [3, 29, 65, 100, 127], -1] = 1.0
    flow = torch.randn((1, 17, 12), generator=generator)
    flow[..., 1] = flow[..., 1].abs()
    ordinals = torch.tensor(
        [
            [
                29_999,
                28_400,
                25_000,
                21_700,
                18_100,
                15_000,
                12_000,
                9_000,
                7_000,
                5_000,
                3_500,
                2_400,
                1_600,
                900,
                400,
                100,
                0,
            ]
        ],
        dtype=torch.int64,
    )
    valid = torch.ones((1, 17), dtype=torch.bool)
    spans = torch.tensor([30_000.0], dtype=torch.float32)
    return local, flow, ordinals, valid, spans


def package(
    checkpoint: Path,
    output: Path,
    manifest_path: Path,
    *,
    source_revision: str,
    training_reference: str,
    evaluation_reference: str,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("model") != MODEL_NAME:
        raise ValueError("checkpoint is not an AHEM v3.4 flow-indexed artifact")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("checkpoint state_dict is absent")

    model = AhemV34FlowIndexedNextPacket()
    model.load_state_dict(state_dict, strict=True)
    graph = _CacheNativeGraph(model.eval())
    inputs = _inputs()
    with torch.inference_mode():
        expected = graph(*inputs).numpy()

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    torch.onnx.export(
        graph,
        inputs,
        temporary,
        input_names=[
            "local_packets",
            "flow_packets",
            "flow_ordinals",
            "flow_valid",
            "context_spans",
        ],
        output_names=["prediction"],
        opset_version=18,
        dynamo=False,
    )
    exported = onnx.load(temporary)
    onnx.checker.check_model(exported, full_check=True)
    session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
    actual = session.run(
        ["prediction"],
        {
            "local_packets": inputs[0].numpy(),
            "flow_packets": inputs[1].numpy(),
            "flow_ordinals": inputs[2].numpy(),
            "flow_valid": inputs[3].numpy(),
            "context_spans": inputs[4].numpy(),
        },
    )[0]
    maximum_error = float(np.max(np.abs(expected - actual)))
    if not np.isfinite(actual).all() or maximum_error > 1e-5:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"ONNX parity failed: maximum absolute error {maximum_error:.9g}"
        )
    os.replace(temporary, output)

    manifest: dict[str, Any] = {
        "format": "argus-model-artifact",
        "format_version": 1,
        "feature_schema_id": FEATURE_SCHEMA,
        "feature_schema_version": 34,
        "model_id": "argus.ahem-v34-flow-indexed",
        "model_version": "3.4.0-exploratory",
        "artifact_sha256": _sha256(output),
        "provenance": {
            "training_dataset_reference": training_reference,
            "evaluation_reference": evaluation_reference,
            "source_revision": source_revision,
        },
        "calibration": {"required": False, "reference": None},
        "readiness": {"minimum_observations": 128, "maximum_age_seconds": None},
        "runtime": {
            "runtime_id": "onnxruntime",
            "minimum_runtime_version": "1.20.0",
            "supported_targets": [
                "x86_64-pc-windows-msvc",
                "x86_64-unknown-linux-gnu",
            ],
        },
    }
    _atomic_json(manifest_path, manifest)
    export_report = {
        "schema": "argus.ahem-v34-onnx-export/v1",
        "artifact_manifest_sha256": _sha256(manifest_path),
        "task": "self_supervised_next_same_flow_packet_forecast_only",
        "security_verdict": False,
        "study_role": payload.get("protocol", {}).get("study_role", "exploratory"),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "onnx_parity_max_abs_error": maximum_error,
        "tensor_contract": {
            "local_packets": {"dtype": "float32", "shape": [1, 128, 13]},
            "flow_packets": {"dtype": "float32", "shape": [1, 17, 12]},
            "flow_ordinals": {"dtype": "int64", "shape": [1, 17]},
            "flow_valid": {"dtype": "bool", "shape": [1, 17]},
            "context_spans": {"dtype": "float32", "shape": [1]},
            "prediction": {"dtype": "float32", "shape": [1, 2]},
        },
    }
    report_path = manifest_path.with_name(manifest_path.name + ".provenance.json")
    _atomic_json(report_path, export_report)
    return {"manifest": manifest, "export": export_report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--training-reference", required=True)
    parser.add_argument("--evaluation-reference", required=True)
    args = parser.parse_args()
    value = package(
        args.checkpoint,
        args.output,
        args.manifest,
        source_revision=args.source_revision,
        training_reference=args.training_reference,
        evaluation_reference=args.evaluation_reference,
    )
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

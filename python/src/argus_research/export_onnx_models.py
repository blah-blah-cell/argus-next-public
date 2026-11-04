"""Export validated PyTorch forecasters and routers to ONNX artifacts with strict manifests."""

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

from .ahem_pretrain import AhemNextPacket, CausalCnnNextPacket, LongTcnNextPacket
from .memory_models import CausalRouter


class _TwoFeatureForecast(torch.nn.Module):
    """Expose only the two forecast coordinates consumed by the live runtime."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.model(context)[..., :2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint_provenance(
    checkpoint_path: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    """Return honest source metadata; never infer a dataset from a filename."""
    checkpoint_hash = sha256_file(checkpoint_path)
    supplied = payload.get("provenance")
    required = {
        "training_dataset_reference",
        "evaluation_reference",
        "source_revision",
    }
    if isinstance(supplied, dict) and required.issubset(supplied):
        values = {name: str(supplied[name]) for name in sorted(required)}
        if all(values.values()):
            return {
                **values,
                "checkpoint_sha256": checkpoint_hash,
                "verified": True,
            }
    reference = f"unverified:checkpoint-sha256:{checkpoint_hash}"
    return {
        "training_dataset_reference": reference,
        "evaluation_reference": reference,
        "source_revision": "unknown",
        "checkpoint_sha256": checkpoint_hash,
        "verified": False,
    }


def _forecast_manifest(
    *,
    checkpoint_path: Path,
    payload: dict[str, Any],
    artifact_hash: str,
    model_id: str,
    minimum_observations: int,
) -> dict[str, Any]:
    return {
        "format": "argus-model-artifact",
        "format_version": 1,
        "feature_schema_id": "packet-causal-v1",
        "feature_schema_version": 1,
        "model_id": model_id,
        "model_version": "1.0.0",
        "artifact_sha256": artifact_hash,
        "provenance": _checkpoint_provenance(checkpoint_path, payload),
        "calibration": {"required": False, "reference": None},
        "readiness": {
            "minimum_observations": minimum_observations,
            "maximum_age_seconds": None,
        },
        "runtime": {
            "runtime_id": "onnxruntime",
            "minimum_runtime_version": "1.20.0",
            "supported_targets": [
                "x86_64-pc-windows-msvc",
                "x86_64-unknown-linux-gnu",
            ],
        },
        "task": "next_packet_forecast_only",
        "security_detector": False,
        "output_contract": {
            "names": ["log_packet_length", "log_interarrival"],
            "loss": "smooth_l1_first_two_features",
        },
    }


def export_cnn_local(checkpoint_path: Path, output_dir: Path) -> dict[str, Any]:
    output_onnx = output_dir / "cnn_local.onnx"
    output_manifest = output_dir / "cnn_local.onnx.manifest.json"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    base_model = CausalCnnNextPacket(12)
    base_model.load_state_dict(payload["state_dict"], strict=True)
    model = _TwoFeatureForecast(base_model).eval()

    dummy_input = torch.randn(1, 31, 12)
    with torch.inference_mode():
        expected = model(dummy_input).numpy()

    temp_onnx = output_onnx.with_suffix(".tmp")
    torch.onnx.export(
        model,
        dummy_input,
        temp_onnx,
        input_names=["context"],
        output_names=["prediction"],
        opset_version=18,
        dynamic_axes={"context": {0: "batch"}, "prediction": {0: "batch"}},
        dynamo=False,
    )
    model_proto = onnx.load(temp_onnx)
    onnx.checker.check_model(model_proto, full_check=True)

    session = ort.InferenceSession(str(temp_onnx), providers=["CPUExecutionProvider"])
    actual = session.run(["prediction"], {"context": dummy_input.numpy()})[0]
    max_error = float(np.max(np.abs(expected - actual)))
    if max_error > 1e-4:
        temp_onnx.unlink(missing_ok=True)
        raise RuntimeError(f"cnn_local parity check failed: max_error={max_error}")
    os.replace(temp_onnx, output_onnx)

    artifact_hash = sha256_file(output_onnx)
    manifest = _forecast_manifest(
        checkpoint_path=checkpoint_path,
        payload=payload,
        artifact_hash=artifact_hash,
        model_id="argus.cnn-local",
        minimum_observations=31,
    )
    write_atomic_json(output_manifest, manifest)
    return {
        "model_id": "argus.cnn-local",
        "onnx_path": str(output_onnx),
        "manifest_path": str(output_manifest),
        "max_abs_error": max_error,
        "artifact_sha256": artifact_hash,
    }


def export_ahem_fixed(checkpoint_path: Path, output_dir: Path) -> dict[str, Any]:
    output_onnx = output_dir / "ahem_fixed.onnx"
    output_manifest = output_dir / "ahem_fixed.onnx.manifest.json"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    base_model = AhemNextPacket(12)
    base_model.load_state_dict(payload["state_dict"], strict=True)
    model = _TwoFeatureForecast(base_model).eval()

    dummy_input = torch.randn(1, 128, 12)
    with torch.inference_mode():
        expected = model(dummy_input).numpy()

    temp_onnx = output_onnx.with_suffix(".tmp")
    torch.onnx.export(
        model,
        dummy_input,
        temp_onnx,
        input_names=["context"],
        output_names=["prediction"],
        opset_version=18,
        dynamic_axes={"context": {0: "batch"}, "prediction": {0: "batch"}},
        dynamo=False,
    )
    model_proto = onnx.load(temp_onnx)
    onnx.checker.check_model(model_proto, full_check=True)

    session = ort.InferenceSession(str(temp_onnx), providers=["CPUExecutionProvider"])
    actual = session.run(["prediction"], {"context": dummy_input.numpy()})[0]
    max_error = float(np.max(np.abs(expected - actual)))
    if max_error > 1e-4:
        temp_onnx.unlink(missing_ok=True)
        raise RuntimeError(f"ahem_fixed parity check failed: max_error={max_error}")
    os.replace(temp_onnx, output_onnx)

    artifact_hash = sha256_file(output_onnx)
    manifest = _forecast_manifest(
        checkpoint_path=checkpoint_path,
        payload=payload,
        artifact_hash=artifact_hash,
        model_id="argus.ahem-fixed",
        minimum_observations=128,
    )
    write_atomic_json(output_manifest, manifest)
    return {
        "model_id": "argus.ahem-fixed",
        "onnx_path": str(output_onnx),
        "manifest_path": str(output_manifest),
        "max_abs_error": max_error,
        "artifact_sha256": artifact_hash,
    }


def export_tcn_long(checkpoint_path: Path, output_dir: Path) -> dict[str, Any]:
    output_onnx = output_dir / "tcn_long.onnx"
    output_manifest = output_dir / "tcn_long.onnx.manifest.json"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    base_model = LongTcnNextPacket(12)
    base_model.load_state_dict(payload["state_dict"], strict=True)
    model = _TwoFeatureForecast(base_model).eval()

    dummy_input = torch.randn(1, 1024, 12)
    with torch.inference_mode():
        expected = model(dummy_input).numpy()

    temp_onnx = output_onnx.with_suffix(".tmp")
    torch.onnx.export(
        model,
        dummy_input,
        temp_onnx,
        input_names=["context"],
        output_names=["prediction"],
        opset_version=18,
        dynamic_axes={"context": {0: "batch"}, "prediction": {0: "batch"}},
        dynamo=False,
    )
    model_proto = onnx.load(temp_onnx)
    onnx.checker.check_model(model_proto, full_check=True)

    session = ort.InferenceSession(str(temp_onnx), providers=["CPUExecutionProvider"])
    actual = session.run(["prediction"], {"context": dummy_input.numpy()})[0]
    max_error = float(np.max(np.abs(expected - actual)))
    if max_error > 1e-3:
        temp_onnx.unlink(missing_ok=True)
        raise RuntimeError(f"tcn_long parity check failed: max_error={max_error}")
    os.replace(temp_onnx, output_onnx)

    artifact_hash = sha256_file(output_onnx)
    manifest = _forecast_manifest(
        checkpoint_path=checkpoint_path,
        payload=payload,
        artifact_hash=artifact_hash,
        model_id="argus.tcn-long",
        minimum_observations=1024,
    )
    write_atomic_json(output_manifest, manifest)
    return {
        "model_id": "argus.tcn-long",
        "onnx_path": str(output_onnx),
        "manifest_path": str(output_manifest),
        "max_abs_error": max_error,
        "artifact_sha256": artifact_hash,
    }


def export_temporal_router(checkpoint_path: Path, output_dir: Path) -> dict[str, Any]:
    output_onnx = output_dir / "temporal_router.onnx"
    output_manifest = output_dir / "temporal_router.onnx.manifest.json"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = CausalRouter(24, 4)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    dummy_input = torch.randn(1, 24)
    with torch.inference_mode():
        expected = model(dummy_input).numpy()

    temp_onnx = output_onnx.with_suffix(".tmp")
    torch.onnx.export(
        model,
        dummy_input,
        temp_onnx,
        input_names=["features"],
        output_names=["logits"],
        opset_version=18,
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    model_proto = onnx.load(temp_onnx)
    onnx.checker.check_model(model_proto, full_check=True)

    session = ort.InferenceSession(str(temp_onnx), providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {"features": dummy_input.numpy()})[0]
    max_error = float(np.max(np.abs(expected - actual)))
    if max_error > 1e-4:
        temp_onnx.unlink(missing_ok=True)
        raise RuntimeError(f"temporal_router parity check failed: max_error={max_error}")
    os.replace(temp_onnx, output_onnx)

    artifact_hash = sha256_file(output_onnx)
    manifest = {
        "format": "argus-model-artifact",
        "format_version": 1,
        "feature_schema_id": "causal-router-features-v1",
        "feature_schema_version": 1,
        "model_id": "argus.temporal-router",
        "model_version": "1.0.0",
        "artifact_sha256": artifact_hash,
        "provenance": _checkpoint_provenance(checkpoint_path, payload),
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
        "task": "forecast_expert_routing_only",
        "security_detector": False,
    }
    write_atomic_json(output_manifest, manifest)
    return {
        "model_id": "argus.temporal-router",
        "onnx_path": str(output_onnx),
        "manifest_path": str(output_manifest),
        "max_abs_error": max_error,
        "artifact_sha256": artifact_hash,
    }


def export_all(models_dir: Path) -> list[dict[str, Any]]:
    results = []
    cnn_local_pt = models_dir / "cnn_local_real_pcap_30k.pt"
    if cnn_local_pt.is_file():
        results.append(export_cnn_local(cnn_local_pt, models_dir))
    ahem_fixed_pt = models_dir / "ahem_fixed_real_pcap_30k.pt"
    if ahem_fixed_pt.is_file():
        results.append(export_ahem_fixed(ahem_fixed_pt, models_dir))
    tcn_long_pt = models_dir / "tcn_long_real_pcap_30k.pt"
    if tcn_long_pt.is_file():
        results.append(export_tcn_long(tcn_long_pt, models_dir))
    router_pt = models_dir / "temporal_router_real_live.pt"
    if router_pt.is_file():
        results.append(export_temporal_router(router_pt, models_dir))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Export validated PyTorch forecasters to ONNX.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="Directory containing PyTorch model checkpoints (required; never guessed)",
    )
    args = parser.parse_args()
    reports = export_all(args.models_dir)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()

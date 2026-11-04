"""Full-precision ONNX export and parity verification for frozen AHEM v6.1."""

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

try:
    from . import ahem_v3 as ahem_v3_module
    from . import ahem_v4 as ahem_v4_module
    from . import ahem_v6 as ahem_v6_module
    from .ahem_v6 import AhemV6StateTransitionNextPacket
    from .flow_v2_features import FEATURE_CONTRACT_SHA256
except ImportError:
    import sys

    src_root = Path(__file__).resolve().parent.parent
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from argus_research import ahem_v3 as ahem_v3_module
    from argus_research import ahem_v4 as ahem_v4_module
    from argus_research import ahem_v6 as ahem_v6_module
    from argus_research.ahem_v6 import AhemV6StateTransitionNextPacket
    from argus_research.flow_v2_features import FEATURE_CONTRACT_SHA256

MANIFEST_SCHEMA = "argus.ahem-v61-onnx-manifest/v1"
MODEL_ID = "argus.ahem-v6.1.state-transition-forecaster"
MODEL_VERSION = "6.1.0"
ARCHITECTURE = "ahem-v6.1-bounded-causal-state-transition-memory"
OPSET_VERSION = 17
PARITY_TOLERANCE = 1e-4

INPUT_NAMES = [
    "local_packets",
    "query_predecessor",
    "query_current",
    "keys",
    "values",
    "successor_dynamic",
    "source_ordinals",
    "valid",
    "anchor_ordinals",
    "context_spans",
    "query_gap",
]
OUTPUT_NAMES = ["prediction", "retrieval_weights", "memory_gate"]


class AhemV61OnnxWrapper(nn.Module):
    """Clean PyTorch export wrapper module for frozen AHEM v6.1.

    Wraps forward inference from projected feature cache
    (``forward_from_projected_cache``) and returns the state transition
    forecast tensor of shape ``[batch, output_dim]``.
    """

    def __init__(self, model: AhemV6StateTransitionNextPacket) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        local_packets: Tensor,
        query_predecessor: Tensor,
        query_current: Tensor,
        keys: Tensor,
        values: Tensor,
        successor_dynamic: Tensor,
        source_ordinals: Tensor,
        valid: Tensor,
        anchor_ordinals: Tensor,
        context_spans: Tensor,
        query_gap: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        output = self.model.forward_from_projected_cache(
            local_packets=local_packets,
            query_predecessor=query_predecessor,
            query_current=query_current,
            keys=keys,
            values=values,
            successor_dynamic=successor_dynamic,
            source_ordinals=source_ordinals,
            valid=valid,
            anchor_ordinals=anchor_ordinals,
            context_spans=context_spans,
            query_gap=query_gap,
        )
        return output.prediction, output.retrieval_weights, output.memory_gate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_sha256() -> str:
    """Hash normalized model sources so line-ending differences are eliminated."""
    digest = hashlib.sha256()
    for module in (ahem_v3_module, ahem_v4_module, ahem_v6_module):
        source_path = Path(module.__file__)
        normalized = source_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        digest.update(source_path.name.encode())
        digest.update(b"\0")
        digest.update(normalized.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def write_atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def make_valid_test_tensors(
    batch_size: int = 1,
    seed: int = 20260906,
    max_transitions: int = 16,
    local_packets_count: int = 128,
    packet_dim: int = 12,
    width: int = 60,
    key_dim: int = 32,
    output_dim: int = 2,
) -> tuple[Tensor, ...]:
    """Construct deterministic, domain-valid tensors for export and parity checking."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    local_packets = torch.randn(batch_size, local_packets_count, packet_dim, generator=generator)
    query_predecessor = torch.randn(batch_size, packet_dim, generator=generator)
    query_current = torch.randn(batch_size, packet_dim, generator=generator)
    keys = torch.randn(batch_size, max_transitions, key_dim, generator=generator)
    values = torch.randn(batch_size, max_transitions, width, generator=generator)
    successor_dynamic = torch.randn(batch_size, max_transitions, output_dim, generator=generator)

    source_ordinals = torch.randint(100, 900, (batch_size, max_transitions), dtype=torch.int64, generator=generator)
    source_ordinals, _ = torch.sort(source_ordinals, dim=-1, descending=True)

    valid = torch.ones(batch_size, max_transitions, dtype=torch.bool)
    if batch_size > 1:
        valid[1, max_transitions // 2 :] = False

    anchor_ordinals = torch.full((batch_size,), 1000, dtype=torch.int64)
    context_spans = torch.full((batch_size,), 10000.0, dtype=torch.float32)
    query_gap = torch.rand(batch_size, generator=generator)

    return (
        local_packets,
        query_predecessor,
        query_current,
        keys,
        values,
        successor_dynamic,
        source_ordinals,
        valid,
        anchor_ordinals,
        context_spans,
        query_gap,
    )


def load_frozen_model(checkpoint_path: Path) -> tuple[AhemV6StateTransitionNextPacket, dict[str, Any]]:
    """Load and validate the frozen AHEM v6.1 PyTorch checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_data, dict):
        raise ValueError("Checkpoint file must be a serialized dictionary payload")

    state_dict = checkpoint_data.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint payload missing valid 'state_dict'")

    model = AhemV6StateTransitionNextPacket(
        packet_dim=checkpoint_data.get("packet_dim", 12),
        output_dim=checkpoint_data.get("output_dim", 2),
        width=checkpoint_data.get("width", 60),
        key_dim=checkpoint_data.get("key_dim", 32),
        local_packets=checkpoint_data.get("local_packets", 128),
        max_transitions=checkpoint_data.get("max_transitions", 16),
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Ensure all parameters are float32 (zero quantization)
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            raise TypeError(f"Model parameter '{name}' has dtype {param.dtype}, expected float32")

    return model, checkpoint_data


def verify_parity(
    wrapper: AhemV61OnnxWrapper,
    onnx_path: Path,
    batch_sizes: tuple[int, ...] = (1, 2, 4),
    tolerance: float = PARITY_TOLERANCE,
) -> dict[str, Any]:
    """Run rigorous parity checks comparing PyTorch vs ONNX Runtime output."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    batch_errors: dict[int, dict[str, float]] = {}

    for b in batch_sizes:
        test_tensors = make_valid_test_tensors(batch_size=b, seed=20260906 + b)
        with torch.inference_mode():
            expected = tuple(tensor.numpy() for tensor in wrapper(*test_tensors))

        feed_dict = {
            name: tensor.numpy()
            for name, tensor in zip(INPUT_NAMES, test_tensors, strict=True)
        }
        actual = session.run(OUTPUT_NAMES, feed_dict)
        batch_errors[b] = {}
        for name, expected_tensor, actual_tensor in zip(
            OUTPUT_NAMES, expected, actual, strict=True
        ):
            if not np.isfinite(actual_tensor).all():
                raise RuntimeError(f"ONNX Runtime produced non-finite {name} for batch size {b}")
            if actual_tensor.shape != expected_tensor.shape:
                raise RuntimeError(f"ONNX Runtime produced wrong {name} shape for batch size {b}")
            max_abs_error = float(np.max(np.abs(expected_tensor - actual_tensor)))
            batch_errors[b][name] = max_abs_error
            if max_abs_error > tolerance:
                raise RuntimeError(
                    f"Parity verification failed for {name} at batch size {b}: "
                    f"max_abs_error={max_abs_error:.9e} > tolerance={tolerance:.9e}"
                )

    overall_max_error = max(error for errors in batch_errors.values() for error in errors.values())
    return {
        "verified": True,
        "max_abs_error": overall_max_error,
        "tolerance": tolerance,
        "batch_errors": {str(k): v for k, v in batch_errors.items()},
    }


def export_ahem_v61(
    checkpoint_path: Path,
    output_onnx_path: Path,
    output_manifest_path: Path | None = None,
    opset_version: int = OPSET_VERSION,
    tolerance: float = PARITY_TOLERANCE,
) -> dict[str, Any]:
    """Export frozen AHEM v6.1 to full-precision ONNX with canonical manifest."""
    model, metadata = load_frozen_model(checkpoint_path)
    wrapper = AhemV61OnnxWrapper(model).eval()

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    temp_onnx = output_onnx_path.with_suffix(output_onnx_path.suffix + ".tmp")
    if temp_onnx.exists():
        temp_onnx.unlink()

    sample_inputs = make_valid_test_tensors(batch_size=1, seed=20260906)
    dynamic_axes = {name: {0: "batch"} for name in INPUT_NAMES}
    dynamic_axes.update({name: {0: "batch"} for name in OUTPUT_NAMES})

    # Full float32 export using specified opset and dynamic batch axes
    torch.onnx.export(
        wrapper,
        sample_inputs,
        temp_onnx,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    # Validate ONNX graph integrity
    onnx_model = onnx.load(str(temp_onnx))
    onnx.checker.check_model(onnx_model, full_check=True)

    # Verify that all initializers (weights) are full float32 (zero quantization)
    for initializer in onnx_model.graph.initializer:
        if initializer.data_type != onnx.TensorProto.FLOAT:
            temp_onnx.unlink(missing_ok=True)
            raise ValueError(
                f"Exported initializer '{initializer.name}' has non-float32 data type {initializer.data_type}"
            )

    # Parity verification against ONNX Runtime across dynamic batch sizes
    parity_result = verify_parity(wrapper, temp_onnx, batch_sizes=(1, 2, 4), tolerance=tolerance)

    # Commit ONNX file atomically
    os.replace(temp_onnx, output_onnx_path)
    artifact_hash = sha256_file(output_onnx_path)
    source_checkpoint_hash = sha256_file(checkpoint_path)

    # Generate canonical manifest
    manifest: dict[str, Any] = {
        "format": "argus-model-artifact",
        "format_version": 1,
        "schema": MANIFEST_SCHEMA,
        "model_id": metadata.get("model_id", MODEL_ID),
        "model_version": MODEL_VERSION,
        "architecture": metadata.get("architecture", ARCHITECTURE),
        "task": metadata.get("supervision", "next_same_flow_dynamic_event_forecast_only"),
        "security_detector": metadata.get("security_detector", False),
        "calibrated_security_verdict": metadata.get("calibrated_security_verdict", False),
        "artifact_sha256": artifact_hash,
        "source_checkpoint_sha256": source_checkpoint_hash,
        "training_source_checkpoint_sha256": metadata.get("source_checkpoint_sha256", source_checkpoint_hash),
        "final_result_sha256": metadata.get("final_result_sha256", ""),
        "model_implementation_sha256": implementation_sha256(),
        "model_implementation_files": metadata.get(
            "model_implementation_files", ["ahem_v3.py", "ahem_v4.py", "ahem_v6.py"]
        ),
        "feature_schema": metadata.get("feature_schema", "packet-causal-flow-v2"),
        "feature_schema_id": metadata.get("feature_schema_id", "packet-causal-flow-v2"),
        "feature_schema_version": int(metadata.get("feature_schema_version", 1)),
        "provenance": {
            "training_dataset_reference": str(
                metadata.get(
                    "training_dataset_reference",
                    f"unverified:checkpoint-sha256:{source_checkpoint_hash}",
                )
            ),
            "evaluation_reference": str(
                metadata.get(
                    "evaluation_reference",
                    f"unverified:final-result-sha256:{metadata.get('final_result_sha256', source_checkpoint_hash)}",
                )
            ),
            "source_revision": str(metadata.get("source_revision", "seed-20260906-epoch-48")),
            "checkpoint_sha256": source_checkpoint_hash,
            "verified": False,
        },
        "calibration": {
            "required": False,
            "reference": None,
        },
        "readiness": {
            "minimum_observations": 128,
            "maximum_age_seconds": None,
        },
        "feature_contract_sha256": metadata.get("feature_contract_sha256", FEATURE_CONTRACT_SHA256),
        "runtime_adapter": metadata.get("runtime_adapter", "argus.causal-flow/v1"),
        "runtime": {
            "runtime_id": "onnxruntime",
            "minimum_runtime_version": "1.20.0",
            "opset_version": opset_version,
            "quantization": "none",
            "precision": "float32",
            "dynamic_axes": ["batch"],
            "supported_targets": [
                "x86_64-pc-windows-msvc",
                "x86_64-unknown-linux-gnu",
            ],
        },
        "architecture_metadata": {
            "parameters": metadata.get("parameters", 40851),
            "packet_dim": metadata.get("packet_dim", 12),
            "output_dim": metadata.get("output_dim", 2),
            "width": metadata.get("width", 60),
            "key_dim": metadata.get("key_dim", 32),
            "local_packets": metadata.get("local_packets", 128),
            "max_transitions": metadata.get("max_transitions", 16),
            "maximum_context_packets": metadata.get("maximum_context_packets", 10000),
            "minimum_completed_transitions": metadata.get("minimum_completed_transitions", 2),
            "loss": metadata.get(
                "loss", "(smooth_l1(log_length)+4*smooth_l1(log_per_flow_interarrival))/5"
            ),
            "selected_seed": metadata.get("selected_seed", 20260906),
            "selected_best_epoch": metadata.get("selected_best_epoch", 48),
            "selection": metadata.get(
                "selection", "lowest development validation loss before final holdout access"
            ),
        },
        "tensor_contract": {
            "inputs": {
                "local_packets": {
                    "dtype": "float32",
                    "shape": ["batch", 128, 12],
                    "description": "Local causal packet feature window",
                },
                "query_predecessor": {
                    "dtype": "float32",
                    "shape": ["batch", 12],
                    "description": "Predecessor packet features in query flow",
                },
                "query_current": {
                    "dtype": "float32",
                    "shape": ["batch", 12],
                    "description": "Current packet features in query flow",
                },
                "keys": {
                    "dtype": "float32",
                    "shape": ["batch", 16, 32],
                    "description": "Projected transition state keys",
                },
                "values": {
                    "dtype": "float32",
                    "shape": ["batch", 16, 60],
                    "description": "Projected transition values",
                },
                "successor_dynamic": {
                    "dtype": "float32",
                    "shape": ["batch", 16, 2],
                    "description": "Dynamic successor ground truth targets",
                },
                "source_ordinals": {
                    "dtype": "int64",
                    "shape": ["batch", 16],
                    "description": "Global packet sequence ordinals of transition sources",
                },
                "valid": {
                    "dtype": "bool",
                    "shape": ["batch", 16],
                    "description": "Boolean mask indicating valid transition slots",
                },
                "anchor_ordinals": {
                    "dtype": "int64",
                    "shape": ["batch"],
                    "description": "Query anchor ordinal for relative recency computation",
                },
                "context_spans": {
                    "dtype": "float32",
                    "shape": ["batch"],
                    "description": "Active horizon context span for recency normalization",
                },
                "query_gap": {
                    "dtype": "float32",
                    "shape": ["batch"],
                    "description": "Normalized gap between query predecessor and current",
                },
            },
            "outputs": {
                "prediction": {
                    "dtype": "float32",
                    "shape": ["batch", 2],
                    "description": "Predicted next packet log-length and log-interarrival",
                },
                "retrieval_weights": {
                    "dtype": "float32",
                    "shape": ["batch", 16],
                    "description": "Attention weights over valid cached transitions",
                },
                "memory_gate": {
                    "dtype": "float32",
                    "shape": ["batch", 1],
                    "description": "Learned interpolation weight for retrieved memory",
                },
            },
        },
        "parity_verification": parity_result,
    }

    canonical_manifest_path = output_manifest_path or output_onnx_path.with_name("ahem_v61.manifest.json")
    write_atomic_json(canonical_manifest_path, manifest)

    # Also maintain sidecar ahem_v61.onnx.manifest.json for runtime loaders expecting that pattern
    sidecar_manifest_path = output_onnx_path.with_suffix(output_onnx_path.suffix + ".manifest.json")
    if sidecar_manifest_path != canonical_manifest_path:
        write_atomic_json(sidecar_manifest_path, manifest)

    return {
        "onnx_path": str(output_onnx_path),
        "manifest_path": str(canonical_manifest_path),
        "sidecar_manifest_path": str(sidecar_manifest_path),
        "artifact_sha256": artifact_hash,
        "max_abs_error": parity_result["max_abs_error"],
        "parity_verified": True,
    }


def find_default_paths() -> tuple[Path, Path, Path]:
    """Find canonical paths for checkpoint, ONNX output, and manifest."""
    search_dirs = [
        Path.cwd(),
        Path.cwd().parent,
        Path(__file__).resolve().parents[3],
        Path(__file__).resolve().parents[2],
    ]
    for directory in search_dirs:
        candidate = directory / "argus-dev" / "models" / "ahem_v61_forecast_only.pt"
        if candidate.is_file():
            models_dir = candidate.parent
            return candidate, models_dir / "ahem_v61.onnx", models_dir / "ahem_v61.manifest.json"

    fallback = Path("argus-dev/models")
    return (
        fallback / "ahem_v61_forecast_only.pt",
        fallback / "ahem_v61.onnx",
        fallback / "ahem_v61.manifest.json",
    )


def main() -> None:
    default_ckpt, default_out, default_manifest = find_default_paths()
    parser = argparse.ArgumentParser(description="Export frozen AHEM v6.1 to full-precision ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_ckpt,
        help="Path to frozen ahem_v61_forecast_only.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_out,
        help="Destination path for ahem_v61.onnx",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest,
        help="Destination path for ahem_v61.manifest.json",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=OPSET_VERSION,
        help="ONNX opset version (default: 17)",
    )
    args = parser.parse_args()

    result = export_ahem_v61(
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output,
        output_manifest_path=args.manifest,
        opset_version=args.opset,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

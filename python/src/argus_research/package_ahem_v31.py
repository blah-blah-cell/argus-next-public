"""Package a qualified AHEM-v3.1 checkpoint for forecast-only shadow use."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from .ahem_v3 import AhemV31NextPacket
from .artifact_manifest import (
    ARTIFACT_FORMAT,
    ARTIFACT_FORMAT_VERSION,
    CalibrationRequirements,
    ModelArtifactManifest,
    Provenance,
    ReadinessRequirements,
    RuntimeCompatibility,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def package(
    checkpoint: Path,
    development_metrics: Path,
    external_metrics: Path,
    destination: Path,
    source_revision: str,
) -> ModelArtifactManifest:
    development = _read_json(development_metrics)
    external = _read_json(external_metrics)
    if development.get("real_data") is not True:
        raise ValueError("development benchmark is not marked as real data")
    if external.get("strict_external_win") is not True:
        raise ValueError("external AHEM-v3.1 comparison did not pass every gate")
    if external.get("not_intrusion_detection_efficacy") is not True:
        raise ValueError(
            "external evaluation lacks the forecast-only truthfulness flag"
        )
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = AhemV31NextPacket(12)
    model.load_state_dict(source["state_dict"], strict=True)
    packaged = {
        "state_dict": model.state_dict(),
        "artifact_type": "causal_temporal_forecaster",
        "architecture": "ahem-v3.1-adaptive-chunk-memory",
        "feature_schema": "packet-causal-v1",
        "supervision": "next_packet_only",
        "security_detector": False,
        "calibrated_security_verdict": False,
        "trained_on_real_data": True,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "local_packets": model.local_packets,
        "chunk_size": model.chunk_size,
        "maximum_cached_chunks": 466,
        "flat_retrieval_until_chunks": model.flat_retrieval_until_chunks,
        "development_metrics_sha256": _sha256(development_metrics),
        "external_metrics_sha256": _sha256(external_metrics),
        "source_revision": source_revision,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(packaged, temporary)
    os.replace(temporary, destination)
    manifest = ModelArtifactManifest(
        format=ARTIFACT_FORMAT,
        format_version=ARTIFACT_FORMAT_VERSION,
        feature_schema_id="packet-causal-v1",
        feature_schema_version=1,
        model_id="argus.ahem-v3.1.forecaster",
        model_version="3.1.0-research1",
        artifact_sha256=_sha256(destination),
        provenance=Provenance(
            training_dataset_reference=f"sha256:{_sha256(development_metrics)}",
            evaluation_reference=f"sha256:{_sha256(external_metrics)}",
            source_revision=source_revision,
        ),
        calibration=CalibrationRequirements(required=False, reference=None),
        readiness=ReadinessRequirements(
            minimum_observations=model.local_packets + model.chunk_size + 1,
            maximum_age_seconds=None,
        ),
        runtime=RuntimeCompatibility(
            runtime_id="argus-python-shadow",
            minimum_runtime_version="0.1.0",
            supported_targets=("x86_64-pc-windows-msvc",),
        ),
    )
    manifest.validate()
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--development-metrics", type=Path, required=True)
    parser.add_argument("--external-metrics", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    manifest = package(
        args.checkpoint,
        args.development_metrics,
        args.external_metrics,
        args.destination,
        args.source_revision,
    )
    print(manifest.to_json())


if __name__ == "__main__":
    main()

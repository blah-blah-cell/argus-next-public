"""Research-side contracts for Argus Next; no training implementation is included."""

from .artifact_manifest import (
    ARTIFACT_FORMAT,
    ARTIFACT_FORMAT_VERSION,
    CalibrationRequirements,
    ModelArtifactManifest,
    Provenance,
    ReadinessRequirements,
    RuntimeCompatibility,
)

__all__ = [
    "ARTIFACT_FORMAT",
    "ARTIFACT_FORMAT_VERSION",
    "CalibrationRequirements",
    "ModelArtifactManifest",
    "Provenance",
    "ReadinessRequirements",
    "RuntimeCompatibility",
]

"""Canonical JSON-ready model-artifact manifest, implemented using only the stdlib."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re

ARTIFACT_FORMAT = "argus-model-artifact"
ARTIFACT_FORMAT_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Provenance:
    training_dataset_reference: str
    evaluation_reference: str
    source_revision: str


@dataclass(frozen=True)
class CalibrationRequirements:
    required: bool
    reference: str | None


@dataclass(frozen=True)
class ReadinessRequirements:
    minimum_observations: int
    maximum_age_seconds: int | None


@dataclass(frozen=True)
class RuntimeCompatibility:
    runtime_id: str
    minimum_runtime_version: str
    supported_targets: tuple[str, ...]


@dataclass(frozen=True)
class ModelArtifactManifest:
    format: str
    format_version: int
    feature_schema_id: str
    feature_schema_version: int
    model_id: str
    model_version: str
    artifact_sha256: str
    provenance: Provenance
    calibration: CalibrationRequirements
    readiness: ReadinessRequirements
    runtime: RuntimeCompatibility

    def validate(self) -> None:
        if self.format != ARTIFACT_FORMAT:
            raise ValueError("unsupported format")
        if self.format_version != ARTIFACT_FORMAT_VERSION:
            raise ValueError("unsupported format version")
        identifiers = (
            ("feature_schema_id", self.feature_schema_id),
            ("model_id", self.model_id),
            ("model_version", self.model_version),
            ("source_revision", self.provenance.source_revision),
            ("runtime_id", self.runtime.runtime_id),
            ("minimum_runtime_version", self.runtime.minimum_runtime_version),
        )
        for name, value in identifiers:
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"invalid identifier: {name}")
        if self.feature_schema_version <= 0:
            raise ValueError("feature_schema_version must be positive")
        if not _SHA256.fullmatch(self.artifact_sha256):
            raise ValueError("artifact_sha256 must be lowercase SHA-256 hex")
        references = (
            ("training_dataset_reference", self.provenance.training_dataset_reference),
            ("evaluation_reference", self.provenance.evaluation_reference),
        )
        for name, value in references:
            if not value or any(char.isspace() for char in value):
                raise ValueError(f"invalid reference: {name}")
        calibration = self.calibration
        has_valid_calibration = (
            calibration.reference is not None
            and bool(calibration.reference)
            and not any(char.isspace() for char in calibration.reference)
        )
        if calibration.required != has_valid_calibration:
            raise ValueError(
                "calibration reference must be present exactly when required"
            )
        maximum_age = self.readiness.maximum_age_seconds
        if self.readiness.minimum_observations < 0 or (
            maximum_age is not None and maximum_age <= 0
        ):
            raise ValueError("invalid readiness requirements")
        targets = self.runtime.supported_targets
        invalid_target = any(
            not target or any(char.isspace() for char in target) for target in targets
        )
        if not targets or len(set(targets)) != len(targets) or invalid_target:
            raise ValueError(
                "runtime targets must be nonempty, unique, and whitespace-free"
            )

    def to_json(self) -> str:
        """Validate then return a canonical, compact JSON manifest."""
        self.validate()
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

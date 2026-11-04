//! Strict, dependency-free contract for a trained-model artifact manifest.
//!
//! This crate deliberately contains no model bytes, inference implementation,
//! training code, or serialization dependency. Producers may encode a validated
//! [`ModelArtifactManifest`] as documented in `docs/MODEL_ARTIFACT_CONTRACT.md`.

#![forbid(unsafe_code)]

use std::fmt;

pub const ARTIFACT_FORMAT: &str = "argus-model-artifact";
pub const ARTIFACT_FORMAT_VERSION: u32 = 1;

/// Metadata required to load a model artifact safely and reproduce its claims.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelArtifactManifest {
    pub format: String,
    pub format_version: u32,
    pub feature_schema_id: String,
    pub feature_schema_version: u32,
    pub model_id: String,
    pub model_version: String,
    pub artifact_sha256: String,
    pub provenance: Provenance,
    pub calibration: CalibrationRequirements,
    pub readiness: ReadinessRequirements,
    pub runtime: RuntimeCompatibility,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Provenance {
    pub training_dataset_reference: String,
    pub evaluation_reference: String,
    pub source_revision: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CalibrationRequirements {
    pub required: bool,
    /// A stable reference to calibration parameters included with the artifact.
    pub reference: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadinessRequirements {
    pub minimum_observations: u64,
    pub maximum_age_seconds: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeCompatibility {
    pub runtime_id: String,
    pub minimum_runtime_version: String,
    pub supported_targets: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ManifestValidationError {
    UnsupportedFormat,
    UnsupportedFormatVersion,
    EmptyField(&'static str),
    InvalidIdentifier(&'static str),
    InvalidChecksum,
    MissingCalibrationReference,
    UnexpectedCalibrationReference,
    EmptyTargetList,
    DuplicateTarget,
    InvalidTarget,
    ZeroMaximumAge,
}

impl fmt::Display for ManifestValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid model artifact manifest: {self:?}")
    }
}

impl std::error::Error for ManifestValidationError {}

impl ModelArtifactManifest {
    /// Validates every required field before an artifact is accepted for loading.
    ///
    /// # Errors
    /// Returns a precise error for the first violated artifact-contract invariant.
    pub fn validate(&self) -> Result<(), ManifestValidationError> {
        if self.format != ARTIFACT_FORMAT {
            return Err(ManifestValidationError::UnsupportedFormat);
        }
        if self.format_version != ARTIFACT_FORMAT_VERSION {
            return Err(ManifestValidationError::UnsupportedFormatVersion);
        }
        required_identifier(&self.feature_schema_id, "feature_schema_id")?;
        nonzero(self.feature_schema_version, "feature_schema_version")?;
        required_identifier(&self.model_id, "model_id")?;
        required_identifier(&self.model_version, "model_version")?;
        if !is_sha256(&self.artifact_sha256) {
            return Err(ManifestValidationError::InvalidChecksum);
        }
        required_reference(
            &self.provenance.training_dataset_reference,
            "training_dataset_reference",
        )?;
        required_reference(
            &self.provenance.evaluation_reference,
            "evaluation_reference",
        )?;
        required_identifier(&self.provenance.source_revision, "source_revision")?;
        validate_calibration(&self.calibration)?;
        validate_readiness(&self.readiness)?;
        validate_runtime(&self.runtime)
    }
}

fn required_identifier(value: &str, field: &'static str) -> Result<(), ManifestValidationError> {
    if value.is_empty() {
        return Err(ManifestValidationError::EmptyField(field));
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(ManifestValidationError::InvalidIdentifier(field));
    }
    Ok(())
}

fn required_reference(value: &str, field: &'static str) -> Result<(), ManifestValidationError> {
    if value.is_empty() {
        return Err(ManifestValidationError::EmptyField(field));
    }
    if value.chars().any(char::is_whitespace) {
        return Err(ManifestValidationError::InvalidIdentifier(field));
    }
    Ok(())
}

fn nonzero(value: u32, field: &'static str) -> Result<(), ManifestValidationError> {
    (value != 0)
        .then_some(())
        .ok_or(ManifestValidationError::EmptyField(field))
}

fn validate_calibration(value: &CalibrationRequirements) -> Result<(), ManifestValidationError> {
    match (value.required, value.reference.as_deref()) {
        (true, Some(reference))
            if !reference.is_empty() && !reference.chars().any(char::is_whitespace) =>
        {
            Ok(())
        }
        (true, _) => Err(ManifestValidationError::MissingCalibrationReference),
        (false, None) => Ok(()),
        (false, Some(_)) => Err(ManifestValidationError::UnexpectedCalibrationReference),
    }
}

fn validate_readiness(value: &ReadinessRequirements) -> Result<(), ManifestValidationError> {
    if value.maximum_age_seconds == Some(0) {
        return Err(ManifestValidationError::ZeroMaximumAge);
    }
    Ok(())
}

fn validate_runtime(value: &RuntimeCompatibility) -> Result<(), ManifestValidationError> {
    required_identifier(&value.runtime_id, "runtime.runtime_id")?;
    required_identifier(
        &value.minimum_runtime_version,
        "runtime.minimum_runtime_version",
    )?;
    if value.supported_targets.is_empty() {
        return Err(ManifestValidationError::EmptyTargetList);
    }
    for (index, target) in value.supported_targets.iter().enumerate() {
        if target.is_empty() || target.chars().any(char::is_whitespace) {
            return Err(ManifestValidationError::InvalidTarget);
        }
        if value.supported_targets[..index].contains(target) {
            return Err(ManifestValidationError::DuplicateTarget);
        }
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> ModelArtifactManifest {
        ModelArtifactManifest {
            format: ARTIFACT_FORMAT.into(),
            format_version: ARTIFACT_FORMAT_VERSION,
            feature_schema_id: "argus.flow.v1".into(),
            feature_schema_version: 1,
            model_id: "argus.anomaly-detector".into(),
            model_version: "1.0.0".into(),
            artifact_sha256: "a".repeat(64),
            provenance: Provenance {
                training_dataset_reference: "dataset:argus-training-2026-01".into(),
                evaluation_reference: "evaluation:argus-eval-2026-01".into(),
                source_revision: "abc123".into(),
            },
            calibration: CalibrationRequirements {
                required: true,
                reference: Some("calibration:platt-v1".into()),
            },
            readiness: ReadinessRequirements {
                minimum_observations: 10,
                maximum_age_seconds: Some(60),
            },
            runtime: RuntimeCompatibility {
                runtime_id: "onnxruntime".into(),
                minimum_runtime_version: "1.20.0".into(),
                supported_targets: vec!["x86_64-pc-windows-msvc".into()],
            },
        }
    }

    #[test]
    fn accepts_complete_manifest() {
        assert_eq!(manifest().validate(), Ok(()));
    }

    #[test]
    fn rejects_checksum_that_is_not_lowercase_sha256() {
        let mut value = manifest();
        value.artifact_sha256 = "A".repeat(64);
        assert_eq!(
            value.validate(),
            Err(ManifestValidationError::InvalidChecksum)
        );
    }

    #[test]
    fn calibration_reference_must_match_requirement() {
        let mut value = manifest();
        value.calibration.reference = None;
        assert_eq!(
            value.validate(),
            Err(ManifestValidationError::MissingCalibrationReference)
        );
        value.calibration.required = false;
        value.calibration.reference = Some("calibration:x".into());
        assert_eq!(
            value.validate(),
            Err(ManifestValidationError::UnexpectedCalibrationReference)
        );
    }

    #[test]
    fn runtime_targets_are_nonempty_unique_and_whitespace_free() {
        let mut value = manifest();
        value
            .runtime
            .supported_targets
            .push("x86_64-pc-windows-msvc".into());
        assert_eq!(
            value.validate(),
            Err(ManifestValidationError::DuplicateTarget)
        );
        value.runtime.supported_targets = vec![" ".into()];
        assert_eq!(
            value.validate(),
            Err(ManifestValidationError::InvalidTarget)
        );
    }

    #[test]
    fn format_and_schema_versions_are_strict() {
        let mut value = manifest();
        value.format_version = 2;
        assert_eq!(
            value.validate(),
            Err(ManifestValidationError::UnsupportedFormatVersion)
        );
        value.format_version = 1;
        value.feature_schema_version = 0;
        assert_eq!(
            value.validate(),
            Err(ManifestValidationError::EmptyField(
                "feature_schema_version"
            ))
        );
    }
}

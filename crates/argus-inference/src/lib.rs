//! Fail-closed native ONNX Runtime inference for admitted Argus experts.
//!
//! This crate never downloads a runtime or model and never substitutes weights.
//! The caller supplies an ONNX Runtime dynamic library, a model artifact, its
//! validated manifest, and an exact tensor contract.

#![forbid(unsafe_code)]

use std::{
    fmt,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    sync::Mutex,
};

use argus_features::{AHEM_V34_SCHEMA_ID, AhemV34Inputs};
use argus_model_artifact::{
    CalibrationRequirements, ManifestValidationError, ModelArtifactManifest, Provenance,
    ReadinessRequirements, RuntimeCompatibility,
};
use argus_moe::{Expert, ExpertOutput};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::{TensorElementType, TensorRef, ValueType},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

pub const RUNTIME_ID: &str = "onnxruntime";
pub const RUNTIME_PATH_ENVIRONMENT: &str = "ARGUS_ONNX_RUNTIME";
pub const CPU_THREADS_ENVIRONMENT: &str = "ARGUS_ONNX_CPU_THREADS";

/// ONNX CPU kernels in the currently admitted laptop-sized graphs benchmark
/// faster with one thread. Operators may override this for larger graphs,
/// without changing model artifacts or the prediction contract.
#[must_use]
pub fn cpu_intra_threads() -> usize {
    std::env::var(CPU_THREADS_ENVIRONMENT)
        .ok()
        .as_deref()
        .and_then(parse_cpu_threads)
        .unwrap_or(1)
}

fn parse_cpu_threads(value: &str) -> Option<usize> {
    value
        .parse::<usize>()
        .ok()
        .filter(|threads| (1..=8).contains(threads))
}

/// Finds an operator-configured or app-adjacent ONNX Runtime library without
/// loading it. Compatibility is still verified during model admission.
///
/// # Errors
/// Returns an unavailable error when no real runtime library exists.
pub fn discover_runtime_library(configured: Option<&Path>) -> Result<PathBuf, InferenceError> {
    let configured = configured.map(Path::to_path_buf).or_else(|| {
        std::env::var_os(RUNTIME_PATH_ENVIRONMENT)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    });
    if let Some(path) = configured {
        return path.is_file().then_some(path).ok_or_else(|| {
            InferenceError::RuntimeLoad(format!(
                "{RUNTIME_PATH_ENVIRONMENT} does not point to a file"
            ))
        });
    }
    let adjacent = std::env::current_exe().ok().and_then(|executable| {
        executable
            .parent()
            .map(|parent| parent.join(runtime_file_name()))
    });
    if let Some(path) = adjacent.filter(|path| path.is_file()) {
        return Ok(path);
    }
    if let Ok(executable) = std::env::current_exe()
        && let Some(directory) = executable.parent()
    {
        for candidate in directory.ancestors().take(4) {
            // App-adjacent venvs support extracted packages; checkout ancestors
            // must be recognizable Cargo output paths, never the process CWD.
            let Ok(relative) = directory.strip_prefix(candidate) else {
                continue;
            };
            let checkout = matches!(
                relative.to_str(),
                Some(
                    "target/debug"
                        | "target/release"
                        | "target/debug/deps"
                        | "target/release/deps"
                        | "target\\debug"
                        | "target\\release"
                        | "target\\debug\\deps"
                        | "target\\release\\deps"
                )
            ) && candidate.join("Cargo.toml").is_file()
                && candidate.join("python/pyproject.toml").is_file();
            if candidate == directory || checkout {
                let runtime = candidate
                    .join("python/.venv/Lib/site-packages/onnxruntime/capi")
                    .join(runtime_file_name());
                if runtime.is_file() {
                    return Ok(runtime);
                }
            }
        }
    }
    Err(InferenceError::RuntimeLoad(format!(
        "set {RUNTIME_PATH_ENVIRONMENT} or place {} beside the application",
        runtime_file_name()
    )))
}

#[derive(Deserialize)]
struct ManifestDocument {
    format: String,
    format_version: u32,
    feature_schema_id: Option<String>,
    feature_schema: Option<String>,
    #[serde(default = "default_feature_schema_version")]
    feature_schema_version: u32,
    model_id: String,
    model_version: String,
    artifact_sha256: String,
    #[serde(default)]
    provenance: ProvenanceDocument,
    #[serde(default)]
    calibration: CalibrationDocument,
    #[serde(default)]
    readiness: ReadinessDocument,
    runtime: RuntimeDocument,
}

const fn default_feature_schema_version() -> u32 {
    1
}

#[derive(Default, Deserialize)]
struct ProvenanceDocument {
    #[serde(default = "default_provenance_ref")]
    training_dataset_reference: String,
    #[serde(default = "default_provenance_ref")]
    evaluation_reference: String,
    #[serde(default = "default_source_rev")]
    source_revision: String,
}

fn default_provenance_ref() -> String {
    "unverified:provenance".to_owned()
}

fn default_source_rev() -> String {
    "unverified".to_owned()
}

#[derive(Default, Deserialize)]
struct CalibrationDocument {
    #[serde(default)]
    required: bool,
    #[serde(default)]
    reference: Option<String>,
}

#[derive(Default, Deserialize)]
struct ReadinessDocument {
    #[serde(default = "default_minimum_observations")]
    minimum_observations: u64,
    #[serde(default)]
    maximum_age_seconds: Option<u64>,
}

const fn default_minimum_observations() -> u64 {
    128
}

#[derive(Deserialize)]
struct RuntimeDocument {
    runtime_id: String,
    #[serde(default = "default_minimum_runtime_version")]
    minimum_runtime_version: String,
    #[serde(default = "default_supported_targets")]
    supported_targets: Vec<String>,
}

fn default_minimum_runtime_version() -> String {
    "1.20.0".to_owned()
}

fn default_supported_targets() -> Vec<String> {
    vec![
        "x86_64-pc-windows-msvc".to_owned(),
        "x86_64-unknown-linux-gnu".to_owned(),
    ]
}

/// Reads and validates a model manifest before any runtime or model is loaded.
///
/// Unknown extension fields are deliberately ignored, while every field in the
/// stable artifact contract remains mandatory.
///
/// # Errors
/// Returns an artifact-read error for missing or malformed JSON and the exact
/// manifest validation error for a structurally invalid contract.
pub fn read_manifest(path: &Path) -> Result<ModelArtifactManifest, InferenceError> {
    let bytes =
        std::fs::read(path).map_err(|error| InferenceError::ArtifactRead(error.to_string()))?;
    let document: ManifestDocument = serde_json::from_slice(&bytes)
        .map_err(|error| InferenceError::ArtifactRead(error.to_string()))?;
    let feature_schema_id = document
        .feature_schema_id
        .or(document.feature_schema)
        .ok_or_else(|| InferenceError::ArtifactRead("missing feature_schema_id".to_owned()))?;
    let manifest = ModelArtifactManifest {
        format: document.format,
        format_version: document.format_version,
        feature_schema_id,
        feature_schema_version: document.feature_schema_version,
        model_id: document.model_id,
        model_version: document.model_version,
        artifact_sha256: document.artifact_sha256,
        provenance: Provenance {
            training_dataset_reference: document.provenance.training_dataset_reference,
            evaluation_reference: document.provenance.evaluation_reference,
            source_revision: document.provenance.source_revision,
        },
        calibration: CalibrationRequirements {
            required: document.calibration.required,
            reference: document.calibration.reference,
        },
        readiness: ReadinessRequirements {
            minimum_observations: document.readiness.minimum_observations,
            maximum_age_seconds: document.readiness.maximum_age_seconds,
        },
        runtime: RuntimeCompatibility {
            runtime_id: document.runtime.runtime_id,
            minimum_runtime_version: document.runtime.minimum_runtime_version,
            supported_targets: document.runtime.supported_targets,
        },
    };
    manifest.validate()?;
    Ok(manifest)
}

const fn runtime_file_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

/// Exact single-request tensor boundary. Dynamic model dimensions are rejected
/// at admission so the live path cannot guess a shape.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorContract {
    pub input_name: String,
    pub input_shape: Vec<i64>,
    pub output_name: String,
    pub output_shape: Vec<i64>,
    pub probability_index: usize,
}

/// Names for the fixed five-input AHEM v3.4 cache-native graph boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AhemV34TensorContract {
    pub local_packets: String,
    pub flow_packets: String,
    pub flow_ordinals: String,
    pub flow_valid: String,
    pub context_spans: String,
    pub prediction: String,
}

impl Default for AhemV34TensorContract {
    fn default() -> Self {
        Self {
            local_packets: "local_packets".to_owned(),
            flow_packets: "flow_packets".to_owned(),
            flow_ordinals: "flow_ordinals".to_owned(),
            flow_valid: "flow_valid".to_owned(),
            context_spans: "context_spans".to_owned(),
            prediction: "prediction".to_owned(),
        }
    }
}

impl AhemV34TensorContract {
    /// Rejects empty or duplicate graph names before a runtime is loaded.
    ///
    /// # Errors
    /// Returns an invalid-contract error for ambiguous names.
    pub fn validate(&self) -> Result<(), InferenceError> {
        let names = [
            self.local_packets.as_str(),
            self.flow_packets.as_str(),
            self.flow_ordinals.as_str(),
            self.flow_valid.as_str(),
            self.context_spans.as_str(),
            self.prediction.as_str(),
        ];
        if names.iter().any(|name| name.is_empty()) {
            return Err(InferenceError::InvalidTensorContract(
                "AHEM graph names must be non-empty".to_owned(),
            ));
        }
        for (index, name) in names.iter().enumerate() {
            if names[..index].contains(name) {
                return Err(InferenceError::InvalidTensorContract(
                    "AHEM graph names must be unique".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AhemV34Forecast {
    pub log_frame_length: f32,
    pub log_per_flow_interarrival_seconds: f32,
}

pub const AHEM_V61_LOCAL_PACKETS: usize = 128;
pub const AHEM_V61_PACKET_DIM: usize = 12;
pub const AHEM_V61_KEY_DIM: usize = 32;
pub const AHEM_V61_VALUE_DIM: usize = 60;
pub const AHEM_V61_OUTPUT_DIM: usize = 2;
pub const AHEM_V61_MAX_TRANSITIONS: usize = 16;
pub const AHEM_V61_SCHEMA_ID: &str = "packet-causal-flow-v2";
pub const AHEM_V61_SCHEMA_VERSION: u32 = 1;

/// Names for the 11-input AHEM v6.1 cache-native graph boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AhemV61TensorContract {
    pub local_packets: String,
    pub query_predecessor: String,
    pub query_current: String,
    pub keys: String,
    pub values: String,
    pub successor_dynamic: String,
    pub source_ordinals: String,
    pub valid: String,
    pub anchor_ordinals: String,
    pub context_spans: String,
    pub query_gap: String,
    pub prediction: String,
    pub retrieval_weights: String,
    pub memory_gate: String,
}

impl Default for AhemV61TensorContract {
    fn default() -> Self {
        Self {
            local_packets: "local_packets".to_owned(),
            query_predecessor: "query_predecessor".to_owned(),
            query_current: "query_current".to_owned(),
            keys: "keys".to_owned(),
            values: "values".to_owned(),
            successor_dynamic: "successor_dynamic".to_owned(),
            source_ordinals: "source_ordinals".to_owned(),
            valid: "valid".to_owned(),
            anchor_ordinals: "anchor_ordinals".to_owned(),
            context_spans: "context_spans".to_owned(),
            query_gap: "query_gap".to_owned(),
            prediction: "prediction".to_owned(),
            retrieval_weights: "retrieval_weights".to_owned(),
            memory_gate: "memory_gate".to_owned(),
        }
    }
}

impl AhemV61TensorContract {
    /// Rejects empty or duplicate graph names before a runtime is loaded.
    ///
    /// # Errors
    /// Returns an invalid-contract error for ambiguous names.
    pub fn validate(&self) -> Result<(), InferenceError> {
        let names = [
            self.local_packets.as_str(),
            self.query_predecessor.as_str(),
            self.query_current.as_str(),
            self.keys.as_str(),
            self.values.as_str(),
            self.successor_dynamic.as_str(),
            self.source_ordinals.as_str(),
            self.valid.as_str(),
            self.anchor_ordinals.as_str(),
            self.context_spans.as_str(),
            self.query_gap.as_str(),
            self.prediction.as_str(),
            self.retrieval_weights.as_str(),
            self.memory_gate.as_str(),
        ];
        if names.iter().any(|name| name.is_empty()) {
            return Err(InferenceError::InvalidTensorContract(
                "AHEM v6.1 graph names must be non-empty".to_owned(),
            ));
        }
        for (index, name) in names.iter().enumerate() {
            if names[..index].contains(name) {
                return Err(InferenceError::InvalidTensorContract(
                    "AHEM v6.1 graph names must be unique".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AhemV61Forecast {
    pub predicted_log_packet_length: f32,
    pub predicted_log_interarrival: f32,
    pub retrieval_weights: [f32; AHEM_V61_MAX_TRANSITIONS],
    pub memory_gate: f32,
}

impl AhemV61Forecast {
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.predicted_log_packet_length.is_finite()
            && self.predicted_log_interarrival.is_finite()
            && (0.0..=1.0).contains(&self.memory_gate)
            && self
                .retrieval_weights
                .iter()
                .all(|weight| weight.is_finite() && (0.0..=1.0).contains(weight))
    }

    #[must_use]
    pub fn is_close(&self, other: &Self, epsilon: f32) -> bool {
        (self.predicted_log_packet_length - other.predicted_log_packet_length).abs() <= epsilon
            && (self.predicted_log_interarrival - other.predicted_log_interarrival).abs() <= epsilon
            && (self.memory_gate - other.memory_gate).abs() <= epsilon
            && self
                .retrieval_weights
                .iter()
                .zip(other.retrieval_weights.iter())
                .all(|(left, right)| (left - right).abs() <= epsilon)
    }
}

/// Fixed-shape single-sample inputs for AHEM v6.1 cache-native graph.
#[derive(Clone, Debug, PartialEq)]
pub struct AhemV61Inputs {
    pub local_packets: [[f32; AHEM_V61_PACKET_DIM]; AHEM_V61_LOCAL_PACKETS],
    pub query_predecessor: [f32; AHEM_V61_PACKET_DIM],
    pub query_current: [f32; AHEM_V61_PACKET_DIM],
    pub keys: [[f32; AHEM_V61_KEY_DIM]; AHEM_V61_MAX_TRANSITIONS],
    pub values: [[f32; AHEM_V61_VALUE_DIM]; AHEM_V61_MAX_TRANSITIONS],
    pub successor_dynamic: [[f32; AHEM_V61_OUTPUT_DIM]; AHEM_V61_MAX_TRANSITIONS],
    pub source_ordinals: [i64; AHEM_V61_MAX_TRANSITIONS],
    pub valid: [bool; AHEM_V61_MAX_TRANSITIONS],
    pub anchor_ordinals: [i64; 1],
    pub context_spans: [f32; 1],
    pub query_gap: [f32; 1],
}

impl Default for AhemV61Inputs {
    fn default() -> Self {
        Self {
            local_packets: [[0.0; AHEM_V61_PACKET_DIM]; AHEM_V61_LOCAL_PACKETS],
            query_predecessor: [0.0; AHEM_V61_PACKET_DIM],
            query_current: [0.0; AHEM_V61_PACKET_DIM],
            keys: [[0.0; AHEM_V61_KEY_DIM]; AHEM_V61_MAX_TRANSITIONS],
            values: [[0.0; AHEM_V61_VALUE_DIM]; AHEM_V61_MAX_TRANSITIONS],
            successor_dynamic: [[0.0; AHEM_V61_OUTPUT_DIM]; AHEM_V61_MAX_TRANSITIONS],
            source_ordinals: [0; AHEM_V61_MAX_TRANSITIONS],
            valid: [false; AHEM_V61_MAX_TRANSITIONS],
            anchor_ordinals: [0],
            context_spans: [1.0],
            query_gap: [0.0],
        }
    }
}

impl AhemV61Inputs {
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.local_packets.iter().flatten().all(|v| v.is_finite())
            && self.query_predecessor.iter().all(|v| v.is_finite())
            && self.query_current.iter().all(|v| v.is_finite())
            && self.keys.iter().flatten().all(|v| v.is_finite())
            && self.values.iter().flatten().all(|v| v.is_finite())
            && self
                .successor_dynamic
                .iter()
                .flatten()
                .all(|v| v.is_finite())
            && self.context_spans.iter().all(|v| v.is_finite())
            && self.query_gap.iter().all(|v| v.is_finite())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeedforwardTensorContract {
    pub input_name: String,
    pub context_packets: usize,
    pub feature_dim: usize,
    pub output_name: String,
}

impl Default for FeedforwardTensorContract {
    fn default() -> Self {
        Self {
            input_name: "context".to_owned(),
            context_packets: 31,
            feature_dim: 12,
            output_name: "prediction".to_owned(),
        }
    }
}

impl FeedforwardTensorContract {
    /// Rejects empty names or non-positive dimensions.
    ///
    /// # Errors
    /// Returns an invalid-contract error if names are empty or equal, or dimensions are zero.
    pub fn validate(&self) -> Result<(), InferenceError> {
        if self.input_name.is_empty() || self.output_name.is_empty() {
            return Err(InferenceError::InvalidTensorContract(
                "feedforward input and output names must be non-empty".to_owned(),
            ));
        }
        if self.input_name == self.output_name {
            return Err(InferenceError::InvalidTensorContract(
                "feedforward input and output names must be distinct".to_owned(),
            ));
        }
        if self.context_packets == 0 || self.feature_dim == 0 {
            return Err(InferenceError::InvalidTensorContract(
                "feedforward dimensions must be positive".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeedforwardForecast {
    pub predicted_log_packet_length: f32,
    pub predicted_log_interarrival: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RouterLogits {
    pub logits: [f32; 4],
}

impl RouterLogits {
    #[must_use]
    pub fn softmax(&self) -> [f32; 4] {
        let max = self
            .logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exp = [0.0_f32; 4];
        let mut sum = 0.0_f32;
        for (slot, logit) in exp.iter_mut().zip(self.logits) {
            *slot = (logit - max).exp();
            sum += *slot;
        }
        if sum > 0.0 {
            for value in &mut exp {
                *value /= sum;
            }
        }
        exp
    }

    #[must_use]
    pub fn argmax(&self) -> usize {
        let mut best_idx = 0;
        let mut best_val = self.logits[0];
        for (i, &val) in self.logits.iter().enumerate().skip(1) {
            if val > best_val {
                best_val = val;
                best_idx = i;
            }
        }
        best_idx
    }
}

impl TensorContract {
    /// Validates names, shapes, and the selected probability element.
    ///
    /// # Errors
    /// Returns an error if any dimension is non-positive, a name is empty, a
    /// shape overflows, or the probability index is outside the output tensor.
    pub fn validate(&self) -> Result<(), InferenceError> {
        if self.input_name.is_empty() || self.output_name.is_empty() {
            return Err(InferenceError::InvalidTensorContract(
                "input and output names must be non-empty".to_owned(),
            ));
        }
        let _ = element_count(&self.input_shape, "input")?;
        let output_count = element_count(&self.output_shape, "output")?;
        if self.probability_index >= output_count {
            return Err(InferenceError::InvalidTensorContract(
                "probability index is outside the output tensor".to_owned(),
            ));
        }
        Ok(())
    }

    fn input_elements(&self) -> Result<usize, InferenceError> {
        element_count(&self.input_shape, "input")
    }
}

#[derive(Debug)]
pub enum InferenceError {
    Manifest(ManifestValidationError),
    UnsupportedRuntime(String),
    UnsupportedTarget(String),
    RuntimeTooOld { required: String, provided: String },
    UncalibratedArtifact,
    ArtifactRead(String),
    ArtifactChecksumMismatch,
    InvalidTensorContract(String),
    RuntimeLoad(String),
    SessionLoad(String),
    ModelIoMismatch(String),
    InvalidFeatures,
    SessionUnavailable,
    Inference(String),
    InvalidOutput,
}

impl fmt::Display for InferenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manifest(error) => write!(formatter, "{error}"),
            Self::UnsupportedRuntime(runtime) => {
                write!(formatter, "unsupported inference runtime: {runtime}")
            }
            Self::UnsupportedTarget(target) => {
                write!(formatter, "model does not support target {target}")
            }
            Self::RuntimeTooOld { required, provided } => write!(
                formatter,
                "ONNX Runtime {provided} is older than required {required}"
            ),
            Self::UncalibratedArtifact => {
                formatter.write_str("artifact does not declare required probability calibration")
            }
            Self::ArtifactRead(detail) => write!(formatter, "artifact read failed: {detail}"),
            Self::ArtifactChecksumMismatch => formatter.write_str("artifact checksum mismatch"),
            Self::InvalidTensorContract(detail) => {
                write!(formatter, "invalid tensor contract: {detail}")
            }
            Self::RuntimeLoad(detail) => write!(formatter, "ONNX Runtime load failed: {detail}"),
            Self::SessionLoad(detail) => write!(formatter, "ONNX session load failed: {detail}"),
            Self::ModelIoMismatch(detail) => write!(formatter, "model I/O mismatch: {detail}"),
            Self::InvalidFeatures => {
                formatter.write_str("features violate the admitted tensor contract")
            }
            Self::SessionUnavailable => {
                formatter.write_str("inference session lock is unavailable")
            }
            Self::Inference(detail) => write!(formatter, "ONNX inference failed: {detail}"),
            Self::InvalidOutput => formatter.write_str("model output is not finite"),
        }
    }
}

impl std::error::Error for InferenceError {}

impl From<ManifestValidationError> for InferenceError {
    fn from(value: ManifestValidationError) -> Self {
        Self::Manifest(value)
    }
}

/// A native ONNX expert admitted only after provenance, checksum, target, and
/// graph I/O validation. ONNX Runtime's built-in CPU provider is the portable
/// fallback when no explicitly configured accelerator handles the graph.
pub struct OnnxProbabilityExpert {
    id: String,
    schema_id: String,
    schema_version: u32,
    contract: TensorContract,
    session: Mutex<Session>,
    model_path: PathBuf,
    execution_provider: String,
}

/// Native cache-native AHEM forecaster. Its output is a next-packet forecast,
/// never an attack probability or security verdict.
pub struct OnnxAhemV34Forecaster {
    id: String,
    contract: AhemV34TensorContract,
    session: Mutex<Session>,
    model_path: PathBuf,
    execution_provider: String,
}

impl OnnxAhemV34Forecaster {
    /// Loads an exact fixed-shape AHEM v3.4 graph after manifest, checksum,
    /// target, runtime, schema, and every graph outlet have been validated.
    ///
    /// # Errors
    /// Fails closed on any artifact or graph-contract mismatch.
    pub fn load(
        runtime_library: &Path,
        model_path: &Path,
        manifest: &ModelArtifactManifest,
        contract: AhemV34TensorContract,
    ) -> Result<Self, InferenceError> {
        manifest.validate()?;
        validate_runtime_compatibility(manifest)?;
        if manifest.feature_schema_id != AHEM_V34_SCHEMA_ID || manifest.feature_schema_version != 34
        {
            return Err(InferenceError::ModelIoMismatch(format!(
                "AHEM requires feature schema {AHEM_V34_SCHEMA_ID}/34"
            )));
        }
        contract.validate()?;
        verify_sha256(model_path, &manifest.artifact_sha256)?;

        ort::init_from(runtime_library)
            .map_err(|error| InferenceError::RuntimeLoad(error.to_string()))?
            .with_name("argus-ahem-v34-inference")
            .commit();
        let session = build_tuned_session(model_path)
            .map_err(|error| InferenceError::SessionLoad(error.to_string()))?;
        validate_ahem_v34_model_io(&session, &contract)?;
        Ok(Self {
            id: manifest.model_id.clone(),
            contract,
            session: Mutex::new(session),
            model_path: model_path.to_path_buf(),
            execution_provider: "CPUExecutionProvider".to_owned(),
        })
    }

    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    #[must_use]
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    #[must_use]
    pub fn execution_provider(&self) -> &str {
        &self.execution_provider
    }

    /// Runs one fixed batch after verifying that the cache produced finite
    /// floating-point inputs and a valid anchor lane.
    ///
    /// # Errors
    /// Returns an error rather than substituting a prediction when inputs,
    /// session execution, or outputs violate the admitted contract.
    pub fn forecast(&self, inputs: &AhemV34Inputs) -> Result<AhemV34Forecast, InferenceError> {
        if inputs
            .local_packets
            .iter()
            .flatten()
            .chain(inputs.flow_packets.iter().flatten())
            .chain(inputs.context_spans.iter())
            .any(|value| !value.is_finite())
            || !inputs.flow_valid[0]
        {
            return Err(InferenceError::InvalidFeatures);
        }
        let local =
            TensorRef::from_array_view((vec![1, 128, 13], inputs.local_packets.as_flattened()))
                .map_err(|error| InferenceError::Inference(error.to_string()))?;
        let flow =
            TensorRef::from_array_view((vec![1, 17, 12], inputs.flow_packets.as_flattened()))
                .map_err(|error| InferenceError::Inference(error.to_string()))?;
        let ordinals = TensorRef::from_array_view((vec![1, 17], inputs.flow_ordinals.as_slice()))
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        let valid = TensorRef::from_array_view((vec![1, 17], inputs.flow_valid.as_slice()))
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        let spans = TensorRef::from_array_view((vec![1], inputs.context_spans.as_slice()))
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| InferenceError::SessionUnavailable)?;
        let outputs = session
            .run(ort::inputs![
                self.contract.local_packets.as_str() => local,
                self.contract.flow_packets.as_str() => flow,
                self.contract.flow_ordinals.as_str() => ordinals,
                self.contract.flow_valid.as_str() => valid,
                self.contract.context_spans.as_str() => spans,
            ])
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        let output = outputs.get(&self.contract.prediction).ok_or_else(|| {
            InferenceError::ModelIoMismatch("declared AHEM prediction disappeared".to_owned())
        })?;
        let (shape, values) = output
            .try_extract_tensor::<f32>()
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        if shape.as_ref() != [1, 2] {
            return Err(InferenceError::ModelIoMismatch(
                "AHEM prediction shape changed".to_owned(),
            ));
        }
        let forecast = AhemV34Forecast {
            log_frame_length: values[0],
            log_per_flow_interarrival_seconds: values[1],
        };
        if !forecast.log_frame_length.is_finite()
            || !forecast.log_per_flow_interarrival_seconds.is_finite()
            || forecast.log_frame_length < 0.0
            || forecast.log_per_flow_interarrival_seconds < 0.0
        {
            return Err(InferenceError::InvalidOutput);
        }
        Ok(forecast)
    }
}

/// Native AHEM v6.1 state-transition forecaster wrapping ONNX Runtime.
/// Its outputs are packet forecasts, not attack probabilities.
pub struct AhemV61Expert {
    id: String,
    contract: AhemV61TensorContract,
    session: Mutex<Session>,
    model_path: PathBuf,
    execution_provider: String,
}

pub type AhemV61Forecaster = AhemV61Expert;
pub type OnnxAhemV61Expert = AhemV61Expert;
pub type OnnxAhemV61Forecaster = AhemV61Expert;

impl AhemV61Expert {
    /// Loads an exact AHEM v6.1 graph after manifest, checksum, runtime, and
    /// graph outlet validation.
    ///
    /// Honors `ARGUS_ONNX_CPU_THREADS`.
    ///
    /// # Errors
    /// Fails closed on any artifact or contract mismatch.
    pub fn load(
        runtime_library: &Path,
        model_path: &Path,
        manifest: &ModelArtifactManifest,
        contract: AhemV61TensorContract,
    ) -> Result<Self, InferenceError> {
        Self::load_with_threads(runtime_library, model_path, manifest, contract, None)
    }

    /// Loads an exact AHEM v6.1 graph with an optional thread pool override,
    /// falling back to `ARGUS_ONNX_CPU_THREADS` or default (1 thread).
    ///
    /// # Errors
    /// Fails closed on any artifact or contract mismatch.
    pub fn load_with_threads(
        runtime_library: &Path,
        model_path: &Path,
        manifest: &ModelArtifactManifest,
        contract: AhemV61TensorContract,
        intra_threads: Option<usize>,
    ) -> Result<Self, InferenceError> {
        manifest.validate()?;
        validate_runtime_compatibility(manifest)?;
        if manifest.feature_schema_id != AHEM_V61_SCHEMA_ID
            || manifest.feature_schema_version != AHEM_V61_SCHEMA_VERSION
        {
            return Err(InferenceError::ModelIoMismatch(format!(
                "AHEM v6.1 requires feature schema {AHEM_V61_SCHEMA_ID}/{AHEM_V61_SCHEMA_VERSION}"
            )));
        }
        contract.validate()?;
        verify_sha256(model_path, &manifest.artifact_sha256)?;

        ort::init_from(runtime_library)
            .map_err(|error| InferenceError::RuntimeLoad(error.to_string()))?
            .with_name("argus-ahem-v61-inference")
            .commit();

        let threads = intra_threads.unwrap_or_else(cpu_intra_threads);
        let session = build_tuned_session_with_threads(model_path, threads)
            .map_err(|error| InferenceError::SessionLoad(error.to_string()))?;
        validate_ahem_v61_model_io(&session, &contract)?;

        Ok(Self {
            id: manifest.model_id.clone(),
            contract,
            session: Mutex::new(session),
            model_path: model_path.to_path_buf(),
            execution_provider: "CPUExecutionProvider".to_owned(),
        })
    }

    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    #[must_use]
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    #[must_use]
    pub fn execution_provider(&self) -> &str {
        &self.execution_provider
    }

    #[must_use]
    pub fn contract(&self) -> &AhemV61TensorContract {
        &self.contract
    }

    /// Runs zero-copy forecast on a single AHEM v6.1 input structure.
    ///
    /// # Errors
    /// Returns an error if inputs contain non-finite values or session execution fails.
    #[allow(clippy::too_many_lines)]
    pub fn forecast(&self, inputs: &AhemV61Inputs) -> Result<AhemV61Forecast, InferenceError> {
        if !inputs.is_finite() {
            return Err(InferenceError::InvalidFeatures);
        }

        let local_packets_dim = usize_to_i64(AHEM_V61_LOCAL_PACKETS, "local packets")?;
        let packet_dim = usize_to_i64(AHEM_V61_PACKET_DIM, "packet dim")?;
        let transitions_dim = usize_to_i64(AHEM_V61_MAX_TRANSITIONS, "transitions")?;
        let key_dim = usize_to_i64(AHEM_V61_KEY_DIM, "key dim")?;
        let value_dim = usize_to_i64(AHEM_V61_VALUE_DIM, "value dim")?;
        let output_dim = usize_to_i64(AHEM_V61_OUTPUT_DIM, "output dim")?;

        let local = TensorRef::from_array_view((
            [1, local_packets_dim, packet_dim],
            inputs.local_packets.as_flattened(),
        ))
        .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let query_pred =
            TensorRef::from_array_view((vec![1, packet_dim], inputs.query_predecessor.as_slice()))
                .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let query_curr =
            TensorRef::from_array_view((vec![1, packet_dim], inputs.query_current.as_slice()))
                .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let keys = TensorRef::from_array_view((
            vec![1, transitions_dim, key_dim],
            inputs.keys.as_flattened(),
        ))
        .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let values = TensorRef::from_array_view((
            vec![1, transitions_dim, value_dim],
            inputs.values.as_flattened(),
        ))
        .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let successor = TensorRef::from_array_view((
            vec![1, transitions_dim, output_dim],
            inputs.successor_dynamic.as_flattened(),
        ))
        .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let ordinals = TensorRef::from_array_view((
            vec![1, transitions_dim],
            inputs.source_ordinals.as_slice(),
        ))
        .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let valid = TensorRef::from_array_view((vec![1, transitions_dim], inputs.valid.as_slice()))
            .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let anchor = TensorRef::from_array_view((vec![1], inputs.anchor_ordinals.as_slice()))
            .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let spans = TensorRef::from_array_view((vec![1], inputs.context_spans.as_slice()))
            .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let gap = TensorRef::from_array_view((vec![1], inputs.query_gap.as_slice()))
            .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| InferenceError::SessionUnavailable)?;

        let outputs = session
            .run(ort::inputs![
                self.contract.local_packets.as_str() => local,
                self.contract.query_predecessor.as_str() => query_pred,
                self.contract.query_current.as_str() => query_curr,
                self.contract.keys.as_str() => keys,
                self.contract.values.as_str() => values,
                self.contract.successor_dynamic.as_str() => successor,
                self.contract.source_ordinals.as_str() => ordinals,
                self.contract.valid.as_str() => valid,
                self.contract.anchor_ordinals.as_str() => anchor,
                self.contract.context_spans.as_str() => spans,
                self.contract.query_gap.as_str() => gap,
            ])
            .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let output = outputs.get(&self.contract.prediction).ok_or_else(|| {
            InferenceError::ModelIoMismatch("declared AHEM v6.1 prediction disappeared".to_owned())
        })?;

        let (shape, vals) = output
            .try_extract_tensor::<f32>()
            .map_err(|error| InferenceError::Inference(error.to_string()))?;

        if shape.as_ref() != [1, 2] {
            return Err(InferenceError::ModelIoMismatch(
                "AHEM v6.1 prediction shape changed".to_owned(),
            ));
        }

        let weights = outputs
            .get(&self.contract.retrieval_weights)
            .ok_or_else(|| {
                InferenceError::ModelIoMismatch(
                    "AHEM v6.1 retrieval weights disappeared".to_owned(),
                )
            })?;
        let (weights_shape, weights_values) = weights
            .try_extract_tensor::<f32>()
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        if weights_shape.as_ref() != [1, transitions_dim] {
            return Err(InferenceError::ModelIoMismatch(
                "AHEM v6.1 retrieval weights shape changed".to_owned(),
            ));
        }
        let mut retrieval_weights = [0.0; AHEM_V61_MAX_TRANSITIONS];
        retrieval_weights.copy_from_slice(weights_values);

        let gate = outputs.get(&self.contract.memory_gate).ok_or_else(|| {
            InferenceError::ModelIoMismatch("AHEM v6.1 memory gate disappeared".to_owned())
        })?;
        let (gate_shape, gate_values) = gate
            .try_extract_tensor::<f32>()
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        if gate_shape.as_ref() != [1, 1] {
            return Err(InferenceError::ModelIoMismatch(
                "AHEM v6.1 memory gate shape changed".to_owned(),
            ));
        }

        let forecast = AhemV61Forecast {
            predicted_log_packet_length: vals[0],
            predicted_log_interarrival: vals[1],
            retrieval_weights,
            memory_gate: gate_values[0],
        };
        if !forecast.is_valid() {
            return Err(InferenceError::InvalidOutput);
        }
        Ok(forecast)
    }

    /// Runs batched inference on multiple AHEM v6.1 input structures.
    ///
    /// # Errors
    /// Returns an error if any input contains non-finite values or session execution fails.
    #[allow(clippy::too_many_lines)]
    pub fn forecast_batch(
        &self,
        inputs: &[AhemV61Inputs],
    ) -> Result<Vec<AhemV61Forecast>, InferenceError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        if inputs.len() == 1 {
            return self.forecast(&inputs[0]).map(|forecast| vec![forecast]);
        }
        let batch_size = inputs.len();
        let batch_dim = usize_to_i64(batch_size, "batch size")?;

        for input in inputs {
            if !input.is_finite() {
                return Err(InferenceError::InvalidFeatures);
            }
        }

        let local_packets_dim = usize_to_i64(AHEM_V61_LOCAL_PACKETS, "local packets")?;
        let packet_dim = usize_to_i64(AHEM_V61_PACKET_DIM, "packet dim")?;
        let transitions_dim = usize_to_i64(AHEM_V61_MAX_TRANSITIONS, "transitions")?;
        let key_dim = usize_to_i64(AHEM_V61_KEY_DIM, "key dim")?;
        let value_dim = usize_to_i64(AHEM_V61_VALUE_DIM, "value dim")?;
        let output_dim = usize_to_i64(AHEM_V61_OUTPUT_DIM, "output dim")?;

        let mut local_buf =
            Vec::with_capacity(batch_size * AHEM_V61_LOCAL_PACKETS * AHEM_V61_PACKET_DIM);
        let mut query_pred_buf = Vec::with_capacity(batch_size * AHEM_V61_PACKET_DIM);
        let mut query_curr_buf = Vec::with_capacity(batch_size * AHEM_V61_PACKET_DIM);
        let mut keys_buf =
            Vec::with_capacity(batch_size * AHEM_V61_MAX_TRANSITIONS * AHEM_V61_KEY_DIM);
        let mut values_buf =
            Vec::with_capacity(batch_size * AHEM_V61_MAX_TRANSITIONS * AHEM_V61_VALUE_DIM);
        let mut successor_buf =
            Vec::with_capacity(batch_size * AHEM_V61_MAX_TRANSITIONS * AHEM_V61_OUTPUT_DIM);
        let mut ordinals_buf = Vec::with_capacity(batch_size * AHEM_V61_MAX_TRANSITIONS);
        let mut valid_buf = Vec::with_capacity(batch_size * AHEM_V61_MAX_TRANSITIONS);
        let mut anchor_buf = Vec::with_capacity(batch_size);
        let mut spans_buf = Vec::with_capacity(batch_size);
        let mut gap_buf = Vec::with_capacity(batch_size);

        for inp in inputs {
            local_buf.extend_from_slice(inp.local_packets.as_flattened());
            query_pred_buf.extend_from_slice(&inp.query_predecessor);
            query_curr_buf.extend_from_slice(&inp.query_current);
            keys_buf.extend_from_slice(inp.keys.as_flattened());
            values_buf.extend_from_slice(inp.values.as_flattened());
            successor_buf.extend_from_slice(inp.successor_dynamic.as_flattened());
            ordinals_buf.extend_from_slice(&inp.source_ordinals);
            valid_buf.extend_from_slice(&inp.valid);
            anchor_buf.push(inp.anchor_ordinals[0]);
            spans_buf.push(inp.context_spans[0]);
            gap_buf.push(inp.query_gap[0]);
        }

        let local = TensorRef::from_array_view((
            vec![batch_dim, local_packets_dim, packet_dim],
            local_buf.as_slice(),
        ))
        .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let query_pred =
            TensorRef::from_array_view((vec![batch_dim, packet_dim], query_pred_buf.as_slice()))
                .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let query_curr =
            TensorRef::from_array_view((vec![batch_dim, packet_dim], query_curr_buf.as_slice()))
                .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let keys = TensorRef::from_array_view((
            vec![batch_dim, transitions_dim, key_dim],
            keys_buf.as_slice(),
        ))
        .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let values = TensorRef::from_array_view((
            vec![batch_dim, transitions_dim, value_dim],
            values_buf.as_slice(),
        ))
        .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let successor = TensorRef::from_array_view((
            vec![batch_dim, transitions_dim, output_dim],
            successor_buf.as_slice(),
        ))
        .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let ordinals =
            TensorRef::from_array_view((vec![batch_dim, transitions_dim], ordinals_buf.as_slice()))
                .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let valid =
            TensorRef::from_array_view((vec![batch_dim, transitions_dim], valid_buf.as_slice()))
                .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let anchor = TensorRef::from_array_view((vec![batch_dim], anchor_buf.as_slice()))
            .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let spans = TensorRef::from_array_view((vec![batch_dim], spans_buf.as_slice()))
            .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let gap = TensorRef::from_array_view((vec![batch_dim], gap_buf.as_slice()))
            .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| InferenceError::SessionUnavailable)?;

        let outputs = session
            .run(ort::inputs![
                self.contract.local_packets.as_str() => local,
                self.contract.query_predecessor.as_str() => query_pred,
                self.contract.query_current.as_str() => query_curr,
                self.contract.keys.as_str() => keys,
                self.contract.values.as_str() => values,
                self.contract.successor_dynamic.as_str() => successor,
                self.contract.source_ordinals.as_str() => ordinals,
                self.contract.valid.as_str() => valid,
                self.contract.anchor_ordinals.as_str() => anchor,
                self.contract.context_spans.as_str() => spans,
                self.contract.query_gap.as_str() => gap,
            ])
            .map_err(|error| InferenceError::Inference(error.to_string()))?;

        let output = outputs.get(&self.contract.prediction).ok_or_else(|| {
            InferenceError::ModelIoMismatch("declared AHEM v6.1 prediction disappeared".to_owned())
        })?;

        let (shape, vals) = output
            .try_extract_tensor::<f32>()
            .map_err(|error| InferenceError::Inference(error.to_string()))?;

        if shape.as_ref() != [batch_dim, 2] {
            return Err(InferenceError::ModelIoMismatch(
                "AHEM v6.1 prediction shape changed".to_owned(),
            ));
        }

        let weights = outputs
            .get(&self.contract.retrieval_weights)
            .ok_or_else(|| {
                InferenceError::ModelIoMismatch(
                    "AHEM v6.1 retrieval weights disappeared".to_owned(),
                )
            })?;
        let (weights_shape, weights_values) = weights
            .try_extract_tensor::<f32>()
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        if weights_shape.as_ref() != [batch_dim, transitions_dim] {
            return Err(InferenceError::ModelIoMismatch(
                "AHEM v6.1 retrieval weights shape changed".to_owned(),
            ));
        }

        let gate = outputs.get(&self.contract.memory_gate).ok_or_else(|| {
            InferenceError::ModelIoMismatch("AHEM v6.1 memory gate disappeared".to_owned())
        })?;
        let (gate_shape, gate_values) = gate
            .try_extract_tensor::<f32>()
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        if gate_shape.as_ref() != [batch_dim, 1] {
            return Err(InferenceError::ModelIoMismatch(
                "AHEM v6.1 memory gate shape changed".to_owned(),
            ));
        }

        let mut results = Vec::with_capacity(batch_size);
        for (index, chunk) in vals.chunks_exact(2).enumerate() {
            let mut retrieval_weights = [0.0; AHEM_V61_MAX_TRANSITIONS];
            let start = index * AHEM_V61_MAX_TRANSITIONS;
            retrieval_weights
                .copy_from_slice(&weights_values[start..start + AHEM_V61_MAX_TRANSITIONS]);
            let forecast = AhemV61Forecast {
                predicted_log_packet_length: chunk[0],
                predicted_log_interarrival: chunk[1],
                retrieval_weights,
                memory_gate: gate_values[index],
            };
            if !forecast.is_valid() {
                return Err(InferenceError::InvalidOutput);
            }
            results.push(forecast);
        }
        Ok(results)
    }
}

/// Native feedforward forecaster for fixed or dynamic-batch causal temporal
/// models (such as CNN-Local, AHEM-Fixed, and TCN-Long).
pub struct OnnxFeedforwardForecaster {
    id: String,
    contract: FeedforwardTensorContract,
    session: Mutex<Session>,
    model_path: PathBuf,
    execution_provider: String,
}

impl OnnxFeedforwardForecaster {
    /// Loads an exact feedforward graph after manifest, checksum, runtime,
    /// and graph outlet validation.
    ///
    /// # Errors
    /// Fails closed on any artifact or contract mismatch.
    pub fn load(
        runtime_library: &Path,
        model_path: &Path,
        manifest: &ModelArtifactManifest,
        contract: FeedforwardTensorContract,
    ) -> Result<Self, InferenceError> {
        manifest.validate()?;
        validate_runtime_compatibility(manifest)?;
        contract.validate()?;
        verify_sha256(model_path, &manifest.artifact_sha256)?;

        ort::init_from(runtime_library)
            .map_err(|error| InferenceError::RuntimeLoad(error.to_string()))?
            .with_name("argus-feedforward-inference")
            .commit();
        let session = build_tuned_session(model_path)
            .map_err(|error| InferenceError::SessionLoad(error.to_string()))?;
        validate_feedforward_model_io(&session, &contract)?;

        Ok(Self {
            id: manifest.model_id.clone(),
            contract,
            session: Mutex::new(session),
            model_path: model_path.to_path_buf(),
            execution_provider: "CPUExecutionProvider".to_owned(),
        })
    }

    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    #[must_use]
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    #[must_use]
    pub fn execution_provider(&self) -> &str {
        &self.execution_provider
    }

    /// Runs single-sample inference on a flattened context window.
    ///
    /// # Errors
    /// Returns an error if context elements violate dimensions, contain non-finite
    /// values, or session execution fails.
    pub fn forecast(&self, context: &[f32]) -> Result<FeedforwardForecast, InferenceError> {
        let expected = self
            .contract
            .context_packets
            .saturating_mul(self.contract.feature_dim);
        if context.len() != expected || context.iter().any(|v| !v.is_finite()) {
            return Err(InferenceError::InvalidFeatures);
        }
        let context_packets = usize_to_i64(self.contract.context_packets, "context packets")?;
        let feature_dim = usize_to_i64(self.contract.feature_dim, "feature dimension")?;
        let input = TensorRef::from_array_view((vec![1, context_packets, feature_dim], context))
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| InferenceError::SessionUnavailable)?;
        let outputs = session
            .run(ort::inputs![self.contract.input_name.as_str() => input])
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        let output = outputs.get(&self.contract.output_name).ok_or_else(|| {
            InferenceError::ModelIoMismatch(
                "declared feedforward prediction disappeared".to_owned(),
            )
        })?;
        let (shape, values) = output
            .try_extract_tensor::<f32>()
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        if shape.as_ref() != [1, 2] {
            return Err(InferenceError::ModelIoMismatch(
                "feedforward prediction shape changed".to_owned(),
            ));
        }
        let forecast = FeedforwardForecast {
            predicted_log_packet_length: values[0],
            predicted_log_interarrival: values[1],
        };
        if !forecast.predicted_log_packet_length.is_finite()
            || !forecast.predicted_log_interarrival.is_finite()
        {
            return Err(InferenceError::InvalidOutput);
        }
        Ok(forecast)
    }

    /// Runs batched inference on a flattened contiguous slice of context windows.
    ///
    /// # Errors
    /// Returns an error if context length does not match `batch_size * context * feature_dim`,
    /// contains non-finite values, or session execution fails.
    pub fn forecast_batch(
        &self,
        contexts: &[f32],
        batch_size: usize,
    ) -> Result<Vec<FeedforwardForecast>, InferenceError> {
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        let single_len = self
            .contract
            .context_packets
            .saturating_mul(self.contract.feature_dim);
        let expected = batch_size.saturating_mul(single_len);
        if contexts.len() != expected || contexts.iter().any(|v| !v.is_finite()) {
            return Err(InferenceError::InvalidFeatures);
        }
        let batch_dimension = usize_to_i64(batch_size, "batch size")?;
        let context_packets = usize_to_i64(self.contract.context_packets, "context packets")?;
        let feature_dim = usize_to_i64(self.contract.feature_dim, "feature dimension")?;
        let input = TensorRef::from_array_view((
            vec![batch_dimension, context_packets, feature_dim],
            contexts,
        ))
        .map_err(|error| InferenceError::Inference(error.to_string()))?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| InferenceError::SessionUnavailable)?;
        let outputs = session
            .run(ort::inputs![self.contract.input_name.as_str() => input])
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        let output = outputs.get(&self.contract.output_name).ok_or_else(|| {
            InferenceError::ModelIoMismatch(
                "declared feedforward prediction disappeared".to_owned(),
            )
        })?;
        let (shape, values) = output
            .try_extract_tensor::<f32>()
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        if shape.as_ref() != [batch_dimension, 2] {
            return Err(InferenceError::ModelIoMismatch(
                "feedforward prediction shape changed".to_owned(),
            ));
        }
        let mut results = Vec::with_capacity(batch_size);
        for chunk in values.chunks_exact(2) {
            let forecast = FeedforwardForecast {
                predicted_log_packet_length: chunk[0],
                predicted_log_interarrival: chunk[1],
            };
            if !forecast.predicted_log_packet_length.is_finite()
                || !forecast.predicted_log_interarrival.is_finite()
            {
                return Err(InferenceError::InvalidOutput);
            }
            results.push(forecast);
        }
        Ok(results)
    }
}

/// Native router forecaster executing `temporal_router.onnx`.
pub struct OnnxRouterForecaster {
    id: String,
    session: Mutex<Session>,
    model_path: PathBuf,
    execution_provider: String,
}

impl OnnxRouterForecaster {
    /// Loads the temporal router graph after manifest and checksum validation.
    ///
    /// # Errors
    /// Fails closed on artifact or contract mismatch.
    pub fn load(
        runtime_library: &Path,
        model_path: &Path,
        manifest: &ModelArtifactManifest,
    ) -> Result<Self, InferenceError> {
        manifest.validate()?;
        validate_runtime_compatibility(manifest)?;
        verify_sha256(model_path, &manifest.artifact_sha256)?;

        ort::init_from(runtime_library)
            .map_err(|error| InferenceError::RuntimeLoad(error.to_string()))?
            .with_name("argus-router-inference")
            .commit();
        let session = build_tuned_session(model_path)
            .map_err(|error| InferenceError::SessionLoad(error.to_string()))?;
        validate_router_model_io(&session)?;

        Ok(Self {
            id: manifest.model_id.clone(),
            session: Mutex::new(session),
            model_path: model_path.to_path_buf(),
            execution_provider: "CPUExecutionProvider".to_owned(),
        })
    }

    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    #[must_use]
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    #[must_use]
    pub fn execution_provider(&self) -> &str {
        &self.execution_provider
    }

    /// Evaluates 24 causal features (last packet + prefix mean) and returns 4-expert logits.
    ///
    /// # Errors
    /// Returns an error on non-finite features or session failure.
    pub fn route(&self, features: &[f32; 24]) -> Result<RouterLogits, InferenceError> {
        if features.iter().any(|v| !v.is_finite()) {
            return Err(InferenceError::InvalidFeatures);
        }
        let input = TensorRef::from_array_view((vec![1, 24], features.as_slice()))
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| InferenceError::SessionUnavailable)?;
        let outputs = session
            .run(ort::inputs!["features" => input])
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        let output = outputs.get("logits").ok_or_else(|| {
            InferenceError::ModelIoMismatch("declared router logits disappeared".to_owned())
        })?;
        let (shape, values) = output
            .try_extract_tensor::<f32>()
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        if shape.as_ref() != [1, 4] {
            return Err(InferenceError::ModelIoMismatch(
                "router logits shape changed".to_owned(),
            ));
        }
        let logits = [values[0], values[1], values[2], values[3]];
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(InferenceError::InvalidOutput);
        }
        Ok(RouterLogits { logits })
    }
}

fn build_tuned_session(model_path: &Path) -> ort::Result<Session> {
    build_tuned_session_with_threads(model_path, cpu_intra_threads())
}

/// Builds an ONNX session tuned for CPU inference with an explicit intra-threads count.
///
/// # Errors
/// Returns an error if ONNX session configuration or file loading fails.
pub fn build_tuned_session_with_threads(
    model_path: &Path,
    intra_threads: usize,
) -> ort::Result<Session> {
    let builder = Session::builder()?;
    let builder = builder.with_optimization_level(GraphOptimizationLevel::All)?;
    let builder = builder.with_intra_threads(intra_threads)?;
    let builder = builder.with_inter_threads(1)?;
    let builder = builder.with_intra_op_spinning(false)?;
    let mut builder = builder.with_inter_op_spinning(false)?;
    builder.commit_from_file(model_path)
}

impl OnnxProbabilityExpert {
    /// Loads an exact model/runtime pair without downloading or generating data.
    ///
    /// # Errors
    /// Fails before inference if the manifest, checksum, target, runtime, or
    /// model tensor interface does not match the declared contract.
    pub fn load(
        runtime_library: &Path,
        model_path: &Path,
        manifest: &ModelArtifactManifest,
        contract: TensorContract,
    ) -> Result<Self, InferenceError> {
        manifest.validate()?;
        validate_compatibility(manifest)?;
        contract.validate()?;
        verify_sha256(model_path, &manifest.artifact_sha256)?;

        ort::init_from(runtime_library)
            .map_err(|error| InferenceError::RuntimeLoad(error.to_string()))?
            .with_name("argus-native-inference")
            .commit();

        let session = build_tuned_session(model_path)
            .map_err(|error| InferenceError::SessionLoad(error.to_string()))?;
        validate_model_io(&session, &contract)?;

        Ok(Self {
            id: manifest.model_id.clone(),
            schema_id: manifest.feature_schema_id.clone(),
            schema_version: manifest.feature_schema_version,
            contract,
            session: Mutex::new(session),
            model_path: model_path.to_path_buf(),
            execution_provider: "CPUExecutionProvider".to_owned(),
        })
    }

    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    #[must_use]
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    #[must_use]
    pub fn execution_provider(&self) -> &str {
        &self.execution_provider
    }

    fn infer_probability(&self, features: &[f32]) -> Result<f32, InferenceError> {
        if features.len() != self.contract.input_elements()?
            || features.iter().any(|value| !value.is_finite())
        {
            return Err(InferenceError::InvalidFeatures);
        }
        let input = TensorRef::from_array_view((self.contract.input_shape.clone(), features))
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| InferenceError::SessionUnavailable)?;
        let outputs = session
            .run(ort::inputs![self.contract.input_name.as_str() => input])
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        let output = outputs.get(&self.contract.output_name).ok_or_else(|| {
            InferenceError::ModelIoMismatch("declared output disappeared".to_owned())
        })?;
        let (shape, values) = output
            .try_extract_tensor::<f32>()
            .map_err(|error| InferenceError::Inference(error.to_string()))?;
        if shape.as_ref() != self.contract.output_shape.as_slice() {
            return Err(InferenceError::ModelIoMismatch(
                "runtime output shape changed".to_owned(),
            ));
        }
        let probability = values[self.contract.probability_index];
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(InferenceError::InvalidOutput);
        }
        Ok(probability)
    }
}

impl Expert for OnnxProbabilityExpert {
    fn id(&self) -> &str {
        &self.id
    }

    fn feature_schema(&self) -> (&str, u32) {
        (&self.schema_id, self.schema_version)
    }

    fn calibrated(&self) -> bool {
        true
    }

    fn infer(&self, features: &[f32]) -> Result<ExpertOutput, String> {
        self.infer_probability(features)
            .map(|attack_probability| ExpertOutput { attack_probability })
            .map_err(|error| error.to_string())
    }
}

fn validate_compatibility(manifest: &ModelArtifactManifest) -> Result<(), InferenceError> {
    validate_runtime_compatibility(manifest)?;
    if !manifest.calibration.required {
        return Err(InferenceError::UncalibratedArtifact);
    }
    Ok(())
}

fn validate_runtime_compatibility(manifest: &ModelArtifactManifest) -> Result<(), InferenceError> {
    if manifest.runtime.runtime_id != RUNTIME_ID {
        return Err(InferenceError::UnsupportedRuntime(
            manifest.runtime.runtime_id.clone(),
        ));
    }
    let target = current_target();
    if !manifest
        .runtime
        .supported_targets
        .iter()
        .any(|candidate| candidate == target)
    {
        return Err(InferenceError::UnsupportedTarget(target.to_owned()));
    }
    let provided = format!("1.{}.0", ort::MINOR_VERSION);
    if version_tuple(&provided) < version_tuple(&manifest.runtime.minimum_runtime_version) {
        return Err(InferenceError::RuntimeTooOld {
            required: manifest.runtime.minimum_runtime_version.clone(),
            provided,
        });
    }
    Ok(())
}

fn validate_model_io(session: &Session, contract: &TensorContract) -> Result<(), InferenceError> {
    let input = session
        .inputs()
        .iter()
        .find(|outlet| outlet.name() == contract.input_name)
        .ok_or_else(|| InferenceError::ModelIoMismatch("declared input is absent".to_owned()))?;
    validate_outlet(input.dtype(), &contract.input_shape, "input")?;
    let output = session
        .outputs()
        .iter()
        .find(|outlet| outlet.name() == contract.output_name)
        .ok_or_else(|| InferenceError::ModelIoMismatch("declared output is absent".to_owned()))?;
    validate_outlet(output.dtype(), &contract.output_shape, "output")
}

fn validate_ahem_v34_model_io(
    session: &Session,
    contract: &AhemV34TensorContract,
) -> Result<(), InferenceError> {
    let inputs = [
        (
            contract.local_packets.as_str(),
            TensorElementType::Float32,
            &[1, 128, 13][..],
        ),
        (
            contract.flow_packets.as_str(),
            TensorElementType::Float32,
            &[1, 17, 12][..],
        ),
        (
            contract.flow_ordinals.as_str(),
            TensorElementType::Int64,
            &[1, 17][..],
        ),
        (
            contract.flow_valid.as_str(),
            TensorElementType::Bool,
            &[1, 17][..],
        ),
        (
            contract.context_spans.as_str(),
            TensorElementType::Float32,
            &[1][..],
        ),
    ];
    for (name, element_type, shape) in inputs {
        let input = session
            .inputs()
            .iter()
            .find(|outlet| outlet.name() == name)
            .ok_or_else(|| {
                InferenceError::ModelIoMismatch(format!("AHEM input {name} is absent"))
            })?;
        validate_typed_outlet(input.dtype(), element_type, shape, name)?;
    }
    let output = session
        .outputs()
        .iter()
        .find(|outlet| outlet.name() == contract.prediction)
        .ok_or_else(|| InferenceError::ModelIoMismatch("AHEM prediction is absent".to_owned()))?;
    validate_typed_outlet(
        output.dtype(),
        TensorElementType::Float32,
        &[1, 2],
        "prediction",
    )
}

fn validate_ahem_v61_outlet(
    dtype: &ValueType,
    element_type: TensorElementType,
    expected: &[i64],
    alternative: Option<&[i64]>,
    label: &str,
) -> Result<(), InferenceError> {
    match dtype {
        ValueType::Tensor { ty, shape, .. }
            if *ty == element_type
                && (shapes_compatible(shape.as_ref(), expected)
                    || alternative.is_some_and(|alt| shapes_compatible(shape.as_ref(), alt))) =>
        {
            Ok(())
        }
        _ => Err(InferenceError::ModelIoMismatch(format!(
            "{label} must be a {element_type:?} tensor with compatible shape {expected:?}"
        ))),
    }
}

fn validate_ahem_v61_model_io(
    session: &Session,
    contract: &AhemV61TensorContract,
) -> Result<(), InferenceError> {
    let inputs = [
        (
            contract.local_packets.as_str(),
            TensorElementType::Float32,
            &[-1, 128, 12][..],
            None,
        ),
        (
            contract.query_predecessor.as_str(),
            TensorElementType::Float32,
            &[-1, 12][..],
            None,
        ),
        (
            contract.query_current.as_str(),
            TensorElementType::Float32,
            &[-1, 12][..],
            None,
        ),
        (
            contract.keys.as_str(),
            TensorElementType::Float32,
            &[-1, 16, 32][..],
            None,
        ),
        (
            contract.values.as_str(),
            TensorElementType::Float32,
            &[-1, 16, 60][..],
            None,
        ),
        (
            contract.successor_dynamic.as_str(),
            TensorElementType::Float32,
            &[-1, 16, 2][..],
            None,
        ),
        (
            contract.source_ordinals.as_str(),
            TensorElementType::Int64,
            &[-1, 16][..],
            None,
        ),
        (
            contract.valid.as_str(),
            TensorElementType::Bool,
            &[-1, 16][..],
            None,
        ),
        (
            contract.anchor_ordinals.as_str(),
            TensorElementType::Int64,
            &[-1][..],
            Some(&[-1, 1][..]),
        ),
        (
            contract.context_spans.as_str(),
            TensorElementType::Float32,
            &[-1][..],
            Some(&[-1, 1][..]),
        ),
        (
            contract.query_gap.as_str(),
            TensorElementType::Float32,
            &[-1][..],
            Some(&[-1, 1][..]),
        ),
    ];
    for (name, element_type, shape, alt_shape) in inputs {
        let input = session
            .inputs()
            .iter()
            .find(|outlet| outlet.name() == name)
            .ok_or_else(|| {
                InferenceError::ModelIoMismatch(format!("AHEM v6.1 input {name} is absent"))
            })?;
        validate_ahem_v61_outlet(input.dtype(), element_type, shape, alt_shape, name)?;
    }
    let transitions_dim = usize_to_i64(AHEM_V61_MAX_TRANSITIONS, "transitions")?;
    for (name, shape) in [
        (contract.prediction.as_str(), &[-1, 2][..]),
        (
            contract.retrieval_weights.as_str(),
            &[-1, transitions_dim][..],
        ),
        (contract.memory_gate.as_str(), &[-1, 1][..]),
    ] {
        let output = session
            .outputs()
            .iter()
            .find(|outlet| outlet.name() == name)
            .ok_or_else(|| {
                InferenceError::ModelIoMismatch(format!("AHEM v6.1 output {name} is absent"))
            })?;
        validate_typed_outlet(output.dtype(), TensorElementType::Float32, shape, name)?;
    }
    Ok(())
}

fn validate_feedforward_model_io(
    session: &Session,
    contract: &FeedforwardTensorContract,
) -> Result<(), InferenceError> {
    let input = session
        .inputs()
        .iter()
        .find(|outlet| outlet.name() == contract.input_name)
        .ok_or_else(|| {
            InferenceError::ModelIoMismatch(format!(
                "feedforward input {} is absent",
                contract.input_name
            ))
        })?;
    let context_packets = usize_to_i64(contract.context_packets, "context packets")?;
    let feature_dim = usize_to_i64(contract.feature_dim, "feature dimension")?;
    validate_typed_outlet(
        input.dtype(),
        TensorElementType::Float32,
        &[-1, context_packets, feature_dim],
        &contract.input_name,
    )?;
    let output = session
        .outputs()
        .iter()
        .find(|outlet| outlet.name() == contract.output_name)
        .ok_or_else(|| {
            InferenceError::ModelIoMismatch(format!(
                "feedforward output {} is absent",
                contract.output_name
            ))
        })?;
    validate_typed_outlet(
        output.dtype(),
        TensorElementType::Float32,
        &[-1, 2],
        &contract.output_name,
    )
}

fn validate_router_model_io(session: &Session) -> Result<(), InferenceError> {
    let input = session
        .inputs()
        .iter()
        .find(|outlet| outlet.name() == "features")
        .ok_or_else(|| {
            InferenceError::ModelIoMismatch("router input features is absent".to_owned())
        })?;
    validate_typed_outlet(
        input.dtype(),
        TensorElementType::Float32,
        &[-1, 24],
        "features",
    )?;
    let output = session
        .outputs()
        .iter()
        .find(|outlet| outlet.name() == "logits")
        .ok_or_else(|| {
            InferenceError::ModelIoMismatch("router output logits is absent".to_owned())
        })?;
    validate_typed_outlet(
        output.dtype(),
        TensorElementType::Float32,
        &[-1, 4],
        "logits",
    )
}

fn shapes_compatible(actual: &[i64], expected: &[i64]) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    actual
        .iter()
        .zip(expected.iter())
        .all(|(&a, &e)| e == -1 || a == -1 || a == e)
}

fn validate_outlet(dtype: &ValueType, expected: &[i64], label: &str) -> Result<(), InferenceError> {
    validate_typed_outlet(dtype, TensorElementType::Float32, expected, label)
}

fn validate_typed_outlet(
    dtype: &ValueType,
    element_type: TensorElementType,
    expected: &[i64],
    label: &str,
) -> Result<(), InferenceError> {
    match dtype {
        ValueType::Tensor { ty, shape, .. }
            if *ty == element_type && shapes_compatible(shape.as_ref(), expected) =>
        {
            Ok(())
        }
        _ => Err(InferenceError::ModelIoMismatch(format!(
            "{label} must be a {element_type:?} tensor with compatible shape {expected:?}"
        ))),
    }
}

fn verify_sha256(path: &Path, expected: &str) -> Result<(), InferenceError> {
    let mut file =
        File::open(path).map_err(|error| InferenceError::ArtifactRead(error.to_string()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 8 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| InferenceError::ArtifactRead(error.to_string()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual != expected {
        return Err(InferenceError::ArtifactChecksumMismatch);
    }
    Ok(())
}

fn element_count(shape: &[i64], label: &str) -> Result<usize, InferenceError> {
    if shape.is_empty() || shape.iter().any(|dimension| *dimension <= 0) {
        return Err(InferenceError::InvalidTensorContract(format!(
            "{label} shape must contain only positive dimensions"
        )));
    }
    shape.iter().try_fold(1_usize, |total, dimension| {
        usize::try_from(*dimension)
            .ok()
            .and_then(|value| total.checked_mul(value))
            .ok_or_else(|| {
                InferenceError::InvalidTensorContract(format!("{label} shape is too large"))
            })
    })
}

fn usize_to_i64(value: usize, label: &str) -> Result<i64, InferenceError> {
    i64::try_from(value).map_err(|_| {
        InferenceError::InvalidTensorContract(format!("{label} exceeds the runtime dimension"))
    })
}

fn version_tuple(value: &str) -> (u64, u64, u64) {
    let mut parts = value
        .split('.')
        .map(|part| part.parse::<u64>().unwrap_or(0));
    (
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
    )
}

const fn current_target() -> &'static str {
    if cfg!(all(target_arch = "x86_64", target_os = "windows")) {
        "x86_64-pc-windows-msvc"
    } else if cfg!(all(target_arch = "aarch64", target_os = "windows")) {
        "aarch64-pc-windows-msvc"
    } else if cfg!(all(target_arch = "x86_64", target_os = "linux")) {
        "x86_64-unknown-linux-gnu"
    } else if cfg!(all(target_arch = "aarch64", target_os = "linux")) {
        "aarch64-unknown-linux-gnu"
    } else {
        "unsupported-target"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argus_model_artifact::{
        ARTIFACT_FORMAT, ARTIFACT_FORMAT_VERSION, CalibrationRequirements, Provenance,
        ReadinessRequirements, RuntimeCompatibility,
    };

    fn manifest() -> ModelArtifactManifest {
        ModelArtifactManifest {
            format: ARTIFACT_FORMAT.to_owned(),
            format_version: ARTIFACT_FORMAT_VERSION,
            feature_schema_id: "argus.causal-flow".to_owned(),
            feature_schema_version: 1,
            model_id: "argus.test".to_owned(),
            model_version: "1.0.0".to_owned(),
            artifact_sha256: "a".repeat(64),
            provenance: Provenance {
                training_dataset_reference: "dataset:test".to_owned(),
                evaluation_reference: "evaluation:test".to_owned(),
                source_revision: "abc123".to_owned(),
            },
            calibration: CalibrationRequirements {
                required: true,
                reference: Some("calibration:test".to_owned()),
            },
            readiness: ReadinessRequirements {
                minimum_observations: 1,
                maximum_age_seconds: None,
            },
            runtime: RuntimeCompatibility {
                runtime_id: RUNTIME_ID.to_owned(),
                minimum_runtime_version: "1.20.0".to_owned(),
                supported_targets: vec![current_target().to_owned()],
            },
        }
    }

    #[test]
    fn exact_tensor_contract_accepts_bounded_shapes() {
        let contract = TensorContract {
            input_name: "features".to_owned(),
            input_shape: vec![1, 12],
            output_name: "probability".to_owned(),
            output_shape: vec![1, 2],
            probability_index: 1,
        };
        assert!(contract.validate().is_ok());
        assert!(matches!(contract.input_elements(), Ok(12)));
    }

    #[test]
    fn ahem_contract_names_are_exact_and_unambiguous() {
        let contract = AhemV34TensorContract::default();
        assert!(contract.validate().is_ok());
        let mut duplicate = contract;
        duplicate.prediction.clone_from(&duplicate.local_packets);
        assert!(matches!(
            duplicate.validate(),
            Err(InferenceError::InvalidTensorContract(_))
        ));
    }

    #[test]
    fn dynamic_or_ambiguous_shapes_are_rejected() {
        let contract = TensorContract {
            input_name: "features".to_owned(),
            input_shape: vec![-1, 12],
            output_name: "probability".to_owned(),
            output_shape: vec![1],
            probability_index: 0,
        };
        assert!(matches!(
            contract.validate(),
            Err(InferenceError::InvalidTensorContract(_))
        ));
    }

    #[test]
    fn probability_index_must_exist() {
        let contract = TensorContract {
            input_name: "features".to_owned(),
            input_shape: vec![1, 12],
            output_name: "probability".to_owned(),
            output_shape: vec![1],
            probability_index: 1,
        };
        assert!(matches!(
            contract.validate(),
            Err(InferenceError::InvalidTensorContract(_))
        ));
    }

    #[test]
    fn semantic_versions_compare_numerically() {
        assert!(version_tuple("1.27.0") > version_tuple("1.9.0"));
    }

    #[test]
    fn configured_runtime_must_be_a_real_file() {
        assert!(matches!(
            discover_runtime_library(Some(Path::new("definitely-missing-onnxruntime.dll"))),
            Err(InferenceError::RuntimeLoad(_))
        ));
    }

    #[test]
    fn cpu_thread_override_is_bounded_and_explicit() {
        assert_eq!(parse_cpu_threads("1"), Some(1));
        assert_eq!(parse_cpu_threads("4"), Some(4));
        assert_eq!(parse_cpu_threads("8"), Some(8));
        assert_eq!(parse_cpu_threads("0"), None);
        assert_eq!(parse_cpu_threads("9"), None);
        assert_eq!(parse_cpu_threads("auto"), None);
    }

    #[test]
    fn uncalibrated_artifact_is_never_admitted_as_an_expert() {
        let mut value = manifest();
        value.calibration.required = false;
        value.calibration.reference = None;
        assert!(value.validate().is_ok());
        assert!(matches!(
            validate_compatibility(&value),
            Err(InferenceError::UncalibratedArtifact)
        ));
    }

    #[test]
    fn forecast_artifact_allows_no_calibration_but_requires_v34_schema() {
        let mut value = manifest();
        value.calibration.required = false;
        value.calibration.reference = None;
        value.feature_schema_id = AHEM_V34_SCHEMA_ID.to_owned();
        value.feature_schema_version = 34;
        assert!(matches!(
            OnnxAhemV34Forecaster::load(
                Path::new("missing-runtime.dll"),
                Path::new("missing-model.onnx"),
                &value,
                AhemV34TensorContract::default(),
            ),
            Err(InferenceError::ArtifactRead(_))
        ));

        value.feature_schema_id = "argus.causal-flow".to_owned();
        assert!(matches!(
            OnnxAhemV34Forecaster::load(
                Path::new("missing-runtime.dll"),
                Path::new("missing-model.onnx"),
                &value,
                AhemV34TensorContract::default(),
            ),
            Err(InferenceError::ModelIoMismatch(_))
        ));
    }

    #[test]
    fn feedforward_contract_validates_dimensions_and_names() {
        let contract = FeedforwardTensorContract::default();
        assert!(contract.validate().is_ok());

        let mut bad_names = contract.clone();
        bad_names.input_name = String::new();
        assert!(matches!(
            bad_names.validate(),
            Err(InferenceError::InvalidTensorContract(_))
        ));

        let mut equal_names = contract.clone();
        equal_names.output_name = equal_names.input_name.clone();
        assert!(matches!(
            equal_names.validate(),
            Err(InferenceError::InvalidTensorContract(_))
        ));

        let mut zero_context = contract;
        zero_context.context_packets = 0;
        assert!(matches!(
            zero_context.validate(),
            Err(InferenceError::InvalidTensorContract(_))
        ));
    }

    #[test]
    fn router_logits_computes_softmax_and_argmax() {
        let logits = RouterLogits {
            logits: [1.0, 3.0, 0.5, 2.0],
        };
        assert_eq!(logits.argmax(), 1);
        let probs = logits.softmax();
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[1] > probs[3]);
        assert!(probs[3] > probs[0]);
        assert!(probs[0] > probs[2]);
    }

    #[test]
    fn shapes_compatible_matches_dynamic_and_static_dimensions() {
        assert!(shapes_compatible(&[-1, 31, 12], &[-1, 31, 12]));
        assert!(shapes_compatible(&[1, 31, 12], &[-1, 31, 12]));
        assert!(!shapes_compatible(&[1, 32, 12], &[-1, 31, 12]));
        assert!(!shapes_compatible(&[1, 31], &[-1, 31, 12]));
    }

    #[test]
    #[ignore = "requires ARGUS_ONNX_RUNTIME and ARGUS_ONNX_MODELS_DIR with real artifacts"]
    #[allow(clippy::too_many_lines)]
    fn real_exported_feedforward_and_router_integration_test() {
        let runtime = discover_runtime_library(None).expect("configured ONNX Runtime");
        let models_dir = std::env::var_os("ARGUS_ONNX_MODELS_DIR")
            .map(PathBuf::from)
            .expect("ARGUS_ONNX_MODELS_DIR");
        for file_name in [
            "cnn_local.onnx",
            "cnn_local.onnx.manifest.json",
            "ahem_fixed.onnx",
            "ahem_fixed.onnx.manifest.json",
            "tcn_long.onnx",
            "tcn_long.onnx.manifest.json",
            "temporal_router.onnx",
            "temporal_router.onnx.manifest.json",
        ] {
            assert!(models_dir.join(file_name).is_file(), "missing {file_name}");
        }
        let cnn_onnx = models_dir.join("cnn_local.onnx");
        let cnn_manifest_path = models_dir.join("cnn_local.onnx.manifest.json");
        if cnn_onnx.is_file() && cnn_manifest_path.is_file() {
            let manifest_content = std::fs::read_to_string(&cnn_manifest_path).unwrap();
            let val: serde_json::Value = serde_json::from_str(&manifest_content).unwrap();
            let sha256 = val["artifact_sha256"].as_str().unwrap();
            let mut manifest = manifest();
            manifest.model_id = "argus.cnn-local".to_owned();
            manifest.artifact_sha256 = sha256.to_owned();
            manifest.calibration.required = false;
            manifest.calibration.reference = None;
            manifest.feature_schema_id = "packet-causal-v1".to_owned();
            manifest.feature_schema_version = 1;

            let contract = FeedforwardTensorContract {
                input_name: "context".to_owned(),
                context_packets: 31,
                feature_dim: 12,
                output_name: "prediction".to_owned(),
            };
            let forecaster =
                OnnxFeedforwardForecaster::load(&runtime, &cnn_onnx, &manifest, contract)
                    .expect("load cnn_local");
            assert_eq!(forecaster.execution_provider(), "CPUExecutionProvider");
            let dummy_input = vec![0.5_f32; 31 * 12];
            let single = forecaster.forecast(&dummy_input).expect("forecast single");
            assert!(single.predicted_log_packet_length.is_finite());
            assert!(single.predicted_log_interarrival.is_finite());
            let batch_input = vec![0.5_f32; 2 * 31 * 12];
            let batch = forecaster
                .forecast_batch(&batch_input, 2)
                .expect("forecast batch");
            assert_eq!(batch.len(), 2);
            assert_forecast_close(single, batch[0]);
        }

        let ahem_onnx = models_dir.join("ahem_fixed.onnx");
        let ahem_manifest_path = models_dir.join("ahem_fixed.onnx.manifest.json");
        if ahem_onnx.is_file() && ahem_manifest_path.is_file() {
            let manifest_content = std::fs::read_to_string(&ahem_manifest_path).unwrap();
            let val: serde_json::Value = serde_json::from_str(&manifest_content).unwrap();
            let sha256 = val["artifact_sha256"].as_str().unwrap();
            let mut manifest = manifest();
            manifest.model_id = "argus.ahem-fixed".to_owned();
            manifest.artifact_sha256 = sha256.to_owned();
            manifest.calibration.required = false;
            manifest.calibration.reference = None;
            manifest.feature_schema_id = "packet-causal-v1".to_owned();
            manifest.feature_schema_version = 1;

            let contract = FeedforwardTensorContract {
                input_name: "context".to_owned(),
                context_packets: 128,
                feature_dim: 12,
                output_name: "prediction".to_owned(),
            };
            let forecaster =
                OnnxFeedforwardForecaster::load(&runtime, &ahem_onnx, &manifest, contract)
                    .expect("load ahem_fixed");
            assert_eq!(forecaster.execution_provider(), "CPUExecutionProvider");
            let dummy_input = vec![0.5_f32; 128 * 12];
            let single = forecaster.forecast(&dummy_input).expect("forecast single");
            assert!(single.predicted_log_packet_length.is_finite());
            assert!(single.predicted_log_interarrival.is_finite());
            let batch_input = vec![0.5_f32; 2 * 128 * 12];
            let batch = forecaster
                .forecast_batch(&batch_input, 2)
                .expect("forecast batch");
            assert_eq!(batch.len(), 2);
            assert_forecast_close(single, batch[0]);
        }

        let tcn_onnx = models_dir.join("tcn_long.onnx");
        let tcn_manifest_path = models_dir.join("tcn_long.onnx.manifest.json");
        if tcn_onnx.is_file() && tcn_manifest_path.is_file() {
            let manifest_content = std::fs::read_to_string(&tcn_manifest_path).unwrap();
            let val: serde_json::Value = serde_json::from_str(&manifest_content).unwrap();
            let sha256 = val["artifact_sha256"].as_str().unwrap();
            let mut manifest = manifest();
            manifest.model_id = "argus.tcn-long".to_owned();
            manifest.artifact_sha256 = sha256.to_owned();
            manifest.calibration.required = false;
            manifest.calibration.reference = None;
            manifest.feature_schema_id = "packet-causal-v1".to_owned();
            manifest.feature_schema_version = 1;

            let contract = FeedforwardTensorContract {
                input_name: "context".to_owned(),
                context_packets: 1024,
                feature_dim: 12,
                output_name: "prediction".to_owned(),
            };
            let forecaster =
                OnnxFeedforwardForecaster::load(&runtime, &tcn_onnx, &manifest, contract)
                    .expect("load tcn_long");
            assert_eq!(forecaster.execution_provider(), "CPUExecutionProvider");
            let dummy_input = vec![0.5_f32; 1024 * 12];
            let single = forecaster.forecast(&dummy_input).expect("forecast single");
            assert!(single.predicted_log_packet_length.is_finite());
            assert!(single.predicted_log_interarrival.is_finite());
            let batch_input = vec![0.5_f32; 2 * 1024 * 12];
            let batch = forecaster
                .forecast_batch(&batch_input, 2)
                .expect("forecast batch");
            assert_eq!(batch.len(), 2);
            assert_forecast_close(single, batch[0]);
        }

        let router_onnx = models_dir.join("temporal_router.onnx");
        let router_manifest_path = models_dir.join("temporal_router.onnx.manifest.json");
        if router_onnx.is_file() && router_manifest_path.is_file() {
            let manifest_content = std::fs::read_to_string(&router_manifest_path).unwrap();
            let val: serde_json::Value = serde_json::from_str(&manifest_content).unwrap();
            let sha256 = val["artifact_sha256"].as_str().unwrap();
            let mut manifest = manifest();
            manifest.model_id = "argus.temporal-router".to_owned();
            manifest.artifact_sha256 = sha256.to_owned();
            manifest.calibration.required = false;
            manifest.calibration.reference = None;
            manifest.feature_schema_id = "causal-router-features-v1".to_owned();
            manifest.feature_schema_version = 1;

            let router =
                OnnxRouterForecaster::load(&runtime, &router_onnx, &manifest).expect("load router");
            assert_eq!(router.execution_provider(), "CPUExecutionProvider");
            let features = [0.1_f32; 24];
            let route_res = router.route(&features).expect("route");
            assert!(route_res.logits.iter().all(|v| v.is_finite()));
            assert!(route_res.argmax() < 4);
        }
    }

    fn assert_forecast_close(left: FeedforwardForecast, right: FeedforwardForecast) {
        assert!(
            (left.predicted_log_packet_length - right.predicted_log_packet_length).abs() < 1e-5
        );
        assert!((left.predicted_log_interarrival - right.predicted_log_interarrival).abs() < 1e-5);
    }

    #[test]
    fn ahem_v61_contract_names_are_exact_and_unambiguous() {
        let contract = AhemV61TensorContract::default();
        assert!(contract.validate().is_ok());

        let mut bad_name = contract.clone();
        bad_name.query_current.clear();
        assert!(matches!(
            bad_name.validate(),
            Err(InferenceError::InvalidTensorContract(_))
        ));

        let mut duplicate = contract;
        duplicate.prediction.clone_from(&duplicate.local_packets);
        assert!(matches!(
            duplicate.validate(),
            Err(InferenceError::InvalidTensorContract(_))
        ));
    }

    #[test]
    fn ahem_v61_inputs_finite_and_default() {
        let inputs = AhemV61Inputs::default();
        assert!(inputs.is_finite());

        let mut bad_local = inputs.clone();
        bad_local.local_packets[0][0] = f32::NAN;
        assert!(!bad_local.is_finite());

        let mut bad_key = inputs.clone();
        bad_key.keys[0][0] = f32::INFINITY;
        assert!(!bad_key.is_finite());

        let mut bad_gap = inputs;
        bad_gap.query_gap[0] = f32::NEG_INFINITY;
        assert!(!bad_gap.is_finite());
    }

    #[test]
    fn ahem_v61_forecast_is_close_and_values() {
        let f1 = AhemV61Forecast {
            predicted_log_packet_length: 5.5,
            predicted_log_interarrival: 0.25,
            retrieval_weights: [0.0; AHEM_V61_MAX_TRANSITIONS],
            memory_gate: 0.25,
        };
        let f2 = AhemV61Forecast {
            predicted_log_packet_length: 5.50001,
            predicted_log_interarrival: 0.25001,
            retrieval_weights: [0.0; AHEM_V61_MAX_TRANSITIONS],
            memory_gate: 0.25001,
        };
        let f3 = AhemV61Forecast {
            predicted_log_packet_length: 6.0,
            predicted_log_interarrival: 1.0,
            retrieval_weights: [0.0; AHEM_V61_MAX_TRANSITIONS],
            memory_gate: 0.25,
        };
        assert!(f1.is_close(&f2, 1e-4));
        assert!(!f1.is_close(&f3, 1e-4));
    }

    #[test]
    fn ahem_v61_expert_load_fails_closed_on_missing_model_or_runtime() {
        let mut value = manifest();
        value.calibration.required = false;
        value.calibration.reference = None;
        value.feature_schema_id = AHEM_V61_SCHEMA_ID.to_owned();
        value.feature_schema_version = AHEM_V61_SCHEMA_VERSION;
        assert!(matches!(
            AhemV61Expert::load(
                Path::new("missing-runtime.dll"),
                Path::new("missing-model.onnx"),
                &value,
                AhemV61TensorContract::default(),
            ),
            Err(InferenceError::ArtifactRead(_))
        ));

        assert!(matches!(
            AhemV61Expert::load_with_threads(
                Path::new("missing-runtime.dll"),
                Path::new("missing-model.onnx"),
                &value,
                AhemV61TensorContract::default(),
                Some(4),
            ),
            Err(InferenceError::ArtifactRead(_))
        ));
    }

    #[test]
    fn ahem_v61_forecaster_rejects_wrong_feature_schema() {
        let mut value = manifest();
        value.calibration.required = false;
        value.calibration.reference = None;
        value.feature_schema_id = "unrelated-features".to_owned();
        value.feature_schema_version = AHEM_V61_SCHEMA_VERSION;
        assert!(matches!(
            AhemV61Expert::load(
                Path::new("missing-runtime.dll"),
                Path::new("missing-model.onnx"),
                &value,
                AhemV61TensorContract::default(),
            ),
            Err(InferenceError::ModelIoMismatch(_))
        ));
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn real_exported_ahem_v61_integration_test() {
        let Ok(runtime) = discover_runtime_library(None) else {
            return; // runtime not present on this machine/environment; gracefully skip
        };

        let models_dir = std::env::var_os("ARGUS_ONNX_MODELS_DIR")
            .map(PathBuf::from)
            .or_else(|| {
                let candidate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
                candidate.is_dir().then_some(candidate)
            });

        let Some(models_dir) = models_dir else {
            return;
        };

        let onnx_path = models_dir.join("ahem_v61.onnx");
        let manifest_path = models_dir.join("ahem_v61.onnx.manifest.json");
        if !onnx_path.is_file() || !manifest_path.is_file() {
            return;
        }

        let manifest = read_manifest(&manifest_path).expect("read manifest");
        let contract = AhemV61TensorContract::default();

        // Test 1: Load expert honoring default CPU threads
        let expert = AhemV61Expert::load(&runtime, &onnx_path, &manifest, contract.clone())
            .expect("load ahem_v61 expert");
        assert_eq!(expert.id(), manifest.model_id);
        assert_eq!(expert.execution_provider(), "CPUExecutionProvider");
        assert_eq!(expert.contract(), &contract);

        // Test 2: Load with explicit thread count
        let expert_multithreaded =
            AhemV61Expert::load_with_threads(&runtime, &onnx_path, &manifest, contract, Some(2))
                .expect("load ahem_v61 with 2 threads");

        // Test 3: Zero-copy single-sample forecast
        let mut inputs = AhemV61Inputs::default();
        inputs.local_packets[0][0] = 1.0;
        inputs.query_current[0] = 1.0;
        inputs.context_spans[0] = 50.0;
        inputs.valid[0] = true;

        let forecast1 = expert.forecast(&inputs).expect("single forecast");
        assert!(forecast1.predicted_log_packet_length.is_finite());
        assert!(forecast1.predicted_log_interarrival.is_finite());
        assert!((0.0..=1.0).contains(&forecast1.memory_gate));
        assert!(
            forecast1
                .retrieval_weights
                .iter()
                .all(|weight| weight.is_finite())
        );

        let forecast_mt = expert_multithreaded
            .forecast(&inputs)
            .expect("single forecast with threads");
        assert!(forecast1.is_close(&forecast_mt, 1e-4));

        // Test 4: Batched forecast
        let batch_inputs = vec![inputs.clone(), inputs.clone(), inputs.clone()];
        let batch_results = expert
            .forecast_batch(&batch_inputs)
            .expect("batch forecast");
        assert_eq!(batch_results.len(), 3);
        for res in &batch_results {
            assert!(forecast1.is_close(res, 1e-4));
        }

        // Test 5: Empty batch returns empty Vec
        let empty_batch = expert.forecast_batch(&[]).expect("empty batch");
        assert!(empty_batch.is_empty());

        // Test 6: Thread safety across concurrent threads
        let shared_expert = std::sync::Arc::new(expert);
        let mut handles = Vec::new();
        for _ in 0..4 {
            let expert_clone = std::sync::Arc::clone(&shared_expert);
            let inp = inputs.clone();
            handles.push(std::thread::spawn(move || {
                expert_clone.forecast(&inp).expect("thread forecast")
            }));
        }
        for handle in handles {
            let res = handle.join().expect("thread join");
            assert!(forecast1.is_close(&res, 1e-4));
        }
    }
}

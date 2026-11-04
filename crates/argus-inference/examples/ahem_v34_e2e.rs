//! Real-PCAP capture -> cache -> native ONNX end-to-end validation utility.

use std::{
    collections::HashMap,
    env, fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Instant,
};

use argus_capture::{CapturePoll, CaptureSource, PcapLimits, PcapReplaySource};
use argus_core::CacheVersion;
use argus_features::{AhemV34Cache, AhemV34CacheConfig, AhemV34FlowIdentity};
use argus_inference::{AhemV34Forecast, AhemV34TensorContract, OnnxAhemV34Forecaster};
use argus_model_artifact::{
    CalibrationRequirements, ModelArtifactManifest, Provenance, ReadinessRequirements,
    RuntimeCompatibility,
};
use serde_json::Value;

fn main() -> Result<(), String> {
    let arguments = env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if arguments.len() != 4 {
        return Err(
            "usage: ahem_v34_e2e <onnxruntime-library> <model.onnx> <manifest.json> <real.pcap-or-pcapng>"
                .to_owned(),
        );
    }
    let manifest = read_manifest(&arguments[2])?;
    let forecaster = OnnxAhemV34Forecaster::load(
        &arguments[0],
        &arguments[1],
        &manifest,
        AhemV34TensorContract::default(),
    )
    .map_err(|error| error.to_string())?;
    let limits = PcapLimits {
        max_file_bytes: NonZeroUsize::new(512 * 1024 * 1024).expect("constant is non-zero"),
        max_records: NonZeroUsize::new(5_000_000).expect("constant is non-zero"),
    };
    let mut source =
        PcapReplaySource::from_path(&arguments[3], limits).map_err(|error| error.to_string())?;
    let capture_id = source.session_metadata().session_id.clone();
    let version = CacheVersion {
        feature_schema: 34,
        model: 34,
    };
    let mut cache = AhemV34Cache::new(&capture_id, version, AhemV34CacheConfig::default())
        .map_err(|error| error.to_string())?;
    let mut pending = HashMap::<AhemV34FlowIdentity, AhemV34Forecast>::new();
    let mut latencies_ns = Vec::new();
    let mut evaluated = 0_u64;
    let mut absolute_error = [0.0_f64; 2];
    let started = Instant::now();
    let mut packets = 0_u64;

    while let CapturePoll::Packet(packet) = source.poll_next().map_err(|error| error.to_string())? {
        let result = cache
            .ingest(&capture_id, version, &packet)
            .map_err(|error| error.to_string())?;
        packets = packets.saturating_add(1);
        if let Some(prior) = pending.remove(&result.flow) {
            absolute_error[0] +=
                f64::from((prior.log_frame_length - result.packet_features[0]).abs());
            absolute_error[1] += f64::from(
                (prior.log_per_flow_interarrival_seconds - result.packet_features[1]).abs(),
            );
            evaluated = evaluated.saturating_add(1);
        }
        if let Some(inputs) = result.inputs {
            let inference_started = Instant::now();
            let forecast = forecaster
                .forecast(&inputs)
                .map_err(|error| error.to_string())?;
            latencies_ns.push(inference_started.elapsed());
            pending.insert(result.flow, forecast);
        }
    }
    source.stop().map_err(|error| error.to_string())?;
    latencies_ns.sort_unstable();
    let p50 = percentile(&latencies_ns, 50);
    let p95 = percentile(&latencies_ns, 95);
    let stats = cache.stats();
    let elapsed = started.elapsed();
    let denominator =
        f64::from(u32::try_from(evaluated.max(1)).map_err(|_| "evaluation count exceeds u32")?);
    println!(
        "model={} packets={} forecasts={} evaluated_next_flow={} elapsed_ms={:.3} p50_inference_ms={:.3} p95_inference_ms={:.3} mae_log_length={:.6} mae_log_interarrival={:.6} local_retained={} flow_lanes={} flow_events={}",
        forecaster.id(),
        packets,
        latencies_ns.len(),
        evaluated,
        elapsed.as_secs_f64() * 1_000.0,
        p50.as_secs_f64() * 1_000.0,
        p95.as_secs_f64() * 1_000.0,
        absolute_error[0] / denominator,
        absolute_error[1] / denominator,
        stats.local_packets,
        stats.flow_lanes,
        stats.flow_events,
    );
    Ok(())
}

fn percentile(values: &[std::time::Duration], percentile: usize) -> std::time::Duration {
    if values.is_empty() {
        return std::time::Duration::ZERO;
    }
    let index = (values.len() - 1).saturating_mul(percentile) / 100;
    values[index]
}

fn read_manifest(path: &Path) -> Result<ModelArtifactManifest, String> {
    let value: Value = serde_json::from_slice(&fs::read(path).map_err(|error| error.to_string())?)
        .map_err(|error| error.to_string())?;
    let calibration_reference = value
        .pointer("/calibration/reference")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let maximum_age_seconds = value
        .pointer("/readiness/maximum_age_seconds")
        .and_then(Value::as_u64);
    let targets = value
        .pointer("/runtime/supported_targets")
        .and_then(Value::as_array)
        .ok_or("manifest supported targets are absent")?
        .iter()
        .map(|target| {
            target
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| "manifest target is not a string".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ModelArtifactManifest {
        format: string_at(&value, "/format")?,
        format_version: u32_at(&value, "/format_version")?,
        feature_schema_id: string_at(&value, "/feature_schema_id")?,
        feature_schema_version: u32_at(&value, "/feature_schema_version")?,
        model_id: string_at(&value, "/model_id")?,
        model_version: string_at(&value, "/model_version")?,
        artifact_sha256: string_at(&value, "/artifact_sha256")?,
        provenance: Provenance {
            training_dataset_reference: string_at(
                &value,
                "/provenance/training_dataset_reference",
            )?,
            evaluation_reference: string_at(&value, "/provenance/evaluation_reference")?,
            source_revision: string_at(&value, "/provenance/source_revision")?,
        },
        calibration: CalibrationRequirements {
            required: bool_at(&value, "/calibration/required")?,
            reference: calibration_reference,
        },
        readiness: ReadinessRequirements {
            minimum_observations: u64_at(&value, "/readiness/minimum_observations")?,
            maximum_age_seconds,
        },
        runtime: RuntimeCompatibility {
            runtime_id: string_at(&value, "/runtime/runtime_id")?,
            minimum_runtime_version: string_at(&value, "/runtime/minimum_runtime_version")?,
            supported_targets: targets,
        },
    })
}

fn string_at(value: &Value, pointer: &str) -> Result<String, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("manifest string {pointer} is absent"))
}

fn u64_at(value: &Value, pointer: &str) -> Result<u64, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("manifest integer {pointer} is absent"))
}

fn u32_at(value: &Value, pointer: &str) -> Result<u32, String> {
    u32::try_from(u64_at(value, pointer)?).map_err(|_| format!("{pointer} exceeds u32"))
}

fn bool_at(value: &Value, pointer: &str) -> Result<bool, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("manifest boolean {pointer} is absent"))
}

//! Explicit CPU-only preview. This is not the Python-backed AHEM v6.1 pipeline.
use std::{fs, io::Write, num::NonZeroUsize, path::Path};

use argus_capture::{PcapLimits, PcapReplaySource, RawCapturePoll, RawCaptureSource};
use serde::Deserialize;
use serde_json::{Value, json};

pub const FILE: &str = "runtime-mode.json";
pub const MODELS: [&str; 3] = ["cnn_local", "ahem_fixed", "tcn_long"];

struct Cleanup(std::path::PathBuf);
impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    schema: String,
    mode: String,
}

pub fn enabled_at(root: &Path) -> Result<bool, String> {
    let path = root.join(FILE);
    if !path.exists() {
        return Ok(false);
    }
    if fs::metadata(&path)
        .map_err(|error| error.to_string())?
        .len()
        > 1_024
    {
        return Err("runtime profile exceeds size limit".into());
    }
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let profile: Profile = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    if profile.schema != "argus.runtime-mode/v1" || profile.mode != "native-preview" {
        return Err("unsupported runtime profile; refusing Python fallback".into());
    }
    Ok(true)
}

pub fn root() -> Result<std::path::PathBuf, String> {
    std::env::current_exe()
        .map_err(|error| error.to_string())?
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "executable parent unavailable".into())
}

pub fn enabled() -> Result<bool, String> {
    enabled_at(&root()?)
}

pub fn response(mut native: Value, disabled: &[String]) -> Value {
    let mut models = native["models"].take();
    if let Some(entries) = models.as_object_mut() {
        for id in super::SELECTABLE_MODEL_IDS {
            entries.entry(id).or_insert_with(|| {
                json!({
                    "state": if disabled.iter().any(|name| name == id) { "disabled" } else { "unavailable" },
                    "detail": "Not available in the native preview. No replacement model was selected.",
                })
            });
        }
    }
    let executed = models.as_object().map_or(0, |entries| {
        entries
            .values()
            .filter(|entry| entry["state"] == "forecast_only")
            .count()
    });
    json!({
        "state": "active", "runtime_backend": "onnxruntime",
        "execution_provider": "CPUExecutionProvider", "performance_mode": "native-preview",
        "runtime_notice": "Legacy native experts only; AHEM v6.1 memory preparation still requires Python.",
        "models": models, "native_onnx": native,
        "router": {"state": "inactive", "detail": "No Python router or model substitution."},
        "forecast_fusion": {"primary_research_state": "unavailable"},
        "telemetry": {"executed_models": executed, "total_inference_ms": native["total_ms"]},
    })
}

/// Headless mechanical test only: never opens `WebView`, captures live traffic,
/// reads preferences, installs tasks, or invokes Defender.
pub fn exit_if_check_requested() {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if arguments
        .first()
        .is_none_or(|arg| arg != "--native-preview-check")
    {
        return;
    }
    let result = if arguments.len() == 2 {
        check(Path::new(&arguments[1]))
    } else {
        Err("usage: --native-preview-check <new absolute result.json>".into())
    };
    if let Err(error) = result {
        eprintln!("native preview check failed: {error}");
        std::process::exit(1);
    }
    std::process::exit(0);
}

fn check(output: &Path) -> Result<(), String> {
    if !output.is_absolute() {
        return Err("check output must be absolute".into());
    }
    let root = root()?;
    if !enabled_at(&root)? {
        return Err("native preview profile is required".into());
    }
    // Reserve without overwriting anything, including on error.
    let mut report = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|error| error.to_string())?;
    let temporary = std::env::temp_dir().join(format!("argus-native-check-{}", std::process::id()));
    fs::create_dir(&temporary).map_err(|error| error.to_string())?;
    let _cleanup = Cleanup(temporary.clone());
    let capture = temporary.join("generated.pcap");
    let features = temporary.join("features.csv");
    let template = temporary.join("template.pcap");
    let bytes = argus_capture::generate_simulation_pcap("nominal_baseline_traffic", 500)?;
    fs::write(&template, bytes).map_err(|error| error.to_string())?;
    let mut source = PcapReplaySource::from_path(
        &template,
        PcapLimits {
            max_file_bytes: NonZeroUsize::new(1_048_576).expect("positive limit"),
            max_records: NonZeroUsize::new(500).expect("positive limit"),
        },
    )
    .map_err(|error| error.to_string())?;
    let mut frames = Vec::new();
    while let RawCapturePoll::Packet(packet) =
        source.poll_next_raw().map_err(|error| error.to_string())?
    {
        frames.push(packet.bytes);
    }
    source.stop_raw().map_err(|error| error.to_string())?;
    if frames.is_empty() {
        return Err("generated template contained no packets".into());
    }
    // Extend only this explicit mechanical fixture beyond the UI's 500-packet
    // safety cap so the 1,024-packet TCN warms up. These are not measured traffic.
    argus_capture::write_pcap_file(
        &capture,
        (0_usize..1_200).map(|index| {
            let frame = &frames[index % frames.len()];
            let ordinal = u64::try_from(index).expect("bounded fixture ordinal");
            (
                1_000_000_000 + ordinal * 20_000_000,
                frame.as_slice(),
                frame.len(),
            )
        }),
    )
    .map_err(|error| error.to_string())?;
    let child = super::background_command(root.join("argus-nids.exe"))
        .args([
            std::ffi::OsStr::new("--export-features"),
            capture.as_os_str(),
            features.as_os_str(),
        ])
        .output()
        .map_err(|error| error.to_string())?;
    if !child.status.success() {
        return Err(format!("native feature export failed: {}", child.status));
    }
    let mut worker = super::CnnWorker::default();
    let result = worker.infer(
        &root.join("models"),
        &features,
        "generated-package-check",
        true,
        None,
        false,
    )?;
    for id in MODELS {
        if result["models"][id]["state"] != "forecast_only" {
            return Err(format!("{id} did not return a forecast"));
        }
    }
    if worker.child.is_some() || result["models"]["ahem_v61"]["state"] != "unavailable" {
        return Err("native-only boundary violated".into());
    }
    let summary = json!({"schema": "argus.native-preview-check/v1", "passed": true,
        "generated_capture": true, "python_started": false, "result": result});
    report
        .write_all(
            serde_json::to_string_pretty(&summary)
                .map_err(|error| error.to_string())?
                .as_bytes(),
        )
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_response_never_labels_legacy_output_as_ahem_v61() {
        let result = response(
            json!({"models": {"cnn_local": {"state": "forecast_only"}}, "total_ms": 2.0}),
            &[],
        );
        assert_eq!(result["models"]["ahem_v61"]["state"], "unavailable");
        assert_eq!(result["telemetry"]["executed_models"], 1);
        assert_eq!(
            result["forecast_fusion"]["primary_research_state"],
            "unavailable"
        );
    }
    #[test]
    fn malformed_or_unknown_profile_refuses_fallback() {
        let directory =
            std::env::temp_dir().join(format!("argus-profile-test-{}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        assert!(!enabled_at(&directory).unwrap());
        fs::write(directory.join(FILE), b"not json").unwrap();
        assert!(enabled_at(&directory).is_err());
        fs::write(
            directory.join(FILE),
            br#"{"schema":"argus.runtime-mode/v1","mode":"native-preview"}"#,
        )
        .unwrap();
        assert!(enabled_at(&directory).unwrap());
        fs::write(
            directory.join(FILE),
            br#"{"schema":"argus.runtime-mode/v1","mode":"python"}"#,
        )
        .unwrap();
        assert!(enabled_at(&directory).is_err());
        fs::remove_dir_all(directory).unwrap();
    }
}

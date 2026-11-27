//! Read-only, on-demand host context. Never attached to imported packet evidence.
use serde_json::Value;

pub async fn snapshot(enabled: bool) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || collect(enabled))
        .await
        .map_err(|e| format!("Device context worker failed: {e}"))?
}

#[cfg(not(windows))]
fn collect(_enabled: bool) -> Result<Value, String> {
    Err("Local device context currently requires Windows.".into())
}

#[cfg(windows)]
fn collect(enabled: bool) -> Result<Value, String> {
    use std::{
        io::Read,
        process::Stdio,
        thread,
        time::{Duration, Instant},
    };
    let root = std::env::var_os("SystemRoot").ok_or("Windows system directory unavailable")?;
    let executable =
        std::path::PathBuf::from(root).join("System32/WindowsPowerShell/v1.0/powershell.exe");
    let script = format!(
        "& {{ {} }} -ReadDefender:${}",
        include_str!("device_context.ps1"),
        if enabled { "true" } else { "false" }
    );
    let mut child = super::background_command(executable)
        .args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            &script,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Cannot read Windows context: {e}"))?;
    let stdout = child.stdout.take().ok_or("Missing device context output")?;
    let reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout
            .take(1_048_577)
            .read_to_end(&mut bytes)
            .map(|_| bytes)
    });
    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(50)),
            other => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("Device context timed out or failed: {other:?}"));
            }
        }
    };
    let bytes = reader
        .join()
        .map_err(|_| "Device context reader failed")?
        .map_err(|e| e.to_string())?;
    if !status.success() || bytes.len() > 1_048_576 {
        return Err("Windows context failed or exceeded its output limit.".into());
    }
    parse_snapshot(&bytes)
}

#[cfg(any(windows, test))]
fn parse_snapshot(bytes: &[u8]) -> Result<Value, String> {
    let value: Value =
        serde_json::from_slice(bytes).map_err(|e| format!("Invalid device context: {e}"))?;
    if value["schema"] != "argus.device-context/v1"
        || !value["connections"].is_array()
        || !value["detections"].is_array()
    {
        return Err("Unexpected device context schema".into());
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    #[test]
    fn rejects_invalid_snapshot() {
        assert!(super::parse_snapshot(b"{}").is_err());
        assert!(super::parse_snapshot(b"not json").is_err());
    }
    #[test]
    fn accepts_partial_provider_failure() {
        assert!(super::parse_snapshot(br#"{"schema":"argus.device-context/v1","connections":[],"detections":[],"defenderError":"Access denied"}"#).is_ok());
    }
}

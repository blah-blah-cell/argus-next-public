//! Fixed, one-shot elevated actions. No arbitrary command text or retained admin service.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DefenderMode {
    #[default]
    Off,
    ReadOnly,
    Controls,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DefenderAction {
    QuickScan,
    FullScan,
    UpdateSignatures,
}

impl DefenderAction {
    fn argument(self) -> &'static str {
        match self {
            Self::QuickScan => "quick_scan",
            Self::FullScan => "full_scan",
            Self::UpdateSignatures => "update_signatures",
        }
    }
    fn script(self) -> &'static str {
        match self {
            Self::QuickScan => "Start-MpScan -ScanType QuickScan -ErrorAction Stop",
            Self::FullScan => "Start-MpScan -ScanType FullScan -ErrorAction Stop",
            Self::UpdateSignatures => "Update-MpSignature -ErrorAction Stop",
        }
    }
}

pub async fn run(action: DefenderAction) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        use std::sync::atomic::{AtomicBool, Ordering};
        static BUSY: AtomicBool = AtomicBool::new(false);
        struct Release;
        impl Drop for Release {
            fn drop(&mut self) {
                BUSY.store(false, Ordering::Release);
            }
        }
        if BUSY.swap(true, Ordering::AcqRel) {
            return Err("A Defender action is already awaiting approval or completion.".into());
        }
        let _release = Release;
        run_blocking(action)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[cfg(windows)]
fn run_blocking(action: DefenderAction) -> Result<(), String> {
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let shell = powershell()?;
    let script = format!(
        "$ErrorActionPreference='Stop'; try {{ $p=Start-Process -FilePath {} -Verb RunAs -WindowStyle Hidden -ArgumentList @('--defender-action','{}') -Wait -PassThru; exit $p.ExitCode }} catch {{ exit 1223 }}",
        super::quote_powershell(&executable),
        action.argument()
    );
    let status = super::background_command(shell)
        .args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            &script,
        ])
        .status()
        .map_err(|e| e.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err("Defender action was cancelled, denied, or failed. Refresh Defender status or review Windows Security for details; no clean-scan result is implied.".into())
    }
}

#[cfg(not(windows))]
fn run_blocking(_action: DefenderAction) -> Result<(), String> {
    Err("Defender controls require Windows.".into())
}

#[cfg(windows)]
fn powershell() -> Result<std::path::PathBuf, String> {
    Ok(
        std::path::PathBuf::from(std::env::var_os("SystemRoot").ok_or("SystemRoot unavailable")?)
            .join("System32/WindowsPowerShell/v1.0/powershell.exe"),
    )
}

pub fn exit_if_helper() {
    if std::env::args().nth(1).as_deref() != Some("--defender-action") {
        return;
    }
    let action = std::env::args()
        .nth(2)
        .and_then(|a| serde_json::from_value::<DefenderAction>(serde_json::Value::String(a)).ok());
    let code = action.map_or(2, helper);
    std::process::exit(code);
}

#[cfg(windows)]
fn helper(action: DefenderAction) -> i32 {
    let Ok(shell) = powershell() else {
        return 1;
    };
    let script = format!(
        "$ErrorActionPreference='Stop'; try {{ if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {{exit 5}}; {}; exit 0 }} catch {{exit 1}}",
        action.script()
    );
    super::background_command(shell)
        .args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            &script,
        ])
        .status()
        .ok()
        .and_then(|s| s.code())
        .unwrap_or(1)
}

#[cfg(not(windows))]
fn helper(_action: DefenderAction) -> i32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actions_are_closed_allowlist() {
        for value in ["disable", "quick_scan; anything", "restore", ""] {
            assert!(
                serde_json::from_value::<DefenderAction>(serde_json::Value::String(value.into()))
                    .is_err()
            );
        }
        assert_eq!(DefenderAction::QuickScan.argument(), "quick_scan");
        assert!(
            !DefenderAction::FullScan
                .script()
                .contains("Set-MpPreference")
        );
        assert_eq!(DefenderMode::default(), DefenderMode::Off);
    }
}

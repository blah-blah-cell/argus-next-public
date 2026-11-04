use std::{
    env, fmt, fs, io,
    num::{NonZeroU32, NonZeroU64},
    path::{Path, PathBuf},
    process::{Command, Output},
    thread,
    time::Duration,
};

use crate::{CaptureError, PcapLimits, PcapReplaySource};

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

fn pktmon_command(executable: &Path) -> Command {
    let command = Command::new(executable);
    #[cfg(windows)]
    let command = {
        use std::os::windows::process::CommandExt;

        let mut command = command;
        command.creation_flags(CREATE_NO_WINDOW);
        command
    };
    command
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PktmonReadiness {
    Unavailable,
    PrivilegeRequired,
    StatusReachable,
    ProbeFailed { detail: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PktmonProbe {
    pub executable: Option<PathBuf>,
    pub readiness: PktmonReadiness,
}

impl PktmonProbe {
    /// Probes the documented Pktmon status command without starting capture.
    #[must_use]
    pub fn inspect() -> Self {
        let Some(executable) = locate_pktmon() else {
            return Self {
                executable: None,
                readiness: PktmonReadiness::Unavailable,
            };
        };
        let readiness = match pktmon_command(&executable).arg("status").output() {
            Ok(output) if output.status.success() => PktmonReadiness::StatusReachable,
            Ok(output) if is_privilege_error(&output) => PktmonReadiness::PrivilegeRequired,
            Ok(output) => PktmonReadiness::ProbeFailed {
                detail: checked_output_detail(&output),
            },
            Err(error) => PktmonReadiness::ProbeFailed {
                detail: error.to_string(),
            },
        };
        Self {
            executable: Some(executable),
            readiness,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PktmonRecordConfig {
    pub duration_seconds: NonZeroU64,
    pub etl_base_path: PathBuf,
    pub pcapng_path: PathBuf,
    pub file_size_megabytes: NonZeroU32,
}

impl PktmonRecordConfig {
    /// Validates a bounded stopped-session recording configuration.
    ///
    /// # Errors
    ///
    /// Returns [`PktmonError::InvalidConfig`] for a duration above 30 seconds, non-absolute paths,
    /// wrong extensions, the same output path, or missing output directories.
    pub fn validate(&self) -> Result<(), PktmonError> {
        if self.duration_seconds.get() > 30 {
            return Err(PktmonError::InvalidConfig(
                "duration_seconds must be between 1 and 30".to_owned(),
            ));
        }
        validate_output_path(&self.etl_base_path, "etl")?;
        validate_output_path(&self.pcapng_path, "pcapng")?;
        if self.etl_base_path == self.pcapng_path {
            return Err(PktmonError::InvalidConfig(
                "ETL and PCAPNG paths must differ".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Completed stopped-session recording. The ETL is retained for audit/recovery.
#[derive(Debug)]
pub struct PktmonRecording {
    pub etl_path: PathBuf,
    pub pcapng_path: PathBuf,
    pub replay: PcapReplaySource,
}

impl PktmonRecording {
    /// Records a bounded Pktmon session, stops it, converts its sole ETL output, validates the
    /// resulting PCAPNG, and opens deterministic replay.
    ///
    /// This is not a live [`crate::CaptureSource`]. It blocks for the configured duration and
    /// returns only after Pktmon has stopped and conversion has succeeded.
    ///
    /// # Errors
    ///
    /// Returns [`PktmonError`] for missing privileges, unsafe/existing paths, any failed command,
    /// absent/ambiguous/empty outputs, or replay parse failure. No failure is converted to success.
    pub fn record(config: &PktmonRecordConfig, limits: PcapLimits) -> Result<Self, PktmonError> {
        config.validate()?;
        let probe = PktmonProbe::inspect();
        let executable = probe.executable.ok_or(PktmonError::Unavailable)?;
        match probe.readiness {
            PktmonReadiness::StatusReachable => {}
            PktmonReadiness::PrivilegeRequired => return Err(PktmonError::PrivilegeRequired),
            PktmonReadiness::Unavailable => return Err(PktmonError::Unavailable),
            PktmonReadiness::ProbeFailed { detail } => {
                return Err(PktmonError::ProbeFailed(detail));
            }
        }
        ensure_outputs_absent(config)?;

        let start = pktmon_command(&executable)
            .args([
                "start",
                "--capture",
                "--comp",
                "nics",
                "--pkt-size",
                "0",
                "--file-name",
            ])
            .arg(&config.etl_base_path)
            .args([
                "--file-size",
                &config.file_size_megabytes.get().to_string(),
                "--log-mode",
                // The CLI accepts one audit ETL path and must not produce a
                // numbered multi-file set. Pktmon's circular mode writes the
                // requested single file, bounded by `file_size_megabytes`.
                "circular",
            ])
            .output()
            .map_err(|error| PktmonError::Io {
                operation: "start",
                error,
            })?;
        check_command("start", &start)?;

        thread::sleep(Duration::from_secs(config.duration_seconds.get()));
        let stop = pktmon_command(&executable)
            .arg("stop")
            .output()
            .map_err(|error| PktmonError::Io {
                operation: "stop",
                error,
            })?;
        check_command("stop", &stop)?;

        let etl_path = sole_etl_output(&config.etl_base_path)?;
        require_nonempty(&etl_path, "ETL")?;
        let conversion = pktmon_command(&executable)
            .arg("etl2pcap")
            .arg(&etl_path)
            .arg("--out")
            .arg(&config.pcapng_path)
            .output()
            .map_err(|error| PktmonError::Io {
                operation: "etl2pcap",
                error,
            })?;
        check_command("etl2pcap", &conversion)?;
        require_nonempty(&config.pcapng_path, "PCAPNG")?;
        let replay = PcapReplaySource::from_path(&config.pcapng_path, limits)
            .map_err(PktmonError::Replay)?;
        Ok(Self {
            etl_path,
            pcapng_path: config.pcapng_path.clone(),
            replay,
        })
    }
}

#[derive(Debug)]
pub enum PktmonError {
    Unavailable,
    PrivilegeRequired,
    ProbeFailed(String),
    InvalidConfig(String),
    OutputAlreadyExists(PathBuf),
    Io {
        operation: &'static str,
        error: io::Error,
    },
    CommandFailed {
        operation: &'static str,
        code: Option<i32>,
        detail: String,
    },
    MissingOutput {
        kind: &'static str,
        path: PathBuf,
    },
    AmbiguousEtlOutputs(Vec<PathBuf>),
    Replay(CaptureError),
}

impl fmt::Display for PktmonError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable => formatter.write_str("Pktmon is unavailable"),
            Self::PrivilegeRequired => {
                formatter.write_str("Pktmon requires an elevated Windows process")
            }
            Self::ProbeFailed(detail) => {
                write!(formatter, "Pktmon readiness probe failed: {detail}")
            }
            Self::InvalidConfig(detail) => write!(formatter, "invalid Pktmon config: {detail}"),
            Self::OutputAlreadyExists(path) => write!(
                formatter,
                "refusing to overwrite existing Pktmon output {}",
                path.display()
            ),
            Self::Io { operation, error } => {
                write!(formatter, "Pktmon {operation} failed: {error}")
            }
            Self::CommandFailed {
                operation,
                code,
                detail,
            } => {
                write!(
                    formatter,
                    "Pktmon {operation} exited with {code:?}: {detail}"
                )
            }
            Self::MissingOutput { kind, path } => {
                write!(
                    formatter,
                    "Pktmon {kind} output is missing or empty: {}",
                    path.display()
                )
            }
            Self::AmbiguousEtlOutputs(paths) => {
                write!(
                    formatter,
                    "Pktmon produced {} ETL files; refusing partial replay",
                    paths.len()
                )
            }
            Self::Replay(error) => write!(formatter, "Pktmon PCAPNG validation failed: {error}"),
        }
    }
}

impl std::error::Error for PktmonError {}

fn locate_pktmon() -> Option<PathBuf> {
    let system_root = env::var_os("SystemRoot")?;
    let path = PathBuf::from(system_root)
        .join("System32")
        .join("PktMon.exe");
    path.is_file().then_some(path)
}

fn validate_output_path(path: &Path, extension: &str) -> Result<(), PktmonError> {
    if !path.is_absolute() {
        return Err(PktmonError::InvalidConfig(format!(
            "{} path must be absolute",
            extension.to_uppercase()
        )));
    }
    if !path
        .extension()
        .is_some_and(|value| value.eq_ignore_ascii_case(extension))
    {
        return Err(PktmonError::InvalidConfig(format!(
            "output must use .{extension} extension"
        )));
    }
    if !path.parent().is_some_and(Path::is_dir) {
        return Err(PktmonError::InvalidConfig(
            "output directory does not exist".to_owned(),
        ));
    }
    Ok(())
}

fn ensure_outputs_absent(config: &PktmonRecordConfig) -> Result<(), PktmonError> {
    if config.pcapng_path.exists() {
        return Err(PktmonError::OutputAlreadyExists(config.pcapng_path.clone()));
    }
    let candidates = etl_candidates(&config.etl_base_path)?;
    if let Some(path) = candidates.into_iter().next() {
        return Err(PktmonError::OutputAlreadyExists(path));
    }
    Ok(())
}

fn sole_etl_output(base: &Path) -> Result<PathBuf, PktmonError> {
    let candidates = etl_candidates(base)?;
    match candidates.as_slice() {
        [path] => Ok(path.clone()),
        [] => Err(PktmonError::MissingOutput {
            kind: "ETL",
            path: base.to_path_buf(),
        }),
        _ => Err(PktmonError::AmbiguousEtlOutputs(candidates)),
    }
}

fn etl_candidates(base: &Path) -> Result<Vec<PathBuf>, PktmonError> {
    let parent = base.parent().ok_or_else(|| {
        PktmonError::InvalidConfig("ETL output has no parent directory".to_owned())
    })?;
    let stem = base
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| PktmonError::InvalidConfig("ETL filename is not Unicode".to_owned()))?
        .to_ascii_lowercase();
    let mut candidates = Vec::new();
    for entry in fs::read_dir(parent).map_err(|error| PktmonError::Io {
        operation: "list output directory",
        error,
    })? {
        let path = entry
            .map_err(|error| PktmonError::Io {
                operation: "read output entry",
                error,
            })?
            .path();
        let matches_extension = path
            .extension()
            .is_some_and(|value| value.eq_ignore_ascii_case("etl"));
        let matches_stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.to_ascii_lowercase().starts_with(&stem));
        if matches_extension && matches_stem {
            candidates.push(path);
        }
    }
    candidates.sort();
    Ok(candidates)
}

fn require_nonempty(path: &Path, kind: &'static str) -> Result<(), PktmonError> {
    let metadata = fs::metadata(path).map_err(|_| PktmonError::MissingOutput {
        kind,
        path: path.to_path_buf(),
    })?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(PktmonError::MissingOutput {
            kind,
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

fn check_command(operation: &'static str, output: &Output) -> Result<(), PktmonError> {
    if output.status.success() {
        return Ok(());
    }
    if is_privilege_error(output) {
        return Err(PktmonError::PrivilegeRequired);
    }
    Err(PktmonError::CommandFailed {
        operation,
        code: output.status.code(),
        detail: checked_output_detail(output),
    })
}

fn is_privilege_error(output: &Output) -> bool {
    let detail = checked_output_detail(output).to_ascii_lowercase();
    detail.contains("access is denied")
        || detail.contains("requires elevation")
        || detail.contains("administrator")
}

fn checked_output_detail(output: &Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if stderr.is_empty() {
        String::from_utf8_lossy(&output.stdout).trim().to_owned()
    } else {
        stderr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_relative_and_overlong_recordings() {
        let relative = PktmonRecordConfig {
            duration_seconds: NonZeroU64::new(1).expect("non-zero"),
            etl_base_path: PathBuf::from("capture.etl"),
            pcapng_path: PathBuf::from("capture.pcapng"),
            file_size_megabytes: NonZeroU32::new(64).expect("non-zero"),
        };
        assert!(matches!(
            relative.validate(),
            Err(PktmonError::InvalidConfig(_))
        ));

        let mut overlong = relative;
        overlong.duration_seconds = NonZeroU64::new(31).expect("non-zero");
        assert!(matches!(
            overlong.validate(),
            Err(PktmonError::InvalidConfig(_))
        ));
    }

    #[test]
    fn current_machine_probe_never_claims_live_capture() {
        let probe = PktmonProbe::inspect();
        assert!(matches!(
            probe.readiness,
            PktmonReadiness::Unavailable
                | PktmonReadiness::PrivilegeRequired
                | PktmonReadiness::StatusReachable
                | PktmonReadiness::ProbeFailed { .. }
        ));
    }
}

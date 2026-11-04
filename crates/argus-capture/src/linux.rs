//! Linux packet capture adapter, `/proc/net` socket-to-process correlator, and bounded `tcpdump`/`dumpcap` recorder.
//!
//! Provides zero-mock Linux network telemetry:
//! - Probes `/sys/class/net`, `/proc/net/dev`, and external capture binaries (`tcpdump`, `dumpcap`).
//! - Parses `/proc/net/{tcp,tcp6,udp,udp6}` and correlates socket inodes with `/proc/[pid]/fd`
//!   to attribute network flows to local Linux processes (`pid`, `comm`, `cmdline`).
//! - Executes bounded `tcpdump` / `dumpcap` PCAP recordings with strict BPF filter validation.

use std::{
    collections::HashMap,
    fs,
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    num::{NonZeroU32, NonZeroUsize},
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

use serde::{Deserialize, Serialize};

use crate::{
    CaptureError,
    pcap::{PcapLimits, PcapLoadReport, PcapReplaySource},
};

/// Readiness state of the Linux packet capture subsystem.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LinuxCaptureReadiness {
    /// `tcpdump` or `dumpcap` is installed and has `CAP_NET_RAW` / root privileges, or `/proc/net` is readable.
    Ready {
        tool: String,
        interfaces: Vec<String>,
        has_raw_capture_privilege: bool,
    },
    /// `/proc/net` socket inspection is available, but `tcpdump`/`dumpcap` lacks `CAP_NET_RAW` or is not installed.
    SocketInspectionOnly {
        reason: String,
        interfaces: Vec<String>,
    },
    /// Not running on Linux or `/proc/net` is inaccessible.
    Unavailable { reason: String },
}

/// Configuration for a bounded Linux `tcpdump`/`dumpcap` recording.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinuxRecordConfig {
    pub output_directory: PathBuf,
    pub duration: Duration,
    pub interface: String,
    pub snaplen: NonZeroU32,
    pub bpf_filter: Option<String>,
    pub limits: PcapLimits,
}

impl LinuxRecordConfig {
    /// Creates a bounded 5-second Ethernet recording on `eth0` (override for this host).
    #[allow(clippy::missing_panics_doc)] // Only nonzero compile-time literals are unwrapped.
    #[must_use]
    pub fn new(output_directory: impl Into<PathBuf>) -> Self {
        Self {
            output_directory: output_directory.into(),
            duration: Duration::from_secs(5),
            interface: "eth0".to_owned(),
            snaplen: NonZeroU32::new(65_535).expect("nonzero snaplen"),
            bpf_filter: None,
            limits: PcapLimits {
                max_file_bytes: NonZeroUsize::new(64 * 1024 * 1024).expect("nonzero limit"),
                max_records: NonZeroUsize::new(250_000).expect("nonzero limit"),
            },
        }
    }

    /// Attaches an optional BPF capture filter after validating it contains no unsafe shell tokens.
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError::InvalidMetadata`] if the filter is empty or contains disallowed control characters.
    pub fn with_bpf_filter(mut self, filter: impl Into<String>) -> Result<Self, CaptureError> {
        let filter = filter.into();
        validate_bpf_filter(&filter)?;
        self.bpf_filter = Some(filter);
        Ok(self)
    }
}

/// Completed Linux PCAP recording and normalized replay source.
#[derive(Clone, Debug)]
pub struct LinuxRecording {
    pub pcap_path: PathBuf,
    pub replay: PcapReplaySource,
}

impl LinuxRecording {
    #[must_use]
    pub const fn report(&self) -> PcapLoadReport {
        self.replay.report()
    }
}

/// Local Linux process owning a network socket.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LinuxProcessOwner {
    pub pid: u32,
    pub name: String,
    pub cmdline: String,
}

/// Observed kernel socket entry from `/proc/net/{tcp,tcp6,udp,udp6}`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LinuxSocketObservation {
    pub protocol: String,
    pub local_address: IpAddr,
    pub local_port: u16,
    pub remote_address: IpAddr,
    pub remote_port: u16,
    pub state: String,
    pub tx_queue: u64,
    pub rx_queue: u64,
    pub uid: u32,
    pub inode: u64,
    pub process: Option<LinuxProcessOwner>,
}

/// Linux interface byte/packet counters from `/proc/net/dev`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LinuxInterfaceStats {
    pub name: String,
    pub rx_bytes: u64,
    pub rx_packets: u64,
    pub rx_errors: u64,
    pub rx_dropped: u64,
    pub tx_bytes: u64,
    pub tx_packets: u64,
    pub tx_errors: u64,
    pub tx_dropped: u64,
}

/// Validates that a user-supplied BPF filter string is safe to pass as a single argument to `tcpdump` or `dumpcap`.
///
/// # Errors
///
/// Returns [`CaptureError::InvalidMetadata`] if the filter exceeds length limits or contains control characters.
pub fn validate_bpf_filter(filter: &str) -> Result<(), CaptureError> {
    let trimmed = filter.trim();
    if trimmed.is_empty() {
        return Err(CaptureError::InvalidMetadata("BPF filter cannot be empty"));
    }
    if trimmed.len() > 512 {
        return Err(CaptureError::InvalidMetadata(
            "BPF filter exceeds 512-character safety limit",
        ));
    }
    if trimmed.chars().any(|ch| {
        ch.is_control() || matches!(ch, ';' | '|' | '`' | '$' | '\n' | '\r' | '\0' | '>' | '<')
    }) {
        return Err(CaptureError::InvalidMetadata(
            "BPF filter contains disallowed shell or control characters",
        ));
    }
    Ok(())
}

/// Probes Linux network interfaces, `/proc/net` readability, and `tcpdump`/`dumpcap` availability.
#[derive(Clone, Debug, Default)]
pub struct LinuxCaptureProbe;

impl LinuxCaptureProbe {
    #[must_use]
    pub fn check() -> LinuxCaptureReadiness {
        let interfaces = list_linux_interfaces();
        let proc_net_ok =
            Path::new("/proc/net/dev").exists() && Path::new("/proc/net/tcp").exists();

        if !proc_net_ok && cfg!(not(target_os = "linux")) {
            return LinuxCaptureReadiness::Unavailable {
                reason: "Linux /proc/net telemetry is not present on this OS".to_owned(),
            };
        }

        let tool = find_capture_tool();
        let is_root = unsafe_geteuid_is_root();

        match tool {
            Some(tool_path) if is_root => LinuxCaptureReadiness::Ready {
                tool: tool_path,
                interfaces,
                has_raw_capture_privilege: true,
            },
            Some(tool_path) => {
                // Check if tool has cap_net_raw via getcap
                let has_caps = Command::new("getcap")
                    .arg(&tool_path)
                    .output()
                    .ok()
                    .and_then(|out| String::from_utf8(out.stdout).ok())
                    .is_some_and(|text| text.contains("cap_net_raw"));
                if has_caps {
                    LinuxCaptureReadiness::Ready {
                        tool: tool_path,
                        interfaces,
                        has_raw_capture_privilege: true,
                    }
                } else {
                    LinuxCaptureReadiness::SocketInspectionOnly {
                        reason: format!(
                            "{tool_path} is installed, but requires root or cap_net_raw+ep for raw wire capture; using /proc/net live kernel socket telemetry"
                        ),
                        interfaces,
                    }
                }
            }
            None if proc_net_ok => LinuxCaptureReadiness::SocketInspectionOnly {
                reason:
                    "tcpdump/dumpcap not found; using /proc/net kernel socket & interface telemetry"
                        .to_owned(),
                interfaces,
            },
            None => LinuxCaptureReadiness::Unavailable {
                reason: "Neither tcpdump/dumpcap nor /proc/net is accessible".to_owned(),
            },
        }
    }
}

fn unsafe_geteuid_is_root() -> bool {
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("Uid:") {
                let mut parts = rest.split_whitespace();
                let _real = parts.next();
                if let Some(effective) = parts.next() {
                    return effective == "0";
                }
            }
        }
    }
    false
}

fn find_capture_tool() -> Option<String> {
    for candidate in [
        "/usr/bin/tcpdump",
        "/usr/sbin/tcpdump",
        "/bin/tcpdump",
        "/sbin/tcpdump",
        "/usr/bin/dumpcap",
        "/usr/local/bin/tcpdump",
    ] {
        if Path::new(candidate).is_file() {
            return Some(candidate.to_owned());
        }
    }
    for name in ["tcpdump", "dumpcap"] {
        if let Ok(out) = Command::new("which").arg(name).output()
            && out.status.success()
        {
            let path = String::from_utf8_lossy(&out.stdout).trim().to_owned();
            if !path.is_empty() {
                return Some(path);
            }
        }
    }
    None
}

/// Lists active Linux network interfaces from `/sys/class/net` or `/proc/net/dev`.
#[must_use]
pub fn list_linux_interfaces() -> Vec<String> {
    let mut names = Vec::new();
    if let Ok(entries) = fs::read_dir("/sys/class/net") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.is_empty() {
                names.push(name);
            }
        }
    }
    if names.is_empty()
        && let Ok(dev) = fs::read_to_string("/proc/net/dev")
    {
        for stats in parse_proc_net_dev(&dev) {
            names.push(stats.name);
        }
    }
    names.sort();
    names.dedup();
    names
}

/// Parses `/proc/net/dev` content into structured per-interface byte/packet counters.
#[must_use]
pub fn parse_proc_net_dev(contents: &str) -> Vec<LinuxInterfaceStats> {
    let mut results = Vec::new();
    for line in contents.lines().skip(2) {
        let Some((iface, rest)) = line.split_once(':') else {
            continue;
        };
        let name = iface.trim().to_owned();
        if name.is_empty() {
            continue;
        }
        let cols: Vec<u64> = rest
            .split_whitespace()
            .filter_map(|tok| tok.parse::<u64>().ok())
            .collect();
        if cols.len() < 12 {
            continue;
        }
        results.push(LinuxInterfaceStats {
            name,
            rx_bytes: cols[0],
            rx_packets: cols[1],
            rx_errors: cols[2],
            rx_dropped: cols[3],
            tx_bytes: cols[8],
            tx_packets: cols[9],
            tx_errors: cols[10],
            tx_dropped: cols[11],
        });
    }
    results
}

/// Reads `/proc/[pid]/fd` across running processes to map socket `inode -> LinuxProcessOwner`.
#[must_use]
pub fn build_inode_process_map() -> HashMap<u64, LinuxProcessOwner> {
    let mut map = HashMap::new();
    let Ok(proc_entries) = fs::read_dir("/proc") else {
        return map;
    };
    for entry in proc_entries.flatten() {
        let file_name = entry.file_name();
        let Some(pid_str) = file_name.to_str() else {
            continue;
        };
        let Ok(pid) = pid_str.parse::<u32>() else {
            continue;
        };
        let fd_dir = entry.path().join("fd");
        let Ok(fds) = fs::read_dir(&fd_dir) else {
            continue;
        };
        let mut inodes = Vec::new();
        for fd in fds.flatten() {
            if let Ok(target) = fs::read_link(fd.path())
                && let Some(target_str) = target.to_str()
                && let Some(inode) = parse_socket_link_inode(target_str)
            {
                inodes.push(inode);
            }
        }
        if inodes.is_empty() {
            continue;
        }
        let name = fs::read_to_string(entry.path().join("comm"))
            .map_or_else(|_| format!("pid-{pid}"), |s| s.trim().to_owned());
        let cmdline = fs::read(entry.path().join("cmdline"))
            .map(|bytes| {
                bytes
                    .split(|b| *b == 0)
                    .filter(|part| !part.is_empty())
                    .map(|part| String::from_utf8_lossy(part).into_owned())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_default();
        let owner = LinuxProcessOwner {
            pid,
            name: name.clone(),
            cmdline: if cmdline.is_empty() { name } else { cmdline },
        };
        for inode in inodes {
            map.insert(inode, owner.clone());
        }
    }
    map
}

#[must_use]
pub fn parse_socket_link_inode(target: &str) -> Option<u64> {
    if let Some(rest) = target.strip_prefix("socket:[") {
        let num = rest.strip_suffix(']')?;
        return num.parse::<u64>().ok();
    }
    if let Some(rest) = target.strip_prefix("[0000]:") {
        return rest.parse::<u64>().ok();
    }
    None
}

/// Inspects active Linux sockets across `/proc/net/{tcp,tcp6,udp,udp6}` and attributes them to owning processes.
#[must_use]
pub fn inspect_linux_sockets() -> Vec<LinuxSocketObservation> {
    let inode_map = build_inode_process_map();
    let mut sockets = Vec::new();
    for (path, proto, ipv6) in [
        ("/proc/net/tcp", "TCP", false),
        ("/proc/net/tcp6", "TCP", true),
        ("/proc/net/udp", "UDP", false),
        ("/proc/net/udp6", "UDP", true),
    ] {
        if let Ok(content) = fs::read_to_string(path) {
            sockets.extend(parse_proc_net_table(&content, proto, ipv6, &inode_map));
        }
    }
    sockets
}

/// Parses a `/proc/net/{tcp,tcp6,udp,udp6}` table and enriches each socket with its owning process if known.
#[must_use]
pub fn parse_proc_net_table<S: std::hash::BuildHasher>(
    content: &str,
    protocol: &str,
    is_ipv6: bool,
    inode_map: &HashMap<u64, LinuxProcessOwner, S>,
) -> Vec<LinuxSocketObservation> {
    let mut results = Vec::new();
    for line in content.lines().skip(1) {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 10 {
            continue;
        }
        let Some((local_address, local_port)) = parse_proc_endpoint(cols[1], is_ipv6) else {
            continue;
        };
        let Some((remote_address, remote_port)) = parse_proc_endpoint(cols[2], is_ipv6) else {
            continue;
        };
        let state_code = u8::from_str_radix(cols[3], 16).unwrap_or(0);
        let state = tcp_state_name(state_code).to_owned();
        let (tx_queue, rx_queue) = cols[4].split_once(':').map_or((0, 0), |(tx, rx)| {
            (
                u64::from_str_radix(tx, 16).unwrap_or(0),
                u64::from_str_radix(rx, 16).unwrap_or(0),
            )
        });
        let uid = cols[7].parse::<u32>().unwrap_or(0);
        let inode = cols[9].parse::<u64>().unwrap_or(0);
        let process = inode_map.get(&inode).cloned();
        results.push(LinuxSocketObservation {
            protocol: protocol.to_owned(),
            local_address,
            local_port,
            remote_address,
            remote_port,
            state,
            tx_queue,
            rx_queue,
            uid,
            inode,
            process,
        });
    }
    results
}

fn parse_proc_endpoint(token: &str, is_ipv6: bool) -> Option<(IpAddr, u16)> {
    let (addr_hex, port_hex) = token.split_once(':')?;
    let port = u16::from_str_radix(port_hex, 16).ok()?;
    let addr = if is_ipv6 {
        IpAddr::V6(parse_hex_ipv6(addr_hex)?)
    } else {
        IpAddr::V4(parse_hex_ipv4(addr_hex)?)
    };
    Some((addr, port))
}

/// Parses a little-endian 8-hex-digit IPv4 address from `/proc/net/tcp`.
#[must_use]
pub fn parse_hex_ipv4(hex: &str) -> Option<Ipv4Addr> {
    if hex.len() != 8 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let raw = u32::from_str_radix(hex, 16).ok()?;
    let octets = raw.to_le_bytes();
    Some(Ipv4Addr::new(octets[0], octets[1], octets[2], octets[3]))
}

/// Parses a `/proc/net/tcp6` 32-hex-digit IPv6 address (four little-endian 32-bit words).
#[must_use]
pub fn parse_hex_ipv6(hex: &str) -> Option<Ipv6Addr> {
    if hex.len() != 32 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let mut octets = [0_u8; 16];
    for word_idx in 0..4 {
        let slice = &hex[word_idx * 8..(word_idx + 1) * 8];
        let word = u32::from_str_radix(slice, 16).ok()?;
        octets[word_idx * 4..(word_idx + 1) * 4].copy_from_slice(&word.to_le_bytes());
    }
    Some(Ipv6Addr::from(octets))
}

#[must_use]
pub const fn tcp_state_name(code: u8) -> &'static str {
    match code {
        0x01 => "ESTABLISHED",
        0x02 => "SYN_SENT",
        0x03 => "SYN_RECV",
        0x04 => "FIN_WAIT1",
        0x05 => "FIN_WAIT2",
        0x06 => "TIME_WAIT",
        0x07 => "CLOSE",
        0x08 => "CLOSE_WAIT",
        0x09 => "LAST_ACK",
        0x0A => "LISTEN",
        0x0B => "CLOSING",
        0x0C => "NEW_SYN_RECV",
        _ => "UNKNOWN",
    }
}

/// Records a bounded Linux PCAP using `tcpdump` or `dumpcap` and loads it into [`PcapReplaySource`].
///
/// # Errors
///
/// Returns [`CaptureError`] if the configuration is invalid, the capture binary fails, or the resulting PCAP is unreadable.
pub fn record_linux_pcap(config: &LinuxRecordConfig) -> Result<LinuxRecording, CaptureError> {
    if !cfg!(target_os = "linux") {
        return Err(CaptureError::InvalidMetadata(
            "Linux recording requires Linux",
        ));
    }
    if config.duration.is_zero() || config.duration > Duration::from_mins(5) {
        return Err(CaptureError::InvalidMetadata(
            "capture duration must be within 1..=300 seconds",
        ));
    }
    let tool = find_capture_tool().ok_or_else(|| {
        CaptureError::SourceFailure("neither tcpdump nor dumpcap is installed".to_owned())
    })?;
    if let Some(filter) = &config.bpf_filter {
        validate_bpf_filter(filter)?;
    }
    fs::create_dir_all(&config.output_directory).map_err(|err| {
        CaptureError::SourceFailure(format!("failed to create Linux capture directory: {err}"))
    })?;
    let pcap_path = reserve_capture_path(&config.output_directory)?;

    let secs = config.duration.as_secs().max(1).to_string();
    let snaplen = config.snaplen.get().to_string();

    let status = if tool.ends_with("dumpcap") {
        let mut cmd = Command::new(&tool);
        cmd.args([
            "-i",
            &config.interface,
            "-a",
            &format!("duration:{secs}"),
            "-s",
            &snaplen,
            "-P", // Write classic libpcap format
            "-w",
        ])
        .arg(&pcap_path);
        if let Some(filter) = &config.bpf_filter {
            cmd.args(["-f", filter]);
        }
        cmd.output().map_err(|err| {
            CaptureError::SourceFailure(format!("dumpcap execution failed: {err}"))
        })?
    } else {
        let mut cmd = Command::new("timeout");
        cmd.arg("--kill-after=2s")
            .arg(format!("{secs}s"))
            .arg(&tool)
            .args(["-i", &config.interface, "-nn", "-U", "-s", &snaplen, "-w"])
            .arg(&pcap_path);
        if let Some(filter) = &config.bpf_filter {
            cmd.arg(filter);
        }
        cmd.output().map_err(|err| {
            CaptureError::SourceFailure(format!("tcpdump execution failed: {err}"))
        })?
    };

    // timeout returns 124 when the bounded duration completes normally
    let exit_code = status.status.code().unwrap_or(-1);
    if !status.status.success() && exit_code != 124 {
        let stderr = String::from_utf8_lossy(&status.stderr);
        return Err(CaptureError::SourceFailure(format!(
            "Linux capture command failed (exit {exit_code}): {}",
            stderr.trim()
        )));
    }

    let replay = PcapReplaySource::from_path(&pcap_path, config.limits)?;
    Ok(LinuxRecording { pcap_path, replay })
}

fn reserve_capture_path(directory: &Path) -> Result<PathBuf, CaptureError> {
    let path = directory.join(format!("argus-linux-{}.pcap", uuid::Uuid::new_v4()));
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|error| CaptureError::SourceFailure(error.to_string()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_reserve_distinct_paths_without_overwriting() {
        let directory =
            std::env::temp_dir().join(format!("argus-capture-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&directory).expect("temporary test directory");
        let first = reserve_capture_path(&directory).expect("first reservation");
        fs::write(&first, b"preserve capture").expect("test sentinel");
        let second = reserve_capture_path(&directory).expect("second reservation");
        assert_ne!(first, second);
        assert_eq!(
            fs::read(&first).expect("original capture"),
            b"preserve capture"
        );
        fs::remove_file(first).expect("remove owned test file");
        fs::remove_file(second).expect("remove owned test file");
        fs::remove_dir(directory).expect("remove owned test directory");
    }

    #[test]
    fn parses_proc_net_ipv4_and_ipv6_addresses() {
        // 0100007F -> 127.0.0.1
        assert_eq!(parse_hex_ipv4("0100007F"), Some(Ipv4Addr::LOCALHOST));
        // 6401A8C0 -> 192.168.1.100
        assert_eq!(
            parse_hex_ipv4("6401A8C0"),
            Some(Ipv4Addr::new(192, 168, 1, 100))
        );

        // ::1 in /proc/net/tcp6 is 00000000000000000000000001000000
        assert_eq!(
            parse_hex_ipv6("00000000000000000000000001000000"),
            Some(Ipv6Addr::LOCALHOST)
        );
    }

    #[test]
    fn parses_proc_net_tcp_and_correlates_process_owner() {
        let sample = "\
  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0100007F:0BB8 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 424242 1 0000000000000000 100 0 0 10 0
   1: 6401A8C0:C350 08080808:01BB 01 00000010:00000000 00:00000000 00000000  1000        0 424243 1 0000000000000000 20 4 30 10 -1
";
        let mut map = HashMap::new();
        map.insert(
            424_243,
            LinuxProcessOwner {
                pid: 1337,
                name: "argus-sensor".to_owned(),
                cmdline: "/usr/bin/argus-sensor --live".to_owned(),
            },
        );

        let rows = parse_proc_net_table(sample, "TCP", false, &map);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].local_address, IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_eq!(rows[0].local_port, 3000);
        assert_eq!(rows[0].state, "LISTEN");
        assert!(rows[0].process.is_none());

        assert_eq!(
            rows[1].local_address,
            IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100))
        );
        assert_eq!(rows[1].local_port, 50000);
        assert_eq!(
            rows[1].remote_address,
            IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8))
        );
        assert_eq!(rows[1].remote_port, 443);
        assert_eq!(rows[1].state, "ESTABLISHED");
        assert_eq!(rows[1].tx_queue, 16);
        assert_eq!(rows[1].process.as_ref().map(|p| p.pid), Some(1337));
    }

    #[test]
    fn parses_proc_net_dev_counters() {
        let sample = "\
Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo:   12400     120    0    0    0     0          0         0    12400     120    0    0    0     0       0          0
  eth0: 9876543    8123    1    2    0     0          0         0  5432100    6500    0    1    0     0       0          0
";
        let stats = parse_proc_net_dev(sample);
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].name, "lo");
        assert_eq!(stats[1].name, "eth0");
        assert_eq!(stats[1].rx_bytes, 9_876_543);
        assert_eq!(stats[1].rx_packets, 8123);
        assert_eq!(stats[1].rx_dropped, 2);
        assert_eq!(stats[1].tx_bytes, 5_432_100);
        assert_eq!(stats[1].tx_dropped, 1);
    }

    #[test]
    fn validates_bpf_filters_and_rejects_shell_metacharacters() {
        assert!(validate_bpf_filter("tcp port 443 or udp port 53").is_ok());
        assert!(validate_bpf_filter("host 192.168.1.1 and not port 22").is_ok());
        assert!(validate_bpf_filter("").is_err());
        assert!(validate_bpf_filter("tcp port 80; rm -rf /").is_err());
        assert!(validate_bpf_filter("udp | cat /etc/passwd").is_err());
        assert!(validate_bpf_filter("port 53 `whoami`").is_err());
    }
}

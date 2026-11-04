#![cfg_attr(windows, windows_subsystem = "windows")]
#![forbid(unsafe_code)]

mod query_cache;

use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet, VecDeque},
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    num::NonZeroUsize,
    path::{Path, PathBuf},
    process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use argus_capture::{
    AdapterOutcome, CaptureSource, FlowTopologyRecord, NetworkTopology, PcapLimits,
    PcapReplaySource, PktmonProbe, PktmonReadiness, RawPacket, SimulationScenario, TopologyEdge,
    generate_simulation_pcap, inspect_frame, list_available_scenarios,
};
use argus_core::{FlowInstanceId, TransportProtocol};
use argus_detector::{AnomalyConfig, AnomalyRecord, AnomalySeverity, AnomalyTracker};
use argus_inference::{FeedforwardTensorContract, OnnxFeedforwardForecaster};
use argus_nids::{FlowSummary, Pipeline, PipelineConfig, RunReport};
use query_cache::{DossierStateFingerprint, IpcQueryCache, TopologyStateFingerprint};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{
    Manager,
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
};

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

const TEMPORAL_EXPERT_FILES: [(&str, &str); 4] = [
    ("ahem_fixed", "ahem_fixed_real_pcap_30k.pt"),
    ("cnn_local", "cnn_local_real_pcap_30k.pt"),
    ("lstm", "lstm_real_pcap_30k.pt"),
    ("tcn_long", "tcn_long_real_pcap_30k.pt"),
];
const FORECAST_ROUTER_FILE: &str = "temporal_router_real_live.pt";
const SELECTABLE_MODEL_IDS: [&str; 14] = [
    "ahem_fixed",
    "cnn_local",
    "lstm",
    "tcn_long",
    "ahem_v31",
    "cnn_cse",
    "lstm_cse",
    "gru_cse",
    "tcn_cse",
    "ahem_flash",
    "ahem_pro",
    "ahem_v61",
    "ahem_v63_experimental",
    "ahem_e2e_experimental",
];
const SECURITY_DETECTOR_FILE: &str = "flow_detector_calibrated_cse_cic_ids2018.pt";
const AHEM_V61_FILE: &str = "ahem_v61_forecast_only.pt";
const AHEM_V61_ARTIFACT_SHA256: &str =
    "2457076fd716e4c342c2d359e0046b46ea574aa79ce4a596c2291c2e87943c9e";
const AHEM_V63_EXPERIMENTAL_FILE: &str = "challengers/ahem_v63_experimental_seed_20260914.pt";
const AHEM_V63_EXPERIMENTAL_REPORT: &str =
    "challengers/ahem_v63_experimental_seed_20260914.report.json";
const AHEM_V63_EXPERIMENTAL_SHA256: &str =
    "fef9c75861b33699009ec4fe717d1d83b94a86954a2b59cd5b602fdd63605bf5";
const LIVE_HISTORY_CAPACITY: usize = 5_000;
const LIVE_HISTORY_UI_CAPACITY: usize = 500;
const LIVE_SEGMENT_KEY_CAPACITY: usize = 2_048;
const LIVE_SEGMENT_SAMPLE_CAPACITY: usize = 180;
const LIVE_SEGMENT_SECONDS: u64 = 5;
const SENSOR_TASK_NAME: &str = "ArgusNextPublicSensor";
const SENSOR_INSTALL_HELPER_ARG: &str = "--install-sensor-task";
const TRAY_OPEN_ID: &str = "open-argus";
const TRAY_QUIT_ID: &str = "quit-argus";

fn background_command<S: AsRef<std::ffi::OsStr>>(program: S) -> Command {
    let mut command = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;

        command.creation_flags(CREATE_NO_WINDOW);
    }
    command
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DesktopStatus {
    runtime: &'static str,
    capture_readiness: String,
    model_backend: String,
    native_inference_backend: String,
    moe_engine: String,
    admitted_experts: usize,
    feature_schema: &'static str,
    admission_policy: String,
    security_verdict_enabled: bool,
    forecast_router_enabled: bool,
    models: Vec<ModelStatus>,
    tools: Vec<ExternalToolStatus>,
    accelerator: Option<String>,
    research_evidence: ResearchEvidenceStatus,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ResearchEvidenceStatus {
    model_id: &'static str,
    evidence_state: &'static str,
    runtime_state: &'static str,
    role: &'static str,
    endpoint: &'static str,
    dataset: &'static str,
    test_coordinates: usize,
    flow_clusters: usize,
    seeds: usize,
    candidate_loss: f64,
    comparator_loss: f64,
    relative_reduction_percent: f64,
    interval_low_percent: f64,
    interval_high_percent: f64,
    parameters: usize,
    comparator_parameters: usize,
    artifact_sha256: Option<String>,
    security_detector: bool,
    caveat: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ExternalToolStatus {
    id: &'static str,
    available: bool,
    executable: Option<String>,
    role: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeDiagnostics {
    healthy: bool,
    data_root: String,
    retained_capture_files: usize,
    retained_capture_bytes: u64,
    checks: Vec<DiagnosticCheck>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DiagnosticCheck {
    id: &'static str,
    label: &'static str,
    state: &'static str,
    detail: String,
    recovery: Option<&'static str>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelStatus {
    id: &'static str,
    file_name: &'static str,
    state: &'static str,
    detail: String,
}

#[derive(Debug, Default, Eq, PartialEq)]
struct AdmissionRuntime {
    policy: String,
    security_verdict_enabled: bool,
    forecast_router_enabled: bool,
}

#[derive(Default)]
struct RollingMonitor {
    process_id: Option<u32>,
    executable: Option<PathBuf>,
    stop_file: Option<PathBuf>,
    stop_requested: bool,
    startup_error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimePreferences {
    version: u32,
    keep_monitoring_after_exit: bool,
    #[serde(default)]
    boost_mode: bool,
    #[serde(default)]
    notification_policy: NotificationPolicy,
    #[serde(default = "default_disabled_models")]
    disabled_models: Vec<String>,
    #[serde(default, skip_deserializing, skip_serializing_if = "Vec::is_empty")]
    selectable_models: Vec<SelectableModel>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SelectableModel {
    id: &'static str,
    available: bool,
    detail: &'static str,
}

fn default_disabled_models() -> Vec<String> {
    vec!["ahem_e2e_experimental".to_owned()]
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum NotificationPolicy {
    #[default]
    ErrorsOnly,
    AllActivity,
    Silent,
}

impl Default for RuntimePreferences {
    fn default() -> Self {
        Self {
            version: 1,
            keep_monitoring_after_exit: false,
            boost_mode: false,
            notification_policy: NotificationPolicy::default(),
            disabled_models: default_disabled_models(),
            selectable_models: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct MonitorControl {
    schema: String,
    version: u32,
    process_id: u32,
    executable: PathBuf,
    output_directory: PathBuf,
    stop_file: PathBuf,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RollingStatus {
    running: bool,
    stopping: bool,
    process_id: Option<u32>,
    segment_seconds: u64,
    mode: &'static str,
    sensor: Option<SensorHealth>,
    startup_error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct SensorHealth {
    schema: String,
    version: u32,
    #[serde(default, alias = "process_id")]
    process_id: u32,
    active: bool,
    packets: u64,
    #[serde(alias = "missed_writes")]
    missed_writes: u64,
    #[serde(alias = "missed_reads")]
    missed_reads: u64,
    #[serde(alias = "application_drops")]
    application_drops: u64,
    malformed: u64,
    unsupported: u64,
    #[serde(alias = "duplicate_appearances")]
    duplicate_appearances: u64,
    #[serde(alias = "completed_segments")]
    completed_segments: u64,
    #[serde(alias = "retained_segments")]
    retained_segments: usize,
    #[serde(default, alias = "retention_evictions")]
    retention_evictions: u64,
    #[serde(default, alias = "retention_delete_failures")]
    retention_delete_failures: u64,
    #[serde(default, alias = "retained_backlog")]
    retained_backlog: usize,
    failed: bool,
}

#[derive(Clone, Default, Deserialize, Serialize)]
struct LiveHistory {
    version: u32,
    segment_keys: VecDeque<String>,
    flows: VecDeque<FlowView>,
    #[serde(default)]
    samples: VecDeque<LiveSegmentSample>,
    segments: u64,
    packets: u64,
    bytes: u64,
    evicted_flows: u64,
    #[serde(skip)]
    load_error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct LiveSegmentSample {
    recorded_at_unix_ms: u64,
    packets: u64,
    bytes: u64,
    flows: usize,
    capture_issues: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct LiveHistoryView {
    capacity: usize,
    retained_flows: usize,
    evicted_flows: u64,
    segments: u64,
    packets: u64,
    bytes: u64,
    segment_seconds: u64,
    sample_capacity: usize,
    samples: Vec<LiveSegmentSample>,
    flows: Vec<FlowView>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct AnomalyAlertDto {
    id: String,
    timestamp_ns: u64,
    timestamp_formatted: String,
    flow_endpoint: Option<String>,
    initiator_address: Option<String>,
    flow_generation: Option<u64>,
    category: String,
    severity: String,
    score: f64,
    threshold: f64,
    summary: String,
    detail: String,
    detector: String,
    decision_time: String,
    evidence: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct AlertStatsDto {
    total_alerts: usize,
    critical_count: u64,
    suspicious_count: u64,
    rule_matches: usize,
    diagnostic_events: usize,
    last_alert_time: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct IncidentReportDto {
    report_id: String,
    generated_at: String,
    report_path: String,
    sha256_hash: String,
    total_flows: usize,
    total_alerts: usize,
    critical_alerts: usize,
    suspicious_alerts: usize,
    markdown_content: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct SessionExportDto {
    export_id: String,
    json_path: String,
    csv_path: String,
    json_sha256: String,
    csv_sha256: String,
    flow_count: usize,
    sample_count: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct FlowPayloadDumpDto {
    flow_id: u64,
    packet_count: usize,
    total_payload_bytes: usize,
    mean_entropy: f64,
    packets: Vec<argus_capture::PacketInspection>,
}

struct AlertManager {
    tracker: AnomalyTracker,
    forecast_trackers: HashMap<String, AnomalyTracker>,
    alerts: VecDeque<AnomalyAlertDto>,
    total_critical: u64,
    total_suspicious: u64,
}

impl Default for AlertManager {
    fn default() -> Self {
        Self {
            tracker: AnomalyTracker::new(AnomalyConfig::default()),
            forecast_trackers: HashMap::new(),
            alerts: VecDeque::with_capacity(500),
            total_critical: 0,
            total_suspicious: 0,
        }
    }
}

impl AlertManager {
    fn evaluate_forecast_error(
        &mut self,
        timestamp_ns: u64,
        forecast_error: f64,
        model_name: &str,
    ) -> Option<AnomalyRecord> {
        self.forecast_trackers
            .entry(model_name.to_owned())
            .or_insert_with(|| AnomalyTracker::new(AnomalyConfig::default()))
            .evaluate_forecast_error(timestamp_ns, None, forecast_error, model_name)
    }

    fn record(&mut self, record: AnomalyRecord) {
        match record.severity {
            AnomalySeverity::Critical => {
                self.total_critical = self.total_critical.saturating_add(1);
            }
            AnomalySeverity::Suspicious => {
                self.total_suspicious = self.total_suspicious.saturating_add(1);
            }
            AnomalySeverity::Nominal => {}
        }

        let now_sec = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let hours = (now_sec % 86400) / 3600;
        let minutes = (now_sec % 3600) / 60;
        let secs = now_sec % 60;
        let formatted = format!("{hours:02}:{minutes:02}:{secs:02}");

        let flow_endpoint = record.flow_id.as_ref().map(|f| {
            format!(
                "{}:{} → {}:{}",
                f.key.initiator.address,
                f.key.initiator.port,
                f.key.responder.address,
                f.key.responder.port
            )
        });
        let flow_generation = record.flow_id.as_ref().map(|f| f.generation);
        let initiator_address = record
            .flow_id
            .as_ref()
            .map(|flow| flow.key.initiator.address.to_string());

        let alert_dto = AnomalyAlertDto {
            id: format!("alert-{}-{}", record.timestamp_ns, self.alerts.len()),
            timestamp_ns: record.timestamp_ns,
            timestamp_formatted: formatted,
            flow_endpoint,
            initiator_address,
            flow_generation,
            category: record.category.as_str().to_owned(),
            severity: record.severity.as_str().to_owned(),
            score: record.score,
            threshold: record.threshold,
            summary: record.summary,
            detail: record.detail,
            detector: "argus.anomaly-diagnostics/v1".to_owned(),
            decision_time: "online".to_owned(),
            evidence: Vec::new(),
        };

        if self.alerts.len() >= 500 {
            self.alerts.pop_front();
        }
        self.alerts.push_back(alert_dto);
    }

    fn get_alerts(&self, limit: Option<usize>, severity: Option<&str>) -> Vec<AnomalyAlertDto> {
        let max_items = limit.unwrap_or(50);
        self.alerts
            .iter()
            .rev()
            .filter(|a| severity.is_none_or(|s| a.severity.eq_ignore_ascii_case(s)))
            .take(max_items)
            .cloned()
            .collect()
    }

    fn clear(&mut self) {
        self.alerts.clear();
        self.total_critical = 0;
        self.total_suspicious = 0;
    }

    fn stats(&self) -> AlertStatsDto {
        AlertStatsDto {
            total_alerts: self.alerts.len(),
            critical_count: u64::try_from(
                self.alerts
                    .iter()
                    .filter(|alert| alert.severity.eq_ignore_ascii_case("critical"))
                    .count(),
            )
            .unwrap_or(u64::MAX),
            suspicious_count: u64::try_from(
                self.alerts
                    .iter()
                    .filter(|alert| alert.severity.eq_ignore_ascii_case("suspicious"))
                    .count(),
            )
            .unwrap_or(u64::MAX),
            rule_matches: self
                .alerts
                .iter()
                .filter(|alert| alert.detector == "argus.deterministic-attack-rules/v1")
                .count(),
            diagnostic_events: self
                .alerts
                .iter()
                .filter(|alert| alert.detector != "argus.deterministic-attack-rules/v1")
                .count(),
            last_alert_time: self.alerts.back().map(|a| a.timestamp_formatted.clone()),
        }
    }

    fn record_flow_attack(&mut self, flow: &FlowView) {
        let Some(verdict) = flow.attack_verdict.as_ref() else {
            return;
        };
        let id = format!("attack-{}-{}-{}", flow.session, flow.id, verdict.category);
        if self.alerts.iter().any(|alert| alert.id == id) {
            return;
        }
        let now_sec = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_secs());
        let alert = AnomalyAlertDto {
            id,
            timestamp_ns: flow.last_timestamp_ns,
            timestamp_formatted: format!(
                "{:02}:{:02}:{:02}",
                (now_sec % 86_400) / 3_600,
                (now_sec % 3_600) / 60,
                now_sec % 60
            ),
            flow_endpoint: Some(format!(
                "{}:{} → {}:{}",
                flow.initiator_address,
                flow.initiator_port,
                flow.responder_address,
                flow.responder_port
            )),
            initiator_address: Some(flow.initiator_address.clone()),
            flow_generation: Some(flow.generation),
            category: verdict.category.clone(),
            severity: verdict.severity.clone(),
            score: verdict.score,
            threshold: verdict.threshold,
            summary: verdict.summary.clone(),
            detail: verdict.detail.clone(),
            detector: verdict.detector.clone(),
            decision_time: "online".to_owned(),
            evidence: verdict.evidence.clone(),
        };
        match verdict.severity.as_str() {
            "critical" => self.total_critical = self.total_critical.saturating_add(1),
            "suspicious" => self.total_suspicious = self.total_suspicious.saturating_add(1),
            _ => {}
        }
        if self.alerts.len() >= 500 {
            self.alerts.pop_front();
        }
        self.alerts.push_back(alert);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BlockRuleDto {
    id: String,
    target: String,
    reason: String,
    severity: String,
    created_at_sec: u64,
    created_at_formatted: String,
    dropped_packets: u64,
    dropped_bytes: u64,
    active: bool,
    os_firewall_rule: bool,
    #[serde(default)]
    action: TrafficRuleAction,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TrafficRuleAction {
    Allow,
    #[default]
    Watch,
    Block,
}

impl TrafficRuleAction {
    fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "allow" | "whitelist" => Ok(Self::Allow),
            "watch" | "watchlist" => Ok(Self::Watch),
            "block" | "blacklist" => Ok(Self::Block),
            _ => Err("Rule action must be allow, watch, or block".to_owned()),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ActiveDefenseSettingsDto {
    enabled: bool,
    apply_os_firewall: bool,
}

impl Default for ActiveDefenseSettingsDto {
    fn default() -> Self {
        Self {
            enabled: true,
            apply_os_firewall: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BlacklistStatsDto {
    active_rules_count: usize,
    allow_rules_count: usize,
    watch_rules_count: usize,
    block_rules_count: usize,
    total_blocked_ips: usize,
    total_dropped_packets: u64,
    total_dropped_bytes: u64,
    auto_blocked_count: usize,
    os_rules_active: usize,
}

#[derive(Default)]
struct BlacklistManager {
    settings: ActiveDefenseSettingsDto,
    rules: Vec<BlockRuleDto>,
    blocked_ips: HashSet<String>,
    total_dropped_packets: u64,
    total_dropped_bytes: u64,
}

fn migrate_legacy_rule_action(rule: &mut BlockRuleDto) {
    if matches!(rule.action, TrafficRuleAction::Watch) && rule.os_firewall_rule {
        rule.action = TrafficRuleAction::Block;
    }
}

fn active_defense_store_path() -> Result<PathBuf, String> {
    let reports_dir = reports_directory()?;
    Ok(reports_dir.join("active_defense_rules.json"))
}

impl BlacklistManager {
    fn load() -> Self {
        let Ok(path) = active_defense_store_path() else {
            return Self::default();
        };
        if !path.is_file() {
            return Self::default();
        }
        if let Ok(bytes) = fs::read(&path) {
            #[derive(Deserialize)]
            struct PersistedRules {
                settings: ActiveDefenseSettingsDto,
                rules: Vec<BlockRuleDto>,
            }
            if let Ok(mut persisted) = serde_json::from_slice::<PersistedRules>(&bytes) {
                let mut blocked_ips = HashSet::new();
                let mut total_dropped_pkts = 0;
                let mut total_dropped_b = 0;
                for r in &mut persisted.rules {
                    // Legacy records had no action field. A confirmed firewall rule was a real
                    // block; non-enforced records retain their historical watch-only meaning.
                    migrate_legacy_rule_action(r);
                    if r.active && matches!(r.action, TrafficRuleAction::Block) {
                        blocked_ips.insert(r.target.clone());
                    }
                    total_dropped_pkts += r.dropped_packets;
                    total_dropped_b += r.dropped_bytes;
                }
                return Self {
                    settings: persisted.settings,
                    rules: persisted.rules,
                    blocked_ips,
                    total_dropped_packets: total_dropped_pkts,
                    total_dropped_bytes: total_dropped_b,
                };
            }
        }
        Self::default()
    }

    fn save(&self) {
        if let Ok(path) = active_defense_store_path() {
            #[derive(Serialize)]
            struct PersistedRules<'a> {
                settings: &'a ActiveDefenseSettingsDto,
                rules: &'a [BlockRuleDto],
            }
            let data = PersistedRules {
                settings: &self.settings,
                rules: &self.rules,
            };
            if let Ok(json) = serde_json::to_vec_pretty(&data) {
                let _ = fs::write(path, json);
            }
        }
    }

    fn matching_rule(&self, ip_or_endpoint: &str) -> Option<&BlockRuleDto> {
        let clean_ip = parse_endpoint_ip(ip_or_endpoint)?;
        self.rules
            .iter()
            .filter(|rule| rule.active && rule_target_contains(&rule.target, clean_ip))
            .max_by_key(|rule| {
                (
                    rule_target_specificity(&rule.target).unwrap_or(0),
                    match rule.action {
                        TrafficRuleAction::Watch => 0_u8,
                        TrafficRuleAction::Allow => 1,
                        TrafficRuleAction::Block => 2,
                    },
                    rule.created_at_sec,
                )
            })
    }

    fn active_rule(&self, ip_or_endpoint: &str) -> Option<&BlockRuleDto> {
        self.settings
            .enabled
            .then(|| self.matching_rule(ip_or_endpoint))
            .flatten()
    }

    fn is_watchlisted(&self, ip_or_endpoint: &str) -> bool {
        self.active_rule(ip_or_endpoint)
            .is_some_and(|rule| matches!(rule.action, TrafficRuleAction::Watch))
    }

    fn is_allowlisted(&self, ip_or_endpoint: &str) -> bool {
        self.active_rule(ip_or_endpoint)
            .is_some_and(|rule| matches!(rule.action, TrafficRuleAction::Allow))
    }

    fn is_blocklisted(&self, ip_or_endpoint: &str) -> bool {
        self.active_rule(ip_or_endpoint)
            .is_some_and(|rule| matches!(rule.action, TrafficRuleAction::Block))
    }

    fn is_blocked(&self, ip_or_endpoint: &str) -> bool {
        // Disabling Argus rule matching does not remove existing Windows
        // Firewall rules. Keep reporting enforcement so the UI never claims
        // traffic is permitted while the host firewall still blocks it.
        self.enforced_rule(ip_or_endpoint).is_some()
    }

    fn enforced_rule(&self, ip_or_endpoint: &str) -> Option<&BlockRuleDto> {
        let clean_ip = parse_endpoint_ip(ip_or_endpoint)?;
        self.rules
            .iter()
            .filter(|rule| {
                rule.active
                    && rule.os_firewall_rule
                    && matches!(rule.action, TrafficRuleAction::Block)
                    && rule_target_contains(&rule.target, clean_ip)
            })
            .max_by_key(|rule| rule_target_specificity(&rule.target).unwrap_or(0))
    }

    fn rule_action(&self, ip_or_endpoint: &str) -> Option<TrafficRuleAction> {
        self.active_rule(ip_or_endpoint).map(|rule| rule.action)
    }

    fn upsert_rule(
        &mut self,
        target: &str,
        reason: String,
        severity: Option<String>,
        action: TrafficRuleAction,
    ) -> Result<BlockRuleDto, String> {
        let clean_target = normalized_rule_target(target)
            .ok_or_else(|| "Target must be a valid IPv4/IPv6 address or CIDR network".to_owned())?;

        if let Some(pos) = self.rules.iter().position(|r| r.target == clean_target) {
            let had_firewall = self.rules[pos].os_firewall_rule;
            let wants_firewall =
                matches!(action, TrafficRuleAction::Block) && self.settings.apply_os_firewall;
            if had_firewall && !wants_firewall {
                delete_os_firewall_rule(&clean_target)?;
            } else if !had_firewall && wants_firewall {
                add_os_firewall_rule(&clean_target)?;
            }
            self.rules[pos].os_firewall_rule = wants_firewall;
            self.rules[pos].active = true;
            self.rules[pos].reason = reason;
            self.rules[pos].severity = severity.unwrap_or_else(|| "Manual".to_owned());
            self.rules[pos].action = action;
            if matches!(action, TrafficRuleAction::Block) {
                self.blocked_ips.insert(clean_target.clone());
            } else {
                self.blocked_ips.remove(&clean_target);
            }
            let updated = self.rules[pos].clone();
            self.save();
            return Ok(updated);
        }

        let now_sec = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let hours = (now_sec % 86400) / 3600;
        let minutes = (now_sec % 3600) / 60;
        let secs = now_sec % 60;
        let formatted = format!("{hours:02}:{minutes:02}:{secs:02}");
        let os_applied =
            matches!(action, TrafficRuleAction::Block) && self.settings.apply_os_firewall;
        if os_applied {
            add_os_firewall_rule(&clean_target)?;
        }
        let rule = BlockRuleDto {
            id: format!("rule-{now_sec}-{}", self.rules.len() + 1),
            target: clean_target.clone(),
            reason,
            severity: severity.unwrap_or_else(|| "Manual".to_owned()),
            created_at_sec: now_sec,
            created_at_formatted: formatted,
            dropped_packets: 0,
            dropped_bytes: 0,
            active: true,
            os_firewall_rule: os_applied,
            action,
        };
        if matches!(action, TrafficRuleAction::Block) {
            self.blocked_ips.insert(clean_target);
        }
        self.rules.push(rule.clone());
        self.save();
        Ok(rule)
    }

    fn block_ip(
        &mut self,
        target: &str,
        reason: String,
        severity: Option<String>,
    ) -> Result<BlockRuleDto, String> {
        self.upsert_rule(target, reason, severity, TrafficRuleAction::Block)
    }

    fn unblock_ip(&mut self, rule_id_or_target: &str) -> Result<bool, String> {
        let position = self
            .rules
            .iter()
            .position(|rule| rule.id == rule_id_or_target || rule.target == rule_id_or_target)
            .ok_or_else(|| "No matching block rule found".to_owned())?;
        if self.rules[position].os_firewall_rule {
            delete_os_firewall_rule(&self.rules[position].target)?;
        }
        let target = self.rules[position].target.clone();
        self.rules[position].active = false;
        self.rules[position].os_firewall_rule = false;
        self.blocked_ips.remove(&target);
        self.save();
        Ok(true)
    }

    fn stats(&self) -> BlacklistStatsDto {
        let active_count = self.rules.iter().filter(|r| r.active).count();
        let allow_count = self
            .rules
            .iter()
            .filter(|r| r.active && matches!(r.action, TrafficRuleAction::Allow))
            .count();
        let watch_count = self
            .rules
            .iter()
            .filter(|r| r.active && matches!(r.action, TrafficRuleAction::Watch))
            .count();
        let block_count = self
            .rules
            .iter()
            .filter(|r| r.active && matches!(r.action, TrafficRuleAction::Block))
            .count();
        let auto_count = self
            .rules
            .iter()
            .filter(|r| r.active && r.severity != "Manual")
            .count();
        let os_count = self
            .rules
            .iter()
            .filter(|r| r.active && r.os_firewall_rule)
            .count();
        BlacklistStatsDto {
            active_rules_count: active_count,
            allow_rules_count: allow_count,
            watch_rules_count: watch_count,
            block_rules_count: block_count,
            total_blocked_ips: self.blocked_ips.len(),
            total_dropped_packets: self.total_dropped_packets,
            total_dropped_bytes: self.total_dropped_bytes,
            auto_blocked_count: auto_count,
            os_rules_active: os_count,
        }
    }
}

fn normalized_ip(value: &str) -> Option<String> {
    parse_endpoint_ip(value).map(|ip| ip.to_string())
}

fn parse_endpoint_ip(value: &str) -> Option<IpAddr> {
    let trimmed = value.trim();
    trimmed
        .parse::<IpAddr>()
        .or_else(|_| trimmed.parse::<SocketAddr>().map(|endpoint| endpoint.ip()))
        .ok()
}

fn normalized_rule_target(value: &str) -> Option<String> {
    let trimmed = value.trim();
    let Some((address, prefix)) = trimmed.split_once('/') else {
        return normalized_ip(trimmed);
    };
    let ip = address.parse::<IpAddr>().ok()?;
    let prefix = prefix.parse::<u8>().ok()?;
    match ip {
        IpAddr::V4(ip) if prefix <= 32 => {
            let mask = if prefix == 0 {
                0
            } else {
                u32::MAX << (32 - prefix)
            };
            let network = Ipv4Addr::from(u32::from(ip) & mask);
            Some(format!("{network}/{prefix}"))
        }
        IpAddr::V6(ip) if prefix <= 128 => {
            let mask = if prefix == 0 {
                0
            } else {
                u128::MAX << (128 - prefix)
            };
            let network = Ipv6Addr::from(u128::from(ip) & mask);
            Some(format!("{network}/{prefix}"))
        }
        _ => None,
    }
}

fn rule_target_specificity(target: &str) -> Option<u8> {
    if let Some((address, prefix)) = target.split_once('/') {
        let ip = address.parse::<IpAddr>().ok()?;
        let prefix = prefix.parse::<u8>().ok()?;
        return match ip {
            IpAddr::V4(_) if prefix <= 32 => Some(prefix),
            IpAddr::V6(_) if prefix <= 128 => Some(prefix),
            _ => None,
        };
    }
    match target.parse::<IpAddr>().ok()? {
        IpAddr::V4(_) => Some(32),
        IpAddr::V6(_) => Some(128),
    }
}

fn rule_target_contains(target: &str, candidate: IpAddr) -> bool {
    let Some(canonical_target) = normalized_rule_target(target) else {
        return false;
    };
    let Some((network, prefix)) = canonical_target.split_once('/') else {
        return canonical_target == candidate.to_string();
    };
    let Ok(prefix) = prefix.parse::<u8>() else {
        return false;
    };
    match (network.parse::<IpAddr>(), candidate) {
        (Ok(IpAddr::V4(network)), IpAddr::V4(candidate)) if prefix <= 32 => {
            let mask = if prefix == 0 {
                0
            } else {
                u32::MAX << (32 - prefix)
            };
            u32::from(network) & mask == u32::from(candidate) & mask
        }
        (Ok(IpAddr::V6(network)), IpAddr::V6(candidate)) if prefix <= 128 => {
            let mask = if prefix == 0 {
                0
            } else {
                u128::MAX << (128 - prefix)
            };
            u128::from(network) & mask == u128::from(candidate) & mask
        }
        _ => false,
    }
}

fn add_os_firewall_rule(ip: &str) -> Result<(), String> {
    #[cfg(windows)]
    {
        let rule_name = format!("Argus-ActiveDefense-Block-{ip}");
        let mut cmd = background_command("netsh");
        cmd.args(["advfirewall", "firewall", "add", "rule"])
            .arg(format!("name={rule_name}"))
            .arg("dir=in")
            .arg("action=block")
            .arg(format!("remoteip={ip}"));
        let status = cmd.status().map_err(|e| e.to_string())?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("netsh firewall rule exited with {status}"))
        }
    }
    #[cfg(not(windows))]
    {
        let _ = ip;
        Ok(())
    }
}

fn delete_os_firewall_rule(ip: &str) -> Result<(), String> {
    #[cfg(windows)]
    {
        let rule_name = format!("Argus-ActiveDefense-Block-{ip}");
        let mut cmd = background_command("netsh");
        cmd.args(["advfirewall", "firewall", "delete", "rule"])
            .arg(format!("name={rule_name}"));
        let status = cmd.status().map_err(|error| error.to_string())?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("netsh firewall rule deletion exited with {status}"))
        }
    }
    #[cfg(not(windows))]
    {
        let _ = ip;
        Err("OS firewall integration is unavailable on this platform".to_owned())
    }
}

impl LiveHistory {
    fn fingerprint(&self) -> TopologyStateFingerprint {
        TopologyStateFingerprint {
            segments: self.segments,
            packets: self.packets,
            bytes: self.bytes,
            flows_len: self.flows.len(),
            evicted_flows: self.evicted_flows,
        }
    }

    fn load() -> Self {
        let path = match live_history_path() {
            Ok(path) => path,
            Err(error) => {
                return Self {
                    version: 1,
                    load_error: Some(error),
                    ..Self::default()
                };
            }
        };
        if !path.is_file() {
            return Self {
                version: 1,
                ..Self::default()
            };
        }
        match fs::read_to_string(&path)
            .map_err(|error| error.to_string())
            .and_then(|contents| {
                serde_json::from_str::<Self>(&contents).map_err(|error| error.to_string())
            }) {
            Ok(mut history) if history.version == 1 => {
                let mut unique = HashSet::new();
                history
                    .segment_keys
                    .retain(|key| unique.insert(key.clone()));
                while history.segment_keys.len() > LIVE_SEGMENT_KEY_CAPACITY {
                    history.segment_keys.pop_front();
                }
                while history.flows.len() > LIVE_HISTORY_CAPACITY {
                    history.flows.pop_front();
                    history.evicted_flows = history.evicted_flows.saturating_add(1);
                }
                while history.samples.len() > LIVE_SEGMENT_SAMPLE_CAPACITY {
                    history.samples.pop_front();
                }
                history.sanitize_loaded_flows(&BlacklistManager::load());
                history
            }
            Ok(_) => Self {
                version: 1,
                load_error: Some("live history has an unsupported version".to_owned()),
                ..Self::default()
            },
            Err(error) => Self {
                version: 1,
                load_error: Some(format!("live history could not be loaded: {error}")),
                ..Self::default()
            },
        }
    }

    fn record(&mut self, segment_key: &str, run: &RunView) -> Result<(), String> {
        if let Some(error) = &self.load_error {
            return Err(error.clone());
        }
        let mut next = self.clone();
        if !next.append_run(segment_key, run)? {
            return Ok(());
        }
        next.persist()?;
        *self = next;
        Ok(())
    }

    fn sanitize_loaded_flows(&mut self, blacklist: &BlacklistManager) {
        for flow in &mut self.flows {
            flow.threat_status = capture_quality_status(flow.lost_packets, flow.truncated_packets);
            flow.is_watchlisted = blacklist.is_watchlisted(&flow.initiator_address)
                || blacklist.is_watchlisted(&flow.responder_address);
            flow.is_blocked = blacklist.is_blocked(&flow.initiator_address)
                || blacklist.is_blocked(&flow.responder_address);
            flow.is_allowlisted = blacklist.is_allowlisted(&flow.initiator_address)
                || blacklist.is_allowlisted(&flow.responder_address);
            flow.is_blocklisted = blacklist.is_blocklisted(&flow.initiator_address)
                || blacklist.is_blocklisted(&flow.responder_address);
        }
        apply_attack_detection(self.flows.make_contiguous(), blacklist);
    }

    fn append_run(&mut self, segment_key: &str, run: &RunView) -> Result<bool, String> {
        if self.segment_keys.iter().any(|key| key == segment_key) {
            return Ok(false);
        }
        self.segment_keys.push_back(segment_key.to_owned());
        while self.segment_keys.len() > LIVE_SEGMENT_KEY_CAPACITY {
            self.segment_keys.pop_front();
        }
        self.segments = checked_history_add(self.segments, 1, "history segment count")?;
        self.packets = checked_history_add(
            self.packets,
            run.health.processed_packets,
            "history packet count",
        )?;
        let segment_bytes = run.flows.iter().try_fold(0_u64, |total, flow| {
            checked_history_add(total, flow.original_bytes, "segment byte count")
        })?;
        self.bytes = checked_history_add(self.bytes, segment_bytes, "history byte count")?;
        let capture_issues = [
            run.health.reported_lost_packets,
            run.health.truncated_packets,
            run.health.reordered_packets,
            run.health.queue_overflows,
            run.health.parse_errors,
            run.health.source_errors,
        ]
        .into_iter()
        .fold(0_u64, u64::saturating_add);
        let recorded_at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| {
                u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
            });
        self.samples.push_back(LiveSegmentSample {
            recorded_at_unix_ms,
            packets: run.health.processed_packets,
            bytes: segment_bytes,
            flows: run.health.flows,
            capture_issues,
        });
        while self.samples.len() > LIVE_SEGMENT_SAMPLE_CAPACITY {
            self.samples.pop_front();
        }
        self.flows.extend(run.flows.iter().cloned());
        while self.flows.len() > LIVE_HISTORY_CAPACITY {
            self.flows.pop_front();
            self.evicted_flows =
                checked_history_add(self.evicted_flows, 1, "history eviction count")?;
        }
        Ok(true)
    }

    fn view(&self) -> Result<LiveHistoryView, String> {
        if let Some(error) = &self.load_error {
            return Err(error.clone());
        }
        Ok(LiveHistoryView {
            capacity: LIVE_HISTORY_CAPACITY,
            retained_flows: self.flows.len(),
            evicted_flows: self.evicted_flows,
            segments: self.segments,
            packets: self.packets,
            bytes: self.bytes,
            segment_seconds: LIVE_SEGMENT_SECONDS,
            sample_capacity: LIVE_SEGMENT_SAMPLE_CAPACITY,
            samples: self.samples.iter().cloned().collect(),
            flows: self
                .flows
                .iter()
                .rev()
                .take(LIVE_HISTORY_UI_CAPACITY)
                .cloned()
                .collect(),
        })
    }

    fn persist(&self) -> Result<(), String> {
        let path = live_history_path()?;
        let parent = path
            .parent()
            .ok_or("live history has no parent directory")?;
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        let encoded = serde_json::to_vec(self).map_err(|error| error.to_string())?;
        replace_file(&path, &encoded)
    }
}

fn checked_history_add(left: u64, right: u64, name: &str) -> Result<u64, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("{name} overflow"))
}

struct CnnWorker {
    child: Option<Child>,
    input: Option<ChildStdin>,
    output: Option<BufReader<ChildStdout>>,
    error: Option<BufReader<ChildStderr>>,
    native: Option<NativeOnnxWorker>,
    native_start_error: Option<String>,
    latency_samples_ms: VecDeque<f64>,
    boost_mode: Option<bool>,
    disabled_models: Vec<String>,
}

impl Default for CnnWorker {
    fn default() -> Self {
        Self {
            child: None,
            input: None,
            output: None,
            error: None,
            native: None,
            native_start_error: None,
            latency_samples_ms: VecDeque::new(),
            boost_mode: None,
            disabled_models: default_disabled_models(),
        }
    }
}

#[derive(Default)]
struct SimulationWorker(Arc<Mutex<CnnWorker>>);

struct NativeModel {
    name: &'static str,
    context_packets: usize,
    forecaster: OnnxFeedforwardForecaster,
}

struct NativeOnnxWorker {
    models: Vec<NativeModel>,
    rejected: HashMap<&'static str, String>,
    history: VecDeque<[f32; 12]>,
    history_scope: Option<String>,
    last_timestamp_ns: Option<u64>,
}

#[derive(Deserialize)]
struct RuntimeFeatureRow {
    schema_id: String,
    schema_version: u32,
    source_capture_ordinal: u64,
    monotonic_timestamp_ns: u64,
    opaque_flow_id: u64,
    ip_version: u8,
    transport_protocol: String,
    tcp_syn: u8,
    tcp_ack: u8,
    tcp_fin: u8,
    tcp_rst: u8,
    tcp_push: u8,
    tcp_urg: u8,
    packet_length_bytes: u64,
}

impl NativeOnnxWorker {
    #[cfg(test)]
    fn load(model_dir: &Path) -> Result<Self, String> {
        Self::load_selected(model_dir, &[])
    }

    fn load_selected(model_dir: &Path, disabled_models: &[String]) -> Result<Self, String> {
        let runtime = discover_native_runtime_library()?;
        let mut models = Vec::new();
        let mut rejected = HashMap::new();
        for (name, stem, context_packets) in [
            ("cnn_local", "cnn_local", 31),
            ("ahem_fixed", "ahem_fixed", 128),
            ("tcn_long", "tcn_long", 1_024),
        ] {
            if disabled_models.iter().any(|disabled| disabled == name) {
                continue;
            }
            let model_path = model_dir.join(format!("{stem}.onnx"));
            let manifest_path = model_dir.join(format!("{stem}.onnx.manifest.json"));
            let loaded = argus_inference::read_manifest(&manifest_path).and_then(|manifest| {
                OnnxFeedforwardForecaster::load(
                    &runtime,
                    &model_path,
                    &manifest,
                    FeedforwardTensorContract {
                        input_name: "context".to_owned(),
                        context_packets,
                        feature_dim: 12,
                        output_name: "prediction".to_owned(),
                    },
                )
            });
            match loaded {
                Ok(forecaster) => models.push(NativeModel {
                    name,
                    context_packets,
                    forecaster,
                }),
                Err(error) => {
                    rejected.insert(name, error.to_string());
                }
            }
        }
        if models.is_empty() {
            return Err(format!(
                "no native ONNX model was admitted: {}",
                rejected
                    .iter()
                    .map(|(name, error)| format!("{name}: {error}"))
                    .collect::<Vec<_>>()
                    .join("; ")
            ));
        }
        Ok(Self {
            models,
            rejected,
            history: VecDeque::with_capacity(30_000),
            history_scope: None,
            last_timestamp_ns: None,
        })
    }

    fn model_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.models.iter().map(|model| model.name)
    }

    #[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
    fn infer(
        &mut self,
        features_path: &Path,
        history_scope: &str,
        predict: bool,
        routed_experts: Option<&HashSet<String>>,
    ) -> Result<serde_json::Value, String> {
        let started = Instant::now();
        if self.history_scope.as_deref() != Some(history_scope) {
            self.history.clear();
            self.last_timestamp_ns = None;
            self.history_scope = Some(history_scope.to_owned());
        }
        let mut reader =
            csv::Reader::from_path(features_path).map_err(|error| error.to_string())?;
        let rows = reader
            .deserialize::<RuntimeFeatureRow>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        if rows.is_empty() {
            return Err("native ONNX input contains no feature rows".to_owned());
        }
        if rows.iter().any(|row| {
            row.schema_id != "argus.causal-flow"
                || row.schema_version != 1
                || !matches!(row.ip_version, 4 | 6)
        }) {
            return Err("native ONNX input violates argus.causal-flow/v1".to_owned());
        }
        if self
            .last_timestamp_ns
            .is_some_and(|previous| rows[0].monotonic_timestamp_ns < previous)
        {
            self.history.clear();
            self.last_timestamp_ns = None;
        }
        for row in &rows {
            let delta_seconds = self.last_timestamp_ns.map_or(0.0, |previous| {
                std::time::Duration::from_nanos(
                    row.monotonic_timestamp_ns
                        .saturating_sub(previous)
                        .min(60_000_000_000),
                )
                .as_secs_f32()
            });
            self.last_timestamp_ns = Some(row.monotonic_timestamp_ns);
            let protocol = row.transport_protocol.as_str();
            self.history.push_back([
                (row.packet_length_bytes as f32).ln_1p(),
                delta_seconds.ln_1p(),
                f32::from(row.ip_version == 4),
                f32::from(row.ip_version == 6),
                f32::from(protocol == "tcp"),
                f32::from(protocol == "udp"),
                f32::from(row.tcp_syn != 0),
                f32::from(row.tcp_ack != 0),
                f32::from(row.tcp_fin != 0),
                f32::from(row.tcp_rst != 0),
                f32::from(row.tcp_push != 0),
                f32::from(row.tcp_urg != 0),
            ]);
            if self.history.len() > 30_000 {
                self.history.pop_front();
            }
        }
        let mut outputs = serde_json::Map::new();
        if predict {
            for model in &self.models {
                if routed_experts.is_some_and(|selected| !selected.contains(model.name)) {
                    outputs.insert(
                        model.name.to_owned(),
                        serde_json::json!({
                            "state": "skipped_by_router",
                            "runtime_backend": "onnxruntime",
                            "execution_provider": model.forecaster.execution_provider(),
                        }),
                    );
                    continue;
                }
                if self.history.len() <= model.context_packets {
                    outputs.insert(
                        model.name.to_owned(),
                        serde_json::json!({
                            "state": "warming_up",
                            "context_packets": self.history.len().saturating_sub(1),
                            "runtime_backend": "onnxruntime",
                            "execution_provider": model.forecaster.execution_provider(),
                        }),
                    );
                    continue;
                }
                let model_started = Instant::now();
                let start = self.history.len() - model.context_packets - 1;
                let context = self
                    .history
                    .iter()
                    .skip(start)
                    .take(model.context_packets)
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>();
                let target = self
                    .history
                    .back()
                    .ok_or("native history target is absent")?;
                let forecast = model
                    .forecaster
                    .forecast(&context)
                    .map_err(|error| format!("{}: {error}", model.name))?;
                let length_error = smooth_l1(forecast.predicted_log_packet_length, target[0]);
                let time_error = smooth_l1(forecast.predicted_log_interarrival, target[1]);
                outputs.insert(
                    model.name.to_owned(),
                    serde_json::json!({
                        "state": "forecast_only",
                        "context_packets": model.context_packets,
                        "forecast_error": length_error.midpoint(time_error),
                        "loss_contract": "smooth_l1_first_two_features",
                        "predicted_log_packet_length": forecast.predicted_log_packet_length,
                        "observed_log_packet_length": target[0],
                        "predicted_log_interarrival": forecast.predicted_log_interarrival,
                        "observed_log_interarrival": target[1],
                        "execution_ms": model_started.elapsed().as_secs_f64() * 1_000.0,
                        "runtime_backend": "onnxruntime",
                        "execution_provider": model.forecaster.execution_provider(),
                        "model_path": model.forecaster.model_path().file_name().and_then(|name| name.to_str()),
                    }),
                );
            }
        }
        Ok(serde_json::json!({
            "state": "active",
            "runtime_backend": "onnxruntime",
            "execution_provider": "CPUExecutionProvider",
            "models": outputs,
            "rejected_models": self.rejected,
            "history_packets": self.history.len(),
            "source_rows": rows.len(),
            "latest_source_capture_ordinal": rows.last().map(|row| row.source_capture_ordinal),
            "latest_opaque_flow_id": rows.last().map(|row| row.opaque_flow_id),
            "total_ms": started.elapsed().as_secs_f64() * 1_000.0,
        }))
    }
}

fn smooth_l1(prediction: f32, target: f32) -> f32 {
    let difference = (prediction - target).abs();
    if difference < 1.0 {
        0.5 * difference * difference
    } else {
        difference - 0.5
    }
}

impl CnnWorker {
    fn select_models(&mut self, disabled_models: &[String]) {
        if self.disabled_models != disabled_models {
            // Called only while holding the worker lock, never during an in-flight request.
            self.stop_worker();
            self.native = None;
            self.native_start_error = None;
            self.disabled_models = disabled_models.to_vec();
            self.latency_samples_ms.clear();
        }
    }

    fn stop_worker(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.child = None;
        self.input = None;
        self.output = None;
        self.error = None;
    }

    fn ensure_started(&mut self, model_dir: &Path, boost_mode: bool) -> Result<(), String> {
        if self.boost_mode.is_some_and(|active| active != boost_mode) {
            self.stop_worker();
            self.native = None;
            self.native_start_error = None;
        }
        self.boost_mode = Some(boost_mode);
        if !boost_mode && self.native.is_none() && self.native_start_error.is_none() {
            match NativeOnnxWorker::load_selected(model_dir, &self.disabled_models) {
                Ok(runtime) => self.native = Some(runtime),
                Err(error) => self.native_start_error = Some(error),
            }
        }
        if self
            .child
            .as_mut()
            .is_some_and(|child| child.try_wait().ok().flatten().is_none())
        {
            return Ok(());
        }
        self.stop_worker();
        let runtime = model_runtime()?;
        let mut command = background_command(&runtime.python);
        command
            .args([
                "-u",
                "-m",
                "argus_research.experimental_cnn",
                "--ensemble-server",
            ])
            .arg(model_dir)
            .arg("--learning-directory")
            .arg(daily_learning_directory()?)
            .args(["--device", "auto"])
            .args(["--micro-batch-size", if boost_mode { "256" } else { "32" }]);
        for name in &self.disabled_models {
            command.args(["--disable-model", name]);
        }
        if !self
            .disabled_models
            .iter()
            .any(|name| name == "ahem_e2e_experimental")
        {
            command.arg("--enable-ahem-e2e-shadow");
        }
        if !boost_mode && let Some(native) = &self.native {
            for name in native.model_names() {
                command.args(["--skip-model", name]);
            }
        }
        let worker_threads = if boost_mode {
            std::thread::available_parallelism()
                .map_or(2, NonZeroUsize::get)
                .clamp(2, 8)
                .to_string()
        } else {
            "1".to_owned()
        };
        let mut child = command
            .env("PYTHONPATH", &runtime.source)
            .env("OMP_NUM_THREADS", &worker_threads)
            .env("MKL_NUM_THREADS", &worker_threads)
            .env("OPENBLAS_NUM_THREADS", &worker_threads)
            .env("ARGUS_TORCH_CPU_THREADS", &worker_threads)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| {
                format!(
                    "model worker could not start {}: {error}",
                    runtime.python.display()
                )
            })?;
        self.input = child.stdin.take();
        self.output = child.stdout.take().map(BufReader::new);
        self.error = child.stderr.take().map(BufReader::new);
        self.child = Some(child);
        Ok(())
    }

    fn infer(
        &mut self,
        model_dir: &Path,
        features: &Path,
        history_scope: &str,
        predict: bool,
        learning_segment_key: Option<&str>,
        boost_mode: bool,
    ) -> Result<serde_json::Value, String> {
        let request_started = Instant::now();
        if boost_mode {
            // The Python CUDA path owns all compatible experts in boost mode;
            // the CPU-only native path would otherwise duplicate their work.
            self.native = None;
        }
        let worker_start = Instant::now();
        self.ensure_started(model_dir, boost_mode)?;
        let worker_start_ms = worker_start.elapsed().as_secs_f64() * 1_000.0;
        let ipc_started = Instant::now();
        let request = serde_json::json!({
            "features": features,
            "history_scope": history_scope,
            "predict": predict,
            "learning_segment_key": learning_segment_key,
            "boost_mode": boost_mode,
            "micro_batch_size": if boost_mode { 256 } else { 32 },
        });
        let input = self
            .input
            .as_mut()
            .ok_or("CNN worker stdin is unavailable")?;
        writeln!(input, "{request}").map_err(|error| error.to_string())?;
        input.flush().map_err(|error| error.to_string())?;
        let mut response = String::new();
        self.output
            .as_mut()
            .ok_or("CNN worker stdout is unavailable")?
            .read_line(&mut response)
            .map_err(|error| error.to_string())?;
        if response.is_empty() {
            let status = self
                .child
                .as_mut()
                .and_then(|child| child.try_wait().ok().flatten());
            let mut detail = String::new();
            if status.is_some()
                && let Some(error) = self.error.as_mut()
            {
                let _ = error.read_to_string(&mut detail);
            }
            let detail = detail.trim();
            return Err(if detail.is_empty() {
                format!("model worker ended without a response (status {status:?})")
            } else {
                format!("model worker failed (status {status:?}): {detail}")
            });
        }
        let worker_ipc_ms = ipc_started.elapsed().as_secs_f64() * 1_000.0;
        let mut value: serde_json::Value =
            serde_json::from_str(&response).map_err(|error| error.to_string())?;
        if let Some(error) = value.get("error").and_then(serde_json::Value::as_str) {
            return Err(format!("experimental CNN failed: {error}"));
        }
        if !boost_mode {
            self.merge_native(&mut value, features, history_scope, predict);
        }
        let total_ms = request_started.elapsed().as_secs_f64() * 1_000.0;
        if let Some(latency) = value
            .get_mut("latency_breakdown")
            .and_then(serde_json::Value::as_object_mut)
        {
            latency.insert("total_ms".to_owned(), serde_json::Value::from(total_ms));
            latency.insert(
                "worker_start_ms".to_owned(),
                serde_json::Value::from(worker_start_ms),
            );
            latency.insert(
                "worker_ipc_ms".to_owned(),
                serde_json::Value::from(worker_ipc_ms),
            );
        }
        if let Some(telemetry) = value
            .get_mut("telemetry")
            .and_then(serde_json::Value::as_object_mut)
        {
            telemetry.insert(
                "total_inference_ms".to_owned(),
                serde_json::Value::from(total_ms),
            );
        }
        self.latency_samples_ms.push_back(total_ms);
        while self.latency_samples_ms.len() > 256 {
            self.latency_samples_ms.pop_front();
        }
        if let Some(object) = value.as_object_mut() {
            object.insert(
                "runtime_latency".to_owned(),
                latency_summary(&self.latency_samples_ms),
            );
        }
        Ok(value)
    }

    fn merge_native(
        &mut self,
        value: &mut serde_json::Value,
        features: &Path,
        history_scope: &str,
        predict: bool,
    ) {
        // An active route to a Python-only expert must skip native forecasts.
        // None means the router was inactive and direct execution is allowed.
        let routed_experts = selected_forecast_experts(value);
        if let Some(native) = self.native.as_mut() {
            match native.infer(features, history_scope, predict, routed_experts.as_ref()) {
                Ok(native_result) => merge_native_success(value, native_result),
                Err(error) => insert_native_status(
                    value,
                    serde_json::json!({"state": "request_failed", "error": error}),
                ),
            }
        } else if let Some(error) = &self.native_start_error {
            insert_native_status(
                value,
                serde_json::json!({"state": "unavailable", "error": error}),
            );
        }
    }
}

fn selected_forecast_experts(value: &serde_json::Value) -> Option<HashSet<String>> {
    value
        .get("router")
        .filter(|router| router.get("state").and_then(serde_json::Value::as_str) == Some("active"))
        .map(|router| {
            router
                .get("selected_experts")
                .and_then(serde_json::Value::as_array)
                .map(|experts| {
                    experts
                        .iter()
                        .filter_map(serde_json::Value::as_str)
                        .map(str::to_owned)
                        .collect::<HashSet<_>>()
                })
                .filter(|experts| !experts.is_empty())
                .or_else(|| {
                    router
                        .get("selected_expert")
                        .and_then(serde_json::Value::as_str)
                        .map(|expert| HashSet::from([expert.to_owned()]))
                })
                .unwrap_or_default()
        })
}

fn insert_native_status(value: &mut serde_json::Value, status: serde_json::Value) {
    if let Some(object) = value.as_object_mut() {
        object.insert("native_onnx".to_owned(), status);
    }
}

fn merge_native_success(value: &mut serde_json::Value, mut native_result: serde_json::Value) {
    let native_ms = native_result
        .get("total_ms")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    let native_executed = native_result
        .get("models")
        .and_then(serde_json::Value::as_object)
        .map_or(0_u64, |models| {
            u64::try_from(
                models
                    .values()
                    .filter(|forecast| {
                        forecast.get("state").and_then(serde_json::Value::as_str)
                            == Some("forecast_only")
                    })
                    .count(),
            )
            .unwrap_or(u64::MAX)
        });
    if let Some(native_models) = native_result
        .get_mut("models")
        .and_then(serde_json::Value::as_object_mut)
        && let Some(models) = value
            .get_mut("models")
            .and_then(serde_json::Value::as_object_mut)
    {
        for (name, forecast) in std::mem::take(native_models) {
            models.insert(name, forecast);
        }
    }
    if let Some(object) = value.as_object_mut() {
        object.insert("native_onnx".to_owned(), native_result);
        object.insert(
            "execution_provider".to_owned(),
            serde_json::Value::String("mixed native ONNX + PyTorch".to_owned()),
        );
    }
    if let Some(telemetry) = value
        .get_mut("telemetry")
        .and_then(serde_json::Value::as_object_mut)
    {
        let python_executed = telemetry
            .get("executed_models")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        telemetry.insert(
            "executed_models".to_owned(),
            serde_json::Value::from(python_executed.saturating_add(native_executed)),
        );
        telemetry.insert(
            "native_onnx_ms".to_owned(),
            serde_json::Value::from(native_ms),
        );
    }
    if let Some(latency) = value
        .get_mut("latency_breakdown")
        .and_then(serde_json::Value::as_object_mut)
    {
        latency.insert(
            "native_onnx_ms".to_owned(),
            serde_json::Value::from(native_ms),
        );
    }
}

fn latency_summary(samples: &VecDeque<f64>) -> serde_json::Value {
    let mut ordered = samples.iter().copied().collect::<Vec<_>>();
    ordered.sort_by(f64::total_cmp);
    serde_json::json!({
        "samples": ordered.len(),
        "capacity": 256,
        "p50_ms": integer_percentile(&ordered, 50),
        "p95_ms": integer_percentile(&ordered, 95),
        "p99_ms": integer_percentile(&ordered, 99),
    })
}

fn integer_percentile(ordered: &[f64], percentile: usize) -> Option<f64> {
    if ordered.is_empty() {
        return None;
    }
    let rank = ordered.len().saturating_mul(percentile).saturating_add(99) / 100;
    let index = rank.saturating_sub(1).min(ordered.len() - 1);
    ordered.get(index).copied()
}

struct ModelRuntime {
    python: PathBuf,
    source: PathBuf,
    torch_version: String,
    cuda_available: bool,
    accelerator: String,
    vram_capacity: String,
}

fn model_runtime() -> Result<ModelRuntime, String> {
    let source = if let Some(configured) = std::env::var_os("ARGUS_PYTHONPATH") {
        PathBuf::from(configured)
    } else {
        std::env::current_exe()
            .ok()
            .and_then(|executable| executable.parent().map(|parent| parent.join("python/src")))
            .filter(|path| path.join("argus_research/experimental_cnn.py").is_file())
            .unwrap_or_else(|| workspace_root().join("python/src"))
    };
    if !source.join("argus_research/experimental_cnn.py").is_file() {
        return Err(format!(
            "model runtime source is absent at {}; set ARGUS_PYTHONPATH to the Argus Python src directory",
            source.display()
        ));
    }

    let python = if let Some(configured) = std::env::var_os("ARGUS_PYTHON") {
        PathBuf::from(configured)
    } else {
        let packaged_python = std::env::current_exe()
            .ok()
            .and_then(|executable| {
                executable
                    .parent()
                    .map(|parent| parent.join("python/.venv/Scripts/python.exe"))
            })
            .filter(|path| path.is_file());
        let workspace_python = workspace_root().join("python/.venv/Scripts/python.exe");
        if let Some(packaged) = packaged_python {
            packaged
        } else if workspace_python.is_file() {
            workspace_python
        } else {
            find_external_tool("ARGUS_PYTHON", "python.exe", &[]).ok_or_else(|| {
                "Python was not found; set ARGUS_PYTHON to a Python executable with PyTorch"
                    .to_owned()
            })?
        }
    };
    if !python.is_file() {
        return Err(format!(
            "model Python executable is absent at {}; set ARGUS_PYTHON to a valid executable",
            python.display()
        ));
    }

    let probe = background_command(&python)
        .args([
            "-c",
            "import torch; import argus_research.experimental_cnn; dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'; vram = (str(round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)) + ' GB') if torch.cuda.is_available() else 'N/A'; print(str(torch.__version__) + '|' + str(torch.cuda.is_available()) + '|' + dev + '|' + vram)",
        ])
        .env("PYTHONPATH", &source)
        .env("OMP_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .map_err(|error| format!("model runtime probe could not start: {error}"))?;
    if !probe.status.success() {
        let detail = String::from_utf8_lossy(&probe.stderr);
        return Err(format!(
            "model runtime probe failed with {}: {}",
            probe.status,
            detail.trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&probe.stdout);
    let mut parts = stdout.trim().split('|');
    let torch_version = parts.next().unwrap_or("unknown").to_owned();
    let cuda_available = parts.next().is_some_and(|v| v.eq_ignore_ascii_case("true"));
    let accelerator = parts.next().unwrap_or("CPU").to_owned();
    let vram_capacity = parts.next().unwrap_or("N/A").to_owned();
    Ok(ModelRuntime {
        python,
        source,
        torch_version,
        cuda_available,
        accelerator,
        vram_capacity,
    })
}

fn discover_native_runtime_library() -> Result<PathBuf, String> {
    argus_inference::discover_runtime_library(None)
        .map_err(|error| error.to_string())
        .or_else(|original| {
            let python = model_runtime()?;
            let environment = python
                .python
                .parent()
                .and_then(Path::parent)
                .ok_or_else(|| original.clone())?;
            let candidate = environment.join("Lib/site-packages/onnxruntime/capi/onnxruntime.dll");
            candidate.is_file().then_some(candidate).ok_or(original)
        })
}

impl Drop for CnnWorker {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

struct TemporaryRuntimeFeature {
    path: PathBuf,
}

struct TemporaryGeneratedCapture {
    paths: Vec<PathBuf>,
}

impl TemporaryGeneratedCapture {
    fn new(paths: impl IntoIterator<Item = PathBuf>) -> Self {
        Self {
            paths: paths.into_iter().collect(),
        }
    }
}

impl Drop for TemporaryGeneratedCapture {
    fn drop(&mut self) {
        for path in &self.paths {
            let _ = fs::remove_file(path);
        }
    }
}

impl TemporaryRuntimeFeature {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TemporaryRuntimeFeature {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RunView {
    source: String,
    session: String,
    detector: DetectorView,
    health: HealthView,
    flows: Vec<FlowView>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DetectorView {
    id: String,
    trained: bool,
    feature_schema: String,
    artifact: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct HealthView {
    source_records: u64,
    normalized_packets: u64,
    processed_packets: u64,
    flows: usize,
    alerts: u64,
    unsupported_records: u64,
    malformed_records: u64,
    reported_lost_packets: u64,
    truncated_packets: u64,
    reordered_packets: u64,
    queue_overflows: u64,
    parse_errors: u64,
    source_errors: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct AttackVerdictDto {
    category: String,
    severity: String,
    score: f64,
    threshold: f64,
    summary: String,
    detail: String,
    detector: String,
    evidence: Vec<String>,
    confidence: String,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
#[allow(clippy::struct_excessive_bools)] // Serialized UI contract preserves independent observed states.
struct FlowView {
    uid: String,
    session: String,
    source: String,
    id: u64,
    protocol: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    service: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    threat_status: Option<String>,
    generation: u64,
    initiator_address: String,
    initiator_port: u16,
    responder_address: String,
    responder_port: u16,
    first_timestamp_ns: u64,
    last_timestamp_ns: u64,
    duration_ms: f64,
    packets: u64,
    captured_bytes: u64,
    original_bytes: u64,
    initiator_packets: u64,
    responder_packets: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    initiator_bytes: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    responder_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    initiator_mac: Option<[u8; 6]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    responder_mac: Option<[u8; 6]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tls_sni: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dns_qname: Option<String>,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    dns_messages: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    dns_queries: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    http_requests: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    tor_port_matches: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    tcp_syn_packets: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    tcp_syn_fin_packets: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    tcp_xmas_packets: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    tcp_null_packets: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    tcp_syn_rst_packets: u64,
    lost_packets: u64,
    truncated_packets: u64,
    reordered_packets: u64,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    is_watchlisted: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    is_blocked: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    is_allowlisted: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    is_blocklisted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    rule_action: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    matched_rule_target: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    attack_verdict: Option<AttackVerdictDto>,
}

#[allow(clippy::trivially_copy_pass_by_ref)] // serde skip predicates receive references.
const fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

#[tauri::command]
async fn desktop_status() -> Result<DesktopStatus, String> {
    tauri::async_runtime::spawn_blocking(desktop_status_blocking)
        .await
        .map_err(|error| error.to_string())
}

fn desktop_status_blocking() -> DesktopStatus {
    let capture_readiness = match PktmonProbe::inspect().readiness {
        PktmonReadiness::Unavailable => "unavailable",
        PktmonReadiness::PrivilegeRequired => "available; elevation required",
        PktmonReadiness::StatusReachable => "available",
        PktmonReadiness::ProbeFailed { .. } => "probe failed",
    };
    let ahem_v61_artifact = verify_ahem_v61_runtime_artifact(&model_directory());
    let ahem_v63_experimental = verify_ahem_v63_experimental_artifact(&model_directory());
    let models = model_statuses(&ahem_v61_artifact, &ahem_v63_experimental);
    let admission = load_admission_runtime();
    let runtime = model_runtime();
    let native_inference_backend = discover_native_runtime_library().map_or_else(
        |error| format!("optional native ONNX backend unavailable: {error}"),
        |path| format!("native ONNX Runtime candidate: {}", path.display()),
    );
    let (model_backend, accelerator) = if admission.security_verdict_enabled {
        ("calibrated security detector admitted".to_owned(), None)
    } else {
        match &runtime {
            Ok(rt) => {
                let acc = if rt.cuda_available {
                    format!("CUDA ({} · {})", rt.accelerator, rt.vram_capacity)
                } else {
                    "CPU (1 thread Balanced; up to 8 Boost)".to_owned()
                };
                (
                    format!(
                        "model worker verified (PyTorch {} · {acc}); awaiting real inference telemetry",
                        rt.torch_version
                    ),
                    Some(acc),
                )
            }
            Err(error) => (format!("model worker unavailable: {error}"), None),
        }
    };
    DesktopStatus {
        runtime: "ready",
        capture_readiness: capture_readiness.to_owned(),
        model_backend,
        native_inference_backend,
        moe_engine: if admission.forecast_router_enabled {
            "legacy forecast-router artifacts hash-admitted; live execution must be checked in model telemetry; security MoE not connected"
        } else {
            "legacy forecast router not admitted; security MoE not connected"
        }
        .to_owned(),
        admitted_experts: if admission.forecast_router_enabled {
            TEMPORAL_EXPERT_FILES.len()
        } else {
            0
        },
        feature_schema: "argus.causal-flow/v1",
        admission_policy: admission.policy,
        security_verdict_enabled: admission.security_verdict_enabled,
        forecast_router_enabled: admission.forecast_router_enabled,
        models,
        tools: external_tool_statuses(),
        accelerator,
        research_evidence: ahem_v61_research_evidence(&ahem_v61_artifact),
    }
}

fn ahem_v61_research_evidence(
    artifact_verification: &Result<String, String>,
) -> ResearchEvidenceStatus {
    let artifact_hash = artifact_verification.as_ref().ok().cloned();
    let verified = artifact_hash.is_some();
    ResearchEvidenceStatus {
        model_id: "AHEM v6.1",
        evidence_state: "SUPPORTED",
        runtime_state: if verified {
            "HASH-VERIFIED FORECAST ARTIFACT"
        } else {
            "FORECAST ARTIFACT UNAVAILABLE OR REJECTED"
        },
        role: "bounded next-same-flow forecasting",
        endpoint: "weighted Smooth L1 on log frame length and per-flow interarrival",
        dataset: "CSE-CIC-IDS2018 / untouched Friday-23-02-2018 test day",
        test_coordinates: 512,
        flow_clusters: 501,
        seeds: 3,
        candidate_loss: 0.248_937,
        comparator_loss: 0.435_422,
        relative_reduction_percent: 42.81,
        interval_low_percent: 35.17,
        interval_high_percent: 50.30,
        parameters: 40_851,
        comparator_parameters: 41_142,
        artifact_sha256: artifact_hash,
        security_detector: false,
        caveat: "Supported for the frozen forecasting endpoint only; no attack-detection, latency, or cross-organization superiority claim.",
    }
}

fn verify_ahem_v61_runtime_artifact(directory: &Path) -> Result<String, String> {
    verify_ahem_v61_runtime_artifact_at(&directory.join(AHEM_V61_FILE), AHEM_V61_ARTIFACT_SHA256)
}

fn verify_ahem_v63_experimental_artifact(directory: &Path) -> Result<String, String> {
    let checkpoint = directory.join(AHEM_V63_EXPERIMENTAL_FILE);
    if !checkpoint.is_file() {
        return Err(format!("artifact is absent at {}", checkpoint.display()));
    }
    let actual_hash = file_sha256(&checkpoint).map_err(|error| error.to_string())?;
    if actual_hash != AHEM_V63_EXPERIMENTAL_SHA256 {
        return Err("experimental artifact SHA-256 differs".to_owned());
    }
    let report_path = directory.join(AHEM_V63_EXPERIMENTAL_REPORT);
    let contents = fs::read_to_string(&report_path)
        .map_err(|error| format!("experimental report could not be read: {error}"))?;
    let report: serde_json::Value = serde_json::from_str(&contents)
        .map_err(|error| format!("experimental report is invalid JSON: {error}"))?;
    if report.get("schema").and_then(serde_json::Value::as_str)
        != Some("argus.ahem-v6.3-local-pilot/v1")
        || report.get("study_role").and_then(serde_json::Value::as_str)
            != Some("source_disjoint_exploratory_vm_training")
        || report
            .get("promotion_eligible")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
        || report
            .get("publication_evidence")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
        || report
            .get("checkpoint")
            .and_then(|value| value.get("sha256"))
            .and_then(serde_json::Value::as_str)
            != Some(AHEM_V63_EXPERIMENTAL_SHA256)
    {
        return Err("experimental report contract differs".to_owned());
    }
    Ok(actual_hash)
}

fn verify_ahem_v61_runtime_artifact_at(
    checkpoint: &Path,
    expected_hash: &str,
) -> Result<String, String> {
    if !checkpoint.is_file() {
        return Err(format!("artifact is absent at {}", checkpoint.display()));
    }
    let manifest_path = checkpoint.with_extension("pt.manifest.json");
    let contents = fs::read_to_string(&manifest_path)
        .map_err(|error| format!("runtime manifest could not be read: {error}"))?;
    let manifest: serde_json::Value = serde_json::from_str(&contents)
        .map_err(|error| format!("runtime manifest is invalid JSON: {error}"))?;
    for (field, expected) in [
        ("schema", "argus.ahem-v6-runtime-manifest/v1"),
        ("model_id", "argus.ahem-v6.1.state-transition-forecaster"),
        ("artifact_sha256", expected_hash),
    ] {
        if manifest.get(field).and_then(serde_json::Value::as_str) != Some(expected) {
            return Err(format!("runtime manifest {field} differs"));
        }
    }
    if manifest
        .get("security_detector")
        .and_then(serde_json::Value::as_bool)
        != Some(false)
    {
        return Err("runtime manifest security_detector must be false".to_owned());
    }
    let actual_hash = file_sha256(checkpoint).map_err(|error| error.to_string())?;
    if actual_hash != expected_hash {
        return Err("runtime artifact SHA-256 differs".to_owned());
    }
    Ok(actual_hash)
}

#[tauri::command]
#[allow(clippy::too_many_lines)]
async fn runtime_diagnostics() -> Result<RuntimeDiagnostics, String> {
    tauri::async_runtime::spawn_blocking(runtime_diagnostics_blocking)
        .await
        .map_err(|error| error.to_string())
}

#[allow(clippy::too_many_lines)]
fn runtime_diagnostics_blocking() -> RuntimeDiagnostics {
    let mut checks = Vec::new();
    let data_root = development_root();
    let capture_root = capture_directory();
    match &capture_root {
        Ok(directory) => {
            let probe = directory.join(format!(".write-probe-{}", std::process::id()));
            let writable =
                fs::write(&probe, b"argus-write-probe").and_then(|()| fs::remove_file(&probe));
            checks.push(match writable {
                Ok(()) => diagnostic_ok(
                    "storage",
                    "Local data storage",
                    format!("Writable at {}", directory.display()),
                ),
                Err(error) => diagnostic_error(
                    "storage",
                    "Local data storage",
                    format!("{}: {error}", directory.display()),
                    "Set ARGUS_DEV_ROOT to a writable absolute directory and restart Argus.",
                ),
            });
        }
        Err(error) => checks.push(diagnostic_error(
            "storage",
            "Local data storage",
            error.clone(),
            "Set ARGUS_DEV_ROOT to a writable absolute directory and restart Argus.",
        )),
    }

    let pktmon = PktmonProbe::inspect();
    checks.push(match pktmon.readiness {
        PktmonReadiness::StatusReachable => diagnostic_ok(
            "pktmon",
            "Windows Packet Monitor",
            "Pktmon is installed and its status is reachable".to_owned(),
        ),
        PktmonReadiness::PrivilegeRequired => {
            // The app runs unprivileged so `pktmon status` always requires elevation.
            // If the elevated scheduled-task sensor is already active (confirmed via
            // sensor-health.json), the privilege constraint is handled — report ok.
            let sensor_live = capture_directory()
                .ok()
                .map(|d| d.join("live"))
                .and_then(|live| read_sensor_health(&live))
                .is_some_and(|h| h.active && !h.failed);
            if sensor_live {
                diagnostic_ok(
                    "pktmon",
                    "Windows Packet Monitor",
                    "Elevated sensor active — live capture running via registered task".to_owned(),
                )
            } else {
                diagnostic_warning(
                    "pktmon",
                    "Windows Packet Monitor",
                    "Pktmon is installed; live capture requires the registered elevated sensor"
                        .to_owned(),
                    "Start the rolling monitor once and approve the Windows administrator prompt.",
                )
            }
        }
        PktmonReadiness::Unavailable => diagnostic_error(
            "pktmon",
            "Windows Packet Monitor",
            "pktmon.exe is unavailable on this Windows installation".to_owned(),
            "Use a supported Windows build with Pktmon or replay an existing PCAP/PCAPNG file.",
        ),
        PktmonReadiness::ProbeFailed { detail } => diagnostic_error(
            "pktmon",
            "Windows Packet Monitor",
            detail,
            "Run `pktmon status` in an elevated terminal and resolve the reported Windows error.",
        ),
    });

    let sensor = sensor_executable();
    checks.push(match &sensor {
        Ok(executable) => diagnostic_ok(
            "sensor",
            "Native live sensor",
            executable.display().to_string(),
        ),
        Err(error) => diagnostic_error(
            "sensor",
            "Native live sensor",
            error.clone(),
            "Build native/pktmon-stream or reinstall the Argus portable package.",
        ),
    });

    checks.push(match model_runtime() {
        Ok(runtime) => diagnostic_ok(
            "model-runtime",
            "Local model worker",
            format!(
                "Verified {} (PyTorch {}, accelerator: {}, VRAM: {}) with runtime source {}",
                runtime.python.display(),
                runtime.torch_version,
                runtime.accelerator,
                runtime.vram_capacity,
                runtime.source.display()
            ),
        ),
        Err(error) => diagnostic_error(
            "model-runtime",
            "Local model worker",
            error,
            "Set ARGUS_PYTHON to a valid PyTorch Python executable and ARGUS_PYTHONPATH to this repository's python/src directory.",
        ),
    });

    checks.push(match verify_ahem_v61_runtime_artifact(&model_directory()) {
        Ok(hash) => diagnostic_warning(
            "ahem-v61-artifact",
            "Frozen AHEM v6.1 artifact",
            format!("Checkpoint hash verified ({hash}); model loading and inference have not been checked here."),
            "Run a forecast and check the model activity and rejected-model details.",
        ),
        Err(error) => diagnostic_warning(
            "ahem-v61-artifact",
            "Frozen AHEM v6.1 artifact",
            format!("Unavailable or quarantined: {error}"),
            "Restore the exact hash-bound forecast artifact and manifest, then restart Argus.",
        ),
    });

    checks.push(match discover_native_runtime_library() {
        Ok(runtime) => diagnostic_ok(
            "native-inference",
            "Native ONNX inference",
            format!(
                "Runtime candidate found at {}; graph compatibility is checked at model admission",
                runtime.display()
            ),
        ),
        Err(error) => diagnostic_warning(
            "native-inference",
            "Native ONNX inference",
            error,
            "Optionally set ARGUS_ONNX_RUNTIME to a compatible ONNX Runtime library. The current Python model worker remains available.",
        ),
    });

    if let (Ok(executable), Ok(directory)) = (&sensor, &capture_root) {
        let output = directory.join("live");
        let stop = output.join("sensor.stop");
        checks.push(if sensor_task_is_current(executable, &output, &stop) {
            diagnostic_ok(
                "sensor-task",
                "Elevated sensor registration",
                format!("Scheduled task {SENSOR_TASK_NAME} matches this installation"),
            )
        } else {
            diagnostic_warning(
                "sensor-task",
                "Elevated sensor registration",
                "The task is absent or points to another Argus installation".to_owned(),
                "Start the rolling monitor; Argus will request elevation and repair the task.",
            )
        });
    }

    let tools = external_tool_statuses();
    let available_tools = tools
        .iter()
        .filter(|tool| tool.available)
        .map(|tool| tool.id)
        .collect::<Vec<_>>();
    checks.push(if available_tools.is_empty() {
        diagnostic_warning(
            "tools",
            "Packet inspection tools",
            "Wireshark and TShark were not found; Argus capture and analysis still work".to_owned(),
            "Optionally install Wireshark for on-demand deep packet inspection.",
        )
    } else {
        diagnostic_ok(
            "tools",
            "Packet inspection tools",
            format!("Available: {}", available_tools.join(", ")),
        )
    });

    let (retained_capture_files, retained_capture_bytes) = capture_root
        .as_deref()
        .map(capture_storage_usage)
        .unwrap_or_default();
    let healthy = checks.iter().all(|check| check.state != "error");
    RuntimeDiagnostics {
        healthy,
        data_root: data_root.map_or_else(
            |error| format!("unavailable: {error}"),
            |path| path.display().to_string(),
        ),
        retained_capture_files,
        retained_capture_bytes,
        checks,
    }
}

fn diagnostic_ok(id: &'static str, label: &'static str, detail: String) -> DiagnosticCheck {
    DiagnosticCheck {
        id,
        label,
        state: "ok",
        detail,
        recovery: None,
    }
}

fn diagnostic_warning(
    id: &'static str,
    label: &'static str,
    detail: String,
    recovery: &'static str,
) -> DiagnosticCheck {
    DiagnosticCheck {
        id,
        label,
        state: "warning",
        detail,
        recovery: Some(recovery),
    }
}

fn diagnostic_error(
    id: &'static str,
    label: &'static str,
    detail: String,
    recovery: &'static str,
) -> DiagnosticCheck {
    DiagnosticCheck {
        id,
        label,
        state: "error",
        detail,
        recovery: Some(recovery),
    }
}

fn capture_storage_usage(directory: &Path) -> (usize, u64) {
    let mut pending = vec![directory.to_path_buf()];
    let mut files = 0_usize;
    let mut bytes = 0_u64;
    while let Some(current) = pending.pop() {
        let Ok(entries) = fs::read_dir(current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| {
                extension.eq_ignore_ascii_case("pcap")
                    || extension.eq_ignore_ascii_case("pcapng")
                    || extension.eq_ignore_ascii_case("etl")
            }) {
                files = files.saturating_add(1);
                bytes = bytes.saturating_add(entry.metadata().map_or(0, |metadata| metadata.len()));
            }
        }
    }
    (files, bytes)
}

fn admission_runtime_from_value(value: &serde_json::Value) -> AdmissionRuntime {
    AdmissionRuntime {
        policy: value
            .pointer("/policy/version")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unverified")
            .to_owned(),
        security_verdict_enabled: value
            .pointer("/runtime/security_verdict_enabled")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        forecast_router_enabled: value
            .pointer("/runtime/forecast_router_enabled")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
    }
}

fn load_admission_runtime() -> AdmissionRuntime {
    let directory = model_directory();
    let path = directory.join("argus_admission_report.json");
    fs::read_to_string(path)
        .ok()
        .and_then(|contents| serde_json::from_str(&contents).ok())
        .as_ref()
        .map_or_else(AdmissionRuntime::default, |value| {
            integrity_bound_admission_runtime(value, &directory)
        })
}

fn integrity_bound_admission_runtime(
    value: &serde_json::Value,
    directory: &Path,
) -> AdmissionRuntime {
    let mut runtime = admission_runtime_from_value(value);
    if runtime.security_verdict_enabled {
        let artifact_valid = artifact_matches_report(
            value,
            "/decisions/security_detector/artifact_sha256",
            &directory.join(SECURITY_DETECTOR_FILE),
        );
        runtime.security_verdict_enabled = false;
        runtime.policy.push_str(if artifact_valid {
            " / security detector execution path unavailable"
        } else {
            " / security artifact integrity failed"
        });
    }
    if runtime.forecast_router_enabled
        && (!artifact_matches_report(
            value,
            "/decisions/forecast_router/artifact_sha256",
            &directory.join(FORECAST_ROUTER_FILE),
        ) || !TEMPORAL_EXPERT_FILES.iter().all(|(id, file_name)| {
            forecast_expert_hash(value, id).is_some_and(|expected| {
                file_sha256(&directory.join(file_name)).is_ok_and(|actual| actual == expected)
            })
        }))
    {
        runtime.forecast_router_enabled = false;
        runtime
            .policy
            .push_str(" / forecast artifact integrity failed");
    }
    runtime
}

fn forecast_expert_hash<'a>(value: &'a serde_json::Value, expert_id: &str) -> Option<&'a str> {
    value
        .pointer("/decisions/forecast_router/checks")?
        .as_array()?
        .iter()
        .find(|check| {
            check.get("name").and_then(serde_json::Value::as_str) == Some("expert_hashes")
        })?
        .get("observed")?
        .get(expert_id)?
        .as_str()
}

fn artifact_matches_report(value: &serde_json::Value, pointer: &str, path: &Path) -> bool {
    let Some(expected) = value.pointer(pointer).and_then(serde_json::Value::as_str) else {
        return false;
    };
    file_sha256(path).is_ok_and(|actual| actual == expected)
}

fn file_sha256(path: &Path) -> Result<String, std::io::Error> {
    let mut source = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let count = source.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn external_tool_statuses() -> Vec<ExternalToolStatus> {
    [
        (
            "wireshark",
            find_external_tool(
                "ARGUS_WIRESHARK",
                "Wireshark.exe",
                &[r"C:\Program Files\Wireshark\Wireshark.exe"],
            ),
            "on-demand retained-capture inspection",
        ),
        (
            "tshark",
            find_external_tool(
                "ARGUS_TSHARK",
                "tshark.exe",
                &[r"C:\Program Files\Wireshark\tshark.exe"],
            ),
            "optional deep-protocol extraction",
        ),
    ]
    .into_iter()
    .map(|(id, executable, role)| ExternalToolStatus {
        id,
        available: executable.is_some(),
        executable: executable.map(|path| path.display().to_string()),
        role,
    })
    .collect()
}

fn find_external_tool(environment: &str, name: &str, standard_paths: &[&str]) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(environment).map(PathBuf::from)
        && path.is_file()
    {
        return Some(path);
    }
    for path in standard_paths.iter().map(PathBuf::from) {
        if path.is_file() {
            return Some(path);
        }
    }
    let output = background_command("where.exe").arg(name).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(PathBuf::from)
        .find(|path| path.is_file())
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
async fn recent_captures(
    cache: tauri::State<'_, Arc<IpcQueryCache>>,
) -> Result<Vec<String>, String> {
    const TTL: Duration = Duration::from_millis(1500);
    if let Some(captures) = cache.get_recent_captures(TTL) {
        return Ok(captures);
    }
    let captures = tauri::async_runtime::spawn_blocking(recent_captures_blocking)
        .await
        .map_err(|error| error.to_string())??;
    cache.set_recent_captures(captures.clone());
    Ok(captures)
}

fn recent_captures_blocking() -> Result<Vec<String>, String> {
    let directory = capture_directory()?;
    let mut captures = Vec::new();
    collect_captures(&directory, &mut captures)?;
    let rolling = directory.join("rolling");
    if rolling.is_dir() {
        collect_captures(&rolling, &mut captures)?;
    }
    let live = directory.join("live");
    if live.is_dir() {
        collect_captures(&live, &mut captures)?;
    }
    captures.sort_unstable_by_key(|capture| Reverse(capture.0));
    Ok(captures.into_iter().map(|(_, path)| path).collect())
}

fn collect_captures(
    directory: &Path,
    captures: &mut Vec<(SystemTime, String)>,
) -> Result<(), String> {
    for entry in fs::read_dir(directory).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        if path.extension().is_some_and(|extension| {
            extension.eq_ignore_ascii_case("pcapng") || extension.eq_ignore_ascii_case("pcap")
        }) {
            let modified = entry
                .metadata()
                .and_then(|metadata| metadata.modified())
                .unwrap_or(UNIX_EPOCH);
            captures.push((modified, path.display().to_string()));
        }
    }
    Ok(())
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
async fn replay_capture(
    path: String,
    history: tauri::State<'_, Arc<Mutex<LiveHistory>>>,
    alerts: tauri::State<'_, Arc<Mutex<AlertManager>>>,
) -> Result<RunView, String> {
    if generated_live_capture_path(Path::new(&path))? && !Path::new(&path).is_file() {
        return Err(
            "live capture expired before analysis; the sensor eviction counter preserves this loss"
                .to_owned(),
        );
    }
    let history_key = generated_live_capture_key(Path::new(&path))?;
    let run = tauri::async_runtime::spawn_blocking(move || replay_path(Path::new(&path)))
        .await
        .map_err(|error| error.to_string())??;
    if let Some(history_key) = history_key {
        history
            .lock()
            .map_err(|_| "live history lock poisoned")?
            .record(&history_key, &run)?;
    }
    if let Ok(mut alert_mgr) = alerts.lock() {
        for flow in &run.flows {
            if let Some(record) = alert_mgr.tracker.evaluate_flow_quality(
                flow.last_timestamp_ns,
                None,
                flow.lost_packets,
                flow.truncated_packets,
            ) {
                alert_mgr.record(record);
            }
            alert_mgr.record_flow_attack(flow);
        }
    }
    Ok(run)
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn get_recent_alerts(
    limit: Option<usize>,
    severity: Option<String>,
    alerts: tauri::State<'_, Arc<Mutex<AlertManager>>>,
) -> Result<Vec<AnomalyAlertDto>, String> {
    let alerts = alerts.lock().map_err(|_| "alert manager lock poisoned")?;
    Ok(alerts.get_alerts(limit, severity.as_deref()))
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn clear_alerts(alerts: tauri::State<'_, Arc<Mutex<AlertManager>>>) -> Result<(), String> {
    let mut alerts = alerts.lock().map_err(|_| "alert manager lock poisoned")?;
    alerts.clear();
    Ok(())
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn get_alert_stats(
    alerts: tauri::State<'_, Arc<Mutex<AlertManager>>>,
) -> Result<AlertStatsDto, String> {
    let alerts = alerts.lock().map_err(|_| "alert manager lock poisoned")?;
    Ok(alerts.stats())
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
async fn live_history(
    history: tauri::State<'_, Arc<Mutex<LiveHistory>>>,
) -> Result<LiveHistoryView, String> {
    let history = Arc::clone(history.inner());
    tauri::async_runtime::spawn_blocking(move || {
        history
            .lock()
            .map_err(|_| "live history lock poisoned")?
            .view()
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
fn daily_learning_status() -> Result<serde_json::Value, String> {
    let path = daily_learning_directory()?.join("status.json");
    if !path.is_file() {
        return Ok(serde_json::json!({
            "schema": "argus.daily-learning-status/v1",
            "state": "collecting",
            "task": "self_supervised_next_packet_forecast_only",
            "securityLabelsLearned": false,
            "autoPromotion": false,
            "retainedPackets": 0,
            "newPacketsSinceTraining": 0,
            "detail": "waiting for real live packet features"
        }));
    }
    let contents = fs::read_to_string(path).map_err(|error| error.to_string())?;
    serde_json::from_str(&contents).map_err(|error| error.to_string())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReleaseCaptureStatus {
    Deleted,
    AlreadyAbsent,
    Deferred,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReleaseCaptureResult {
    pub status: ReleaseCaptureStatus,
    pub path: String,
    pub attempts: u32,
    pub file_existed: bool,
    pub path_still_exists: bool,
    pub error: Option<String>,
}

#[tauri::command]
fn release_live_capture(path: String) -> Result<ReleaseCaptureResult, String> {
    let candidate = PathBuf::from(&path);
    if !candidate.exists() {
        return Ok(ReleaseCaptureResult {
            status: ReleaseCaptureStatus::AlreadyAbsent,
            path,
            attempts: 0,
            file_existed: false,
            path_still_exists: false,
            error: None,
        });
    }
    if !generated_live_capture_path(&candidate)? {
        return Err("only generated live capture segments can be released".to_owned());
    }
    let canonical = dunce::canonicalize(&candidate).map_err(|error| error.to_string())?;

    let mut attempts = 0_u32;
    let max_attempts = 3_u32;
    let mut last_error = None;

    while attempts < max_attempts {
        attempts += 1;
        match fs::remove_file(&canonical) {
            Ok(()) => {
                IpcQueryCache::invalidate_global_recent_captures();
                let still_exists = canonical.exists();
                return Ok(ReleaseCaptureResult {
                    status: if still_exists {
                        ReleaseCaptureStatus::Deferred
                    } else {
                        ReleaseCaptureStatus::Deleted
                    },
                    path,
                    attempts,
                    file_existed: true,
                    path_still_exists: still_exists,
                    error: if still_exists {
                        Some("file unlinked but still reports existing on disk".to_owned())
                    } else {
                        None
                    },
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(ReleaseCaptureResult {
                    status: ReleaseCaptureStatus::AlreadyAbsent,
                    path,
                    attempts,
                    file_existed: true,
                    path_still_exists: false,
                    error: None,
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {
                last_error = Some(error.to_string());
                if attempts < max_attempts {
                    std::thread::sleep(std::time::Duration::from_millis(50 * u64::from(attempts)));
                }
            }
            Err(error) => {
                let still_exists = canonical.exists();
                return Ok(ReleaseCaptureResult {
                    status: ReleaseCaptureStatus::Failed,
                    path,
                    attempts,
                    file_existed: true,
                    path_still_exists: still_exists,
                    error: Some(error.to_string()),
                });
            }
        }
    }

    let still_exists = canonical.exists();
    if !still_exists {
        IpcQueryCache::invalidate_global_recent_captures();
    }
    Ok(ReleaseCaptureResult {
        status: if still_exists {
            ReleaseCaptureStatus::Deferred
        } else {
            ReleaseCaptureStatus::Deleted
        },
        path,
        attempts,
        file_existed: true,
        path_still_exists: still_exists,
        error: last_error,
    })
}

fn cleanup_generated_live_captures() -> Result<usize, String> {
    let live = capture_directory()?.join("live");
    if !live.is_dir() {
        return Ok(0);
    }
    let mut removed = 0;
    for entry in fs::read_dir(&live).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        let is_pcap = path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("pcap"));
        let is_temporary_pcap = path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("tmp"))
            && path
                .file_stem()
                .and_then(|stem| Path::new(stem).extension())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("pcap"));
        let generated_segment = name.starts_with("live-") && (is_pcap || is_temporary_pcap);
        if generated_segment {
            match fs::remove_file(&path) {
                Ok(()) => {
                    if !path.exists() {
                        removed += 1;
                    }
                }
                Err(error)
                    if error.kind() == std::io::ErrorKind::NotFound
                        || error.kind() == std::io::ErrorKind::PermissionDenied => {}
                Err(error) => return Err(error.to_string()),
            }
        }
    }
    Ok(removed)
}

fn cleanup_generated_temporary_captures() -> Result<usize, String> {
    cleanup_generated_temporary_captures_in(&capture_directory()?)
}

fn cleanup_generated_temporary_captures_in(directory: &Path) -> Result<usize, String> {
    if !directory.is_dir() {
        return Ok(0);
    }
    let mut removed = 0_usize;
    for entry in fs::read_dir(directory).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        let generated_simulation = name.starts_with("sim_")
            && path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("pcap"));
        let generated_manual = name
            .strip_prefix("argus-")
            .and_then(|rest| rest.split_once('.'))
            .is_some_and(|(stamp, extension)| {
                !stamp.is_empty()
                    && stamp.bytes().all(|byte| byte.is_ascii_digit())
                    && matches!(extension.to_ascii_lowercase().as_str(), "etl" | "pcapng")
            });
        if generated_simulation || generated_manual {
            match fs::remove_file(&path) {
                Ok(()) => {
                    if !path.exists() {
                        removed = removed.saturating_add(1);
                    }
                }
                Err(error)
                    if error.kind() == std::io::ErrorKind::NotFound
                        || error.kind() == std::io::ErrorKind::PermissionDenied => {}
                Err(error) => return Err(error.to_string()),
            }
        }
    }
    Ok(removed)
}

#[tauri::command]
fn open_capture_in_wireshark(path: String) -> Result<(), String> {
    let capture = dunce::canonicalize(PathBuf::from(path)).map_err(|error| error.to_string())?;
    if !capture.is_file()
        || !capture.extension().is_some_and(|extension| {
            extension.eq_ignore_ascii_case("pcap") || extension.eq_ignore_ascii_case("pcapng")
        })
    {
        return Err("Wireshark can open only an existing PCAP/PCAPNG file".to_owned());
    }
    let executable = find_external_tool(
        "ARGUS_WIRESHARK",
        "Wireshark.exe",
        &[r"C:\Program Files\Wireshark\Wireshark.exe"],
    )
    .ok_or("Wireshark is not installed or configured")?;
    Command::new(executable)
        .arg(capture)
        .spawn()
        .map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
async fn capture_30_seconds() -> Result<RunView, String> {
    let result = tauri::async_runtime::spawn_blocking(capture_and_replay)
        .await
        .map_err(|error| error.to_string())??;
    IpcQueryCache::invalidate_global_recent_captures();
    Ok(result)
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
async fn start_rolling_capture(
    monitor: tauri::State<'_, Arc<Mutex<RollingMonitor>>>,
) -> Result<RollingStatus, String> {
    let monitor = Arc::clone(monitor.inner());
    let status =
        tauri::async_runtime::spawn_blocking(move || start_rolling_capture_blocking(&monitor))
            .await
            .map_err(|error| error.to_string())??;
    IpcQueryCache::invalidate_global_recent_captures();
    Ok(status)
}

fn start_rolling_capture_blocking(
    monitor: &Mutex<RollingMonitor>,
) -> Result<RollingStatus, String> {
    let result = start_rolling_capture_inner(monitor);
    if let Err(error) = &result
        && let Ok(mut monitor) = monitor.lock()
    {
        monitor.startup_error = Some(error.clone());
    }
    result
}

fn start_rolling_capture_inner(monitor: &Mutex<RollingMonitor>) -> Result<RollingStatus, String> {
    let (output_directory, stop_file, executable) = {
        let mut monitor = monitor.lock().map_err(|_| "monitor lock poisoned")?;
        restore_monitor(&mut monitor)?;
        if monitor_running(&monitor) {
            return Ok(rolling_status_view(&monitor, true));
        }
        clear_monitor(&mut monitor)?;
        cleanup_generated_live_captures()?;
        let output_directory = capture_directory()?.join("live");
        fs::create_dir_all(&output_directory).map_err(|error| error.to_string())?;
        let stop_file = output_directory.join("sensor.stop");
        let _ = fs::remove_file(&stop_file);
        let _ = fs::remove_file(output_directory.join("cleanup-on-stop"));
        let _ = fs::remove_file(output_directory.join("sensor-health.json"));
        let executable = sensor_executable()?;
        ensure_sensor_task(&executable, &output_directory, &stop_file)?;
        (output_directory, stop_file, executable)
    };

    let health_file = output_directory.join("sensor-health.json");
    let script = format!(
        "$task=Get-ScheduledTask -TaskName {} -ErrorAction Stop; Start-ScheduledTask -InputObject $task; $health={}; for($i=0;$i -lt 100;$i++){{ if(Test-Path -LiteralPath $health){{ try{{$h=Get-Content -LiteralPath $health -Raw | ConvertFrom-Json; if($h.active -eq $true){{ if($h.process_id -and [uint32]$h.process_id -gt 0){{[uint32]$h.process_id; exit 0}} $proc=Get-CimInstance Win32_Process -Filter \"Name='argus-pktmon-stream.exe'\" -ErrorAction SilentlyContinue | Select-Object -First 1; if($proc -and $proc.ProcessId -gt 0){{[uint32]$proc.ProcessId; exit 0}} }} }}catch{{}} }}; Start-Sleep -Milliseconds 100 }}; exit 1",
        quote_powershell_text(SENSOR_TASK_NAME),
        quote_powershell(&health_file),
    );
    let output = background_command("powershell.exe")
        .args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            &script,
        ])
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "live sensor task was not started: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let process_id = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u32>()
        .map_err(|_| "elevated sensor did not return a process ID")?;

    let mut monitor = monitor.lock().map_err(|_| "monitor lock poisoned")?;
    monitor.process_id = Some(process_id);
    monitor.executable = Some(executable);
    monitor.stop_file = Some(stop_file);
    monitor.stop_requested = false;
    monitor.startup_error = None;
    persist_monitor_control(&monitor)?;
    Ok(rolling_status_view(&monitor, true))
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn stop_rolling_capture(
    monitor: tauri::State<'_, Arc<Mutex<RollingMonitor>>>,
) -> Result<RollingStatus, String> {
    let mut monitor = monitor.lock().map_err(|_| "monitor lock poisoned")?;
    let Some(stop_file) = monitor.stop_file.as_ref() else {
        return Ok(rolling_status_view(&monitor, false));
    };
    fs::write(stop_file, b"stop\n").map_err(|error| error.to_string())?;
    monitor.stop_requested = true;
    IpcQueryCache::invalidate_global_recent_captures();
    Ok(rolling_status_view(&monitor, true))
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
async fn rolling_capture_status(
    monitor: tauri::State<'_, Arc<Mutex<RollingMonitor>>>,
) -> Result<RollingStatus, String> {
    let monitor = Arc::clone(monitor.inner());
    tauri::async_runtime::spawn_blocking(move || rolling_capture_status_blocking(&monitor))
        .await
        .map_err(|error| error.to_string())?
}

fn rolling_capture_status_blocking(
    monitor: &Mutex<RollingMonitor>,
) -> Result<RollingStatus, String> {
    let mut monitor = monitor.lock().map_err(|_| "monitor lock poisoned")?;
    restore_monitor(&mut monitor)?;
    let running = monitor_running(&monitor);
    if !running {
        clear_monitor(&mut monitor)?;
    }
    Ok(rolling_status_view(&monitor, running))
}

fn monitor_control_path() -> Result<PathBuf, String> {
    Ok(capture_directory()?.join("live/active-monitor.txt"))
}

fn runtime_preferences_path() -> Result<PathBuf, String> {
    Ok(development_root()?.join("runtime-preferences.json"))
}

fn load_runtime_preferences() -> RuntimePreferences {
    runtime_preferences_path()
        .ok()
        .and_then(|path| fs::read_to_string(path).ok())
        .and_then(|contents| serde_json::from_str::<RuntimePreferences>(&contents).ok())
        .filter(|preferences| preferences.version == 1)
        .unwrap_or_default()
}

fn persist_runtime_preferences(preferences: &RuntimePreferences) -> Result<(), String> {
    let encoded = serde_json::to_vec_pretty(preferences).map_err(|error| error.to_string())?;
    replace_file(&runtime_preferences_path()?, &encoded)
}

fn stop_sensor_on_exit(preferences: &RuntimePreferences) -> bool {
    !preferences.keep_monitoring_after_exit
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn runtime_preferences(
    preferences: tauri::State<'_, Arc<Mutex<RuntimePreferences>>>,
) -> Result<RuntimePreferences, String> {
    preferences
        .lock()
        .map(|preferences| preferences_with_model_catalog(preferences.clone()))
        .map_err(|_| "runtime preferences lock poisoned".to_owned())
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn set_keep_monitoring_after_exit(
    enabled: bool,
    preferences: tauri::State<'_, Arc<Mutex<RuntimePreferences>>>,
) -> Result<RuntimePreferences, String> {
    let mut preferences = preferences
        .lock()
        .map_err(|_| "runtime preferences lock poisoned".to_owned())?;
    let mut updated = preferences.clone();
    updated.keep_monitoring_after_exit = enabled;
    persist_runtime_preferences(&updated)?;
    *preferences = updated.clone();
    Ok(updated)
}

fn update_model_selection(
    preferences: &RuntimePreferences,
    model_id: &str,
    enabled: bool,
) -> Result<RuntimePreferences, String> {
    if !SELECTABLE_MODEL_IDS.contains(&model_id) {
        return Err(format!("unsupported selectable model: {model_id}"));
    }
    let mut updated = preferences.clone();
    updated.disabled_models.retain(|id| id != model_id);
    if !enabled {
        updated.disabled_models.push(model_id.to_owned());
    }
    updated.disabled_models.sort();
    updated.disabled_models.dedup();
    Ok(updated)
}

fn preferences_with_model_catalog(mut preferences: RuntimePreferences) -> RuntimePreferences {
    let directory = model_directory();
    let files = [
        "ahem_fixed_real_pcap_30k.pt",
        "cnn_local_real_pcap_30k.pt",
        "lstm_real_pcap_30k.pt",
        "tcn_long_real_pcap_30k.pt",
        "ahem_v31_forecast_only.pt",
        "cnn_cse_cic_ids2018_best.pt",
        "lstm_cse_cic_ids2018_best.pt",
        "gru_cse_cic_ids2018_best.pt",
        "tcn_cse_cic_ids2018_best.pt",
        "ahem_flash_cse_cic_ids2018_best.pt",
        "ahem_pro_cse_cic_ids2018_best.pt",
        AHEM_V61_FILE,
        AHEM_V63_EXPERIMENTAL_FILE,
        "challengers/ahem_e2e_experimental_step3808.pt",
    ];
    preferences.selectable_models = SELECTABLE_MODEL_IDS
        .iter()
        .zip(files)
        .map(|(&id, file)| {
            let available = directory.join(file).is_file();
            SelectableModel {
                id,
                available,
                detail: if available {
                    "Checkpoint present; admission and successful execution remain unverified."
                } else {
                    "Checkpoint absent from the runtime model directory."
                },
            }
        })
        .collect();
    preferences
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn set_model_enabled(
    model_id: String,
    enabled: bool,
    preferences: tauri::State<'_, Arc<Mutex<RuntimePreferences>>>,
) -> Result<RuntimePreferences, String> {
    let mut preferences = preferences
        .lock()
        .map_err(|_| "runtime preferences lock poisoned".to_owned())?;
    let updated = update_model_selection(&preferences, &model_id, enabled)?;
    // Commit the file before publishing the new selection. The next request
    // safely recreates its worker; ongoing forecasts are not interrupted.
    persist_runtime_preferences(&updated)?;
    *preferences = updated.clone();
    Ok(preferences_with_model_catalog(updated))
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn reset_model_preferences(
    preferences: tauri::State<'_, Arc<Mutex<RuntimePreferences>>>,
) -> Result<RuntimePreferences, String> {
    let mut preferences = preferences
        .lock()
        .map_err(|_| "runtime preferences lock poisoned".to_owned())?;
    let mut updated = preferences.clone();
    updated.disabled_models = default_disabled_models();
    persist_runtime_preferences(&updated)?;
    *preferences = updated.clone();
    Ok(preferences_with_model_catalog(updated))
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn set_boost_mode(
    enabled: bool,
    preferences: tauri::State<'_, Arc<Mutex<RuntimePreferences>>>,
    worker: tauri::State<'_, Arc<Mutex<CnnWorker>>>,
) -> Result<RuntimePreferences, String> {
    let mut preferences = preferences
        .lock()
        .map_err(|_| "runtime preferences lock poisoned".to_owned())?;
    let mut updated = preferences.clone();
    updated.boost_mode = enabled;
    persist_runtime_preferences(&updated)?;
    *preferences = updated.clone();
    drop(preferences);

    let mut worker = worker
        .lock()
        .map_err(|_| "model worker lock poisoned".to_owned())?;
    worker.stop_worker();
    worker.native = None;
    worker.native_start_error = None;
    worker.boost_mode = None;
    Ok(updated)
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn set_notification_policy(
    policy: String,
    preferences: tauri::State<'_, Arc<Mutex<RuntimePreferences>>>,
) -> Result<RuntimePreferences, String> {
    let notification_policy = match policy.as_str() {
        "errors_only" => NotificationPolicy::ErrorsOnly,
        "all_activity" => NotificationPolicy::AllActivity,
        "silent" => NotificationPolicy::Silent,
        _ => return Err("notification policy is unsupported".to_owned()),
    };
    let mut preferences = preferences
        .lock()
        .map_err(|_| "runtime preferences lock poisoned".to_owned())?;
    let mut updated = preferences.clone();
    updated.notification_policy = notification_policy;
    persist_runtime_preferences(&updated)?;
    *preferences = updated.clone();
    Ok(updated)
}

fn persist_monitor_control(monitor: &RollingMonitor) -> Result<(), String> {
    let process_id = monitor.process_id.ok_or("monitor process ID is absent")?;
    let executable = monitor
        .executable
        .as_ref()
        .ok_or("monitor executable is absent")?;
    let stop_file = monitor
        .stop_file
        .as_ref()
        .ok_or("monitor stop file is absent")?;
    let output_directory = stop_file
        .parent()
        .ok_or("monitor stop path has no parent")?;
    let control = MonitorControl {
        schema: "argus.rolling-monitor-control".to_owned(),
        version: 1,
        process_id,
        executable: executable.clone(),
        output_directory: output_directory.to_path_buf(),
        stop_file: stop_file.clone(),
    };
    let encoded = serde_json::to_vec_pretty(&control).map_err(|error| error.to_string())?;
    replace_file(&monitor_control_path()?, &encoded)
}

fn load_monitor_control() -> Result<Option<MonitorControl>, String> {
    let path = monitor_control_path()?;
    if !path.is_file() {
        return Ok(None);
    }
    let contents = fs::read_to_string(&path).map_err(|error| error.to_string())?;
    let control: MonitorControl = if let Ok(control) = serde_json::from_str(&contents) {
        control
    } else {
        let _ = fs::remove_file(&path);
        return Ok(None);
    };
    if let Ok(valid) = validate_monitor_control(control) {
        Ok(Some(valid))
    } else {
        let _ = fs::remove_file(&path);
        Ok(None)
    }
}

fn validate_monitor_control(control: MonitorControl) -> Result<MonitorControl, String> {
    if control.schema != "argus.rolling-monitor-control" || control.version != 1 {
        return Err("rolling control schema is unsupported".to_owned());
    }
    let expected_output = capture_directory()?.join("live");
    let expected_stop = expected_output.join("sensor.stop");
    if control.process_id == 0
        || control.output_directory != expected_output
        || control.stop_file != expected_stop
        || control
            .executable
            .file_name()
            .is_none_or(|name| !name.eq_ignore_ascii_case("argus-pktmon-stream.exe"))
    {
        return Err("rolling control does not match the Argus live sensor scope".to_owned());
    }
    Ok(control)
}

fn process_is_running(control: &MonitorControl) -> bool {
    let script = format!(
        "$p=Get-CimInstance Win32_Process -Filter \"ProcessId={}\" -ErrorAction SilentlyContinue; $exe={}; $stop={}; if($p -and $p.ExecutablePath -ieq $exe -and $p.CommandLine -and $p.CommandLine.IndexOf($stop,[StringComparison]::OrdinalIgnoreCase) -ge 0){{exit 0}}; exit 1",
        control.process_id,
        quote_powershell(&control.executable),
        quote_powershell(&control.stop_file),
    );
    background_command("powershell.exe")
        .args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            &script,
        ])
        .status()
        .is_ok_and(|status| status.success())
}

fn monitor_running(monitor: &RollingMonitor) -> bool {
    let (Some(process_id), Some(executable), Some(stop_file)) = (
        monitor.process_id,
        monitor.executable.as_ref(),
        monitor.stop_file.as_ref(),
    ) else {
        return false;
    };
    let Some(output_directory) = stop_file.parent() else {
        return false;
    };
    if sensor_health_is_fresh(output_directory) {
        return true;
    }
    process_is_running(&MonitorControl {
        schema: "argus.rolling-monitor-control".to_owned(),
        version: 1,
        process_id,
        executable: executable.clone(),
        output_directory: output_directory.to_path_buf(),
        stop_file: stop_file.clone(),
    })
}

fn sensor_health_is_fresh(directory: &Path) -> bool {
    let path = directory.join("sensor-health.json");
    let fresh = path
        .metadata()
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .is_some_and(|age| age.as_secs() <= LIVE_SEGMENT_SECONDS.saturating_add(3));
    fresh && read_sensor_health(directory).is_some_and(|health| health.active && !health.failed)
}

fn restore_monitor(monitor: &mut RollingMonitor) -> Result<(), String> {
    if monitor.process_id.is_some() {
        return Ok(());
    }
    let Some(control) = load_monitor_control()? else {
        return Ok(());
    };
    monitor.process_id = Some(control.process_id);
    monitor.executable = Some(control.executable);
    monitor.stop_file = Some(control.stop_file);
    Ok(())
}

fn clear_monitor(monitor: &mut RollingMonitor) -> Result<(), String> {
    match fs::remove_file(monitor_control_path()?) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.to_string()),
    }
    monitor.process_id = None;
    monitor.executable = None;
    monitor.stop_file = None;
    monitor.stop_requested = false;
    Ok(())
}

fn sensor_executable() -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    if let Some(configured) = std::env::var_os("ARGUS_PKTMON_SENSOR") {
        candidates.push(PathBuf::from(configured));
    }
    if let Ok(current) = std::env::current_exe()
        && let Some(directory) = current.parent()
    {
        candidates.push(directory.join("argus-pktmon-stream.exe"));
    }
    let workspace = workspace_root();
    candidates.push(workspace.join("target/release/argus-pktmon-stream.exe"));
    candidates.push(workspace.join("target/debug/argus-pktmon-stream.exe"));
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .and_then(|path| dunce::canonicalize(&path).ok())
        .ok_or_else(|| {
            "native Pktmon sensor is absent; build it or set ARGUS_PKTMON_SENSOR".to_owned()
        })
}

fn rolling_status_view(monitor: &RollingMonitor, running: bool) -> RollingStatus {
    RollingStatus {
        running,
        stopping: running && monitor.stop_requested,
        process_id: monitor.process_id,
        segment_seconds: LIVE_SEGMENT_SECONDS,
        mode: "pktmon_api_gapless_live_stream",
        sensor: monitor
            .stop_file
            .as_ref()
            .and_then(|path| path.parent())
            .and_then(read_sensor_health),
        startup_error: monitor.startup_error.clone(),
    }
}

fn read_sensor_health(directory: &Path) -> Option<SensorHealth> {
    let contents = fs::read_to_string(directory.join("sensor-health.json")).ok()?;
    let health: SensorHealth = serde_json::from_str(&contents).ok()?;
    (health.schema == "argus.pktmon-live-health" && health.version == 1).then_some(health)
}

fn ensure_sensor_task(executable: &Path, output: &Path, stop_file: &Path) -> Result<(), String> {
    if sensor_task_is_current(executable, output, stop_file) {
        return Ok(());
    }
    let argus_executable = std::env::current_exe().map_err(|error| error.to_string())?;
    let elevated = format!(
        "$p=Start-Process -FilePath {} -Verb RunAs -WindowStyle Hidden -ArgumentList @({}) -Wait -PassThru; exit $p.ExitCode",
        quote_powershell(&argus_executable),
        quote_powershell_text(SENSOR_INSTALL_HELPER_ARG),
    );
    let status = background_command("powershell.exe")
        .args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            &elevated,
        ])
        .status()
        .map_err(|error| error.to_string())?;
    if status.success() && sensor_task_is_current(executable, output, stop_file) {
        Ok(())
    } else {
        Err("Argus sensor registration was cancelled or failed".to_owned())
    }
}

fn run_elevated_sensor_install_helper() -> Result<(), String> {
    let installer = sensor_installer().ok_or("sensor task installer is missing")?;
    let output = capture_directory()?.join("live");
    fs::create_dir_all(&output).map_err(|error| error.to_string())?;
    let stop_file = output.join("sensor.stop");
    let sensor = sensor_executable()?;
    let status = background_command("powershell.exe")
        .arg("-NoLogo")
        .arg("-NoProfile")
        .arg("-NonInteractive")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-File")
        .arg(installer)
        .arg("-TaskName")
        .arg(SENSOR_TASK_NAME)
        .arg("-Sensor")
        .arg(sensor)
        .arg("-OutputDirectory")
        .arg(output)
        .arg("-StopFile")
        .arg(stop_file)
        .arg("-SegmentSeconds")
        .arg(LIVE_SEGMENT_SECONDS.to_string())
        .status()
        .map_err(|error| error.to_string())?;
    status
        .success()
        .then_some(())
        .ok_or_else(|| "Argus could not register its native sensor".to_owned())
}

fn sensor_task_is_current(executable: &Path, output: &Path, stop_file: &Path) -> bool {
    let expected_arguments = format!(
        "\"{}\" \"{}\" {LIVE_SEGMENT_SECONDS}",
        output.display(),
        stop_file.display()
    );
    let task_check = format!(
        "$t=Get-ScheduledTask -TaskName {} -ErrorAction SilentlyContinue; if(-not $t){{exit 1}}; $a=@($t.Actions)[0]; if($a.Execute -eq {} -and $a.Arguments -eq {}){{exit 0}}; exit 1",
        quote_powershell_text(SENSOR_TASK_NAME),
        quote_powershell(executable),
        quote_powershell_text(&expected_arguments)
    );
    background_command("powershell.exe")
        .args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            &task_check,
        ])
        .status()
        .is_ok_and(|status| status.success())
}

fn sensor_installer() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(configured) = std::env::var_os("ARGUS_SENSOR_INSTALLER") {
        candidates.push(PathBuf::from(configured));
    }
    if let Ok(current) = std::env::current_exe()
        && let Some(directory) = current.parent()
    {
        candidates.push(directory.join("install-task.ps1"));
    }
    candidates.push(workspace_root().join("native/pktmon-stream/install-task.ps1"));
    candidates.into_iter().find(|path| path.is_file())
}

fn nids_executable() -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    if let Some(configured) = std::env::var_os("ARGUS_NIDS_EXECUTABLE") {
        candidates.push(PathBuf::from(configured));
    }
    if let Ok(current) = std::env::current_exe()
        && let Some(directory) = current.parent()
    {
        candidates.push(directory.join("argus-nids.exe"));
    }
    let workspace = workspace_root();
    candidates.push(workspace.join("target/release/argus-nids.exe"));
    candidates.push(workspace.join("target/debug/argus-nids.exe"));
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .and_then(|path| dunce::canonicalize(&path).ok())
        .ok_or_else(|| "argus-nids.exe is absent; rebuild or reinstall Argus Next".to_owned())
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
async fn try_experimental_cnn(
    path: String,
    predict: Option<bool>,
    boost_mode: Option<bool>,
    worker: tauri::State<'_, Arc<Mutex<CnnWorker>>>,
    monitor: tauri::State<'_, Arc<Mutex<RollingMonitor>>>,
    alerts: tauri::State<'_, Arc<Mutex<AlertManager>>>,
    preferences: tauri::State<'_, Arc<Mutex<RuntimePreferences>>>,
) -> Result<serde_json::Value, String> {
    let disabled_models = preferences
        .lock()
        .map_err(|_| "runtime preferences lock poisoned".to_owned())?
        .disabled_models
        .clone();
    let worker = Arc::clone(worker.inner());
    let monitor = Arc::clone(monitor.inner());
    let response = tauri::async_runtime::spawn_blocking(move || {
        let capture = PathBuf::from(&path);
        let consume_capture = generated_live_capture_path(&capture)?;
        let _capture_guard =
            consume_capture.then(|| TemporaryGeneratedCapture::new([capture.clone()]));
        let history_scope = model_history_scope(&capture, &monitor)?;
        let mut worker = worker.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => {
                "model inference is busy; request was rejected instead of queued".to_owned()
            }
            std::sync::TryLockError::Poisoned(_) => "model worker lock poisoned".to_owned(),
        })?;
        run_experimental_cnn(
            &capture,
            &history_scope,
            predict.unwrap_or(true),
            boost_mode.unwrap_or(false),
            &disabled_models,
            &mut worker,
        )
    })
    .await
    .map_err(|error| error.to_string())??;

    if let Ok(mut alert_mgr) = alerts.lock()
        && let Some(models) = response.get("models").and_then(|m| m.as_object())
    {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX));
        for (name, model_info) in models {
            if operator_forecast_diagnostic_allowed(name, model_info)
                && let Some(error_val) = model_info
                    .get("forecast_error")
                    .and_then(serde_json::Value::as_f64)
                && let Some(alert) =
                    alert_mgr.evaluate_forecast_error(timestamp_ns, error_val, name)
            {
                alert_mgr.record(alert);
            }
        }
    }

    Ok(response)
}

fn operator_forecast_diagnostic_allowed(name: &str, model_info: &serde_json::Value) -> bool {
    name != "ahem_v63_experimental"
        && name != "ahem_e2e_experimental"
        && model_info
            .get("experimental_shadow")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        && model_info.get("state").and_then(serde_json::Value::as_str) == Some("forecast_only")
}

fn run_experimental_cnn(
    capture: &Path,
    history_scope: &str,
    predict: bool,
    boost_mode: bool,
    disabled_models: &[String],
    worker: &mut CnnWorker,
) -> Result<serde_json::Value, String> {
    let pipeline_started = Instant::now();
    if !capture.is_file() {
        return Err("selected capture does not exist".to_owned());
    }
    let model_dir = model_directory();
    let executable = nids_executable()?;
    let output_root = capture_directory()?.join("experimental");
    fs::create_dir_all(&output_root).map_err(|error| error.to_string())?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let features =
        TemporaryRuntimeFeature::new(output_root.join(format!("cnn-runtime-{stamp}.csv")));
    let export_started = Instant::now();
    let export = background_command(&executable)
        .args(["--export-features"])
        .arg(capture)
        .arg(features.path())
        .output()
        .map_err(|error| error.to_string())?;
    if !export.status.success() {
        return Err(format!(
            "runtime feature export failed: {}",
            String::from_utf8_lossy(&export.stderr).trim()
        ));
    }
    let feature_export_ms = export_started.elapsed().as_secs_f64() * 1_000.0;
    let feature_csv_bytes = fs::metadata(features.path())
        .map_err(|error| error.to_string())?
        .len();
    let learning_segment_key = generated_live_capture_key(capture)?;
    worker.select_models(disabled_models);
    let mut result = worker.infer(
        &model_dir,
        features.path(),
        history_scope,
        predict,
        learning_segment_key.as_deref(),
        boost_mode,
    )?;
    if let Some(latency) = result
        .get_mut("latency_breakdown")
        .and_then(serde_json::Value::as_object_mut)
    {
        latency.insert(
            "feature_export_ms".to_owned(),
            serde_json::Value::from(feature_export_ms),
        );
        latency.insert(
            "feature_csv_bytes".to_owned(),
            serde_json::Value::from(feature_csv_bytes),
        );
        latency.insert(
            "analysis_pipeline_ms".to_owned(),
            serde_json::Value::from(pipeline_started.elapsed().as_secs_f64() * 1_000.0),
        );
    }
    Ok(result)
}

fn cleanup_runtime_feature_csvs() -> Result<usize, String> {
    let captures = capture_directory()?;
    let experimental = cleanup_runtime_feature_csvs_in(&captures.join("experimental"))?;
    let live = cleanup_runtime_feature_csvs_in(&captures.join("live"))?;
    Ok(experimental.saturating_add(live))
}

fn cleanup_runtime_feature_csvs_in(directory: &Path) -> Result<usize, String> {
    if !directory.is_dir() {
        return Ok(0);
    }
    let mut removed = 0_usize;
    for entry in fs::read_dir(directory).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        let generated_runtime_csv = path.is_file()
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    let is_csv = Path::new(name)
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("csv"));
                    is_csv
                        && (name.starts_with("cnn-runtime-")
                            || name == "live-features.csv"
                            || (name.starts_with("live-") && name.ends_with("-features.csv")))
                });
        if generated_runtime_csv {
            fs::remove_file(&path).map_err(|error| error.to_string())?;
            removed = removed.saturating_add(1);
        }
    }
    Ok(removed)
}

fn model_history_scope(capture: &Path, monitor: &Mutex<RollingMonitor>) -> Result<String, String> {
    let canonical = dunce::canonicalize(capture).map_err(|error| error.to_string())?;
    let live_directory = capture_directory()?.join("live");
    let is_live_segment = canonical
        .parent()
        .is_some_and(|parent| parent == live_directory.as_path());
    if is_live_segment {
        let mut monitor = monitor.lock().map_err(|_| "monitor lock poisoned")?;
        restore_monitor(&mut monitor)?;
        if !monitor_running(&monitor) {
            return Err("live capture segment has no active owning sensor session".to_owned());
        }
        let process_id = monitor
            .process_id
            .ok_or("live sensor process ID is absent")?;
        return Ok(format!("live-sensor:{process_id}"));
    }
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    Ok(format!("isolated-replay:{nonce}"))
}

#[allow(clippy::too_many_lines)]
fn model_statuses(
    ahem_v61_artifact: &Result<String, String>,
    ahem_v63_experimental: &Result<String, String>,
) -> Vec<ModelStatus> {
    const MODELS: [(&str, &str, &str); 16] = [
        (
            "cnn-local",
            "cnn_local_real_pcap_30k.pt",
            "compatible temporal forecaster",
        ),
        (
            "ahem-fixed",
            "ahem_fixed_real_pcap_30k.pt",
            "compatible temporal forecaster",
        ),
        (
            "lstm",
            "lstm_real_pcap_30k.pt",
            "compatible temporal forecaster",
        ),
        (
            "tcn-long",
            "tcn_long_real_pcap_30k.pt",
            "compatible temporal forecaster",
        ),
        (
            "ahem-v3.1",
            "ahem_v31_forecast_only.pt",
            "incremental cached next-packet forecaster; never a security verdict",
        ),
        (
            "ahem-v6.1",
            AHEM_V61_FILE,
            "frozen supported next-same-flow forecaster; segment-local cache; never a security verdict",
        ),
        (
            "ahem-v6.3-shadow",
            AHEM_V63_EXPERIMENTAL_FILE,
            "source-disjoint experimental next-same-flow shadow; failed promotion consistency gate",
        ),
        (
            "cnn-cse",
            "cnn_cse_cic_ids2018_best.pt",
            "real-journal local temporal forecaster; never a security verdict",
        ),
        (
            "lstm-cse",
            "lstm_cse_cic_ids2018_best.pt",
            "real-journal 30k recurrent forecaster; never a security verdict",
        ),
        (
            "gru-cse",
            "gru_cse_cic_ids2018_best.pt",
            "real-journal 30k recurrent forecaster; never a security verdict",
        ),
        (
            "tcn-cse",
            "tcn_cse_cic_ids2018_best.pt",
            "real-journal 30k causal convolution forecaster; never a security verdict",
        ),
        (
            "ahem-flash",
            "ahem_flash_cse_cic_ids2018_best.pt",
            "real-journal bounded transition-memory forecaster; retrieval under audit",
        ),
        (
            "ahem-pro",
            "ahem_pro_cse_cic_ids2018_best.pt",
            "real-journal bounded transition-memory forecaster; retrieval under audit",
        ),
        (
            "flow-mlp",
            "flow_mlp_real_cic_ids2018.pt",
            "retrospective schema only",
        ),
        (
            "temporal-router",
            "temporal_router_real_live.pt",
            "causal forecast-loss router; not a security detector",
        ),
        (
            "flow-detector-calibrated",
            "flow_detector_calibrated_cse_cic_ids2018.pt",
            "offline only: held-out ranking passed, cross-family calibration failed",
        ),
    ];
    let directory = model_directory();
    MODELS
        .into_iter()
        .map(|(id, file_name, present_detail)| {
            let present = directory.join(file_name).is_file();
            if id == "ahem-v6.1" {
                return match ahem_v61_artifact {
                    Ok(hash) => ModelStatus {
                        id,
                        file_name,
                        state: "hash-verified",
                        detail: format!(
                            "checkpoint hash verified ({}); runtime loading is not yet checked",
                            &hash[..12]
                        ),
                    },
                    Err(error) => ModelStatus {
                        id,
                        file_name,
                        state: if present {
                            "quarantined"
                        } else {
                            "unavailable"
                        },
                        detail: error.clone(),
                    },
                };
            }
            if id == "ahem-v6.3-shadow" {
                return match ahem_v63_experimental {
                    Ok(hash) => ModelStatus {
                        id,
                        file_name,
                        state: "experimental-shadow",
                        detail: format!(
                            "hash-verified challenger {}; never routed or promoted",
                            &hash[..12]
                        ),
                    },
                    Err(error) => ModelStatus {
                        id,
                        file_name,
                        state: if present {
                            "quarantined"
                        } else {
                            "unavailable"
                        },
                        detail: error.clone(),
                    },
                };
            }
            ModelStatus {
                id,
                file_name,
                state: if present {
                    match id {
                        "flow-mlp" => "rejected",
                        "flow-detector-calibrated" => "offline-only",
                        _ => "artifact-present",
                    }
                } else {
                    "unavailable"
                },
                detail: if present {
                    format!(
                        "{present_detail}; file present only, runtime loading is not yet checked"
                    )
                } else {
                    "artifact not found".to_owned()
                },
            }
        })
        .collect()
}

fn model_directory() -> PathBuf {
    if let Some(configured) = std::env::var_os("ARGUS_MODEL_DIR") {
        return PathBuf::from(configured);
    }
    let managed = development_root().map_or_else(
        |_| workspace_root().join("models"),
        |root| root.join("models"),
    );
    if managed.is_dir() {
        return managed;
    }
    if let Some(packaged) = std::env::current_exe()
        .ok()
        .and_then(|executable| executable.parent().map(|parent| parent.join("models")))
        .filter(|path| path.is_dir())
    {
        return packaged;
    }
    let workspace_models = workspace_root().join("models");
    if workspace_models.is_dir() {
        workspace_models
    } else {
        managed
    }
}

fn capture_and_replay() -> Result<RunView, String> {
    let capture_root = capture_directory()?;
    fs::create_dir_all(&capture_root).map_err(|error| error.to_string())?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_secs();
    let etl = capture_root.join(format!("argus-{stamp}.etl"));
    let pcapng = capture_root.join(format!("argus-{stamp}.pcapng"));
    let _capture_guard = TemporaryGeneratedCapture::new([etl.clone(), pcapng.clone()]);
    let executable = nids_executable()?;
    let script = format!(
        "$p=Start-Process -FilePath {} -Verb RunAs -WindowStyle Hidden -ArgumentList @('--pktmon-record','30',{},{}) -PassThru -Wait; exit $p.ExitCode",
        quote_powershell(&executable),
        quote_powershell(&etl),
        quote_powershell(&pcapng)
    );
    let status = background_command("powershell.exe")
        .args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            &script,
        ])
        .status()
        .map_err(|error| error.to_string())?;
    if !status.success() {
        return Err(format!("elevated Pktmon capture failed with {status}"));
    }
    replay_path(&pcapng)
}

fn replay_path(path: &Path) -> Result<RunView, String> {
    let limits = PcapLimits {
        max_file_bytes: NonZeroUsize::new(512 * 1024 * 1024).expect("non-zero literal"),
        max_records: NonZeroUsize::new(5_000_000).expect("non-zero literal"),
    };
    let mut source =
        PcapReplaySource::from_path(path, limits).map_err(|error| error.to_string())?;
    let report = Pipeline::new(PipelineConfig::default())
        .run(&mut source as &mut dyn CaptureSource)
        .map_err(|error| error.to_string())?;
    Ok(run_view(report))
}

fn run_view(report: RunReport) -> RunView {
    let provenance = report.detector.provenance;
    let source = report.source.source_id;
    let session = report.session.session_id;
    let blacklist = BlacklistManager::load();
    let mut flows: Vec<FlowView> = report
        .flows
        .iter()
        .map(|summary| flow_view(summary, &session, &source, &blacklist))
        .collect();
    apply_attack_detection(&mut flows, &blacklist);
    RunView {
        source: source.clone(),
        session: session.clone(),
        detector: DetectorView {
            id: provenance.detector_id.to_owned(),
            trained: report.detector.trained_model,
            feature_schema: format!(
                "{}/v{}",
                provenance.feature_schema.schema_id, provenance.feature_schema.schema_version
            ),
            artifact: provenance.artifact_id.map(str::to_owned),
        },
        health: HealthView {
            source_records: report.health.source_records,
            normalized_packets: report.health.normalized_packets,
            processed_packets: report.health.processed_packets,
            flows: report.health.flows,
            alerts: report.health.alerts,
            unsupported_records: report.health.unsupported_records,
            malformed_records: report.health.malformed_records,
            reported_lost_packets: report.health.reported_lost_packets,
            truncated_packets: report.health.truncated_packets,
            reordered_packets: report.health.reordered_packets,
            queue_overflows: report.health.queue_overflows,
            parse_errors: report.health.capture_parse_errors,
            source_errors: report.health.capture_source_errors,
        },
        flows,
    }
}

fn classify_service(initiator_port: u16, responder_port: u16) -> Option<String> {
    let check = |port: u16| -> Option<&'static str> {
        match port {
            53 | 853 | 5353 => Some("DNS"),
            80 | 8080 | 8000 => Some("HTTP"),
            443 | 8443 => Some("HTTPS"),
            22 => Some("SSH"),
            123 => Some("NTP"),
            67 | 68 => Some("DHCP"),
            1900 => Some("SSDP"),
            445 => Some("SMB"),
            3389 => Some("RDP"),
            5355 => Some("LLMNR"),
            137..=139 => Some("NetBIOS"),
            _ => None,
        }
    };
    check(responder_port)
        .or_else(|| check(initiator_port))
        .map(ToOwned::to_owned)
}

fn capture_quality_status(lost_packets: u64, truncated_packets: u64) -> Option<String> {
    if lost_packets == 0 && truncated_packets == 0 {
        None
    } else {
        Some(format!(
            "Capture Quality Loss ({lost_packets} lost, {truncated_packets} truncated)"
        ))
    }
}

#[allow(clippy::too_many_arguments)] // Keeps each evidence-backed verdict construction explicit.
fn attack_verdict(
    flow: &FlowView,
    category: &str,
    severity: &str,
    score: f64,
    threshold: f64,
    summary: &str,
    detail: String,
    evidence: Vec<String>,
) -> AttackVerdictDto {
    AttackVerdictDto {
        category: category.to_owned(),
        severity: severity.to_owned(),
        score,
        threshold,
        summary: summary.to_owned(),
        detail,
        detector: "argus.deterministic-attack-rules/v1".to_owned(),
        evidence,
        confidence: if flow.lost_packets == 0 && flow.truncated_packets == 0 {
            "high"
        } else {
            "degraded_capture"
        }
        .to_owned(),
    }
}

/// Applies auditable, online-safe attack signatures to completed causal flow summaries.
/// AHEM forecasts are deliberately excluded: they remain forecast-only until a separately
/// frozen and validated forecast-to-detection bridge is admitted.
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)] // Counts are capture-bounded and used only for display scores.
fn apply_attack_detection(flows: &mut [FlowView], rules: &BlacklistManager) {
    const SCAN_WINDOW_NS: u64 = 60_000_000_000;
    struct ScanProfile {
        destinations: HashSet<String>,
        ports: HashSet<u16>,
        syn_packets: u64,
        trigger_uid: String,
        trigger_timestamp_ns: u64,
        trigger_flow_id: u64,
    }

    let mut session_ends: HashMap<&str, u64> = HashMap::new();
    for flow in flows.iter() {
        session_ends
            .entry(&flow.session)
            .and_modify(|timestamp| *timestamp = (*timestamp).max(flow.last_timestamp_ns))
            .or_insert(flow.last_timestamp_ns);
    }
    let mut scan_profiles: HashMap<(String, String), ScanProfile> = HashMap::new();
    for flow in flows.iter() {
        let session_end = session_ends
            .get(flow.session.as_str())
            .copied()
            .unwrap_or(0);
        let in_scan_window = flow.last_timestamp_ns.saturating_add(SCAN_WINDOW_NS) >= session_end;
        if flow.protocol == "TCP"
            && flow.tcp_syn_packets > 0
            && in_scan_window
            && !rules.is_allowlisted(&flow.initiator_address)
        {
            let profile = scan_profiles
                .entry((flow.session.clone(), flow.initiator_address.clone()))
                .or_insert_with(|| ScanProfile {
                    destinations: HashSet::new(),
                    ports: HashSet::new(),
                    syn_packets: 0,
                    trigger_uid: flow.uid.clone(),
                    trigger_timestamp_ns: flow.last_timestamp_ns,
                    trigger_flow_id: flow.id,
                });
            profile.destinations.insert(flow.responder_address.clone());
            profile.ports.insert(flow.responder_port);
            profile.syn_packets = profile.syn_packets.saturating_add(flow.tcp_syn_packets);
            if (flow.last_timestamp_ns, flow.id)
                > (profile.trigger_timestamp_ns, profile.trigger_flow_id)
            {
                profile.trigger_uid.clone_from(&flow.uid);
                profile.trigger_timestamp_ns = flow.last_timestamp_ns;
                profile.trigger_flow_id = flow.id;
            }
        }
    }

    for flow in flows {
        flow.attack_verdict = None;
        if flow.is_blocklisted {
            let target = if rules.is_blocklisted(&flow.initiator_address) {
                &flow.initiator_address
            } else {
                &flow.responder_address
            };
            flow.attack_verdict = Some(attack_verdict(
                flow,
                "blocklist_match",
                "critical",
                1.0,
                1.0,
                "Explicit blocklist match",
                format!("Endpoint {target} matched an active analyst block rule"),
                vec![format!("blocklisted_endpoint={target}")],
            ));
            continue;
        }
        // Attack signatures describe initiator behavior. An allow rule for the
        // responder must not suppress evidence from an unrelated initiator
        // merely because the protected service is trusted.
        if rules.is_allowlisted(&flow.initiator_address) {
            continue;
        }

        let invalid_flags = flow
            .tcp_syn_fin_packets
            .saturating_add(flow.tcp_xmas_packets)
            .saturating_add(flow.tcp_syn_rst_packets);
        if invalid_flags > 0 {
            flow.attack_verdict = Some(attack_verdict(
                flow,
                "tcp_evasion",
                "critical",
                invalid_flags as f64,
                1.0,
                "TCP evasion signature detected",
                "One or more packets carried impossible or scan-associated TCP flag combinations"
                    .to_owned(),
                vec![
                    format!("syn_fin_packets={}", flow.tcp_syn_fin_packets),
                    format!("xmas_packets={}", flow.tcp_xmas_packets),
                    format!("syn_rst_packets={}", flow.tcp_syn_rst_packets),
                ],
            ));
            continue;
        }
        if flow.tcp_null_packets > 0 {
            flow.attack_verdict = Some(attack_verdict(
                flow,
                "tcp_null_probe",
                "suspicious",
                flow.tcp_null_packets as f64,
                1.0,
                "TCP NULL probe signature detected",
                "TCP packets with no control flags were observed".to_owned(),
                vec![format!("null_flag_packets={}", flow.tcp_null_packets)],
            ));
            continue;
        }
        if let Some(profile) =
            scan_profiles.get(&(flow.session.clone(), flow.initiator_address.clone()))
            && profile.trigger_uid == flow.uid
            && profile.syn_packets >= 20
            && (profile.destinations.len() >= 20 || profile.ports.len() >= 20)
        {
            let breadth = profile.destinations.len().max(profile.ports.len());
            flow.attack_verdict = Some(attack_verdict(
                flow,
                "tcp_scan",
                "critical",
                breadth as f64,
                20.0,
                "TCP scan pattern detected",
                format!(
                    "Initiator contacted {} distinct hosts and {} distinct ports using {syn_packets} SYN packets in this capture window",
                    profile.destinations.len(),
                    profile.ports.len(),
                    syn_packets = profile.syn_packets,
                ),
                vec![
                    format!("distinct_destinations={}", profile.destinations.len()),
                    format!("distinct_ports={}", profile.ports.len()),
                    format!("syn_packets={}", profile.syn_packets),
                    "window_seconds=60".to_owned(),
                ],
            ));
            continue;
        }
        let seconds = flow.duration_ms / 1_000.0;
        let packets_per_second = if seconds > 0.0 {
            flow.packets as f64 / seconds
        } else {
            0.0
        };
        if flow.packets >= 100 && packets_per_second >= 10_000.0 {
            flow.attack_verdict = Some(attack_verdict(
                flow,
                "volumetric_burst",
                "suspicious",
                packets_per_second,
                10_000.0,
                "High-rate packet burst detected",
                format!(
                    "Observed {packets_per_second:.0} packets/s across {} packets",
                    flow.packets
                ),
                vec![
                    format!("packets={}", flow.packets),
                    format!("packets_per_second={packets_per_second:.0}"),
                ],
            ));
        }
    }
}

fn matched_flow_rule_target(
    blacklist: &BlacklistManager,
    initiator: &str,
    responder: &str,
) -> Option<String> {
    let enforced_rule = [
        blacklist.enforced_rule(initiator),
        blacklist.enforced_rule(responder),
    ]
    .into_iter()
    .flatten()
    .max_by_key(|rule| rule_target_specificity(&rule.target).unwrap_or(0));
    let policy_rule = [
        blacklist.active_rule(initiator),
        blacklist.active_rule(responder),
    ]
    .into_iter()
    .flatten()
    .max_by_key(|rule| match rule.action {
        TrafficRuleAction::Watch => 0,
        TrafficRuleAction::Allow => 1,
        TrafficRuleAction::Block => 2,
    });
    enforced_rule
        .or(policy_rule)
        .map(|rule| rule.target.clone())
}

fn flow_view(
    summary: &FlowSummary,
    session: &str,
    source: &str,
    blacklist: &BlacklistManager,
) -> FlowView {
    let init_addr = summary.flow.key.initiator.address.to_string();
    let resp_addr = summary.flow.key.responder.address.to_string();
    let is_watchlisted =
        blacklist.is_watchlisted(&init_addr) || blacklist.is_watchlisted(&resp_addr);
    let is_blocked = blacklist.is_blocked(&init_addr) || blacklist.is_blocked(&resp_addr);
    let is_allowlisted =
        blacklist.is_allowlisted(&init_addr) || blacklist.is_allowlisted(&resp_addr);
    let is_blocklisted =
        blacklist.is_blocklisted(&init_addr) || blacklist.is_blocklisted(&resp_addr);
    let rule_action = [
        blacklist.rule_action(&init_addr),
        blacklist.rule_action(&resp_addr),
    ]
    .into_iter()
    .flatten()
    .max_by_key(|action| match action {
        TrafficRuleAction::Watch => 0,
        TrafficRuleAction::Allow => 1,
        TrafficRuleAction::Block => 2,
    })
    .map(|action| match action {
        TrafficRuleAction::Allow => "allow",
        TrafficRuleAction::Watch => "watch",
        TrafficRuleAction::Block => "block",
    })
    .map(str::to_owned);
    let matched_rule_target = matched_flow_rule_target(blacklist, &init_addr, &resp_addr);

    let protocol = match summary.flow.key.protocol {
        TransportProtocol::Tcp => "TCP",
        TransportProtocol::Udp => "UDP",
        TransportProtocol::Other(1) => "ICMP",
        TransportProtocol::Other(58) => "ICMPv6",
        TransportProtocol::Other(_) => "OTHER",
    };
    let service = classify_service(
        summary.flow.key.initiator.port,
        summary.flow.key.responder.port,
    );
    let threat_status =
        capture_quality_status(summary.reported_lost_packets, summary.truncated_packets);
    FlowView {
        uid: format!("{session}:{}", summary.id),
        session: session.to_owned(),
        source: source.to_owned(),
        id: summary.id,
        protocol: protocol.to_owned(),
        service,
        threat_status,
        generation: summary.flow.generation,
        initiator_address: summary.flow.key.initiator.address.to_string(),
        initiator_port: summary.flow.key.initiator.port,
        responder_address: summary.flow.key.responder.address.to_string(),
        responder_port: summary.flow.key.responder.port,
        first_timestamp_ns: summary.first_timestamp_ns,
        last_timestamp_ns: summary.last_timestamp_ns,
        duration_ms: std::time::Duration::from_nanos(
            summary
                .last_timestamp_ns
                .saturating_sub(summary.first_timestamp_ns),
        )
        .as_secs_f64()
            * 1_000.0,
        packets: summary.packets,
        captured_bytes: summary.captured_bytes,
        original_bytes: summary.original_bytes,
        initiator_packets: summary.initiator_packets,
        responder_packets: summary.responder_packets,
        initiator_bytes: summary.initiator_bytes,
        responder_bytes: summary.responder_bytes,
        initiator_mac: summary.initiator_mac,
        responder_mac: summary.responder_mac,
        tls_sni: summary.tls_sni.clone(),
        dns_qname: summary.dns_qname.clone(),
        dns_messages: summary.dns_messages,
        dns_queries: summary.dns_queries,
        http_requests: summary.http_requests,
        tor_port_matches: summary.tor_port_matches,
        tcp_syn_packets: summary.tcp_syn_packets,
        tcp_syn_fin_packets: summary.tcp_syn_fin_packets,
        tcp_xmas_packets: summary.tcp_xmas_packets,
        tcp_null_packets: summary.tcp_null_packets,
        tcp_syn_rst_packets: summary.tcp_syn_rst_packets,
        lost_packets: summary.reported_lost_packets,
        truncated_packets: summary.truncated_packets,
        reordered_packets: summary.reordered_packets,
        is_watchlisted,
        is_blocked,
        is_allowlisted,
        is_blocklisted,
        rule_action,
        matched_rule_target,
        attack_verdict: None,
    }
}

fn workspace_root() -> PathBuf {
    let executable = std::env::current_exe()
        .expect("cannot locate this executable; refusing working-directory resource discovery");
    application_root_from_executable(&executable)
}

fn application_root_from_executable(executable: &Path) -> PathBuf {
    assert!(
        executable.is_absolute(),
        "application executable path must be absolute"
    );
    let directory = executable
        .parent()
        .expect("application executable has a parent");
    // Only ascend from a Cargo output directory with the expected checkout
    // markers. A copied/packaged executable otherwise uses its own directory.
    for candidate in directory.ancestors().take(4) {
        let relative = directory.strip_prefix(candidate).expect("ancestor prefix");
        if matches!(
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
            && candidate
                .join("desktop/src-tauri/tauri.conf.json")
                .is_file()
            && candidate.join("python/pyproject.toml").is_file()
        {
            return candidate.to_path_buf();
        }
    }
    directory.to_path_buf()
}

fn development_root() -> Result<PathBuf, String> {
    let root = if let Some(configured) = std::env::var_os("ARGUS_DEV_ROOT") {
        let configured = PathBuf::from(configured);
        if !configured.is_absolute() {
            return Err("ARGUS_DEV_ROOT must be an absolute path".to_owned());
        }
        configured
    } else {
        let local = std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .ok_or("LOCALAPPDATA is unavailable")?;
        local.join("ArgusNextPublic")
    };
    fs::create_dir_all(&root).map_err(|error| error.to_string())?;
    dunce::canonicalize(&root).map_err(|error| error.to_string())
}

fn capture_directory() -> Result<PathBuf, String> {
    let captures = if let Some(configured) = std::env::var_os("ARGUS_CAPTURE_ROOT") {
        let configured = PathBuf::from(configured);
        if !configured.is_absolute() {
            return Err("ARGUS_CAPTURE_ROOT must be an absolute path".to_owned());
        }
        configured
    } else {
        let local = std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .ok_or("LOCALAPPDATA is unavailable")?;
        local.join("ArgusNextPublic/captures")
    };
    fs::create_dir_all(&captures).map_err(|error| error.to_string())?;
    dunce::canonicalize(&captures).map_err(|error| error.to_string())
}

fn live_history_path() -> Result<PathBuf, String> {
    Ok(development_root()?.join("state/live-flow-history.json"))
}

fn daily_learning_directory() -> Result<PathBuf, String> {
    Ok(development_root()?.join("learning"))
}

fn generated_live_capture_key(path: &Path) -> Result<Option<String>, String> {
    if !generated_live_capture_path(path)? || !path.is_file() {
        return Ok(None);
    }
    let canonical = dunce::canonicalize(path).map_err(|error| error.to_string())?;
    Ok(Some(canonical.display().to_string()))
}

fn generated_live_capture_path(path: &Path) -> Result<bool, String> {
    generated_live_capture_path_in(path, &capture_directory()?.join("live"))
}

fn generated_live_capture_path_in(path: &Path, live: &Path) -> Result<bool, String> {
    let parent = path
        .parent()
        .ok_or("capture path has no parent directory")?;
    let parent = dunce::canonicalize(parent).map_err(|error| error.to_string())?;
    let live = match dunce::canonicalize(live) {
        Ok(live) => live,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.to_string()),
    };
    Ok(parent == live
        && path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("pcap"))
        && path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("live-")))
}

fn replace_file(path: &Path, contents: &[u8]) -> Result<(), String> {
    let temporary = path.with_extension("json.tmp");
    let backup = path.with_extension("json.bak");
    fs::write(&temporary, contents).map_err(|error| error.to_string())?;
    if backup.exists() {
        fs::remove_file(&backup).map_err(|error| error.to_string())?;
    }
    if path.exists() {
        fs::rename(path, &backup).map_err(|error| error.to_string())?;
    }
    if let Err(error) = fs::rename(&temporary, path) {
        if backup.exists() {
            let _ = fs::rename(&backup, path);
        }
        return Err(error.to_string());
    }
    if backup.exists() {
        fs::remove_file(backup).map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn quote_powershell(path: &Path) -> String {
    quote_powershell_text(&path.display().to_string())
}

fn quote_powershell_text(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn reports_directory() -> Result<PathBuf, String> {
    let reports = development_root()?.join("reports");
    fs::create_dir_all(&reports).map_err(|error| error.to_string())?;
    dunce::canonicalize(&reports).map_err(|error| error.to_string())
}

fn sha256_bytes(contents: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(contents);
    format!("{:x}", hasher.finalize())
}

fn encode_live_session_export(
    history: &LiveHistory,
    generated_at_unix_ms: u64,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    if let Some(error) = &history.load_error {
        return Err(error.clone());
    }
    let json = serde_json::to_vec_pretty(&serde_json::json!({
        "schema": "argus.live-session-export/v1",
        "generatedAtUnixMs": generated_at_unix_ms,
        "rawCaptureIncluded": false,
        "retention": {
            "flowCapacity": LIVE_HISTORY_CAPACITY,
            "sampleCapacity": LIVE_SEGMENT_SAMPLE_CAPACITY,
            "segmentSeconds": LIVE_SEGMENT_SECONDS,
        },
        "totals": {
            "segments": history.segments,
            "packets": history.packets,
            "bytes": history.bytes,
            "retainedFlows": history.flows.len(),
            "evictedFlows": history.evicted_flows,
        },
        "counterSamples": &history.samples,
        "flows": &history.flows,
    }))
    .map_err(|error| error.to_string())?;

    let mut writer = csv::WriterBuilder::new()
        .has_headers(true)
        .from_writer(Vec::new());
    for flow in &history.flows {
        writer.serialize(flow).map_err(|error| error.to_string())?;
    }
    let csv = writer
        .into_inner()
        .map_err(|error| error.into_error().to_string())?;
    Ok((json, csv))
}

#[tauri::command]
async fn export_live_session(
    history: tauri::State<'_, Arc<Mutex<LiveHistory>>>,
) -> Result<SessionExportDto, String> {
    let generated_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| {
            u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
        });
    let (json, csv, flow_count, sample_count) = {
        let history = history
            .lock()
            .map_err(|_| "live history lock poisoned".to_owned())?;
        let (json, csv) = encode_live_session_export(&history, generated_at_unix_ms)?;
        (json, csv, history.flows.len(), history.samples.len())
    };
    let export_id = format!("ARGUS-SESSION-{generated_at_unix_ms}");
    let directory = reports_directory()?;
    let json_path = directory.join(format!("{export_id}.json"));
    let csv_path = directory.join(format!("{export_id}-flows.csv"));
    fs::write(&json_path, &json).map_err(|error| error.to_string())?;
    if let Err(error) = fs::write(&csv_path, &csv) {
        let _ = fs::remove_file(&json_path);
        return Err(error.to_string());
    }
    Ok(SessionExportDto {
        export_id,
        json_path: json_path.display().to_string(),
        csv_path: csv_path.display().to_string(),
        json_sha256: sha256_bytes(&json),
        csv_sha256: sha256_bytes(&csv),
        flow_count,
        sample_count,
    })
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::too_many_lines)]
async fn export_incident_report(
    flow_id: Option<u64>,
    notes: Option<String>,
    alerts: tauri::State<'_, Arc<Mutex<AlertManager>>>,
    history: tauri::State<'_, Arc<Mutex<LiveHistory>>>,
) -> Result<IncidentReportDto, String> {
    use std::fmt::Write as _;

    let alert_items = alerts
        .lock()
        .map_err(|_| "AlertManager lock poisoned")?
        .alerts
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    let flows = history
        .lock()
        .map_err(|_| "LiveHistory lock poisoned")?
        .view()
        .map(|v| v.flows)
        .unwrap_or_default();

    let reports_dir = reports_directory()?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let report_id = format!("ARGUS-INCIDENT-{now}");
    let report_file = reports_dir.join(format!("{report_id}.md"));

    let critical_count = alert_items
        .iter()
        .filter(|a| a.severity.eq_ignore_ascii_case("critical"))
        .count();
    let suspicious_count = alert_items
        .iter()
        .filter(|a| a.severity.eq_ignore_ascii_case("suspicious"))
        .count();
    let rule_matches = alert_items
        .iter()
        .filter(|a| a.detector == "argus.deterministic-attack-rules/v1")
        .count();
    let critical_rule_matches = alert_items
        .iter()
        .filter(|a| {
            a.detector == "argus.deterministic-attack-rules/v1"
                && a.severity.eq_ignore_ascii_case("critical")
        })
        .count();
    let diagnostic_events = alert_items.len().saturating_sub(rule_matches);

    let mut md = String::new();
    md.push_str("# ARGUS NIDS — SECURITY EVIDENCE REPORT\n\n");
    let _ = writeln!(md, "- **Report Identifier:** `{report_id}`");
    let _ = writeln!(md, "- **Generated Timestamp:** Unix epoch {now}s");
    md.push_str("- **Runtime:** Argus deterministic attack rules, endpoint policy, capture quality, and forecast diagnostics\n");
    md.push_str(
        "- **Verdict Contract:** `argus.deterministic-attack-rules/v1` events are auditable signature or analyst-rule matches; AHEM forecast residuals remain diagnostics and are not attack verdicts\n",
    );
    if let Some(user_notes) = notes.as_ref().filter(|n| !n.trim().is_empty()) {
        let _ = writeln!(md, "- **Analyst Assessment Notes:** {user_notes}");
    }
    md.push_str("\n---\n\n");

    md.push_str("## 1. Executive Summary\n\n");
    let _ = writeln!(md, "- **Total Monitored Flows:** {}", flows.len());
    let _ = writeln!(
        md,
        "- **Total Security and Diagnostic Events:** {}",
        alert_items.len()
    );
    let _ = writeln!(
        md,
        "- **Deterministic Rule Matches:** {rule_matches} ({critical_rule_matches} critical)"
    );
    let _ = writeln!(
        md,
        "- **Forecast / Quality Diagnostics:** {diagnostic_events}"
    );
    let _ = writeln!(
        md,
        "- **High-Severity Events of Any Kind:** {critical_count}"
    );
    let _ = writeln!(
        md,
        "- **Review-Severity Events of Any Kind:** {suspicious_count}"
    );
    let posture_str = if critical_rule_matches > 0 {
        "CRITICAL DETERMINISTIC RULE MATCH — ANALYST REVIEW REQUIRED"
    } else if rule_matches > 0 {
        "DETERMINISTIC RULE MATCH — ANALYST REVIEW REQUIRED"
    } else if diagnostic_events > 0 {
        "DIAGNOSTIC DEVIATION ONLY — NOT AN ATTACK VERDICT"
    } else {
        "NO RECORDED EVENTS — NOT A BENIGN-TRAFFIC VERDICT"
    };
    let _ = write!(md, "- **Diagnostic Posture:** {posture_str}\n\n");

    md.push_str("## 2. Security Evidence Timeline\n\n");
    if alert_items.is_empty() {
        md.push_str(
            "_No security or diagnostic events recorded during this observation window._\n\n",
        );
    } else {
        md.push_str("| Timestamp | Severity | Category | Detector | Score | Threshold | Flow | Summary | Evidence |\n");
        md.push_str("|---|---|---|---|---|---|---|---|---|\n");
        for alert in &alert_items {
            let flow_str = alert.flow_endpoint.as_deref().unwrap_or("System/Global");
            let _ = writeln!(
                md,
                "| {} | **{}** | {} | `{}` | {:.2} | {:.2} | `{}` | {} | {} |",
                alert.timestamp_formatted,
                alert.severity,
                alert.category,
                alert.detector,
                alert.score,
                alert.threshold,
                flow_str,
                alert.summary,
                alert.evidence.join("; ")
            );
        }
        md.push('\n');
    }

    md.push_str("## 3. Monitored Flow Inventory & Service Classification\n\n");
    if flows.is_empty() {
        md.push_str("_No active flow records in bounded live memory._\n\n");
    } else {
        let display_flows = if let Some(target_id) = flow_id {
            flows
                .iter()
                .filter(|f| f.id == target_id)
                .collect::<Vec<_>>()
        } else {
            flows.iter().take(50).collect::<Vec<_>>()
        };
        md.push_str("| Flow ID | Protocol | Service | Initiator | Responder | Packets | Bytes | Quality |\n");
        md.push_str("|---|---|---|---|---|---|---|---|\n");
        for flow in display_flows {
            let service_str = flow.service.as_deref().unwrap_or("—");
            let quality = flow.lost_packets + flow.truncated_packets + flow.reordered_packets;
            let quality_str = if quality == 0 {
                "Clean".to_string()
            } else {
                format!("{quality} issues")
            };
            let _ = writeln!(
                md,
                "| {} | {} | {} | `{}:{}` | `{}:{}` | {} | {} | {} |",
                flow.id,
                flow.protocol,
                service_str,
                flow.initiator_address,
                flow.initiator_port,
                flow.responder_address,
                flow.responder_port,
                flow.packets,
                flow.original_bytes,
                quality_str
            );
        }
        md.push('\n');
    }

    md.push_str("## 4. Cryptographic Provenance & Chain of Custody\n\n");
    let mut hasher = Sha256::new();
    hasher.update(md.as_bytes());
    let report_sha256 = format!("{:x}", hasher.finalize());
    let _ = writeln!(md, "- **Report Content SHA-256:** `{report_sha256}`");
    md.push_str("- **Fail-Closed Verification:** All calculations derived strictly from deterministic live memory with zero mock data.\n");

    fs::write(&report_file, &md).map_err(|e| e.to_string())?;

    Ok(IncidentReportDto {
        report_id,
        generated_at: format!("{now}"),
        report_path: report_file.to_string_lossy().into_owned(),
        sha256_hash: report_sha256,
        total_flows: flows.len(),
        total_alerts: alert_items.len(),
        critical_alerts: critical_count,
        suspicious_alerts: suspicious_count,
        markdown_content: md,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DossierReportDto {
    report_id: String,
    generated_at: String,
    report_path: String,
    sha256_hash: String,
    total_nodes: usize,
    total_edges: usize,
    total_bytes: u64,
    total_packets: u64,
    tor_nodes_count: usize,
    malicious_nodes_count: usize,
    html_content: String,
}

fn build_topology_from_flows(flows: &[FlowView]) -> NetworkTopology {
    let records: Vec<FlowTopologyRecord> = flows
        .iter()
        .map(|flow| {
            let is_malicious =
                !flow.is_allowlisted && (flow.is_blocklisted || flow.attack_verdict.is_some());
            let threat_score = if flow.is_blocklisted
                || flow
                    .attack_verdict
                    .as_ref()
                    .is_some_and(|verdict| verdict.severity == "critical")
            {
                1.0_f32
            } else if flow.attack_verdict.is_some() {
                0.7_f32
            } else {
                0.0_f32
            };
            FlowTopologyRecord {
                src_ip: flow.initiator_address.clone(),
                dst_ip: flow.responder_address.clone(),
                src_port: flow.initiator_port,
                dst_port: flow.responder_port,
                protocol: flow.protocol.clone(),
                src_mac: flow.initiator_mac,
                dst_mac: flow.responder_mac,
                bytes: flow.captured_bytes.max(flow.original_bytes),
                initiator_to_responder_bytes: flow.initiator_bytes,
                responder_to_initiator_bytes: flow.responder_bytes,
                packet_count: flow.packets,
                dns_queries: flow.dns_queries,
                http_requests: flow.http_requests,
                tor_port_matches: flow.tor_port_matches,
                threat_score,
                is_malicious,
                sni: flow.tls_sni.clone(),
                dns_qname: flow.dns_qname.clone(),
            }
        })
        .collect();

    argus_capture::build_network_topology(&records)
}

#[allow(clippy::cast_precision_loss)]
fn format_bytes_human(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GiB", b / GB)
    } else if b >= MB {
        format!("{:.2} MiB", b / MB)
    } else if b >= KB {
        format!("{:.2} KiB", b / KB)
    } else {
        format!("{bytes} B")
    }
}

fn escape_html(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            _ => escaped.push(character),
        }
    }
    escaped
}

#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args
)]
fn format_network_dossier_html(
    report_id: &str,
    timestamp: u64,
    topology: &NetworkTopology,
    flows: &[FlowView],
    _alerts: &[AnomalyAlertDto],
) -> String {
    use std::fmt::Write as _;

    let mut html = String::with_capacity(16_384);
    html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
    html.push_str("<meta charset=\"UTF-8\">\n<title>Argus X-Ray Network Dossier</title>\n");
    html.push_str("<style>\n");
    html.push_str(
        "body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",Helvetica,Arial,sans-serif;margin:0;padding:24px;line-height:1.5}\n\
        .header{border-bottom:1px solid #30363d;padding-bottom:16px;margin-bottom:24px}\n\
        .header h1{color:#58a6ff;margin:0 0 8px 0;font-size:24px;letter-spacing:-0.5px}\n\
        .meta{color:#8b949e;font-size:13px}\n\
        .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:32px}\n\
        .card{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:16px}\n\
        .card .title{color:#8b949e;font-size:12px;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px}\n\
        .card .value{font-size:24px;font-weight:600;color:#f0f6fc}\n\
        .card.alert .value{color:#f85149}\n\
        .card.warning .value{color:#d29922}\n\
        .card.success .value{color:#3fb950}\n\
        h2{color:#f0f6fc;font-size:18px;border-bottom:1px solid #21262d;padding-bottom:8px;margin-top:32px}\n\
        table{width:100%;border-collapse:collapse;margin-top:12px;background:#161b22;border-radius:6px;overflow:hidden;border:1px solid #30363d}\n\
        th,td{padding:10px 14px;text-align:left;font-size:13px;border-bottom:1px solid #21262d}\n\
        th{background:#21262d;color:#8b949e;font-weight:600}\n\
        tr:hover{background:#1f242c}\n\
        .badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600}\n\
        .badge-gateway{background:#238636;color:#fff}\n\
        .badge-lan{background:#1f6feb;color:#fff}\n\
        .badge-external{background:#30363d;color:#8b949e}\n\
        .badge-tor{background:#8957e5;color:#fff}\n\
        .badge-threat{background:#da3633;color:#fff}\n\
        .badge-benign{background:#238636;color:#fff}\n\
        .diagram-container{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:20px;text-align:center;margin-top:16px}\n\
        svg{max-width:100%;height:auto}\n"
    );
    html.push_str("</style>\n</head>\n<body>\n");

    // Header
    let _ = writeln!(
        html,
        "<div class=\"header\"><h1>ARGUS X-RAY // FORENSIC NETWORK DOSSIER</h1>\
        <div class=\"meta\">Report ID: <code>{}</code> &bull; Generated: Unix epoch {}s &bull; Reference Model: PcapXray Architecture</div></div>",
        report_id, timestamp
    );

    // KPI Summary
    html.push_str("<div class=\"grid\">\n");
    let _ = writeln!(
        html,
        "<div class=\"card\"><div class=\"title\">Total Endpoints</div><div class=\"value\">{}</div></div>",
        topology.nodes.len()
    );
    let _ = writeln!(
        html,
        "<div class=\"card\"><div class=\"title\">Active Edges</div><div class=\"value\">{}</div></div>",
        topology.edges.len()
    );
    let _ = writeln!(
        html,
        "<div class=\"card\"><div class=\"title\">Total Volume</div><div class=\"value\">{}</div></div>",
        format_bytes_human(topology.total_bytes)
    );
    let _ = writeln!(
        html,
        "<div class=\"card\"><div class=\"title\">Total Packets</div><div class=\"value\">{}</div></div>",
        topology.total_packets
    );
    let tor_card_class = if topology.tor_nodes_count > 0 {
        "card warning"
    } else {
        "card"
    };
    let _ = writeln!(
        html,
        "<div class=\"{}\"><div class=\"title\">Tor-Port Endpoints</div><div class=\"value\">{}</div></div>",
        tor_card_class, topology.tor_nodes_count
    );
    let mal_card_class = if topology.malicious_nodes_count > 0 {
        "card alert"
    } else {
        "card success"
    };
    let _ = writeln!(
        html,
        "<div class=\"{}\"><div class=\"title\">Rule / Policy-Flagged Endpoints</div><div class=\"value\">{}</div></div>",
        mal_card_class, topology.malicious_nodes_count
    );
    html.push_str("</div>\n");

    // SVG Topology Diagram Preview
    html.push_str("<h2>1. Visual Network Topology</h2>\n");
    html.push_str("<div class=\"diagram-container\">\n");
    html.push_str("<svg width=\"900\" height=\"360\" viewBox=\"0 0 900 360\" xmlns=\"http://www.w3.org/2000/svg\">\n");
    html.push_str("<rect width=\"900\" height=\"360\" fill=\"#161b22\" rx=\"6\"/>\n");

    // Render nodes in SVG
    let node_count = topology.nodes.len();
    if node_count == 0 {
        html.push_str("<text x=\"450\" y=\"180\" fill=\"#8b949e\" font-size=\"14\" text-anchor=\"middle\">No communication flows recorded in current session</text>\n");
    } else {
        // Layout: compute circular coordinates
        let center_x = 450.0_f64;
        let center_y = 180.0_f64;
        let radius = 130.0_f64;

        let node_indices: HashMap<&str, usize> = topology
            .nodes
            .iter()
            .enumerate()
            .map(|(idx, n)| (n.id.as_str(), idx))
            .collect();

        for edge in topology.edges.iter().take(48) {
            let src_idx = node_indices.get(edge.source.as_str()).copied().unwrap_or(0);
            let dst_idx = node_indices.get(edge.target.as_str()).copied().unwrap_or(0);

            let src_angle = (src_idx as f64) * (2.0 * std::f64::consts::PI / (node_count as f64));
            let dst_angle = (dst_idx as f64) * (2.0 * std::f64::consts::PI / (node_count as f64));

            let x1 = center_x + radius * src_angle.cos();
            let y1 = center_y + radius * src_angle.sin();
            let x2 = center_x + radius * dst_angle.cos();
            let y2 = center_y + radius * dst_angle.sin();

            let stroke = if edge.is_tor {
                "#bc8cff"
            } else if edge.is_malicious {
                "#f85149"
            } else {
                "#30363d"
            };

            let _ = writeln!(
                html,
                "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"1.5\" opacity=\"0.6\" stroke-dasharray=\"{}\"/>",
                x1,
                y1,
                x2,
                y2,
                stroke,
                if edge.is_tor { "4,3" } else { "none" }
            );
        }

        for (i, node) in topology.nodes.iter().enumerate() {
            let angle = (i as f64) * (2.0 * std::f64::consts::PI / (node_count as f64));
            let x = center_x + radius * angle.cos();
            let y = center_y + radius * angle.sin();

            let (fill, stroke) = if node.is_tor {
                ("#8957e5", "#d2a8ff")
            } else if node.threat_score >= 0.7 {
                ("#da3633", "#ff7b72")
            } else {
                match node.role {
                    argus_capture::NodeRole::Gateway => ("#238636", "#3fb950"),
                    argus_capture::NodeRole::LanHost => ("#1f6feb", "#58a6ff"),
                    argus_capture::NodeRole::TorNode => ("#8957e5", "#d2a8ff"),
                    _ => ("#30363d", "#8b949e"),
                }
            };

            let _ = writeln!(
                html,
                "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"8\" fill=\"{}\" stroke=\"{}\" stroke-width=\"2\"/>",
                x, y, fill, stroke
            );
            let _ = writeln!(
                html,
                "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#c9d1d9\" font-size=\"11\" text-anchor=\"middle\">{}</text>",
                x,
                y + 18.0,
                escape_html(&node.label)
            );
        }
    }
    html.push_str("</svg>\n</div>\n");

    // Network Asset Inventory
    html.push_str("<h2>2. Network Asset Inventory & Attribution</h2>\n");
    html.push_str("<table><thead><tr><th>Endpoint IP</th><th>Entity Identity</th><th>Category</th><th>Role</th><th>Organization / ASN</th><th>Hardware Vendor</th><th>Inbound</th><th>Outbound</th><th>Packets</th><th>Threat Status</th></tr></thead><tbody>\n");
    for node in &topology.nodes {
        let (role_badge, role_name) = match node.role {
            argus_capture::NodeRole::Gateway => ("badge-gateway", "Gateway"),
            argus_capture::NodeRole::LanHost => ("badge-lan", "LAN Host"),
            argus_capture::NodeRole::TorNode => ("badge-tor", "Tor-Port Heuristic"),
            argus_capture::NodeRole::Broadcast => ("badge-external", "Broadcast"),
            argus_capture::NodeRole::Multicast => ("badge-external", "Multicast"),
            argus_capture::NodeRole::ExternalServer => ("badge-external", "External Server"),
        };
        let threat_badge = if node.threat_score >= 0.7 {
            "<span class=\"badge badge-threat\">Rule / Policy Flag</span>"
        } else if node.is_tor {
            "<span class=\"badge badge-tor\">Tor-Port Match Only</span>"
        } else {
            "<span class=\"badge badge-benign\">No Rule / Policy Flag</span>"
        };
        let entity = node.entity_name.as_deref().unwrap_or("Generic Endpoint");
        let category = node.category.as_deref().unwrap_or("Standard");
        let org_asn = match (&node.organization, &node.asn) {
            (Some(org), Some(asn)) => format!("{org} ({asn})"),
            (Some(org), None) => org.clone(),
            (None, Some(asn)) => asn.clone(),
            (None, None) => "-".to_string(),
        };

        let _ = writeln!(
            html,
            "<tr><td><code>{}</code></td><td><strong>{}</strong></td><td>{}</td><td><span class=\"badge {}\">{}</span></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            escape_html(&node.ip),
            escape_html(entity),
            escape_html(category),
            role_badge,
            role_name,
            escape_html(&org_asn),
            escape_html(node.vendor.as_deref().unwrap_or("Generic / Unknown")),
            format_bytes_human(node.total_in_bytes),
            format_bytes_human(node.total_out_bytes),
            node.packet_count,
            threat_badge
        );
    }
    html.push_str("</tbody></table>\n");

    // Protocol & Traffic Distribution
    html.push_str("<h2>3. Protocol & Application Distribution</h2>\n");
    html.push_str("<table><thead><tr><th>Protocol / Service</th><th>Observed Volume</th><th>Telemetry Type</th></tr></thead><tbody>\n");
    let dist = &topology.protocol_distribution;
    let _ = writeln!(
        html,
        "<tr><td>TCP Stream Traffic</td><td>{}</td><td>Transport Layer Volume</td></tr>",
        format_bytes_human(dist.tcp_bytes)
    );
    let _ = writeln!(
        html,
        "<tr><td>UDP Datagram Traffic</td><td>{}</td><td>Transport Layer Volume</td></tr>",
        format_bytes_human(dist.udp_bytes)
    );
    let _ = writeln!(
        html,
        "<tr><td>ICMP Control Messages</td><td>{}</td><td>Network Diagnostic Volume</td></tr>",
        format_bytes_human(dist.icmp_bytes)
    );
    let _ = writeln!(
        html,
        "<tr><td>Parsed DNS Messages</td><td>{} messages</td><td>Decoded DNS Payload</td></tr>",
        dist.dns_queries
    );
    let _ = writeln!(
        html,
        "<tr><td>TLS ClientHello with SNI</td><td>{} flows</td><td>Decoded TLS Payload</td></tr>",
        dist.tls_sessions
    );
    let _ = writeln!(
        html,
        "<tr><td>Parsed HTTP Cleartext Requests</td><td>{} requests</td><td>Decoded HTTP Payload</td></tr>",
        dist.http_requests
    );
    let _ = writeln!(
        html,
        "<tr><td>Tor-Associated Port Packets</td><td>{} packets</td><td>Port Heuristic Only</td></tr>",
        dist.tor_packets
    );
    html.push_str("</tbody></table>\n");

    // Observed port matches and rule/policy flags; neither proves malicious identity.
    html.push_str("<h2>4. Tor-Port and Rule / Policy-Flagged Communications</h2>\n");
    let flagged_edges: Vec<&TopologyEdge> = topology
        .edges
        .iter()
        .filter(|e| e.is_tor || e.is_malicious || e.threat_score >= 0.7)
        .collect();

    if flagged_edges.is_empty() {
        html.push_str("<p style=\"color:#8b949e;font-size:13px;\">No Tor-port matches or rule/policy-flagged communications recorded.</p>\n");
    } else {
        html.push_str("<table><thead><tr><th>Source</th><th>Destination</th><th>Protocol</th><th>Volume</th><th>Classification</th></tr></thead><tbody>\n");
        for edge in flagged_edges {
            let reason = if edge.is_tor && edge.is_malicious {
                "<span class=\"badge badge-threat\">Tor-Port Match & Rule / Policy Flag</span>"
            } else if edge.is_tor {
                "<span class=\"badge badge-tor\">Tor-Port Heuristic</span>"
            } else {
                "<span class=\"badge badge-threat\">Rule / Policy Flag</span>"
            };
            let _ = writeln!(
                html,
                "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td></tr>",
                escape_html(&edge.source),
                escape_html(&edge.target),
                escape_html(&edge.protocol),
                format_bytes_human(edge.total_bytes),
                reason
            );
        }
        html.push_str("</tbody></table>\n");
    }

    // Top 10 Talkers
    html.push_str("<h2>5. Top Talkers</h2>\n");
    html.push_str("<table><thead><tr><th>Initiator</th><th>Responder</th><th>Protocol</th><th>Service</th><th>Packets</th><th>Volume</th></tr></thead><tbody>\n");
    let mut sorted_flows = flows.to_vec();
    sorted_flows.sort_by_key(|b| std::cmp::Reverse(b.captured_bytes.max(b.original_bytes)));
    for flow in sorted_flows.iter().take(10) {
        let _ = writeln!(
            html,
            "<tr><td><code>{}:{}</code></td><td><code>{}:{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            escape_html(&flow.initiator_address),
            flow.initiator_port,
            escape_html(&flow.responder_address),
            flow.responder_port,
            escape_html(&flow.protocol),
            escape_html(flow.service.as_deref().unwrap_or("-")),
            flow.packets,
            format_bytes_human(flow.captured_bytes.max(flow.original_bytes))
        );
    }
    html.push_str("</tbody></table>\n");

    html.push_str("</body>\n</html>\n");
    html
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
async fn get_network_topology(
    history: tauri::State<'_, Arc<Mutex<LiveHistory>>>,
    cache: tauri::State<'_, Arc<IpcQueryCache>>,
) -> Result<NetworkTopology, String> {
    const TTL: Duration = Duration::from_secs(2);
    let (fingerprint, flows) = {
        let h = history.lock().map_err(|_| "LiveHistory lock poisoned")?;
        let fp = h.fingerprint();
        if let Some(topo) = cache.get_topology(fp, TTL) {
            return Ok(topo);
        }
        (fp, h.view().map(|v| v.flows).unwrap_or_default())
    };

    let topology = build_topology_from_flows(&flows);
    cache.set_topology(fingerprint, topology.clone());
    Ok(topology)
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
async fn generate_network_dossier(
    history: tauri::State<'_, Arc<Mutex<LiveHistory>>>,
    alerts: tauri::State<'_, Arc<Mutex<AlertManager>>>,
    cache: tauri::State<'_, Arc<IpcQueryCache>>,
) -> Result<DossierReportDto, String> {
    const TTL: Duration = Duration::from_secs(5);
    let (fingerprint, flows, alert_items) = {
        let h = history.lock().map_err(|_| "LiveHistory lock poisoned")?;
        let a = alerts.lock().map_err(|_| "AlertManager lock poisoned")?;
        let fp = DossierStateFingerprint {
            topology: h.fingerprint(),
            alert_count: a.alerts.len(),
            total_critical: a.total_critical,
            total_suspicious: a.total_suspicious,
        };
        if let Some(dossier) = cache.get_dossier(fp, TTL)
            && Path::new(&dossier.report_path).is_file()
        {
            return Ok(dossier);
        }
        (
            fp,
            h.view().map(|v| v.flows).unwrap_or_default(),
            a.alerts.iter().cloned().collect::<Vec<_>>(),
        )
    };

    let topology = build_topology_from_flows(&flows);

    let reports_dir = reports_directory()?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let report_id = format!("ARGUS-XRAY-{now}");
    let report_file = reports_dir.join(format!("{report_id}.html"));

    let html = format_network_dossier_html(&report_id, now, &topology, &flows, &alert_items);

    let mut hasher = Sha256::new();
    hasher.update(html.as_bytes());
    let sha256_hash = format!("{:x}", hasher.finalize());

    fs::write(&report_file, &html).map_err(|e| e.to_string())?;

    let dossier = DossierReportDto {
        report_id,
        generated_at: format!("{now}"),
        report_path: report_file.to_string_lossy().into_owned(),
        sha256_hash,
        total_nodes: topology.nodes.len(),
        total_edges: topology.edges.len(),
        total_bytes: topology.total_bytes,
        total_packets: topology.total_packets,
        tor_nodes_count: topology.tor_nodes_count,
        malicious_nodes_count: topology.malicious_nodes_count,
        html_content: html,
    };
    cache.set_dossier(fingerprint, dossier.clone());
    Ok(dossier)
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
// Inspected packet count is bounded by the pcap limit, so `f64` conversion is exact.
#[allow(clippy::cast_precision_loss)]
async fn get_flow_payload_hex(
    path: Option<String>,
    flow_id: u64,
    max_packets: Option<usize>,
) -> Result<FlowPayloadDumpDto, String> {
    tauri::async_runtime::spawn_blocking(move || inspect_flow_payload(path, flow_id, max_packets))
        .await
        .map_err(|error| format!("flow inspection worker failed: {error}"))?
}

fn capture_path_or_latest(path: Option<String>) -> Result<PathBuf, String> {
    if let Some(path) = path {
        return Ok(PathBuf::from(path));
    }
    recent_captures_blocking()?
        .into_iter()
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| "No active capture files available".to_owned())
}

fn default_pcap_limits() -> PcapLimits {
    PcapLimits {
        max_file_bytes: NonZeroUsize::new(512 * 1024 * 1024).expect("non-zero literal"),
        max_records: NonZeroUsize::new(5_000_000).expect("non-zero literal"),
    }
}

fn collect_flow_packets(
    target_path: &Path,
    flow_id: u64,
    limit: Option<usize>,
) -> Result<Vec<RawPacket>, String> {
    let mut source = PcapReplaySource::from_path(target_path, default_pcap_limits())
        .map_err(|error| error.to_string())?;
    let mut flow_ids = HashMap::<FlowInstanceId, u64>::new();
    let mut next_flow_id = 0_u64;
    let mut target_seen = false;
    let mut packets = Vec::new();

    while let Some(outcome) = source
        .poll_next_paired()
        .map_err(|error| error.to_string())?
    {
        let AdapterOutcome::Normalized { raw, packet } = outcome else {
            continue;
        };
        if packet.quality.clock_rollback {
            flow_ids.clear();
            next_flow_id = 0;
            target_seen = false;
            packets.clear();
        }
        let current_id = *flow_ids.entry(packet.flow).or_insert_with(|| {
            let assigned = next_flow_id;
            next_flow_id = next_flow_id.saturating_add(1);
            assigned
        });
        if current_id != flow_id {
            continue;
        }
        target_seen = true;
        if limit.is_none_or(|maximum| packets.len() < maximum) {
            packets.push(raw);
        }
    }

    if !target_seen {
        return Err(format!(
            "Flow {flow_id} was not found in the selected capture"
        ));
    }
    Ok(packets)
}

#[allow(clippy::cast_precision_loss)]
fn inspect_flow_payload(
    path: Option<String>,
    flow_id: u64,
    max_packets: Option<usize>,
) -> Result<FlowPayloadDumpDto, String> {
    let limit = max_packets.unwrap_or(20).clamp(1, 100);
    let target_path = capture_path_or_latest(path)?;

    if !target_path.is_file() {
        return Err("Selected capture file does not exist".to_owned());
    }

    let raw_packets = collect_flow_packets(&target_path, flow_id, Some(limit))?;
    let inspected_packets = raw_packets
        .iter()
        .enumerate()
        .map(|(index, raw)| {
            inspect_frame(
                raw.bytes.as_slice(),
                index + 1,
                raw.monotonic_timestamp_ns,
                raw.original_len as usize,
            )
        })
        .collect::<Vec<_>>();
    let mut total_bytes = 0;
    let mut entropy_sum = 0.0;

    for p in &inspected_packets {
        total_bytes += p.captured_length;
        entropy_sum += p.entropy;
    }

    let mean_entropy = if inspected_packets.is_empty() {
        0.0
    } else {
        entropy_sum / (inspected_packets.len() as f64)
    };

    Ok(FlowPayloadDumpDto {
        flow_id,
        packet_count: inspected_packets.len(),
        total_payload_bytes: total_bytes,
        mean_entropy,
        packets: inspected_packets,
    })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ExportFlowPcapResultDto {
    flow_id: u64,
    packet_count: usize,
    total_bytes: usize,
    output_path: String,
    filename: String,
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
async fn export_flow_pcap(
    path: Option<String>,
    flow_id: u64,
) -> Result<ExportFlowPcapResultDto, String> {
    tauri::async_runtime::spawn_blocking(move || export_flow_pcap_blocking(path, flow_id))
        .await
        .map_err(|error| format!("flow export worker failed: {error}"))?
}

fn export_flow_pcap_blocking(
    path: Option<String>,
    flow_id: u64,
) -> Result<ExportFlowPcapResultDto, String> {
    let target_path = capture_path_or_latest(path)?;

    if !target_path.is_file() {
        return Err("Selected capture file does not exist".to_owned());
    }

    let raw_packets = collect_flow_packets(&target_path, flow_id, None)?;
    let total_bytes = raw_packets.iter().map(|packet| packet.bytes.len()).sum();

    let reports_dir = reports_directory()?;
    let now_sec = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let filename = format!("flow_{flow_id}_{now_sec}.pcap");
    let out_file = reports_dir.join(&filename);

    let packets_ref: Vec<(u64, &[u8], usize)> = raw_packets
        .iter()
        .map(|raw| {
            (
                raw.monotonic_timestamp_ns,
                raw.bytes.as_slice(),
                raw.original_len as usize,
            )
        })
        .collect();

    let written_count = argus_capture::write_pcap_file(&out_file, packets_ref)
        .map_err(|e| format!("Failed to write flow PCAP: {e}"))?;

    Ok(ExportFlowPcapResultDto {
        flow_id,
        packet_count: written_count,
        total_bytes,
        output_path: out_file.to_string_lossy().into_owned(),
        filename,
    })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SimulationRunResultDto {
    scenario_id: String,
    title: String,
    technique_id: String,
    technique_name: String,
    tactic: String,
    packet_count: usize,
    flows_count: usize,
    anomalies_observed: usize,
    high_severity_anomalies: usize,
    pcap_path: Option<String>,
    model_result: Option<serde_json::Value>,
    model_error: Option<String>,
    run: RunView,
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn list_simulation_scenarios() -> Vec<SimulationScenario> {
    list_available_scenarios()
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
// Simulation flow counters are bounded by the generated pcap, so `f64` conversion is exact.
#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_precision_loss)]
async fn run_simulation_scenario(
    scenario_id: String,
    packet_count: Option<usize>,
    simulation_worker: tauri::State<'_, SimulationWorker>,
    preferences: tauri::State<'_, Arc<Mutex<RuntimePreferences>>>,
) -> Result<SimulationRunResultDto, String> {
    let disabled_models = preferences
        .lock()
        .map_err(|_| "runtime preferences lock poisoned".to_owned())?
        .disabled_models
        .clone();
    let scenarios = list_available_scenarios();
    let scenario = scenarios
        .iter()
        .find(|s| s.id == scenario_id)
        .ok_or_else(|| format!("Unknown simulation scenario '{scenario_id}'"))?
        .clone();

    let count = packet_count
        .unwrap_or(scenario.default_packet_count)
        .clamp(10, 500);
    let pcap_bytes = generate_simulation_pcap(&scenario_id, count)?;

    let captures_dir = capture_directory()?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_millis();
    let sim_file = captures_dir.join(format!("sim_{scenario_id}_{now}.pcap"));
    let _capture_guard = TemporaryGeneratedCapture::new([sim_file.clone()]);
    fs::write(&sim_file, &pcap_bytes).map_err(|e| e.to_string())?;

    let worker = Arc::clone(&simulation_worker.0);
    let model_path = sim_file.clone();
    let simulation_scope = format!("simulation:{scenario_id}:{now}");
    let (run, model_result) = tauri::async_runtime::spawn_blocking(move || {
        let run = replay_path(&model_path)?;
        let model_result = worker.lock().map_or_else(
            |_| Err("simulation model worker lock poisoned".to_owned()),
            |mut worker| {
                run_experimental_cnn(
                    &model_path,
                    &simulation_scope,
                    true,
                    false,
                    &disabled_models,
                    &mut worker,
                )
            },
        );
        Ok::<_, String>((run, model_result))
    })
    .await
    .map_err(|error| error.to_string())??;

    let anomalies_observed = run
        .flows
        .iter()
        .filter(|flow| flow.attack_verdict.is_some())
        .count();
    let high_severity_anomalies = run
        .flows
        .iter()
        .filter(|flow| {
            flow.attack_verdict
                .as_ref()
                .is_some_and(|verdict| verdict.severity == "critical")
        })
        .count();

    Ok(SimulationRunResultDto {
        scenario_id: scenario.id,
        title: scenario.title,
        technique_id: scenario.technique.technique_id,
        technique_name: scenario.technique.technique_name,
        tactic: scenario.technique.tactic,
        packet_count: count,
        flows_count: run.flows.len(),
        anomalies_observed,
        high_severity_anomalies,
        pcap_path: None,
        model_result: model_result.as_ref().ok().cloned(),
        model_error: model_result.err(),
        run,
    })
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn get_active_defense_settings(
    blacklist: tauri::State<Arc<Mutex<BlacklistManager>>>,
) -> Result<ActiveDefenseSettingsDto, String> {
    let mgr = blacklist.lock().map_err(|e| e.to_string())?;
    Ok(mgr.settings.clone())
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn update_active_defense_settings(
    settings: ActiveDefenseSettingsDto,
    blacklist: tauri::State<Arc<Mutex<BlacklistManager>>>,
) -> Result<ActiveDefenseSettingsDto, String> {
    let mut mgr = blacklist.lock().map_err(|e| e.to_string())?;
    mgr.settings = settings.clone();
    mgr.save();
    Ok(settings)
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn list_blacklist_rules(
    blacklist: tauri::State<Arc<Mutex<BlacklistManager>>>,
) -> Result<Vec<BlockRuleDto>, String> {
    let mgr = blacklist.lock().map_err(|e| e.to_string())?;
    Ok(mgr.rules.clone())
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn block_ip_address(
    target: String,
    reason: String,
    severity: Option<String>,
    blacklist: tauri::State<Arc<Mutex<BlacklistManager>>>,
) -> Result<BlockRuleDto, String> {
    let mut mgr = blacklist.lock().map_err(|e| e.to_string())?;
    mgr.block_ip(&target, reason, severity)
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn upsert_traffic_rule(
    target: String,
    reason: String,
    severity: Option<String>,
    action: String,
    blacklist: tauri::State<Arc<Mutex<BlacklistManager>>>,
) -> Result<BlockRuleDto, String> {
    let action = TrafficRuleAction::parse(&action)?;
    let mut manager = blacklist.lock().map_err(|error| error.to_string())?;
    manager.upsert_rule(&target, reason, severity, action)
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn unblock_ip_address(
    target: String,
    blacklist: tauri::State<Arc<Mutex<BlacklistManager>>>,
) -> Result<bool, String> {
    let mut mgr = blacklist.lock().map_err(|e| e.to_string())?;
    mgr.unblock_ip(&target)
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn clear_all_blacklist_rules(
    blacklist: tauri::State<Arc<Mutex<BlacklistManager>>>,
) -> Result<usize, String> {
    let mut mgr = blacklist.lock().map_err(|e| e.to_string())?;
    let count = mgr.rules.len();
    for r in &mgr.rules {
        if r.active && r.os_firewall_rule {
            let _ = delete_os_firewall_rule(&r.target);
        }
    }
    mgr.rules.clear();
    mgr.blocked_ips.clear();
    mgr.save();
    Ok(count)
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn get_blacklist_stats(
    blacklist: tauri::State<Arc<Mutex<BlacklistManager>>>,
) -> Result<BlacklistStatsDto, String> {
    let mgr = blacklist.lock().map_err(|e| e.to_string())?;
    Ok(mgr.stats())
}

fn show_main_window(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
        let _ = window.set_always_on_top(true);
        let _ = window.set_always_on_top(false);
    }
}

#[cfg(windows)]
fn ensure_desktop_shortcut() -> Result<(), String> {
    let executable =
        dunce::canonicalize(std::env::current_exe().map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    let icon_bytes = include_bytes!("../icons/icon.ico");
    let icon_hash = format!("{:x}", Sha256::digest(icon_bytes));
    let local_root = std::env::var_os("LOCALAPPDATA")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or("Local application data directory unavailable")?;
    if !local_root.is_absolute() {
        return Err("Local application data directory must be absolute".to_owned());
    }
    let icon_directory = local_root.join("ArgusNextPublic").join("icons");
    fs::create_dir_all(&icon_directory).map_err(|error| error.to_string())?;
    let icon_path = icon_directory.join(format!("argus-{icon_hash}.ico"));
    if !icon_path.exists() {
        let temporary =
            icon_directory.join(format!("argus-{icon_hash}-{}.tmp", std::process::id()));
        fs::write(&temporary, icon_bytes).map_err(|error| error.to_string())?;
        fs::rename(&temporary, &icon_path).map_err(|error| error.to_string())?;
    }
    let script = format!(
        "& {{ {} }} -Executable {} -IconPath {}",
        include_str!("../windows/ensure-shortcut.ps1"),
        quote_powershell(&executable),
        quote_powershell(&icon_path),
    );
    let status = background_command("powershell.exe")
        .args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            &script,
        ])
        .status()
        .map_err(|error| error.to_string())?;
    status
        .success()
        .then_some(())
        .ok_or_else(|| "could not create the Argus desktop shortcut".to_owned())
}

#[cfg(not(windows))]
fn ensure_desktop_shortcut() -> Result<(), String> {
    Ok(())
}

fn configure_desktop(app: &mut tauri::App) -> tauri::Result<()> {
    let open = MenuItem::with_id(app, TRAY_OPEN_ID, "Open Argus", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, TRAY_QUIT_ID, "Quit Argus", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&open, &quit])?;
    let mut tray = TrayIconBuilder::with_id("argus-resident")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .tooltip("Argus Next Public — capture starts only on request");
    if let Some(icon) = app.default_window_icon() {
        tray = tray.icon(icon.clone());
    }
    tray.build(app)?;

    std::thread::spawn(|| {
        let _ = ensure_desktop_shortcut();
    });
    // Opening the public app must not capture traffic or request sensor elevation.
    // Capture is started only through the explicit start_rolling_capture command.
    show_main_window(app.handle());
    Ok(())
}

fn exit_after_sensor_install_if_requested() {
    if std::env::args().nth(1).as_deref() == Some(SENSOR_INSTALL_HELPER_ARG) {
        let exit_code = i32::from(run_elevated_sensor_install_helper().is_err());
        std::process::exit(exit_code);
    }
}

#[allow(clippy::too_many_lines)]
fn main() {
    std::panic::set_hook(Box::new(|info| {
        let msg = format!("ARGUS PANIC: {info}\n");
        eprintln!("{msg}");
        if let Ok(dir) = std::env::current_dir() {
            let _ = fs::write(dir.join("argus-panic.log"), msg);
        }
    }));
    exit_after_sensor_install_if_requested();
    let rolling_monitor = Arc::new(Mutex::new(RollingMonitor::default()));
    let runtime_preferences_state = Arc::new(Mutex::new(load_runtime_preferences()));
    let _ = cleanup_runtime_feature_csvs();
    let _ = cleanup_generated_temporary_captures();
    let _ = cleanup_generated_live_captures();
    let query_cache = Arc::new(IpcQueryCache::default());
    query_cache.register_active();
    let application = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(
            |app, _arguments, _working_directory| {
                show_main_window(app);
            },
        ))
        .manage(rolling_monitor)
        .manage(runtime_preferences_state)
        .manage(Arc::new(Mutex::new(CnnWorker::default())))
        .manage(SimulationWorker::default())
        .manage(Arc::new(Mutex::new(LiveHistory::load())))
        .manage(Arc::new(Mutex::new(AlertManager::default())))
        .manage(Arc::new(Mutex::new(BlacklistManager::load())))
        .manage(query_cache)
        .setup(|app| Ok(configure_desktop(app)?))
        .on_menu_event(|app, event| match event.id().as_ref() {
            TRAY_OPEN_ID => show_main_window(app),
            TRAY_QUIT_ID => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|app, event| {
            if matches!(
                event,
                TrayIconEvent::Click {
                    button: MouseButton::Left,
                    button_state: MouseButtonState::Up,
                    ..
                } | TrayIconEvent::DoubleClick {
                    button: MouseButton::Left,
                    ..
                }
            ) {
                show_main_window(app);
            }
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .invoke_handler(tauri::generate_handler![
            desktop_status,
            runtime_diagnostics,
            recent_captures,
            replay_capture,
            live_history,
            daily_learning_status,
            release_live_capture,
            open_capture_in_wireshark,
            capture_30_seconds,
            start_rolling_capture,
            stop_rolling_capture,
            rolling_capture_status,
            runtime_preferences,
            set_keep_monitoring_after_exit,
            set_boost_mode,
            set_model_enabled,
            reset_model_preferences,
            set_notification_policy,
            try_experimental_cnn,
            get_recent_alerts,
            clear_alerts,
            get_alert_stats,
            export_incident_report,
            export_live_session,
            get_flow_payload_hex,
            export_flow_pcap,
            list_simulation_scenarios,
            run_simulation_scenario,
            get_active_defense_settings,
            update_active_defense_settings,
            list_blacklist_rules,
            block_ip_address,
            upsert_traffic_rule,
            unblock_ip_address,
            clear_all_blacklist_rules,
            get_blacklist_stats,
            get_network_topology,
            generate_network_dossier
        ])
        .build(tauri::generate_context!())
        .expect("Argus desktop runtime failed");
    application.run(|handle, event| {
        if matches!(event, tauri::RunEvent::ExitRequested { .. }) {
            let _ = cleanup_runtime_feature_csvs();
            let _ = cleanup_generated_temporary_captures();
            let _ = cleanup_generated_live_captures();
            let cnn_worker = handle.state::<Arc<Mutex<CnnWorker>>>();
            if let Ok(mut worker) = cnn_worker.lock()
                && let Some(mut child) = worker.child.take()
            {
                let _ = child.kill();
                let _ = child.wait();
            }
            let preferences = handle.state::<Arc<Mutex<RuntimePreferences>>>();
            let should_stop = preferences
                .lock()
                .map_or(true, |preferences| stop_sensor_on_exit(&preferences));
            if should_stop {
                let state = handle.state::<Arc<Mutex<RollingMonitor>>>();
                if let Ok(monitor) = state.lock()
                    && let Some(stop_file) = monitor.stop_file.as_ref()
                {
                    if let Some(parent) = stop_file.parent() {
                        let _ = fs::write(parent.join("cleanup-on-stop"), b"cleanup\n");
                    }
                    let _ = fs::write(stop_file, b"stop\n");
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn packaged_resource_root_stays_beside_relocated_executable() {
        let directory = std::env::temp_dir().join("argus-public-relocation-test");
        let executable = directory.join("argus-desktop.exe");
        assert_eq!(
            super::application_root_from_executable(&executable),
            directory
        );
    }

    #[test]
    #[should_panic(expected = "application executable path must be absolute")]
    fn relative_resource_roots_fail_closed() {
        super::application_root_from_executable(std::path::Path::new("argus-desktop.exe"));
    }

    #[test]
    fn source_build_resource_root_is_discovered_from_executable() {
        let executable = std::env::current_exe().expect("test executable");
        let root = super::application_root_from_executable(&executable);
        assert!(root.join("Cargo.toml").is_file());
        assert!(root.join("python/pyproject.toml").is_file());
        assert!(executable.starts_with(root.join("target")));
    }

    use std::{collections::VecDeque, fs, path::PathBuf, time::SystemTime};

    use super::{
        AHEM_V61_FILE, AdmissionRuntime, AlertManager, CnnWorker, DetectorView,
        FORECAST_ROUTER_FILE, FlowView, HealthView, LIVE_HISTORY_CAPACITY,
        LIVE_HISTORY_UI_CAPACITY, LIVE_SEGMENT_SAMPLE_CAPACITY, LiveHistory, MonitorControl,
        NativeOnnxWorker, NotificationPolicy, ReleaseCaptureStatus, RunView, RuntimePreferences,
        SECURITY_DETECTOR_FILE, SensorHealth, TEMPORAL_EXPERT_FILES, admission_runtime_from_value,
        build_topology_from_flows, capture_directory, capture_storage_usage,
        cleanup_generated_temporary_captures_in, cleanup_runtime_feature_csvs_in,
        encode_live_session_export, escape_html, file_sha256, format_network_dossier_html,
        integrity_bound_admission_runtime, release_live_capture, replace_file,
        selected_forecast_experts, smooth_l1, stop_sensor_on_exit, validate_monitor_control,
        verify_ahem_v61_runtime_artifact_at,
    };

    #[test]
    fn clearing_alerts_resets_visible_counts() {
        let mut manager = AlertManager {
            total_critical: 2,
            total_suspicious: 3,
            ..Default::default()
        };
        manager.clear();
        let stats = manager.stats();
        assert_eq!(stats.total_alerts, 0);
        assert_eq!(stats.critical_count, 0);
        assert_eq!(stats.suspicious_count, 0);
        assert!(stats.last_alert_time.is_none());
    }

    #[test]
    fn experimental_shadow_does_not_feed_operator_forecast_diagnostics() {
        let ordinary = serde_json::json!({"state": "forecast_only", "forecast_error": 1.0});
        let shadow = serde_json::json!({"state": "forecast_only", "experimental_shadow": true});
        assert!(super::operator_forecast_diagnostic_allowed(
            "ahem_v61", &ordinary
        ));
        assert!(!super::operator_forecast_diagnostic_allowed(
            "ahem_v63_experimental",
            &ordinary
        ));
        assert!(!super::operator_forecast_diagnostic_allowed(
            "ahem_e2e_experimental",
            &ordinary
        ));
        assert!(!super::operator_forecast_diagnostic_allowed(
            "other_shadow",
            &shadow
        ));
        assert!(!super::operator_forecast_diagnostic_allowed(
            "ahem_v61",
            &serde_json::json!({"state":"error"})
        ));
    }

    #[test]
    fn active_python_only_route_skips_native_experts() {
        let active = serde_json::json!({
            "router": {"state": "active", "selected_expert": "lstm_cse"}
        });
        let selected = selected_forecast_experts(&active).expect("active route");
        assert!(selected.contains("lstm_cse"));
        assert!(!selected.contains("cnn_local"));
        assert!(
            selected_forecast_experts(&serde_json::json!({
                "router": {"state": "warming_up"}
            }))
            .is_none()
        );
    }

    fn history_run(session: &str, flow_count: usize) -> RunView {
        let flows = (0..flow_count)
            .map(|id| FlowView {
                uid: format!("{session}:{id}"),
                session: session.to_owned(),
                source: "live-test.pcap".to_owned(),
                id: u64::try_from(id).expect("test flow ID"),
                protocol: "TCP".to_owned(),
                service: Some("HTTPS".to_owned()),
                threat_status: None,
                generation: 0,
                initiator_address: "192.0.2.1".to_owned(),
                initiator_port: 40_000,
                responder_address: "198.51.100.1".to_owned(),
                responder_port: 443,
                first_timestamp_ns: 1,
                last_timestamp_ns: 2,
                duration_ms: 0.000_001,
                packets: 1,
                captured_bytes: 10,
                original_bytes: 10,
                initiator_packets: 1,
                responder_packets: 0,
                initiator_bytes: 10,
                responder_bytes: 0,
                initiator_mac: None,
                responder_mac: None,
                tls_sni: None,
                dns_qname: None,
                dns_messages: 0,
                dns_queries: 0,
                http_requests: 0,
                tor_port_matches: 0,
                tcp_syn_packets: 0,
                tcp_syn_fin_packets: 0,
                tcp_xmas_packets: 0,
                tcp_null_packets: 0,
                tcp_syn_rst_packets: 0,
                lost_packets: 0,
                truncated_packets: 0,
                reordered_packets: 0,
                is_watchlisted: false,
                is_blocked: false,
                is_allowlisted: false,
                is_blocklisted: false,
                rule_action: None,
                matched_rule_target: None,
                attack_verdict: None,
            })
            .collect();
        RunView {
            source: "live-test.pcap".to_owned(),
            session: session.to_owned(),
            detector: DetectorView {
                id: "baseline".to_owned(),
                trained: false,
                feature_schema: "argus.causal-flow/v1".to_owned(),
                artifact: None,
            },
            health: HealthView {
                source_records: u64::try_from(flow_count).expect("test record count"),
                normalized_packets: u64::try_from(flow_count).expect("test packet count"),
                processed_packets: u64::try_from(flow_count).expect("test packet count"),
                flows: flow_count,
                alerts: 0,
                unsupported_records: 0,
                malformed_records: 0,
                reported_lost_packets: 0,
                truncated_packets: 0,
                reordered_packets: 0,
                queue_overflows: 0,
                parse_errors: 0,
                source_errors: 0,
            },
            flows,
        }
    }

    #[test]
    fn live_history_is_bounded_and_session_idempotent() {
        let mut history = LiveHistory {
            version: 1,
            ..LiveHistory::default()
        };
        let run = history_run("session-one", LIVE_HISTORY_CAPACITY + 1);

        assert!(
            history
                .append_run("live-segment-one", &run)
                .expect("append history")
        );
        let reopened = history_run("new-replay-session", 1);
        assert!(
            !history
                .append_run("live-segment-one", &reopened)
                .expect("deduplicate history")
        );
        assert_eq!(history.segments, 1);
        assert_eq!(history.samples.len(), 1);
        assert_eq!(history.samples[0].flows, LIVE_HISTORY_CAPACITY + 1);
        assert_eq!(history.samples[0].packets, history.packets);
        assert_eq!(history.samples[0].bytes, history.packets * 10);
        assert_eq!(history.samples[0].capture_issues, 0);
        assert_eq!(history.flows.len(), LIVE_HISTORY_CAPACITY);
        assert_eq!(history.evicted_flows, 1);
        assert_eq!(
            history.packets,
            u64::try_from(LIVE_HISTORY_CAPACITY + 1).expect("history capacity")
        );
        let view = history.view().expect("history view");
        assert_eq!(view.flows.len(), LIVE_HISTORY_UI_CAPACITY);
        assert_eq!(
            view.flows[0].id,
            u64::try_from(LIVE_HISTORY_CAPACITY).expect("history capacity")
        );
        assert_eq!(view.samples.len(), 1);
        assert_eq!(view.sample_capacity, LIVE_SEGMENT_SAMPLE_CAPACITY);
    }

    #[test]
    fn live_traffic_samples_are_bounded() {
        let mut history = LiveHistory {
            version: 1,
            ..LiveHistory::default()
        };
        for segment in 0..=LIVE_SEGMENT_SAMPLE_CAPACITY {
            assert!(
                history
                    .append_run(
                        &format!("live-segment-{segment}"),
                        &history_run("sample", 1)
                    )
                    .expect("append sample")
            );
        }
        assert_eq!(history.samples.len(), LIVE_SEGMENT_SAMPLE_CAPACITY);
        assert_eq!(
            history.view().expect("history view").samples.len(),
            LIVE_SEGMENT_SAMPLE_CAPACITY
        );
    }

    #[test]
    fn legacy_live_history_defaults_new_rule_fields_and_removes_old_verdicts() {
        let mut legacy_flow = serde_json::to_value(
            history_run("legacy-session", 1)
                .flows
                .into_iter()
                .next()
                .expect("legacy flow"),
        )
        .expect("serialize legacy flow");
        let object = legacy_flow.as_object_mut().expect("legacy flow object");
        object.remove("isWatchlisted");
        object.remove("isBlocked");
        object.insert(
            "threatStatus".to_owned(),
            serde_json::Value::String("fabricated legacy verdict".to_owned()),
        );

        let flow = serde_json::from_value(legacy_flow).expect("deserialize legacy flow");
        let mut history = LiveHistory {
            version: 1,
            flows: VecDeque::from([flow]),
            ..LiveHistory::default()
        };
        history.sanitize_loaded_flows(&super::BlacklistManager::default());

        let migrated = history.flows.front().expect("migrated flow");
        assert!(!migrated.is_watchlisted);
        assert!(!migrated.is_blocked);
        assert_eq!(migrated.threat_status, None);
    }

    #[test]
    fn legacy_live_history_without_traffic_samples_remains_loadable() {
        let history: LiveHistory = serde_json::from_value(serde_json::json!({
            "version": 1,
            "segment_keys": [],
            "flows": [],
            "segments": 4,
            "packets": 120,
            "bytes": 4096,
            "evicted_flows": 0
        }))
        .expect("deserialize history written before segment telemetry");
        assert!(history.samples.is_empty());
        assert_eq!(history.segments, 4);
    }

    #[test]
    fn live_session_export_contains_bounded_facts_without_raw_capture() {
        let mut history = LiveHistory {
            version: 1,
            ..LiveHistory::default()
        };
        assert!(
            history
                .append_run("segment", &history_run("session", 2))
                .unwrap()
        );
        let (json, csv) = encode_live_session_export(&history, 42).expect("encode export");
        let value: serde_json::Value = serde_json::from_slice(&json).expect("parse export");
        assert_eq!(value["schema"], "argus.live-session-export/v1");
        assert_eq!(value["generatedAtUnixMs"], 42);
        assert_eq!(value["rawCaptureIncluded"], false);
        assert_eq!(value["flows"].as_array().map(Vec::len), Some(2));
        assert_eq!(value["counterSamples"].as_array().map(Vec::len), Some(1));
        let csv = String::from_utf8(csv).expect("UTF-8 CSV");
        assert!(csv.contains("initiatorAddress"));
        assert_eq!(csv.lines().count(), 3);
    }

    #[test]
    fn rolling_monitor_control_is_bound_to_the_live_sensor_scope() {
        let output_directory = capture_directory().expect("capture directory").join("live");
        let valid = MonitorControl {
            schema: "argus.rolling-monitor-control".to_owned(),
            version: 1,
            process_id: 42,
            executable: PathBuf::from(r"C:\Argus\argus-pktmon-stream.exe"),
            stop_file: output_directory.join("sensor.stop"),
            output_directory,
        };

        assert!(validate_monitor_control(valid.clone()).is_ok());
        let mut escaped = valid;
        escaped.stop_file = PathBuf::from(r"C:\unrelated\sensor.stop");
        assert!(validate_monitor_control(escaped).is_err());
    }

    #[test]
    fn monitoring_stops_on_exit_unless_the_user_opts_in() {
        let default_preferences = RuntimePreferences::default();
        assert!(stop_sensor_on_exit(&default_preferences));
        assert_eq!(
            default_preferences.notification_policy,
            NotificationPolicy::ErrorsOnly
        );
        assert!(!stop_sensor_on_exit(&RuntimePreferences {
            keep_monitoring_after_exit: true,
            ..default_preferences
        }));
    }

    #[test]
    fn model_selection_migrates_old_preferences_without_enabling_new_shadow() {
        let preferences: RuntimePreferences = serde_json::from_value(serde_json::json!({
            "version": 1, "keepMonitoringAfterExit": false
        }))
        .expect("old preferences remain supported");
        assert_eq!(preferences.disabled_models, vec!["ahem_e2e_experimental"]);
        assert!(
            !preferences
                .disabled_models
                .iter()
                .any(|id| id == "ahem_v61")
        );
    }

    #[test]
    fn model_selection_roundtrips_and_rejects_non_forecasters() {
        let preferences = RuntimePreferences::default();
        let updated = super::update_model_selection(&preferences, "cnn_local", false)
            .expect("disable known model");
        let updated = super::update_model_selection(&updated, "cnn_local", false)
            .expect("repeat is idempotent");
        assert_eq!(
            updated
                .disabled_models
                .iter()
                .filter(|id| *id == "cnn_local")
                .count(),
            1
        );
        let encoded = serde_json::to_vec(&updated).expect("encode selection");
        let restored: RuntimePreferences =
            serde_json::from_slice(&encoded).expect("restore selection");
        assert_eq!(updated.disabled_models, restored.disabled_models);
        let enabled =
            super::update_model_selection(&restored, "cnn_local", true).expect("enable model");
        assert!(!enabled.disabled_models.iter().any(|id| id == "cnn_local"));
        assert!(super::update_model_selection(&enabled, "security_detector", false).is_err());
        assert!(super::update_model_selection(&enabled, "cnn-local", false).is_err());
    }

    #[test]
    fn model_selection_reconfigures_worker_and_clears_stale_latency() {
        let mut worker = CnnWorker::default();
        worker.latency_samples_ms.push_back(42.0);
        worker.native_start_error = Some("old admission result".to_owned());
        worker.select_models(&["cnn_local".to_owned()]);
        assert_eq!(worker.disabled_models, vec!["cnn_local"]);
        assert!(worker.latency_samples_ms.is_empty());
        assert!(worker.native_start_error.is_none());
        worker.latency_samples_ms.push_back(12.0);
        worker.select_models(&["cnn_local".to_owned()]);
        assert_eq!(
            worker.latency_samples_ms.len(),
            1,
            "unchanged selection preserves worker state"
        );
    }

    #[test]
    fn runtime_feature_cleanup_removes_only_argus_temporary_csvs() {
        let nonce = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("test clock")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("argus-runtime-csv-{nonce}"));
        fs::create_dir(&directory).expect("runtime CSV test directory");
        let temporary = directory.join("cnn-runtime-123.csv");
        let legacy = directory.join("live-61-features.csv");
        let legacy_live = directory.join("live-features.csv");
        let retained = directory.join("user-export.csv");
        fs::write(&temporary, b"temporary").expect("write temporary CSV");
        fs::write(&legacy, b"temporary").expect("write legacy temporary CSV");
        fs::write(&legacy_live, b"temporary").expect("write live temporary CSV");
        fs::write(&retained, b"retained").expect("write retained CSV");

        assert_eq!(
            cleanup_runtime_feature_csvs_in(&directory).expect("clean runtime CSVs"),
            3
        );
        assert!(!temporary.exists());
        assert!(!legacy.exists());
        assert!(!legacy_live.exists());
        assert!(retained.exists());
        fs::remove_dir_all(directory).expect("clean runtime CSV test directory");
    }

    #[test]
    fn capture_classification_handles_missing_live_directory() {
        let nonce = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("argus-classification-{nonce}"));
        fs::create_dir(&root).expect("create isolated capture root");
        let live = root.join("live");
        let external = root.join("simulation.pcap");
        assert!(
            !super::generated_live_capture_path_in(&external, &live)
                .expect("missing live directory")
        );
        fs::create_dir(&live).expect("create live directory");
        assert!(
            super::generated_live_capture_path_in(&live.join("live-1.pcap"), &live)
                .expect("generated capture")
        );
        assert!(
            !super::generated_live_capture_path_in(&external, &live).expect("external capture")
        );
        fs::remove_dir_all(root).expect("clean isolated capture root");
    }

    #[test]
    fn generated_capture_cleanup_preserves_user_owned_captures() {
        let nonce = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("argus-capture-cleanup-{nonce}"));
        fs::create_dir(&directory).expect("create capture cleanup directory");
        for name in [
            "sim_dns_tunnel_123.pcap",
            "argus-123.etl",
            "argus-123.pcapng",
        ] {
            fs::write(directory.join(name), b"generated").expect("write generated capture");
        }
        for name in [
            "user-evidence.pcap",
            "incident.pcapng",
            "argus-not-a-stamp.etl",
        ] {
            fs::write(directory.join(name), b"user").expect("write user capture");
        }

        assert_eq!(
            cleanup_generated_temporary_captures_in(&directory)
                .expect("cleanup generated captures"),
            3
        );
        assert!(directory.join("user-evidence.pcap").is_file());
        assert!(directory.join("incident.pcapng").is_file());
        assert!(directory.join("argus-not-a-stamp.etl").is_file());
        fs::remove_dir_all(directory).expect("remove capture cleanup directory");
    }

    #[test]
    fn native_sensor_health_uses_camel_case_at_the_ui_boundary() {
        let health: SensorHealth = serde_json::from_str(
            r#"{"schema":"argus.pktmon-live-health","version":1,"processId":42,"active":true,"packets":7,"missed_writes":2,"missed_reads":3,"application_drops":4,"malformed":0,"unsupported":1,"duplicate_appearances":5,"completed_segments":6,"retained_segments":2,"retention_evictions":1,"failed":false}"#,
        )
        .expect("native health");
        let value = serde_json::to_value(health).expect("UI health");
        assert_eq!(value["processId"], 42);
        assert_eq!(value["missedWrites"], 2);
        assert_eq!(value["applicationDrops"], 4);
        assert_eq!(value["completedSegments"], 6);
        assert_eq!(value["retentionEvictions"], 1);
    }

    #[test]
    fn live_history_replacement_leaves_no_partial_or_backup_file() {
        let nonce = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("test clock")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("argus-history-{nonce}"));
        fs::create_dir(&directory).expect("history test directory");
        let path = directory.join("live-flow-history.json");
        replace_file(&path, b"first").expect("first history write");
        replace_file(&path, b"second").expect("replacement history write");
        assert_eq!(fs::read(&path).expect("read history"), b"second");
        assert!(!path.with_extension("json.tmp").exists());
        assert!(!path.with_extension("json.bak").exists());
        fs::remove_dir_all(directory).expect("clean history test directory");
    }

    #[test]
    fn diagnostic_storage_usage_counts_only_capture_artifacts() {
        let nonce = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("test clock")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("argus-storage-{nonce}"));
        let nested = directory.join("live");
        fs::create_dir_all(&nested).expect("storage test directory");
        fs::write(directory.join("one.pcapng"), [1_u8; 3]).expect("first capture");
        fs::write(nested.join("two.pcap"), [2_u8; 5]).expect("second capture");
        fs::write(nested.join("source.etl"), [3_u8; 7]).expect("ETL capture");
        fs::write(directory.join("ignore.json"), [4_u8; 11]).expect("ignored state");

        assert_eq!(capture_storage_usage(&directory), (3, 15));
        fs::remove_dir_all(directory).expect("clean storage test directory");
    }

    #[test]
    fn admission_report_enables_only_explicit_capabilities() {
        let value = serde_json::json!({
            "policy": {"version": "argus.model-admission/v1"},
            "runtime": {
                "security_verdict_enabled": false,
                "forecast_router_enabled": true
            }
        });
        assert_eq!(
            admission_runtime_from_value(&value),
            AdmissionRuntime {
                policy: "argus.model-admission/v1".to_owned(),
                security_verdict_enabled: false,
                forecast_router_enabled: true,
            }
        );
    }

    #[test]
    fn malformed_admission_report_fails_closed() {
        assert_eq!(
            admission_runtime_from_value(&serde_json::json!({"runtime": {}})),
            AdmissionRuntime {
                policy: "unverified".to_owned(),
                security_verdict_enabled: false,
                forecast_router_enabled: false,
            }
        );
    }

    #[test]
    fn ahem_runtime_catalog_quarantines_manifest_or_hash_mismatch() {
        let nonce = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("test clock")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("argus-ahem-runtime-{nonce}"));
        fs::create_dir(&directory).expect("create AHEM runtime test directory");
        let checkpoint = directory.join(AHEM_V61_FILE);
        fs::write(&checkpoint, b"test-forecast-weights").expect("write test checkpoint");
        let hash = file_sha256(&checkpoint).expect("hash test checkpoint");
        let manifest_path = checkpoint.with_extension("pt.manifest.json");
        fs::write(
            &manifest_path,
            serde_json::to_vec(&serde_json::json!({
                "schema": "argus.ahem-v6-runtime-manifest/v1",
                "model_id": "argus.ahem-v6.1.state-transition-forecaster",
                "artifact_sha256": hash.clone(),
                "security_detector": false
            }))
            .expect("serialize test manifest"),
        )
        .expect("write test manifest");

        assert_eq!(
            verify_ahem_v61_runtime_artifact_at(&checkpoint, &hash).expect("verify bound artifact"),
            hash
        );
        fs::write(&checkpoint, b"tampered-forecast-weights").expect("tamper test checkpoint");
        assert!(
            verify_ahem_v61_runtime_artifact_at(&checkpoint, &hash)
                .expect_err("tampered artifact must fail")
                .contains("SHA-256 differs")
        );
        fs::remove_dir_all(directory).expect("clean AHEM runtime test directory");
    }

    #[test]
    fn admitted_forecast_capability_is_bound_to_current_artifact_hashes() {
        let nonce = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("test clock")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("argus-admission-{nonce}"));
        fs::create_dir(&directory).expect("create test model directory");
        let router = directory.join(FORECAST_ROUTER_FILE);
        fs::write(&router, b"router").expect("write router");
        let expert_hashes = TEMPORAL_EXPERT_FILES
            .iter()
            .map(|(id, file_name)| {
                let path = directory.join(file_name);
                fs::write(&path, id.as_bytes()).expect("write expert");
                (
                    (*id).to_owned(),
                    serde_json::Value::String(file_sha256(&path).expect("hash expert")),
                )
            })
            .collect::<serde_json::Map<String, serde_json::Value>>();
        let report = serde_json::json!({
            "policy": {"version": "argus.model-admission/v1"},
            "runtime": {
                "security_verdict_enabled": false,
                "forecast_router_enabled": true
            },
            "decisions": {
                "forecast_router": {
                    "artifact_sha256": file_sha256(&router).expect("hash router"),
                    "checks": [{"name": "expert_hashes", "observed": expert_hashes}]
                }
            }
        });

        assert!(integrity_bound_admission_runtime(&report, &directory).forecast_router_enabled);
        fs::write(&router, b"tampered").expect("tamper router");
        let rejected = integrity_bound_admission_runtime(&report, &directory);
        assert!(!rejected.forecast_router_enabled);
        assert!(rejected.policy.contains("integrity failed"));
        fs::remove_dir_all(directory).expect("clean test directory");
    }

    #[test]
    fn security_artifact_cannot_enable_an_unwired_inference_path() {
        let directory =
            std::env::temp_dir().join(format!("argus-security-admission-{}", std::process::id()));
        fs::create_dir_all(&directory).expect("create security admission directory");
        let detector = directory.join(SECURITY_DETECTOR_FILE);
        fs::write(&detector, b"detector").expect("write detector artifact");
        let report = serde_json::json!({
            "policy": {"version": "argus.model-admission/v1"},
            "runtime": {
                "security_verdict_enabled": true,
                "forecast_router_enabled": false
            },
            "decisions": {
                "security_detector": {
                    "artifact_sha256": file_sha256(&detector).expect("hash detector")
                }
            }
        });

        let runtime = integrity_bound_admission_runtime(&report, &directory);
        assert!(!runtime.security_verdict_enabled);
        assert!(runtime.policy.contains("execution path unavailable"));
        fs::remove_dir_all(directory).expect("remove security admission directory");
    }

    #[test]
    fn alert_manager_records_bounded_alerts_and_tracks_severities() {
        use argus_detector::{AlertCategory, AnomalyRecord, AnomalySeverity};
        let mut manager = super::AlertManager::default();

        for i in 0..10 {
            manager.record(AnomalyRecord {
                timestamp_ns: 1000 + i,
                flow_id: None,
                category: AlertCategory::TcpEvasion,
                severity: if i % 2 == 0 {
                    AnomalySeverity::Critical
                } else {
                    AnomalySeverity::Suspicious
                },
                score: 1.0,
                threshold: 1.0,
                summary: format!("Alert {i}"),
                detail: format!("Detail {i}"),
            });
        }

        let stats = manager.stats();
        assert_eq!(stats.total_alerts, 10);
        assert_eq!(stats.critical_count, 5);
        assert_eq!(stats.suspicious_count, 5);

        let critical_only = manager.get_alerts(None, Some("critical"));
        assert_eq!(critical_only.len(), 5);

        let recent_three = manager.get_alerts(Some(3), None);
        assert_eq!(recent_three.len(), 3);
        assert_eq!(recent_three[0].summary, "Alert 9");

        manager.clear();
        assert_eq!(manager.stats().total_alerts, 0);
    }

    #[test]
    fn retained_alert_counts_do_not_include_evicted_events() {
        use argus_detector::{AlertCategory, AnomalyRecord, AnomalySeverity};
        let mut manager = AlertManager::default();
        for index in 0..501 {
            manager.record(AnomalyRecord {
                timestamp_ns: index,
                flow_id: None,
                category: AlertCategory::ForecastResidual,
                severity: if index == 0 {
                    AnomalySeverity::Critical
                } else {
                    AnomalySeverity::Suspicious
                },
                score: 1.0,
                threshold: 1.0,
                summary: format!("Diagnostic {index}"),
                detail: String::new(),
            });
        }
        let stats = manager.stats();
        assert_eq!(stats.total_alerts, 500);
        assert_eq!(stats.critical_count, 0);
        assert_eq!(stats.suspicious_count, 500);
    }

    #[test]
    fn forecast_baselines_are_isolated_by_model_and_loss_contract() {
        let mut manager = super::AlertManager::default();
        for index in 0..9 {
            assert!(
                manager
                    .evaluate_forecast_error(index, 0.01, "legacy-unweighted")
                    .is_none()
            );
            assert!(
                manager
                    .evaluate_forecast_error(index, 10.0, "v61-weighted")
                    .is_none()
            );
        }
        assert_eq!(manager.forecast_trackers.len(), 2);
    }

    #[test]
    fn reports_directory_is_under_development_root() {
        let reports = super::reports_directory().expect("reports directory");
        assert!(reports.is_dir());
        assert!(reports.to_string_lossy().contains("reports"));
    }

    #[test]
    fn packet_inspection_generates_valid_hexdump_and_layer_ranges() {
        let dummy = [
            0x45, 0x00, 0x00, 0x28, 0x12, 0x34, 0x40, 0x00, 0x40, 0x06, 0x00, 0x00, 127, 0, 0, 1,
            127, 0, 0, 1,
        ];
        let dump = argus_capture::format_hexdump(&dummy);
        assert!(dump.contains("0000: "));
        assert!(dump.contains("45 00 00 28"));
    }

    #[test]
    fn simulation_scenarios_are_registered_and_generate_parseable_pcaps() {
        let scenarios = super::list_simulation_scenarios();
        assert_eq!(scenarios.len(), 5);
        for sc in scenarios {
            let pcap = argus_capture::generate_simulation_pcap(&sc.id, 15)
                .expect("simulation pcap succeeds");
            assert!(pcap.len() > 100);
        }
    }

    #[test]
    #[ignore = "requires the installed Windows model runtime and real checkpoints"]
    fn simulation_path_reaches_forecast_worker_without_live_state() {
        let capture = capture_directory()
            .expect("capture directory")
            .join(format!(
                "sim_nominal_baseline_traffic_e2e_{}.pcap",
                std::process::id()
            ));
        let _guard = super::TemporaryGeneratedCapture::new([capture.clone()]);
        let bytes = argus_capture::generate_simulation_pcap("nominal_baseline_traffic", 160)
            .expect("generate simulation capture");
        fs::write(&capture, bytes).expect("write simulation capture");
        let mut isolated = CnnWorker::default();
        let result = super::run_experimental_cnn(
            &capture,
            "simulation:e2e-test",
            true,
            false,
            &[],
            &mut isolated,
        )
        .expect("simulation model inference");
        let models = result["models"].as_object().expect("model states");
        for (name, model) in models {
            println!(
                "MODEL {name}: state={} backend={} scored={} forecast_error={}",
                model["state"].as_str().unwrap_or("unknown"),
                model["runtime_backend"].as_str().unwrap_or("unspecified"),
                model["forecasts_scored"].as_u64().unwrap_or(0),
                model["forecast_error"],
            );
        }
        println!("ROUTER {}", result["router"]);
        println!("REJECTED {}", result["rejected_models"]);
        let completed = models
            .values()
            .filter(|model| model["state"] == "forecast_only")
            .count();
        assert!(completed > 0, "no forecast model executed: {models:?}");
        assert_eq!(
            models
                .get("ahem_v61")
                .and_then(|model| model["state"].as_str()),
            Some("forecast_only"),
            "frozen AHEM v6.1 did not execute: {}",
            result["rejected_models"]
        );
        assert!(
            models.values().any(|model| {
                model["state"] == "forecast_only"
                    && model["forecast_error"].as_f64().is_some_and(f64::is_finite)
            }),
            "no executed model returned a finite forecast error"
        );
        assert!(
            result["history_scope"] == "simulation:e2e-test" || result["history_scope"].is_null()
        );
    }

    #[test]
    fn traffic_rules_distinguish_blocklist_allowlist_and_firewall_enforcement() {
        let mut manager = super::BlacklistManager::default();
        assert!(!manager.is_blocked("192.168.1.100"));

        let rule = manager
            .block_ip(
                "192.168.1.100",
                "TCP Evasion Probe".to_owned(),
                Some("Critical".to_owned()),
            )
            .expect("block succeeds");
        assert_eq!(rule.target, "192.168.1.100");
        assert!(manager.is_blocklisted("192.168.1.100"));
        assert!(manager.is_blocklisted("192.168.1.100:443"));
        assert!(!manager.is_watchlisted("192.168.1.100"));
        assert!(!manager.is_blocked("192.168.1.100"));

        let stats = manager.stats();
        assert_eq!(stats.active_rules_count, 1);
        assert_eq!(stats.total_dropped_packets, 0);
        assert_eq!(stats.total_dropped_bytes, 0);
        assert_eq!(stats.block_rules_count, 1);

        manager
            .upsert_rule(
                "192.168.1.100",
                "trusted lab host".to_owned(),
                Some("Manual".to_owned()),
                super::TrafficRuleAction::Allow,
            )
            .expect("allow succeeds");
        assert!(manager.is_allowlisted("192.168.1.100"));
        assert!(!manager.is_blocklisted("192.168.1.100"));

        let unblocked = manager
            .unblock_ip("192.168.1.100")
            .expect("unblock succeeds");
        assert!(unblocked);
        assert!(!manager.is_allowlisted("192.168.1.100"));

        let subnet = manager
            .upsert_rule(
                "10.20.30.99/24",
                "restricted lab network".to_owned(),
                Some("Critical".to_owned()),
                super::TrafficRuleAction::Block,
            )
            .expect("CIDR block succeeds");
        assert_eq!(subnet.target, "10.20.30.0/24");
        assert!(manager.is_blocklisted("10.20.30.42"));
        assert!(!manager.is_blocklisted("10.20.31.42"));

        manager
            .upsert_rule(
                "10.20.30.42",
                "specific trusted host".to_owned(),
                Some("Manual".to_owned()),
                super::TrafficRuleAction::Allow,
            )
            .expect("specific allow succeeds");
        assert!(manager.is_allowlisted("10.20.30.42:443"));
        assert!(!manager.is_blocklisted("10.20.30.42"));
        assert!(manager.is_blocklisted("10.20.30.43"));

        let ipv6 = manager
            .upsert_rule(
                "2001:db8:abcd::99/64",
                "IPv6 test range".to_owned(),
                None,
                super::TrafficRuleAction::Watch,
            )
            .expect("IPv6 CIDR succeeds");
        assert_eq!(ipv6.target, "2001:db8:abcd::/64");
        assert!(manager.is_watchlisted("[2001:db8:abcd::1234]:443"));
        assert!(!manager.is_watchlisted("2001:db8:abce::1"));
    }

    #[test]
    fn legacy_confirmed_firewall_rule_migrates_to_block_action() {
        let mut rule: super::BlockRuleDto = serde_json::from_value(serde_json::json!({
            "id": "legacy-rule",
            "target": "203.0.113.9",
            "reason": "legacy",
            "severity": "Manual",
            "createdAtSec": 1,
            "createdAtFormatted": "00:00:01",
            "droppedPackets": 0,
            "droppedBytes": 0,
            "active": true,
            "osFirewallRule": true
        }))
        .expect("deserialize legacy rule");
        assert!(matches!(rule.action, super::TrafficRuleAction::Watch));
        super::migrate_legacy_rule_action(&mut rule);
        assert!(matches!(rule.action, super::TrafficRuleAction::Block));

        let mut manager = super::BlacklistManager::default();
        manager.settings.enabled = false;
        manager.rules.push(rule);
        assert!(manager.is_blocked("203.0.113.9"));
        assert!(!manager.is_blocklisted("203.0.113.9"));
    }

    #[test]
    fn deterministic_attack_rules_detect_tcp_evasion_and_honor_allowlist() {
        let mut flow = history_run("attack-session", 1).flows.remove(0);
        flow.tcp_syn_fin_packets = 1;
        let rules = super::BlacklistManager::default();
        super::apply_attack_detection(std::slice::from_mut(&mut flow), &rules);
        let verdict = flow.attack_verdict.as_ref().expect("attack verdict");
        assert_eq!(verdict.category, "tcp_evasion");
        assert_eq!(verdict.severity, "critical");
        assert_eq!(verdict.detector, "argus.deterministic-attack-rules/v1");

        let mut allow_rules = super::BlacklistManager::default();
        allow_rules
            .upsert_rule(
                &flow.initiator_address,
                "trusted test host".to_owned(),
                None,
                super::TrafficRuleAction::Allow,
            )
            .expect("allow rule");
        flow.is_allowlisted = true;
        super::apply_attack_detection(std::slice::from_mut(&mut flow), &allow_rules);
        assert!(flow.attack_verdict.is_none());

        let mut inbound_attack = history_run("attack-session", 2).flows.remove(0);
        inbound_attack.tcp_syn_fin_packets = 1;
        let mut responder_rules = super::BlacklistManager::default();
        responder_rules
            .upsert_rule(
                &inbound_attack.responder_address,
                "trusted protected service".to_owned(),
                None,
                super::TrafficRuleAction::Allow,
            )
            .expect("responder allow rule");
        inbound_attack.is_allowlisted = true;
        super::apply_attack_detection(std::slice::from_mut(&mut inbound_attack), &responder_rules);
        assert_eq!(
            inbound_attack
                .attack_verdict
                .as_ref()
                .expect("responder allow rule cannot suppress attacker evidence")
                .category,
            "tcp_evasion"
        );
    }

    #[test]
    fn scan_detection_is_session_scoped_and_time_bounded() {
        let rules = super::BlacklistManager::default();
        let mut split_sessions = history_run("scan-a", 10).flows;
        split_sessions.extend(history_run("scan-b", 10).flows);
        for (index, flow) in split_sessions.iter_mut().enumerate() {
            flow.responder_port = 10_000 + u16::try_from(index).expect("test port");
            flow.tcp_syn_packets = 1;
            flow.last_timestamp_ns = u64::try_from(index).expect("test timestamp") * 1_000;
        }
        super::apply_attack_detection(&mut split_sessions, &rules);
        assert!(
            split_sessions
                .iter()
                .all(|flow| flow.attack_verdict.is_none()),
            "flows from separate capture sessions cannot combine into a scan"
        );

        let mut one_session = history_run("scan-one", 20).flows;
        for (index, flow) in one_session.iter_mut().enumerate() {
            flow.responder_port = 20_000 + u16::try_from(index).expect("test port");
            flow.tcp_syn_packets = 1;
            flow.last_timestamp_ns = u64::try_from(index).expect("test timestamp") * 1_000;
        }
        super::apply_attack_detection(&mut one_session, &rules);
        assert_eq!(
            one_session
                .iter()
                .filter_map(|flow| flow.attack_verdict.as_ref())
                .filter(|verdict| verdict.category == "tcp_scan")
                .count(),
            1,
            "only the latest triggering flow receives the scan verdict"
        );

        let mut stale = history_run("scan-stale", 20).flows;
        for (index, flow) in stale.iter_mut().enumerate() {
            flow.responder_port = 30_000 + u16::try_from(index).expect("test port");
            flow.tcp_syn_packets = 1;
            flow.last_timestamp_ns = if index == 19 {
                120_000_000_000
            } else {
                u64::try_from(index).expect("test timestamp") * 1_000
            };
        }
        super::apply_attack_detection(&mut stale, &rules);
        assert!(
            stale.iter().all(|flow| flow.attack_verdict.is_none()),
            "SYN observations older than the 60-second window cannot accumulate"
        );
    }

    #[test]
    fn packet_scenario_replay_detects_recon_and_keeps_baseline_nominal() {
        let directory =
            std::env::temp_dir().join(format!("argus-attack-e2e-{}", std::process::id()));
        fs::create_dir_all(&directory).expect("create attack e2e directory");

        let recon_path = directory.join("recon.pcap");
        fs::write(
            &recon_path,
            argus_capture::generate_simulation_pcap("tcp_recon_evasion", 64)
                .expect("generate recon pcap"),
        )
        .expect("write recon pcap");
        let recon = super::replay_path(&recon_path).expect("replay recon pcap");
        assert!(recon.flows.iter().any(|flow| {
            flow.attack_verdict
                .as_ref()
                .is_some_and(|verdict| verdict.category == "tcp_evasion")
        }));

        let baseline_path = directory.join("baseline.pcap");
        fs::write(
            &baseline_path,
            argus_capture::generate_simulation_pcap("nominal_baseline_traffic", 50)
                .expect("generate baseline pcap"),
        )
        .expect("write baseline pcap");
        let baseline = super::replay_path(&baseline_path).expect("replay baseline pcap");
        assert!(
            baseline
                .flows
                .iter()
                .all(|flow| flow.attack_verdict.is_none())
        );

        fs::remove_dir_all(directory).expect("remove attack e2e directory");
    }

    #[test]
    fn flow_packet_collection_filters_by_pipeline_flow_id() {
        let directory =
            std::env::temp_dir().join(format!("argus-flow-filter-test-{}", std::process::id()));
        fs::create_dir_all(&directory).expect("create flow filter test directory");
        let capture = directory.join("multi-flow.pcap");
        let bytes = argus_capture::generate_simulation_pcap("tcp_recon_evasion", 12)
            .expect("generate multi-flow capture");
        fs::write(&capture, bytes).expect("write multi-flow capture");

        let first = super::collect_flow_packets(&capture, 0, None).expect("collect first flow");
        let second = super::collect_flow_packets(&capture, 1, None).expect("collect second flow");
        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1);
        assert_ne!(first[0].bytes, second[0].bytes);
        assert!(super::collect_flow_packets(&capture, 99, None).is_err());

        fs::remove_dir_all(directory).expect("remove flow filter test directory");
    }

    #[test]
    fn native_forecast_loss_matches_smooth_l1_contract() {
        assert!(smooth_l1(2.0, 2.0).abs() < f32::EPSILON);
        assert!((smooth_l1(1.5, 1.0) - 0.125).abs() < f32::EPSILON);
        assert!((smooth_l1(3.0, 1.0) - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    #[ignore = "requires ARGUS_ONNX_RUNTIME, ARGUS_ONNX_MODELS_DIR, and ARGUS_ONNX_FEATURES"]
    fn native_onnx_desktop_path_runs_real_features() {
        let models = std::env::var_os("ARGUS_ONNX_MODELS_DIR")
            .map(PathBuf::from)
            .expect("ARGUS_ONNX_MODELS_DIR");
        let features = std::env::var_os("ARGUS_ONNX_FEATURES")
            .map(PathBuf::from)
            .expect("ARGUS_ONNX_FEATURES");
        let mut worker = NativeOnnxWorker::load(&models).expect("native worker loads");
        let result = worker
            .infer(&features, "native-e2e-test", true, None)
            .expect("native inference succeeds");
        assert_eq!(result["state"], "active");
        assert_eq!(result["models"]["cnn_local"]["state"], "forecast_only");
        assert_eq!(
            result["models"]["cnn_local"]["runtime_backend"],
            "onnxruntime"
        );
    }

    #[test]
    #[ignore = "requires the local model runtime and real ARGUS_ONNX_FEATURES"]
    fn hybrid_desktop_worker_uses_native_and_python_backends() {
        let models = std::env::var_os("ARGUS_ONNX_MODELS_DIR")
            .map(PathBuf::from)
            .expect("ARGUS_ONNX_MODELS_DIR");
        let features = std::env::var_os("ARGUS_ONNX_FEATURES")
            .map(PathBuf::from)
            .expect("ARGUS_ONNX_FEATURES");
        let mut worker = CnnWorker::default();
        let result = worker
            .infer(&models, &features, "hybrid-e2e-test", true, None, false)
            .expect("hybrid inference succeeds");
        assert_eq!(result["native_onnx"]["state"], "active");
        assert_eq!(result["router"]["state"], "active");
        let selected = selected_forecast_experts(&result).expect("active route has a selection");
        assert_eq!(
            result["models"]["cnn_local"]["state"],
            if selected.contains("cnn_local") {
                "forecast_only"
            } else {
                "skipped_by_router"
            }
        );
        assert_eq!(
            result["models"]["cnn_local"]["runtime_backend"],
            "onnxruntime"
        );
        assert_eq!(
            result["models"]["ahem_v61"]["runtime_backend"],
            "pytorch+onnxruntime"
        );
        assert_eq!(result["models"]["ahem_v61"]["forecasts_scored"], 1);
        assert!(
            result["models"]["ahem_v61"]["eligible_forecasts"]
                .as_u64()
                .is_some_and(|count| count > 1)
        );
    }

    #[test]
    fn test_release_live_capture_deletes_existing_file() {
        let live_dir = capture_directory().expect("capture directory").join("live");
        fs::create_dir_all(&live_dir).expect("create live dir");
        let test_file = live_dir.join("live-999999-1.pcap");
        fs::write(&test_file, b"test-pcap-payload").expect("write test file");
        assert!(test_file.exists());

        let result = release_live_capture(test_file.to_string_lossy().into_owned())
            .expect("release succeeds");
        assert_eq!(result.status, ReleaseCaptureStatus::Deleted);
        assert!(result.file_existed);
        assert!(!result.path_still_exists);
        assert!(!test_file.exists());
    }

    #[test]
    fn test_release_live_capture_already_absent() {
        let live_dir = capture_directory().expect("capture directory").join("live");
        let test_file = live_dir.join("live-888888-99.pcap");
        if test_file.exists() {
            let _ = fs::remove_file(&test_file);
        }

        let result = release_live_capture(test_file.to_string_lossy().into_owned())
            .expect("release succeeds");
        assert_eq!(result.status, ReleaseCaptureStatus::AlreadyAbsent);
        assert!(!result.file_existed);
        assert!(!result.path_still_exists);
    }

    #[test]
    fn test_release_live_capture_rejects_non_generated_path() {
        let captures = capture_directory().expect("capture directory");
        let non_generated = captures.join("external-capture.pcap");
        fs::write(&non_generated, b"not generated").expect("write non-generated");
        let error = release_live_capture(non_generated.to_string_lossy().into_owned())
            .expect_err("must reject non-generated capture");
        assert!(error.contains("only generated live capture segments can be released"));
        let _ = fs::remove_file(non_generated);
    }

    #[test]
    fn test_sensor_health_deserializes_retention_backlog_fields() {
        let json = serde_json::json!({
            "schema": "argus.pktmon-live-health",
            "version": 1,
            "process_id": 1234,
            "active": true,
            "packets": 500,
            "missed_writes": 0,
            "missed_reads": 0,
            "application_drops": 0,
            "malformed": 0,
            "unsupported": 0,
            "duplicate_appearances": 0,
            "completed_segments": 20,
            "retained_segments": 10,
            "retention_evictions": 10,
            "retention_delete_failures": 3,
            "retained_backlog": 2,
            "failed": false
        });
        let health: SensorHealth = serde_json::from_value(json).expect("deserialize health");
        assert_eq!(health.retention_delete_failures, 3);
        assert_eq!(health.retained_backlog, 2);
    }

    #[test]
    fn test_build_topology_from_flows_and_html_dossier() {
        let flows = vec![FlowView {
            uid: "test:0".to_owned(),
            session: "test-sess".to_owned(),
            source: "live.pcap".to_owned(),
            id: 1,
            protocol: "TCP".to_owned(),
            service: Some("HTTPS".to_owned()),
            threat_status: None,
            generation: 0,
            initiator_address: "192.168.1.100".to_owned(),
            initiator_port: 52140,
            responder_address: "198.51.100.50".to_owned(),
            responder_port: 9001, // Tor
            first_timestamp_ns: 100,
            last_timestamp_ns: 200,
            duration_ms: 100.0,
            packets: 20,
            captured_bytes: 5000,
            original_bytes: 5000,
            initiator_packets: 10,
            responder_packets: 10,
            initiator_bytes: 3000,
            responder_bytes: 2000,
            initiator_mac: Some([0x00, 0x15, 0x5d, 0x11, 0x22, 0x33]),
            responder_mac: None,
            tls_sni: None,
            dns_qname: None,
            dns_messages: 0,
            dns_queries: 0,
            http_requests: 0,
            tor_port_matches: 20,
            tcp_syn_packets: 0,
            tcp_syn_fin_packets: 0,
            tcp_xmas_packets: 0,
            tcp_null_packets: 0,
            tcp_syn_rst_packets: 0,
            lost_packets: 0,
            truncated_packets: 0,
            reordered_packets: 0,
            is_watchlisted: false,
            is_blocked: false,
            is_allowlisted: false,
            is_blocklisted: false,
            rule_action: None,
            matched_rule_target: None,
            attack_verdict: Some(super::AttackVerdictDto {
                category: "tcp_evasion".to_owned(),
                severity: "critical".to_owned(),
                score: 1.0,
                threshold: 1.0,
                summary: "TCP evasion signature detected".to_owned(),
                detail: "test evidence".to_owned(),
                detector: "argus.deterministic-attack-rules/v1".to_owned(),
                evidence: vec!["syn_fin_packets=1".to_owned()],
                confidence: "high".to_owned(),
            }),
        }];

        let topology = build_topology_from_flows(&flows);
        assert_eq!(topology.nodes.len(), 2);
        assert_eq!(topology.edges.len(), 1);
        assert!(topology.edges[0].is_tor);
        assert!(topology.edges[0].is_malicious);
        assert_eq!(topology.tor_nodes_count, 1);
        assert_eq!(topology.malicious_nodes_count, 2);

        let html =
            format_network_dossier_html("TEST-ARGUS-XRAY", 123_456_789, &topology, &flows, &[]);
        assert!(html.contains("ARGUS X-RAY // FORENSIC NETWORK DOSSIER"));
        assert!(html.contains("192.168.1.100"));
        assert!(html.contains("198.51.100.50"));
        assert!(html.contains("Tor-Port Heuristic"));
        assert!(html.contains("Rule / Policy Flag"));
    }

    #[test]
    fn dossier_html_escapes_network_controlled_text() {
        assert_eq!(
            escape_html("<script>alert('x') & \"y\"</script>"),
            "&lt;script&gt;alert(&#39;x&#39;) &amp; &quot;y&quot;&lt;/script&gt;"
        );
    }

    #[test]
    fn test_flow_view_compact_serialization() {
        let flow = FlowView {
            uid: "flow-1".to_owned(),
            session: "sess".to_owned(),
            source: "test.pcap".to_owned(),
            id: 1,
            protocol: "TCP".to_owned(),
            service: None,
            threat_status: None,
            generation: 0,
            initiator_address: "10.0.0.1".to_owned(),
            initiator_port: 1234,
            responder_address: "10.0.0.2".to_owned(),
            responder_port: 80,
            first_timestamp_ns: 0,
            last_timestamp_ns: 10,
            duration_ms: 0.01,
            packets: 1,
            captured_bytes: 64,
            original_bytes: 64,
            initiator_packets: 1,
            responder_packets: 0,
            initiator_bytes: 64,
            responder_bytes: 0,
            initiator_mac: None,
            responder_mac: None,
            tls_sni: None,
            dns_qname: None,
            dns_messages: 0,
            dns_queries: 0,
            http_requests: 0,
            tor_port_matches: 0,
            tcp_syn_packets: 0,
            tcp_syn_fin_packets: 0,
            tcp_xmas_packets: 0,
            tcp_null_packets: 0,
            tcp_syn_rst_packets: 0,
            lost_packets: 0,
            truncated_packets: 0,
            reordered_packets: 0,
            is_watchlisted: false,
            is_blocked: false,
            is_allowlisted: false,
            is_blocklisted: false,
            rule_action: None,
            matched_rule_target: None,
            attack_verdict: None,
        };
        let json = serde_json::to_string(&flow).expect("serialize flow");
        assert!(!json.contains("service"), "service None must be omitted");
        assert!(
            !json.contains("threatStatus"),
            "threatStatus None must be omitted"
        );
        assert!(
            !json.contains("isWatchlisted"),
            "isWatchlisted false must be omitted"
        );
        assert!(
            !json.contains("isBlocked"),
            "isBlocked false must be omitted"
        );

        let deserialized: FlowView = serde_json::from_str(&json).expect("deserialize flow");
        assert_eq!(deserialized.service, None);
        assert_eq!(deserialized.threat_status, None);
        assert!(!deserialized.is_watchlisted);
        assert!(!deserialized.is_blocked);
        assert_eq!(deserialized.initiator_address, "10.0.0.1");
    }

    #[test]
    fn test_topology_node_compact_serialization() {
        use argus_capture::{NodeRole, TopologyNode};

        let node = TopologyNode {
            id: "10.0.0.1".to_string(),
            label: "10.0.0.1".to_string(),
            ip: "10.0.0.1".to_string(),
            mac: None,
            role: NodeRole::LanHost,
            vendor: None,
            total_in_bytes: 512,
            total_out_bytes: 1024,
            packet_count: 8,
            is_tor: false,
            threat_score: 0.0,
            active_ports: Vec::new(),
            entity_name: None,
            organization: None,
            asn: None,
            category: None,
            device_type: None,
            description: None,
            confidence: None,
        };

        let json = serde_json::to_string(&node).expect("serialize compact node");
        assert!(!json.contains("mac"), "mac should be omitted when None");
        assert!(
            !json.contains("vendor"),
            "vendor should be omitted when None"
        );
        assert!(
            !json.contains("activePorts"),
            "activePorts should be omitted when empty"
        );
        assert!(!json.contains("asn"), "asn should be omitted when None");
        assert!(
            !json.contains("entityName"),
            "entityName should be omitted when None"
        );

        let deserialized: TopologyNode =
            serde_json::from_str(&json).expect("deserialize compact node");
        assert_eq!(deserialized, node);
    }

    #[test]
    fn test_build_topology_prepass_deduplication() {
        let flows: Vec<FlowView> = (0..50)
            .map(|i| FlowView {
                uid: format!("test:{i}"),
                session: "test-sess".to_owned(),
                source: "live.pcap".to_owned(),
                id: i,
                protocol: "TCP".to_owned(),
                service: Some("HTTPS".to_owned()),
                threat_status: None,
                generation: 0,
                initiator_address: "192.168.1.50".to_owned(),
                initiator_port: 40000 + (u16::try_from(i % 10).unwrap_or(0)),
                responder_address: "1.1.1.1".to_owned(),
                responder_port: 443,
                first_timestamp_ns: 100,
                last_timestamp_ns: 200,
                duration_ms: 10.0,
                packets: 2,
                captured_bytes: 100,
                original_bytes: 100,
                initiator_packets: 1,
                responder_packets: 1,
                initiator_bytes: 60,
                responder_bytes: 40,
                initiator_mac: None,
                responder_mac: None,
                tls_sni: None,
                dns_qname: None,
                dns_messages: 0,
                dns_queries: 0,
                http_requests: 0,
                tor_port_matches: 0,
                tcp_syn_packets: 0,
                tcp_syn_fin_packets: 0,
                tcp_xmas_packets: 0,
                tcp_null_packets: 0,
                tcp_syn_rst_packets: 0,
                lost_packets: 0,
                truncated_packets: 0,
                reordered_packets: 0,
                is_watchlisted: false,
                is_blocked: false,
                is_allowlisted: false,
                is_blocklisted: false,
                rule_action: None,
                matched_rule_target: None,
                attack_verdict: None,
            })
            .collect();

        let topology = build_topology_from_flows(&flows);
        assert_eq!(topology.nodes.len(), 2);
        assert_eq!(topology.edges.len(), 1);

        let dns_node = topology
            .nodes
            .iter()
            .find(|n| n.ip == "1.1.1.1")
            .expect("1.1.1.1 node exists");
        assert_eq!(
            dns_node.entity_name.as_deref(),
            Some("Cloudflare Public DNS")
        );
    }
}

use std::{
    collections::HashMap,
    env,
    error::Error,
    fmt::Write as _,
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    num::{NonZeroU32, NonZeroU64, NonZeroUsize},
    path::{Path, PathBuf},
    process::ExitCode,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use argus_capture::{
    CaptureSource, PcapLimits, PcapLoadReport, PcapReplaySource, PktmonProbe, PktmonReadiness,
    PktmonRecordConfig, PktmonRecording,
};
use argus_core::FlowInstanceId;
use argus_features::{FEATURE_FIELDS, FeatureVector};
use argus_nids::{Pipeline, PipelineConfig};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("argus-nids: {error}");
            ExitCode::FAILURE
        }
    }
}

// Argument routing and the two explicit output formats stay together in this small binary.
#[allow(clippy::too_many_lines)]
fn run() -> Result<(), Box<dyn Error>> {
    let mut arguments: Vec<String> = env::args().skip(1).collect();
    let json = arguments
        .first()
        .is_some_and(|argument| argument == "--json");
    if json {
        arguments.remove(0);
    }
    if arguments.as_slice() == ["--pktmon-probe"] {
        print_pktmon_probe(json);
        return Ok(());
    }
    if let [flag, capture, output] = arguments.as_slice()
        && flag == "--export-features"
    {
        return export_real_features(Path::new(capture), Path::new(output));
    }
    if let [flag, seconds, output_directory, stop_file] = arguments.as_slice()
        && flag == "--pktmon-rolling"
    {
        return run_pktmon_rolling(
            seconds.parse::<u64>()?,
            Path::new(output_directory),
            Path::new(stop_file),
        );
    }
    let limits = PcapLimits {
        max_file_bytes: NonZeroUsize::new(512 * 1024 * 1024).expect("non-zero literal"),
        max_records: NonZeroUsize::new(5_000_000).expect("non-zero literal"),
    };
    let (mut source, pcap_report, benchmark): (
        Box<dyn CaptureSource>,
        Option<PcapLoadReport>,
        bool,
    ) = match arguments.as_slice() {
        [flag, path] if flag == "--pcap" || flag == "--benchmark" => {
            let pcap = PcapReplaySource::from_path(Path::new(path), limits)?;
            let load_report = pcap.report();
            (Box::new(pcap), Some(load_report), flag == "--benchmark")
        }
        [flag, seconds, etl, pcapng] if flag == "--pktmon-record" => {
            let config = PktmonRecordConfig {
                duration_seconds: NonZeroU64::new(seconds.parse::<u64>()?)
                    .ok_or("Pktmon duration must be non-zero")?,
                etl_base_path: PathBuf::from(etl),
                pcapng_path: PathBuf::from(pcapng),
                file_size_megabytes: NonZeroU32::new(1_024).expect("non-zero literal"),
            };
            let recording = PktmonRecording::record(&config, limits)?;
            let load_report = recording.replay.report();
            (Box::new(recording.replay), Some(load_report), false)
        }
        _ => {
            return Err(
                "usage: argus-nids [--json] [--pcap <real-pcap-or-pcapng-file> | --benchmark <real-pcap-or-pcapng-file> | --export-features <real-pcap-or-pcapng-file> <absolute-new-csv-path> | --pktmon-probe | --pktmon-record <seconds> <absolute-etl-path> <absolute-pcapng-path>]".into(),
            );
        }
    };
    let started = Instant::now();
    let report = Pipeline::new(PipelineConfig::default()).run(source.as_mut())?;
    let elapsed = started.elapsed();

    if json {
        print_json_report(&report, pcap_report, elapsed, benchmark)?;
        return Ok(());
    }

    println!(
        "source={} session={} kind={:?} transport_tracking={:?}",
        report.source.source_id,
        report.session.session_id,
        report.source.kind,
        report.transport_tracking,
    );
    println!(
        "detector={} v{} trained_model={} feature_schema={} v{}",
        report.detector.provenance.detector_id,
        report.detector.provenance.detector_version,
        report.detector.trained_model,
        report.detector.provenance.feature_schema.schema_id,
        report.detector.provenance.feature_schema.schema_version,
    );
    println!("stop={:?}", report.stop_reason);
    if let Some(pcap) = pcap_report {
        println!(
            "pcap records={} emitted={} unsupported_link={} unsupported_network={} fragmented={}",
            pcap.records_seen,
            pcap.packets_emitted,
            pcap.unsupported_link_packets,
            pcap.unsupported_network_packets,
            pcap.fragmented_packets,
        );
    }
    println!(
        "health source_records={} normalized={} non_ip={} unsupported={} malformed={} captured={} processed={} flows={} alerts={} warming_up={} ready={} lost={} truncated={} reordered={} clock_rollbacks={} queue_high_water={} queue_overflows={} parse_errors={} source_errors={} state_resets={}",
        report.health.source_records,
        report.health.normalized_packets,
        report.health.non_ip_records,
        report.health.unsupported_records,
        report.health.malformed_records,
        report.health.captured_packets,
        report.health.processed_packets,
        report.health.flows,
        report.health.alerts,
        report.health.warming_up_evaluations,
        report.health.ready_evaluations,
        report.health.reported_lost_packets,
        report.health.truncated_packets,
        report.health.reordered_packets,
        report.health.clock_rollbacks,
        report.health.queue_high_water,
        report.health.queue_overflows,
        report.health.capture_parse_errors,
        report.health.capture_source_errors,
        report.health.capture_state_resets,
    );
    for alert in &report.alerts {
        println!(
            "alert flow={:?} timestamp_ns={} evidence={:?}",
            alert.flow, alert.monotonic_timestamp_ns, alert.evidence
        );
    }
    if benchmark {
        let packets_per_second = if elapsed.as_nanos() > 0 {
            u128::from(report.health.processed_packets) * 1_000_000_000 / elapsed.as_nanos()
        } else {
            0
        };
        println!(
            "benchmark elapsed_ms={:.3} packets_per_second={}",
            elapsed.as_secs_f64() * 1_000.0,
            packets_per_second
        );
    }
    Ok(())
}

fn run_pktmon_rolling(
    segment_seconds: u64,
    output_directory: &Path,
    stop_file: &Path,
) -> Result<(), Box<dyn Error>> {
    if !output_directory.is_absolute() || !output_directory.is_dir() {
        return Err("rolling output must be an existing absolute directory".into());
    }
    if !stop_file.is_absolute() || stop_file.exists() {
        return Err("rolling stop signal must be an absent absolute path".into());
    }
    let duration_seconds =
        NonZeroU64::new(segment_seconds).ok_or("rolling segment duration must be non-zero")?;
    let limits = PcapLimits {
        max_file_bytes: NonZeroUsize::new(512 * 1024 * 1024).expect("non-zero literal"),
        max_records: NonZeroUsize::new(5_000_000).expect("non-zero literal"),
    };
    let mut segment = 0_u64;
    while !stop_file.exists() {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let prefix = format!("rolling-{stamp}-{segment}");
        let config = PktmonRecordConfig {
            duration_seconds,
            etl_base_path: output_directory.join(format!("{prefix}.etl")),
            pcapng_path: output_directory.join(format!("{prefix}.pcapng")),
            file_size_megabytes: NonZeroU32::new(1_024).expect("non-zero literal"),
        };
        let recording = PktmonRecording::record(&config, limits)?;
        println!(
            "segment={} capture={}",
            segment,
            recording.pcapng_path.display()
        );
        std::io::stdout().flush()?;
        segment = segment
            .checked_add(1)
            .ok_or("rolling segment counter overflow")?;
        // Yield between conversion and the next checked start. This mode is
        // intentionally called rolling, not gapless continuous capture.
        thread::sleep(Duration::from_millis(25));
    }
    Ok(())
}

fn export_real_features(capture: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    if !output.is_absolute()
        || output
            .extension()
            .is_none_or(|extension| extension != "csv")
    {
        return Err("feature export output must be an absolute .csv path".into());
    }
    if output.exists() {
        return Err(format!("refusing to overwrite feature export {}", output.display()).into());
    }
    let parent = output
        .parent()
        .ok_or("feature export output has no parent")?;
    if !parent.is_dir() {
        return Err(format!(
            "feature export directory does not exist: {}",
            parent.display()
        )
        .into());
    }
    let limits = PcapLimits {
        max_file_bytes: NonZeroUsize::new(512 * 1024 * 1024).expect("non-zero literal"),
        max_records: NonZeroUsize::new(5_000_000).expect("non-zero literal"),
    };
    let mut replay = PcapReplaySource::from_path(capture, limits)?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let mut writer = RuntimeCsvWriter::new(file)?;
    let report = Pipeline::new(PipelineConfig::default())
        .run_with_feature_observer(&mut replay, &mut |feature| writer.write(feature))?;
    writer.finish()?;
    println!(
        "exported schema=argus.causal-flow/v1 rows={} flows={} output={}",
        report.health.processed_packets,
        writer.flow_count(),
        output.display()
    );
    Ok(())
}

struct RuntimeCsvWriter {
    output: BufWriter<File>,
    flows: HashMap<FlowInstanceId, u64>,
}

impl RuntimeCsvWriter {
    fn new(file: File) -> Result<Self, std::io::Error> {
        let mut output = BufWriter::new(file);
        write!(
            output,
            "schema_id,schema_version,source_capture_ordinal,monotonic_timestamp_ns,opaque_flow_id,ip_version,transport_protocol,tcp_syn,tcp_ack,tcp_fin,tcp_rst,tcp_push,tcp_urg,flow_initiator_ip,flow_initiator_port,flow_responder_ip,flow_responder_port,flow_generation"
        )?;
        for field in FEATURE_FIELDS {
            write!(output, ",{}", field.name)?;
        }
        writeln!(output)?;
        Ok(Self {
            output,
            flows: HashMap::new(),
        })
    }

    fn write(&mut self, feature: &FeatureVector) -> Result<(), String> {
        let next_id = u64::try_from(self.flows.len()).map_err(|_| "flow ID overflow")?;
        let flow_id = *self.flows.entry(feature.flow.clone()).or_insert(next_id);
        write!(
            self.output,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            feature.schema_id,
            feature.schema_version,
            feature.values[11],
            feature.monotonic_timestamp_ns,
            flow_id,
            feature.protocol.ip_version,
            match feature.flow.key.protocol {
                argus_core::TransportProtocol::Tcp => "tcp",
                argus_core::TransportProtocol::Udp => "udp",
                argus_core::TransportProtocol::Other(_) => "other",
            },
            u8::from(
                feature
                    .protocol
                    .tcp_flags
                    .is_some_and(|flags| flags & 0x02 != 0)
            ),
            u8::from(
                feature
                    .protocol
                    .tcp_flags
                    .is_some_and(|flags| flags & 0x10 != 0)
            ),
            u8::from(
                feature
                    .protocol
                    .tcp_flags
                    .is_some_and(|flags| flags & 0x01 != 0)
            ),
            u8::from(
                feature
                    .protocol
                    .tcp_flags
                    .is_some_and(|flags| flags & 0x04 != 0)
            ),
            u8::from(
                feature
                    .protocol
                    .tcp_flags
                    .is_some_and(|flags| flags & 0x08 != 0)
            ),
            u8::from(
                feature
                    .protocol
                    .tcp_flags
                    .is_some_and(|flags| flags & 0x20 != 0)
            ),
            feature.flow.key.initiator.address,
            feature.flow.key.initiator.port,
            feature.flow.key.responder.address,
            feature.flow.key.responder.port,
            feature.flow.generation,
        )
        .map_err(|error| error.to_string())?;
        for value in feature.values {
            write!(self.output, ",{value}").map_err(|error| error.to_string())?;
        }
        writeln!(self.output).map_err(|error| error.to_string())
    }

    fn finish(&mut self) -> Result<(), std::io::Error> {
        self.output.flush()
    }

    fn flow_count(&self) -> usize {
        self.flows.len()
    }
}

fn print_pktmon_probe(json: bool) {
    let probe = PktmonProbe::inspect();
    let readiness = match probe.readiness {
        PktmonReadiness::Unavailable => "unavailable",
        PktmonReadiness::PrivilegeRequired => "privilege_required",
        PktmonReadiness::StatusReachable => "status_reachable",
        PktmonReadiness::ProbeFailed { .. } => "probe_failed",
    };
    let executable = probe
        .executable
        .as_ref()
        .map_or_else(String::new, |path| path.display().to_string());
    if json {
        println!(
            "{{\"schema\":\"argus.pktmon-status\",\"version\":1,\"readiness\":\"{readiness}\",\"executable\":\"{}\",\"live_source\":false}}",
            json_escape(&executable)
        );
    } else {
        println!("pktmon readiness={readiness} executable={executable:?} live_source=false");
    }
}

// Keep the versioned JSON field order visible in one place for contract review.
#[allow(clippy::too_many_lines)]
fn print_json_report(
    report: &argus_nids::RunReport,
    pcap: Option<PcapLoadReport>,
    elapsed: std::time::Duration,
    benchmark: bool,
) -> Result<(), std::fmt::Error> {
    let mut output = String::new();
    write!(
        output,
        "{{\"schema\":\"argus.run-status\",\"version\":2,\"source\":{{\"id\":\"{}\",\"session_id\":\"{}\",\"kind\":\"{:?}\"}},\"transport_tracking\":\"observed_flow_packet_ordinal_only_no_tcp_reassembly\",\"detector\":{{\"id\":\"{}\",\"version\":{},\"algorithm\":\"deterministic_threshold_rules\",\"trained_model\":false,\"artifact_id\":null,\"artifact_checksum\":null,\"feature_schema\":\"{}\",\"feature_schema_version\":{},\"threshold_config\":{{\"id\":\"{}\",\"version\":{},\"readiness_packets\":{},\"maximum_packet_bytes\":{},\"maximum_loss_before\":{}}}}},\"health\":{{\"source_records\":{},\"normalized_packets\":{},\"non_ip_records\":{},\"unsupported_records\":{},\"malformed_records\":{},\"captured\":{},\"processed\":{},\"flows\":{},\"alerts\":{},\"warming_up\":{},\"ready\":{},\"reported_lost\":{},\"truncated\":{},\"reordered\":{},\"queue_high_water\":{},\"queue_overflows\":{},\"capture_parse_errors\":{},\"capture_source_errors\":{},\"clock_rollbacks\":{},\"capture_sessions_started\":{},\"capture_sessions_stopped\":{},\"capture_state_resets\":{}}}",
        json_escape(&report.source.source_id),
        json_escape(&report.session.session_id),
        report.source.kind,
        report.detector.provenance.detector_id,
        report.detector.provenance.detector_version,
        report.detector.provenance.feature_schema.schema_id,
        report.detector.provenance.feature_schema.schema_version,
        report.detector.provenance.threshold_config.config_id,
        report.detector.provenance.threshold_config.config_version,
        report
            .detector
            .provenance
            .threshold_config
            .readiness_packets,
        report
            .detector
            .provenance
            .threshold_config
            .maximum_packet_bytes,
        report
            .detector
            .provenance
            .threshold_config
            .maximum_loss_before,
        report.health.source_records,
        report.health.normalized_packets,
        report.health.non_ip_records,
        report.health.unsupported_records,
        report.health.malformed_records,
        report.health.captured_packets,
        report.health.processed_packets,
        report.health.flows,
        report.health.alerts,
        report.health.warming_up_evaluations,
        report.health.ready_evaluations,
        report.health.reported_lost_packets,
        report.health.truncated_packets,
        report.health.reordered_packets,
        report.health.queue_high_water,
        report.health.queue_overflows,
        report.health.capture_parse_errors,
        report.health.capture_source_errors,
        report.health.clock_rollbacks,
        report.health.capture_sessions_started,
        report.health.capture_sessions_stopped,
        report.health.capture_state_resets,
    )?;
    if let Some(pcap) = pcap {
        write!(
            output,
            ",\"pcap\":{{\"records\":{},\"emitted\":{},\"unsupported_link\":{},\"unsupported_network\":{},\"fragmented\":{},\"ipv4_fragments\":{},\"ipv6_fragments\":{}}}",
            pcap.records_seen,
            pcap.packets_emitted,
            pcap.unsupported_link_packets,
            pcap.unsupported_network_packets,
            pcap.fragmented_packets,
            pcap.unsupported_ipv4_fragment_packets,
            pcap.unsupported_ipv6_fragment_packets,
        )?;
    }
    if benchmark {
        let rate = if elapsed.as_nanos() == 0 {
            0
        } else {
            u128::from(report.health.processed_packets) * 1_000_000_000 / elapsed.as_nanos()
        };
        write!(
            output,
            ",\"benchmark\":{{\"kind\":\"pipeline\",\"elapsed_ns\":{},\"packets_per_second\":{rate}}}",
            elapsed.as_nanos()
        )?;
    }
    output.push_str(",\"alerts\":[");
    for (alert_index, alert) in report.alerts.iter().enumerate() {
        if alert_index > 0 {
            output.push(',');
        }
        write!(
            output,
            "{{\"flow_generation\":{},\"timestamp_ns\":{},\"health\":\"nominal\",\"score\":null,\"reason\":\"{}\",\"evidence\":[",
            alert.flow.generation,
            alert.monotonic_timestamp_ns,
            match alert.reason {
                argus_detector::EvaluationReason::InsufficientPackets => "insufficient_packets",
                argus_detector::EvaluationReason::NoThresholdExceeded => "no_threshold_exceeded",
                argus_detector::EvaluationReason::ThresholdExceeded => "threshold_exceeded",
            }
        )?;
        for (index, evidence) in alert.evidence.iter().enumerate() {
            if index > 0 {
                output.push(',');
            }
            write!(
                output,
                "{{\"feature\":\"{}\",\"comparison\":\"greater_than\",\"observed\":{},\"threshold\":{}}}",
                json_escape(evidence.feature),
                evidence.observed,
                evidence.threshold
            )?;
        }
        output.push_str("]}");
    }
    output.push_str("]}");
    println!("{output}");
    Ok(())
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                let _ = write!(escaped, "\\u{:04x}", u32::from(character));
            }
            character => escaped.push(character),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::export_real_features;

    #[test]
    fn runtime_export_carries_observed_flow_identity_without_changing_feature_schema() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "argus-flow-identity-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir(&directory).expect("test directory");
        let capture = directory.join("baseline.pcap");
        let output = directory.join("features.csv");
        fs::write(
            &capture,
            argus_capture::generate_simulation_pcap("nominal_baseline_traffic", 10)
                .expect("capture"),
        )
        .expect("write capture");
        export_real_features(&capture, &output).expect("export");
        let csv = fs::read_to_string(&output).expect("read export");
        let mut lines = csv.lines();
        let header = lines.next().expect("header");
        assert!(header.contains("flow_initiator_ip,flow_initiator_port,flow_responder_ip,flow_responder_port,flow_generation"));
        assert!(header.contains("packet_length_bytes,captured_length_bytes"));
        let first = lines.next().expect("first row");
        assert!(first.starts_with("argus.causal-flow,1,"));
        assert!(first.contains(",192.168.1.50,51234,142.250.190.46,443,0,"));
        fs::remove_dir_all(directory).expect("clean test directory");
    }
}

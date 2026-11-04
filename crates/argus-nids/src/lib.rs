//! Runnable vertical slice for deterministic capture replay.
//!
//! This crate deliberately exposes a small orchestration layer rather than a live capture
//! implementation. It preserves source quality information through bounded handoff, temporal
//! memory, feature extraction, and the deterministic baseline detector.

use std::{collections::HashMap, fmt, num::NonZeroUsize};

use argus_capture::{
    BoundedCaptureBuffer, CaptureError, CapturePoll, CaptureSessionMetadata, CaptureSource,
    CapturedPacket, PushError, SourceMetadata,
};
use argus_core::{CacheVersion, Direction, FlowInstanceId, IngestError, TemporalMemory};
use argus_detector::{
    Alert, BaselineConfig, DetectorProvenance, EvaluationError, Readiness, ThresholdBaseline,
};
use argus_features::{FEATURE_SCHEMA_VERSION, FeatureError, FeatureVector, PerFlowExtractor};

/// Tunable bounded-resource limits for the vertical-slice runner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PipelineConfig {
    pub queue_capacity: NonZeroUsize,
    /// Hard cap for flow-local memories, feature extractors, and summaries in one source run.
    pub max_tracked_flows: NonZeroUsize,
    pub short_term_capacity: NonZeroUsize,
    pub chunk_size: NonZeroUsize,
    pub max_closed_chunks: NonZeroUsize,
    pub baseline: BaselineConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            queue_capacity: NonZeroUsize::new(64).expect("non-zero literal"),
            max_tracked_flows: NonZeroUsize::new(16_384).expect("non-zero literal"),
            short_term_capacity: NonZeroUsize::new(64).expect("non-zero literal"),
            chunk_size: NonZeroUsize::new(16).expect("non-zero literal"),
            max_closed_chunks: NonZeroUsize::new(8).expect("non-zero literal"),
            baseline: BaselineConfig {
                readiness_packets: 3,
                maximum_packet_bytes: 1_500,
                maximum_loss_before: 0,
            },
        }
    }
}

/// Explicit non-model provenance carried with every run and every emitted alert.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DetectorStatus {
    pub provenance: DetectorProvenance,
    pub trained_model: bool,
}

impl DetectorStatus {
    fn from_detector(detector: &ThresholdBaseline) -> Self {
        let provenance = detector.provenance();
        Self {
            provenance,
            trained_model: provenance.trained_model,
        }
    }
}

/// Terminal source state observed by the runner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StopReason {
    EndOfStream,
    SourceStopped,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransportTrackingStatus {
    /// Flow-local observed packet ordinals only; no TCP sequence/reassembly guarantees.
    ObservedPacketOrdinalOnly,
}

/// Counts that describe what the pipeline actually processed.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HealthCounts {
    pub source_records: u64,
    pub normalized_packets: u64,
    pub non_ip_records: u64,
    pub unsupported_records: u64,
    pub malformed_records: u64,
    pub captured_packets: u64,
    pub processed_packets: u64,
    pub flows: usize,
    pub alerts: u64,
    pub warming_up_evaluations: u64,
    pub ready_evaluations: u64,
    pub reported_lost_packets: u64,
    pub truncated_packets: u64,
    pub reordered_packets: u64,
    pub queue_high_water: usize,
    pub queue_overflows: u64,
    pub capture_parse_errors: u64,
    pub capture_source_errors: u64,
    pub clock_rollbacks: u64,
    pub capture_sessions_started: u64,
    pub capture_sessions_stopped: u64,
    pub capture_state_resets: u64,
}

/// Complete successful result from one source run.
#[derive(Clone, Debug, PartialEq)]
pub struct RunReport {
    pub source: SourceMetadata,
    pub session: CaptureSessionMetadata,
    pub detector: DetectorStatus,
    pub stop_reason: StopReason,
    pub transport_tracking: TransportTrackingStatus,
    pub health: HealthCounts,
    pub flows: Vec<FlowSummary>,
    pub alerts: Vec<Alert>,
}

/// Auditable per-flow facts from the current capture session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FlowSummary {
    pub id: u64,
    pub flow: FlowInstanceId,
    pub first_timestamp_ns: u64,
    pub last_timestamp_ns: u64,
    pub packets: u64,
    pub captured_bytes: u64,
    pub original_bytes: u64,
    pub initiator_packets: u64,
    pub responder_packets: u64,
    pub initiator_bytes: u64,
    pub responder_bytes: u64,
    pub initiator_mac: Option<[u8; 6]>,
    pub responder_mac: Option<[u8; 6]>,
    pub tls_sni: Option<String>,
    pub dns_qname: Option<String>,
    pub dns_messages: u64,
    pub dns_queries: u64,
    pub http_requests: u64,
    pub tor_port_matches: u64,
    pub tcp_syn_packets: u64,
    pub tcp_syn_fin_packets: u64,
    pub tcp_xmas_packets: u64,
    pub tcp_null_packets: u64,
    pub tcp_syn_rst_packets: u64,
    pub reported_lost_packets: u64,
    pub truncated_packets: u64,
    pub reordered_packets: u64,
}

/// Errors remain explicit; capture, queue, core, feature, and detector failures are never hidden.
#[derive(Debug)]
pub enum PipelineError {
    Capture(CaptureError),
    QueueFull(PushError),
    Memory(IngestError),
    Features(FeatureError),
    FeatureObserver(String),
    FlowCapacityExceeded { capacity: usize },
    CounterOverflow,
    Detector(EvaluationError),
    Stop(CaptureError),
}

impl fmt::Display for PipelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capture(error) => write!(formatter, "capture failed: {error}"),
            Self::QueueFull(error) => write!(formatter, "bounded handoff failed: {error}"),
            Self::Memory(error) => write!(formatter, "temporal memory failed: {error}"),
            Self::Features(error) => write!(formatter, "feature extraction failed: {error}"),
            Self::FeatureObserver(error) => write!(formatter, "feature export failed: {error}"),
            Self::FlowCapacityExceeded { capacity } => {
                write!(
                    formatter,
                    "flow tracking capacity of {capacity} was exceeded"
                )
            }
            Self::CounterOverflow => formatter.write_str("flow telemetry counter overflow"),
            Self::Detector(error) => write!(formatter, "detector evaluation failed: {error:?}"),
            Self::Stop(error) => write!(formatter, "clean source stop failed: {error}"),
        }
    }
}

impl std::error::Error for PipelineError {}

/// Single-threaded bounded pipeline. Pull-based capture provides upstream backpressure.
#[derive(Debug)]
pub struct Pipeline {
    config: PipelineConfig,
    detector: ThresholdBaseline,
    memories: HashMap<FlowInstanceId, TemporalMemory>,
    extractors: HashMap<FlowInstanceId, PerFlowExtractor>,
    flow_summaries: HashMap<FlowInstanceId, FlowSummary>,
    next_flow_id: u64,
    last_health: HealthCounts,
}

impl Pipeline {
    #[must_use]
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            detector: ThresholdBaseline::new(config.baseline),
            config,
            memories: HashMap::new(),
            extractors: HashMap::new(),
            flow_summaries: HashMap::new(),
            next_flow_id: 0,
            last_health: HealthCounts::default(),
        }
    }

    /// Runs a source to its terminal state and calls its idempotent stop hook exactly once.
    ///
    /// The queue is drained before every subsequent source poll, so no packet is silently
    /// discarded when downstream processing is slower than a source adapter.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] if capture, bounded handoff, memory, feature extraction,
    /// detector evaluation, or source shutdown fails. A prior processing error takes precedence
    /// over a subsequent shutdown error.
    pub fn run(&mut self, source: &mut dyn CaptureSource) -> Result<RunReport, PipelineError> {
        self.run_with_feature_observer(source, &mut |_| Ok(()))
    }

    /// Runs one source while exposing each exact runtime feature vector to a
    /// fallible observer. This is the only supported feature-export path, so
    /// offline training data has the same causal schema as inference.
    ///
    /// # Errors
    ///
    /// Returns all normal pipeline errors or an observer error. A failed
    /// observer stops the run and source cleanly rather than silently omitting
    /// training rows.
    pub fn run_with_feature_observer(
        &mut self,
        source: &mut dyn CaptureSource,
        observer: &mut dyn FnMut(&FeatureVector) -> Result<(), String>,
    ) -> Result<RunReport, PipelineError> {
        self.reset_capture_state();
        self.last_health = HealthCounts {
            capture_sessions_started: 1,
            capture_state_resets: 1,
            ..HealthCounts::default()
        };
        let metadata = source.metadata().clone();
        let session = source.session_metadata().clone();
        let result = self.run_until_terminal(source, metadata, session, observer);
        let stopped = source.stop().map_err(PipelineError::Stop);
        let result = match (result, stopped) {
            (Ok(mut report), Ok(())) => {
                report.health.capture_sessions_stopped = 1;
                report.health.capture_state_resets += 1;
                self.last_health = report.health.clone();
                Ok(report)
            }
            (Err(error), Ok(())) => {
                self.last_health.capture_sessions_stopped = 1;
                self.last_health.capture_state_resets += 1;
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(_), Err(error)) => Err(error),
        };
        self.reset_capture_state();
        result
    }

    /// Last completed or failed run's quality counters. Available even when [`Self::run`] errors.
    #[must_use]
    pub const fn last_health(&self) -> &HealthCounts {
        &self.last_health
    }

    fn reset_capture_state(&mut self) {
        self.memories.clear();
        self.extractors.clear();
        self.flow_summaries.clear();
        self.next_flow_id = 0;
    }

    fn run_until_terminal(
        &mut self,
        source: &mut dyn CaptureSource,
        metadata: SourceMetadata,
        session: CaptureSessionMetadata,
        observer: &mut dyn FnMut(&FeatureVector) -> Result<(), String>,
    ) -> Result<RunReport, PipelineError> {
        let mut queue = BoundedCaptureBuffer::new(self.config.queue_capacity);
        let mut health = self.last_health.clone();
        let mut alerts = Vec::new();

        loop {
            let poll = match source.poll_next() {
                Ok(poll) => poll,
                Err(error) => {
                    apply_capture_counters(&mut health, source);
                    match &error {
                        CaptureError::Malformed(_) => health.capture_parse_errors += 1,
                        _ => health.capture_source_errors += 1,
                    }
                    self.last_health = health;
                    return Err(PipelineError::Capture(error));
                }
            };
            match poll {
                CapturePoll::Packet(packet) => {
                    health.captured_packets = health
                        .captured_packets
                        .checked_add(1)
                        .expect("packet count cannot overflow in a process lifetime");
                    if let Err(error) = queue.try_push(packet) {
                        health.queue_overflows += 1;
                        self.last_health = health;
                        return Err(PipelineError::QueueFull(error));
                    }
                    health.queue_high_water = health.queue_high_water.max(queue.len());
                    if let Err(error) =
                        self.drain_queue(&mut queue, &mut health, &mut alerts, observer)
                    {
                        self.last_health = health;
                        return Err(error);
                    }
                    self.last_health = health.clone();
                }
                CapturePoll::EndOfStream => {
                    self.drain_queue(&mut queue, &mut health, &mut alerts, observer)?;
                    apply_capture_counters(&mut health, source);
                    return Ok(RunReport {
                        source: metadata,
                        session,
                        detector: DetectorStatus::from_detector(&self.detector),
                        stop_reason: StopReason::EndOfStream,
                        transport_tracking: TransportTrackingStatus::ObservedPacketOrdinalOnly,
                        health,
                        flows: self.sorted_flow_summaries(),
                        alerts,
                    });
                }
                CapturePoll::Stopped => {
                    self.drain_queue(&mut queue, &mut health, &mut alerts, observer)?;
                    apply_capture_counters(&mut health, source);
                    return Ok(RunReport {
                        source: metadata,
                        session,
                        detector: DetectorStatus::from_detector(&self.detector),
                        stop_reason: StopReason::SourceStopped,
                        transport_tracking: TransportTrackingStatus::ObservedPacketOrdinalOnly,
                        health,
                        flows: self.sorted_flow_summaries(),
                        alerts,
                    });
                }
            }
        }
    }

    fn drain_queue(
        &mut self,
        queue: &mut BoundedCaptureBuffer,
        health: &mut HealthCounts,
        alerts: &mut Vec<Alert>,
        observer: &mut dyn FnMut(&FeatureVector) -> Result<(), String>,
    ) -> Result<(), PipelineError> {
        while let Some(packet) = queue.pop() {
            self.process_packet(&packet, health, alerts, observer)?;
        }
        Ok(())
    }

    fn process_packet(
        &mut self,
        packet: &CapturedPacket,
        health: &mut HealthCounts,
        alerts: &mut Vec<Alert>,
        observer: &mut dyn FnMut(&FeatureVector) -> Result<(), String>,
    ) -> Result<(), PipelineError> {
        health.reported_lost_packets = health
            .reported_lost_packets
            .checked_add(packet.quality.packets_lost_before)
            .expect("loss count cannot overflow in a process lifetime");
        health.truncated_packets += usize::from(packet.quality.truncated) as u64;
        health.reordered_packets += usize::from(packet.quality.reordered) as u64;
        if packet.quality.clock_rollback {
            health.clock_rollbacks += 1;
            health.capture_state_resets += 1;
            self.reset_capture_state();
        }

        if !self.memories.contains_key(&packet.flow)
            && self.memories.len() >= self.config.max_tracked_flows.get()
        {
            return Err(PipelineError::FlowCapacityExceeded {
                capacity: self.config.max_tracked_flows.get(),
            });
        }

        let flow = packet.flow.clone();
        self.update_flow_summary(packet)?;
        let version = CacheVersion {
            feature_schema: u64::from(FEATURE_SCHEMA_VERSION),
            model: 0,
        };
        let memory = self.memories.entry(flow.clone()).or_insert_with(|| {
            TemporalMemory::new_at_sequence(
                flow.clone(),
                version,
                self.config.short_term_capacity,
                self.config.chunk_size,
                self.config.max_closed_chunks,
                packet.packet.sequence,
            )
        });
        memory
            .ingest(packet.packet.clone(), version)
            .map_err(PipelineError::Memory)?;
        let extractor = self
            .extractors
            .entry(flow)
            .or_insert_with_key(|flow| PerFlowExtractor::new(flow.clone()));
        let features = extractor.ingest(packet).map_err(PipelineError::Features)?;
        observer(&features).map_err(PipelineError::FeatureObserver)?;
        let evaluation = self
            .detector
            .evaluate(&features)
            .map_err(PipelineError::Detector)?;
        match evaluation.readiness {
            Readiness::WarmingUp { .. } => health.warming_up_evaluations += 1,
            Readiness::Ready => health.ready_evaluations += 1,
        }
        // The shipped detector is an explicitly untrained threshold baseline.
        // It may inform development diagnostics, but it is never permitted to
        // emit a security alert. A trained artifact must replace it first.
        if self.detector.provenance().trained_model
            && let Some(alert) = self.detector.alert(&features, &evaluation)
        {
            health.alerts += 1;
            alerts.push(alert);
        }
        health.processed_packets += 1;
        health.flows = self.memories.len();
        Ok(())
    }

    fn update_flow_summary(&mut self, packet: &CapturedPacket) -> Result<(), PipelineError> {
        if !self.flow_summaries.contains_key(&packet.flow) {
            let id = self.next_flow_id;
            self.next_flow_id = self
                .next_flow_id
                .checked_add(1)
                .ok_or(PipelineError::CounterOverflow)?;
            self.flow_summaries.insert(
                packet.flow.clone(),
                FlowSummary {
                    id,
                    flow: packet.flow.clone(),
                    first_timestamp_ns: packet.packet.monotonic_timestamp_ns,
                    last_timestamp_ns: packet.packet.monotonic_timestamp_ns,
                    packets: 0,
                    captured_bytes: 0,
                    original_bytes: 0,
                    initiator_packets: 0,
                    responder_packets: 0,
                    initiator_bytes: 0,
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
                    reported_lost_packets: 0,
                    truncated_packets: 0,
                    reordered_packets: 0,
                },
            );
        }
        let summary = self
            .flow_summaries
            .get_mut(&packet.flow)
            .ok_or(PipelineError::CounterOverflow)?;
        summary.last_timestamp_ns = packet.packet.monotonic_timestamp_ns;
        summary.packets = checked_add(summary.packets, 1)?;
        summary.captured_bytes = checked_add(
            summary.captured_bytes,
            u64::from(packet.packet.captured_len),
        )?;
        summary.original_bytes = checked_add(
            summary.original_bytes,
            u64::from(packet.packet.original_len),
        )?;
        update_directional_observations(summary, packet)?;
        summary.reported_lost_packets = checked_add(
            summary.reported_lost_packets,
            packet.quality.packets_lost_before,
        )?;
        summary.truncated_packets = checked_add(
            summary.truncated_packets,
            u64::from(packet.quality.truncated),
        )?;
        summary.reordered_packets = checked_add(
            summary.reordered_packets,
            u64::from(packet.quality.reordered),
        )?;
        Ok(())
    }

    fn sorted_flow_summaries(&self) -> Vec<FlowSummary> {
        let mut summaries: Vec<_> = self.flow_summaries.values().cloned().collect();
        summaries.sort_unstable_by_key(|summary| summary.id);
        summaries
    }
}

fn update_directional_observations(
    summary: &mut FlowSummary,
    packet: &CapturedPacket,
) -> Result<(), PipelineError> {
    match packet.packet.direction {
        Direction::InitiatorToResponder => {
            summary.initiator_packets = checked_add(summary.initiator_packets, 1)?;
            summary.initiator_bytes = checked_add(
                summary.initiator_bytes,
                u64::from(packet.packet.captured_len),
            )?;
            if summary.initiator_mac.is_none() {
                summary.initiator_mac = packet.observations.source_mac;
            }
            if summary.responder_mac.is_none() {
                summary.responder_mac = packet.observations.destination_mac;
            }
        }
        Direction::ResponderToInitiator => {
            summary.responder_packets = checked_add(summary.responder_packets, 1)?;
            summary.responder_bytes = checked_add(
                summary.responder_bytes,
                u64::from(packet.packet.captured_len),
            )?;
            if summary.responder_mac.is_none() {
                summary.responder_mac = packet.observations.source_mac;
            }
            if summary.initiator_mac.is_none() {
                summary.initiator_mac = packet.observations.destination_mac;
            }
        }
    }
    if summary.tls_sni.is_none() {
        summary.tls_sni.clone_from(&packet.observations.tls_sni);
    }
    if summary.dns_qname.is_none() {
        summary.dns_qname.clone_from(&packet.observations.dns_qname);
    }
    summary.dns_messages = checked_add(
        summary.dns_messages,
        u64::from(packet.observations.dns_message),
    )?;
    summary.dns_queries = checked_add(
        summary.dns_queries,
        u64::from(packet.observations.dns_query),
    )?;
    summary.http_requests = checked_add(
        summary.http_requests,
        u64::from(packet.observations.http_request),
    )?;
    summary.tor_port_matches = checked_add(
        summary.tor_port_matches,
        u64::from(packet.observations.tor_port_match),
    )?;
    if let Some(flags) = packet.protocol.tcp_flags {
        let syn = flags & 0x02 != 0;
        let fin = flags & 0x01 != 0;
        let rst = flags & 0x04 != 0;
        let psh = flags & 0x08 != 0;
        let urg = flags & 0x20 != 0;
        summary.tcp_syn_packets = checked_add(summary.tcp_syn_packets, u64::from(syn))?;
        summary.tcp_syn_fin_packets =
            checked_add(summary.tcp_syn_fin_packets, u64::from(syn && fin))?;
        summary.tcp_xmas_packets =
            checked_add(summary.tcp_xmas_packets, u64::from(fin && psh && urg))?;
        summary.tcp_null_packets = checked_add(summary.tcp_null_packets, u64::from(flags == 0))?;
        summary.tcp_syn_rst_packets =
            checked_add(summary.tcp_syn_rst_packets, u64::from(syn && rst))?;
    }
    Ok(())
}

fn checked_add(left: u64, right: u64) -> Result<u64, PipelineError> {
    left.checked_add(right)
        .ok_or(PipelineError::CounterOverflow)
}

fn apply_capture_counters(health: &mut HealthCounts, source: &dyn CaptureSource) {
    let counters = source.counters();
    health.source_records = counters.source_records;
    health.normalized_packets = counters.normalized_packets;
    health.non_ip_records = counters.non_ip_records;
    health.unsupported_records = counters.unsupported_records;
    health.malformed_records = counters.malformed_records;
    health.reported_lost_packets = counters.packets_lost;
    health.truncated_packets = counters.truncated_packets;
    health.reordered_packets = counters.reordered_packets;
    health.clock_rollbacks = counters.clock_rollbacks;
}

#[cfg(test)]
mod tests {
    use super::*;
    use argus_capture::{
        LinkType, PacketProtocolMetadata, QualityMarkers, ReplaySource, SourceKind,
    };
    use argus_core::{Direction, Endpoint, FlowKey, PacketEvent, TransportProtocol};
    use std::{net::IpAddr, num::NonZeroUsize};

    fn packet(ordinal: u64, sequence: u64, original_len: u32) -> CapturedPacket {
        CapturedPacket {
            capture_ordinal: ordinal,
            flow: FlowInstanceId {
                key: FlowKey {
                    initiator: Endpoint {
                        address: "192.0.2.10".parse::<IpAddr>().expect("IP"),
                        port: 50_000,
                    },
                    responder: Endpoint {
                        address: "198.51.100.20".parse::<IpAddr>().expect("IP"),
                        port: 443,
                    },
                    protocol: TransportProtocol::Tcp,
                },
                generation: 0,
            },
            packet: PacketEvent {
                sequence,
                monotonic_timestamp_ns: 1_000 + ordinal,
                captured_len: original_len,
                original_len,
                direction: Direction::InitiatorToResponder,
            },
            protocol: PacketProtocolMetadata {
                ip_version: 4,
                tcp_flags: Some(0),
            },
            observations: argus_capture::PacketObservationMetadata::default(),
            quality: QualityMarkers::default(),
        }
    }

    fn source(records: Vec<CapturedPacket>) -> ReplaySource {
        ReplaySource::new(
            SourceMetadata::new(
                "unit-replay",
                SourceKind::Replay,
                LinkType::Ethernet,
                NonZeroUsize::new(2_000).expect("non-zero"),
            )
            .expect("metadata"),
            records,
            NonZeroUsize::new(8).expect("non-zero"),
        )
        .expect("source")
    }

    fn packet_for_port(ordinal: u64, port: u16) -> CapturedPacket {
        let mut value = packet(ordinal, 0, 100);
        value.flow.key.initiator.port = port;
        value
    }

    #[test]
    fn untrained_baseline_never_emits_a_security_alert() {
        let mut replay = source(vec![
            packet(0, 0, 100),
            packet(1, 1, 100),
            packet(2, 2, 1_501),
        ]);
        let report = Pipeline::new(PipelineConfig::default())
            .run(&mut replay)
            .expect("run");

        assert_eq!(report.stop_reason, StopReason::EndOfStream);
        assert!(!report.detector.trained_model);
        assert_eq!(report.health.processed_packets, 3);
        assert_eq!(report.health.warming_up_evaluations, 2);
        assert_eq!(report.health.ready_evaluations, 1);
        assert_eq!(report.health.queue_high_water, 1);
        assert_eq!(report.alerts.len(), 0);
        assert_eq!(report.health.alerts, 0);
    }

    #[test]
    fn source_errors_are_reported_and_source_is_stopped() {
        let mut malformed = packet(1, 0, 100);
        malformed.quality.packets_lost_before = 0;
        let mut replay = source(vec![malformed]);
        let mut pipeline = Pipeline::new(PipelineConfig::default());
        assert!(matches!(
            pipeline.run(&mut replay),
            Err(PipelineError::Capture(CaptureError::Malformed(_)))
        ));
        assert_eq!(pipeline.last_health().capture_parse_errors, 1);
        assert_eq!(pipeline.last_health().capture_sessions_stopped, 1);
        assert_eq!(replay.poll_next().expect("stopped"), CapturePoll::Stopped);
    }

    #[test]
    fn flow_tracking_capacity_fails_closed() {
        let config = PipelineConfig {
            max_tracked_flows: NonZeroUsize::new(2).expect("non-zero"),
            ..PipelineConfig::default()
        };
        let mut replay = source(vec![
            packet_for_port(0, 40_000),
            packet_for_port(1, 40_001),
            packet_for_port(2, 40_002),
        ]);
        let mut pipeline = Pipeline::new(config);

        assert!(matches!(
            pipeline.run(&mut replay),
            Err(PipelineError::FlowCapacityExceeded { capacity: 2 })
        ));
        assert_eq!(pipeline.last_health().processed_packets, 2);
    }

    #[test]
    fn capture_sessions_do_not_reuse_temporal_state() {
        let mut pipeline = Pipeline::new(PipelineConfig::default());
        let mut first = source(vec![packet(0, 0, 100)]);
        let mut second = source(vec![packet(0, 0, 100)]);
        let first_report = pipeline.run(&mut first).expect("first session");
        let second_report = pipeline.run(&mut second).expect("second session");
        assert_eq!(first_report.health.processed_packets, 1);
        assert_eq!(second_report.health.processed_packets, 1);
        assert_eq!(second_report.health.capture_state_resets, 2);
    }

    #[test]
    fn clock_rollback_resets_temporal_state_and_preserves_flow_ordinal() {
        let mut after_rollback = packet(1, 1, 100);
        after_rollback.quality.reordered = true;
        after_rollback.quality.clock_rollback = true;
        let mut replay = source(vec![packet(0, 0, 100), after_rollback]);
        let report = Pipeline::new(PipelineConfig::default())
            .run(&mut replay)
            .expect("clock-normalized session");
        assert_eq!(report.health.clock_rollbacks, 1);
        assert_eq!(report.health.capture_state_resets, 3);
        assert_eq!(report.health.processed_packets, 2);
    }

    #[test]
    fn flow_summary_preserves_directional_bytes_and_packet_observations() {
        let mut request = packet(0, 0, 120);
        request.observations.source_mac = Some([0, 1, 2, 3, 4, 5]);
        request.observations.destination_mac = Some([6, 7, 8, 9, 10, 11]);
        request.observations.tls_sni = Some("api.example.test".to_owned());
        request.observations.http_request = true;
        request.observations.dns_message = true;
        request.observations.dns_query = true;

        let mut response = packet(1, 1, 80);
        response.packet.direction = Direction::ResponderToInitiator;
        response.observations.source_mac = Some([6, 7, 8, 9, 10, 11]);
        response.observations.destination_mac = Some([0, 1, 2, 3, 4, 5]);
        response.observations.dns_message = true;

        let mut replay = source(vec![request, response]);
        let report = Pipeline::new(PipelineConfig::default())
            .run(&mut replay)
            .expect("run");
        let summary = report.flows.first().expect("flow summary");
        assert_eq!(summary.initiator_bytes, 120);
        assert_eq!(summary.responder_bytes, 80);
        assert_eq!(summary.initiator_mac, Some([0, 1, 2, 3, 4, 5]));
        assert_eq!(summary.responder_mac, Some([6, 7, 8, 9, 10, 11]));
        assert_eq!(summary.tls_sni.as_deref(), Some("api.example.test"));
        assert_eq!(summary.http_requests, 1);
        assert_eq!(summary.dns_messages, 2);
        assert_eq!(summary.dns_queries, 1);
    }
}

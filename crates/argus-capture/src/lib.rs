//! Platform-neutral packet-capture contracts and deterministic replay.

use std::{
    collections::VecDeque,
    fmt,
    num::{NonZeroU64, NonZeroUsize},
};

use argus_core::{FlowInstanceId, PacketEvent};
use uuid::Uuid;

pub mod asset_db;
pub mod dissector;
pub mod filter;
pub mod inspector;
pub mod linux;
pub mod oui;
mod pcap;
mod pktmon;
pub mod simulator;
pub mod topology;
pub mod writer;

pub use asset_db::{AssetFingerprint, identify_asset};
pub use dissector::{
    DnsSummary, HttpSummary, IcmpSummary, is_known_tor_port, is_tor_traffic, parse_dns_summary,
    parse_http_summary, parse_icmp_summary, parse_tls_sni,
};
pub use filter::PacketFilter;
pub use inspector::{PacketInspection, format_hexdump, inspect_frame};
pub use oui::{format_mac, lookup_vendor, parse_mac};
pub use pcap::{PcapLimits, PcapLoadReport, PcapReplaySource};
pub use pktmon::{PktmonError, PktmonProbe, PktmonReadiness, PktmonRecordConfig, PktmonRecording};
pub use simulator::{
    AttackTechnique, SimulationScenario, generate_simulation_pcap, list_available_scenarios,
};
pub use topology::{
    FlowTopologyRecord, NetworkTopology, NodeRole, ProtocolDistribution, TopologyEdge,
    TopologyNode, build_network_topology, classify_endpoint_role, is_private_or_local_ip,
};
pub use writer::write_pcap_file;

/// Stable identity and capabilities reported by an opened source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceMetadata {
    pub source_id: String,
    pub kind: SourceKind,
    pub link_type: LinkType,
    pub snapshot_length: NonZeroUsize,
}

impl SourceMetadata {
    /// Creates validated source metadata.
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError::InvalidMetadata`] for an empty source identifier.
    pub fn new(
        source_id: impl Into<String>,
        kind: SourceKind,
        link_type: LinkType,
        snapshot_length: NonZeroUsize,
    ) -> Result<Self, CaptureError> {
        let source_id = source_id.into();
        if source_id.trim().is_empty() {
            return Err(CaptureError::InvalidMetadata("source_id must not be empty"));
        }
        Ok(Self {
            source_id,
            kind,
            link_type,
            snapshot_length,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceKind {
    Replay,
    WindowsPktmon,
    WindowsNpcap,
    Other,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinkType {
    Ethernet,
    RawIp,
    Other(u32),
}

/// Identity of one open/recording interval for a capture source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CaptureSessionMetadata {
    pub session_id: String,
}

impl CaptureSessionMetadata {
    /// Creates capture-session metadata whose identifier is stable for the session lifetime.
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError::InvalidMetadata`] when the identifier is empty.
    pub fn new(session_id: impl Into<String>) -> Result<Self, CaptureError> {
        let session_id = session_id.into();
        if session_id.trim().is_empty() {
            return Err(CaptureError::InvalidMetadata(
                "session_id must not be empty",
            ));
        }
        Ok(Self { session_id })
    }

    /// Creates an opaque identifier for one distinct source-open interval.
    ///
    /// The human-readable source identity remains in [`SourceMetadata`]. A random UUID here
    /// prevents a replay with the same filename from inheriting capture-scoped cache state.
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError::InvalidMetadata`] when the prefix is empty.
    pub fn fresh(prefix: &str) -> Result<Self, CaptureError> {
        if prefix.trim().is_empty() {
            return Err(CaptureError::InvalidMetadata(
                "session prefix must not be empty",
            ));
        }
        Self::new(format!("{prefix}:{}", Uuid::new_v4()))
    }
}

/// Native clock value retained before timestamp normalization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawTimestamp {
    pub ticks: u64,
    pub ticks_per_second: NonZeroU64,
}

/// Why a raw record cannot be normalized into the supported IP/TCP/UDP model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnsupportedReason {
    LinkProtocol,
    NetworkProtocol,
    TransportProtocol,
    FragmentedIpv4,
    FragmentedIpv6,
    Other,
}

/// Classification of bytes supplied by a capture provider.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RawPacketClassification {
    SupportedIp,
    NonIp,
    Unsupported(UnsupportedReason),
    /// A provider may deliver a malformed record; the reason must be retained verbatim.
    Malformed(String),
}

/// Explicit state of conversion from a raw provider record to Argus' packet model.
///
/// `Normalized` means structural IP and transport normalization only. It does not claim that TCP
/// sequence numbers have been tracked or validated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormalizationStatus {
    NotAttempted,
    Normalized,
    Unsupported,
    Malformed,
}

/// Provider and adapter quality facts for a canonical raw record.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RawPacketQuality {
    pub packets_lost_before: u64,
    pub reordered: bool,
    pub truncated: bool,
    /// The provider clock moved backwards before monotonic normalization.
    pub clock_rollback: bool,
}

/// Platform-neutral, loss-aware record exchanged by capture adapters.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawPacket {
    pub source: SourceMetadata,
    pub session: CaptureSessionMetadata,
    /// Ordinal in the provider stream, including unsupported and malformed records.
    pub source_ordinal: u64,
    pub raw_timestamp: RawTimestamp,
    /// Nondecreasing adapter-normalized time used for ordering; the raw clock remains above.
    pub monotonic_timestamp_ns: u64,
    pub captured_len: u32,
    pub original_len: u32,
    pub bytes: Vec<u8>,
    pub quality: RawPacketQuality,
    pub classification: RawPacketClassification,
    pub normalization: NormalizationStatus,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RawCapturePoll {
    Packet(RawPacket),
    EndOfStream,
    Stopped,
}

/// Raw provider boundary. Every source record is emitted once; unsupported and malformed records
/// are data with explicit classifications, never successful empty polls or silent drops.
pub trait RawCaptureSource {
    fn metadata(&self) -> &SourceMetadata;
    fn session_metadata(&self) -> &CaptureSessionMetadata;

    /// Polls exactly one raw provider record or a terminal state.
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError`] when the provider fails or its stream framing cannot be read.
    fn poll_next_raw(&mut self) -> Result<RawCapturePoll, CaptureError>;

    /// Requests an idempotent stop of the raw provider stream.
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError`] when a concrete provider cannot be stopped cleanly.
    fn stop_raw(&mut self) -> Result<(), CaptureError>;
}

/// Exhaustive adapter result. Unsupported and malformed inputs remain paired with their raw
/// record, preventing callers from accidentally discarding them through an `Option`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdapterOutcome {
    Normalized {
        raw: RawPacket,
        packet: CapturedPacket,
    },
    Unnormalized(RawPacket),
}

pub trait CaptureAdapter {
    /// Attempts structural normalization while preserving the input record in every outcome.
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError`] for adapter failures. Unsupported or malformed packet bytes are
    /// represented by [`AdapterOutcome::Unnormalized`] instead of errors or dropped records.
    fn normalize(&mut self, packet: RawPacket) -> Result<AdapterOutcome, CaptureError>;
}

/// Explicit source-quality facts attached to a normalized packet.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QualityMarkers {
    /// Number of source packets known to be missing immediately before this packet.
    pub packets_lost_before: u64,
    /// The adapter received fewer bytes than the original on-wire length.
    pub truncated: bool,
    /// The provider reported this packet out of native order; emission was normalized.
    pub reordered: bool,
    /// The raw capture clock moved backwards. Downstream temporal state must be reset.
    pub clock_rollback: bool,
}

/// Protocol facts decoded from bytes at capture time. They are kept separate
/// from the frozen causal-flow feature vector so adapters can reproduce older
/// packet contracts without guessing or reparsing discarded frames.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PacketProtocolMetadata {
    pub ip_version: u8,
    pub tcp_flags: Option<u8>,
}

/// Optional observations decoded directly from the captured frame. These facts
/// are deliberately kept out of the model feature vector: they support
/// evidence-backed topology and reporting without changing the frozen model
/// schema or requiring the desktop to reparse captures that may be retired.
// These independent wire observations are not mutually exclusive states.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PacketObservationMetadata {
    pub source_mac: Option<[u8; 6]>,
    pub destination_mac: Option<[u8; 6]>,
    pub tls_sni: Option<String>,
    pub dns_qname: Option<String>,
    pub dns_message: bool,
    pub dns_query: bool,
    pub http_request: bool,
    pub tor_port_match: bool,
}

/// A normalized packet ready for flow temporal memory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapturedPacket {
    /// Source-global ordinal. It starts at zero and gaps require an exact loss marker.
    pub capture_ordinal: u64,
    pub flow: FlowInstanceId,
    pub packet: PacketEvent,
    pub protocol: PacketProtocolMetadata,
    pub observations: PacketObservationMetadata,
    pub quality: QualityMarkers,
}

/// Monotonic source/adapter counters available to downstream health reporting.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CaptureCounters {
    pub source_records: u64,
    pub normalized_packets: u64,
    pub non_ip_records: u64,
    pub unsupported_records: u64,
    pub malformed_records: u64,
    pub packets_lost: u64,
    pub truncated_packets: u64,
    pub reordered_packets: u64,
    pub clock_rollbacks: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CapturePoll {
    Packet(CapturedPacket),
    EndOfStream,
    Stopped,
}

/// Pull-based source boundary. One poll produces at most one packet, providing natural
/// backpressure; adapters must not hide internal drops.
pub trait CaptureSource {
    fn metadata(&self) -> &SourceMetadata;
    fn session_metadata(&self) -> &CaptureSessionMetadata;
    fn counters(&self) -> CaptureCounters;

    /// Polls one normalized packet or terminal state.
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError`] when source data violates the capture contract or the source
    /// itself fails. Errors must never be converted into empty successful packets.
    fn poll_next(&mut self) -> Result<CapturePoll, CaptureError>;

    /// Requests a clean, idempotent stop.
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError`] if a concrete adapter cannot stop its underlying provider.
    fn stop(&mut self) -> Result<(), CaptureError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CaptureError {
    InvalidMetadata(&'static str),
    ReplayCapacityExceeded { capacity: usize, supplied: usize },
    Malformed(MalformedCapture),
    SourceFailure(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MalformedCapture {
    NonMonotonicOrdinal {
        expected_at_least: u64,
        received: u64,
    },
    UnexplainedOrdinalGap {
        expected: u64,
        received: u64,
        reported_loss: u64,
    },
    CapturedLengthExceedsOriginal,
    TruncationMarkerMismatch,
    SnapshotLengthExceeded {
        snapshot_length: usize,
        captured_length: usize,
    },
    OrdinalOverflow,
}

impl fmt::Display for CaptureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMetadata(message) => {
                write!(formatter, "invalid capture metadata: {message}")
            }
            Self::ReplayCapacityExceeded { capacity, supplied } => write!(
                formatter,
                "replay contains {supplied} records but its capacity is {capacity}"
            ),
            Self::Malformed(reason) => write!(formatter, "malformed capture record: {reason:?}"),
            Self::SourceFailure(message) => write!(formatter, "capture source failed: {message}"),
        }
    }
}

impl std::error::Error for CaptureError {}

/// Deterministic, capacity-bounded replay used for tests and future recorded-capture ingestion.
#[derive(Clone, Debug)]
pub struct ReplaySource {
    metadata: SourceMetadata,
    session: CaptureSessionMetadata,
    records: VecDeque<CapturedPacket>,
    next_ordinal: u64,
    stopped: bool,
    counters: CaptureCounters,
}

impl ReplaySource {
    /// Creates a replay source without pre-validating records, so malformed-input behavior can
    /// be tested through the same polling boundary used by live adapters.
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError::ReplayCapacityExceeded`] when records exceed the configured bound,
    /// or [`CaptureError::InvalidMetadata`] when metadata does not identify a replay source.
    pub fn new(
        metadata: SourceMetadata,
        records: Vec<CapturedPacket>,
        capacity: NonZeroUsize,
    ) -> Result<Self, CaptureError> {
        let session = CaptureSessionMetadata::fresh("replay")?;
        Self::new_with_session(metadata, session, records, capacity)
    }

    pub(crate) fn new_with_session(
        metadata: SourceMetadata,
        session: CaptureSessionMetadata,
        records: Vec<CapturedPacket>,
        capacity: NonZeroUsize,
    ) -> Result<Self, CaptureError> {
        if metadata.kind != SourceKind::Replay {
            return Err(CaptureError::InvalidMetadata(
                "ReplaySource requires SourceKind::Replay",
            ));
        }
        if records.len() > capacity.get() {
            return Err(CaptureError::ReplayCapacityExceeded {
                capacity: capacity.get(),
                supplied: records.len(),
            });
        }
        Ok(Self {
            metadata,
            session,
            records: records.into(),
            next_ordinal: 0,
            stopped: false,
            counters: CaptureCounters::default(),
        })
    }

    fn validate(&self, record: &CapturedPacket) -> Result<(), CaptureError> {
        if record.capture_ordinal < self.next_ordinal {
            return Err(CaptureError::Malformed(
                MalformedCapture::NonMonotonicOrdinal {
                    expected_at_least: self.next_ordinal,
                    received: record.capture_ordinal,
                },
            ));
        }
        let expected = self
            .next_ordinal
            .checked_add(record.quality.packets_lost_before)
            .ok_or(CaptureError::Malformed(MalformedCapture::OrdinalOverflow))?;
        if record.capture_ordinal != expected {
            return Err(CaptureError::Malformed(
                MalformedCapture::UnexplainedOrdinalGap {
                    expected: self.next_ordinal,
                    received: record.capture_ordinal,
                    reported_loss: record.quality.packets_lost_before,
                },
            ));
        }
        if record.packet.captured_len > record.packet.original_len {
            return Err(CaptureError::Malformed(
                MalformedCapture::CapturedLengthExceedsOriginal,
            ));
        }
        let actually_truncated = record.packet.captured_len < record.packet.original_len;
        if record.quality.truncated != actually_truncated {
            return Err(CaptureError::Malformed(
                MalformedCapture::TruncationMarkerMismatch,
            ));
        }
        if record.packet.captured_len as usize > self.metadata.snapshot_length.get() {
            return Err(CaptureError::Malformed(
                MalformedCapture::SnapshotLengthExceeded {
                    snapshot_length: self.metadata.snapshot_length.get(),
                    captured_length: record.packet.captured_len as usize,
                },
            ));
        }
        Ok(())
    }
}

impl CaptureSource for ReplaySource {
    fn metadata(&self) -> &SourceMetadata {
        &self.metadata
    }

    fn session_metadata(&self) -> &CaptureSessionMetadata {
        &self.session
    }

    fn counters(&self) -> CaptureCounters {
        self.counters
    }

    fn poll_next(&mut self) -> Result<CapturePoll, CaptureError> {
        if self.stopped {
            return Ok(CapturePoll::Stopped);
        }
        let Some(record) = self.records.front() else {
            return Ok(CapturePoll::EndOfStream);
        };
        self.validate(record)?;
        let record = self.records.pop_front().expect("front record was present");
        self.counters.source_records += 1;
        self.counters.normalized_packets += 1;
        self.counters.packets_lost += record.quality.packets_lost_before;
        self.counters.truncated_packets += u64::from(record.quality.truncated);
        self.counters.reordered_packets += u64::from(record.quality.reordered);
        self.counters.clock_rollbacks += u64::from(record.quality.clock_rollback);
        self.next_ordinal = record
            .capture_ordinal
            .checked_add(1)
            .ok_or(CaptureError::Malformed(MalformedCapture::OrdinalOverflow))?;
        Ok(CapturePoll::Packet(record))
    }

    fn stop(&mut self) -> Result<(), CaptureError> {
        self.stopped = true;
        self.records.clear();
        Ok(())
    }
}

/// Explicit bounded handoff between a capture producer and downstream processing.
#[derive(Clone, Debug)]
pub struct BoundedCaptureBuffer {
    capacity: NonZeroUsize,
    queue: VecDeque<CapturedPacket>,
}

impl BoundedCaptureBuffer {
    #[must_use]
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            capacity,
            queue: VecDeque::with_capacity(capacity.get()),
        }
    }

    /// Enqueues without blocking or silently dropping.
    ///
    /// # Errors
    ///
    /// Returns [`PushError`] containing the unconsumed packet when the buffer is full.
    pub fn try_push(&mut self, packet: CapturedPacket) -> Result<(), PushError> {
        if self.queue.len() == self.capacity.get() {
            return Err(PushError {
                packet: Box::new(packet),
                capacity: self.capacity.get(),
            });
        }
        self.queue.push_back(packet);
        Ok(())
    }

    pub fn pop(&mut self) -> Option<CapturedPacket> {
        self.queue.pop_front()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PushError {
    pub packet: Box<CapturedPacket>,
    pub capacity: usize,
}

impl fmt::Display for PushError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "capture buffer is full at capacity {}",
            self.capacity
        )
    }
}

impl std::error::Error for PushError {}

#[cfg(test)]
mod tests {
    use super::*;
    use argus_core::{Direction, Endpoint, FlowKey, TransportProtocol};
    use std::{net::IpAddr, num::NonZeroUsize};

    fn metadata() -> SourceMetadata {
        SourceMetadata::new(
            "test-replay",
            SourceKind::Replay,
            LinkType::Ethernet,
            NonZeroUsize::new(128).expect("non-zero"),
        )
        .expect("valid metadata")
    }

    fn packet(capture_ordinal: u64, flow_sequence: u64) -> CapturedPacket {
        CapturedPacket {
            capture_ordinal,
            flow: FlowInstanceId {
                key: FlowKey {
                    initiator: Endpoint {
                        address: "192.0.2.1".parse::<IpAddr>().expect("IP"),
                        port: 50_000,
                    },
                    responder: Endpoint {
                        address: "198.51.100.2".parse::<IpAddr>().expect("IP"),
                        port: 443,
                    },
                    protocol: TransportProtocol::Tcp,
                },
                generation: 0,
            },
            packet: PacketEvent {
                sequence: flow_sequence,
                monotonic_timestamp_ns: 1_000 + capture_ordinal,
                captured_len: 64,
                original_len: 64,
                direction: Direction::InitiatorToResponder,
            },
            protocol: PacketProtocolMetadata {
                ip_version: 4,
                tcp_flags: Some(0),
            },
            observations: PacketObservationMetadata::default(),
            quality: QualityMarkers::default(),
        }
    }

    fn replay(records: Vec<CapturedPacket>) -> ReplaySource {
        ReplaySource::new(metadata(), records, NonZeroUsize::new(8).expect("non-zero"))
            .expect("replay")
    }

    #[test]
    fn replay_is_deterministic_and_ends_cleanly() {
        let expected = vec![packet(0, 0), packet(1, 1)];
        let mut source = replay(expected.clone());
        for packet in expected {
            assert_eq!(
                source.poll_next().expect("poll"),
                CapturePoll::Packet(packet)
            );
        }
        assert_eq!(source.poll_next().expect("poll"), CapturePoll::EndOfStream);
        assert_eq!(source.poll_next().expect("poll"), CapturePoll::EndOfStream);
    }

    #[test]
    fn each_replay_open_gets_a_distinct_opaque_session() {
        let first = replay(Vec::new());
        let second = replay(Vec::new());
        assert!(first.session.session_id.starts_with("replay:"));
        assert!(second.session.session_id.starts_with("replay:"));
        assert_ne!(first.session, second.session);
    }

    #[test]
    fn exact_loss_marker_explains_an_ordinal_gap() {
        let mut after_loss = packet(3, 1);
        after_loss.quality.packets_lost_before = 2;
        let mut source = replay(vec![packet(0, 0), after_loss.clone()]);
        assert!(matches!(source.poll_next(), Ok(CapturePoll::Packet(_))));
        assert_eq!(
            source.poll_next().expect("poll"),
            CapturePoll::Packet(after_loss)
        );
    }

    #[test]
    fn reorder_and_valid_truncation_markers_are_preserved() {
        let mut marked = packet(0, 0);
        marked.packet.captured_len = 60;
        marked.quality.truncated = true;
        marked.quality.reordered = true;
        let mut source = replay(vec![marked.clone()]);
        assert_eq!(
            source.poll_next().expect("poll"),
            CapturePoll::Packet(marked)
        );
    }

    #[test]
    fn unexplained_gap_is_a_hard_error_and_record_is_not_consumed() {
        let mut source = replay(vec![packet(1, 0)]);
        let expected = Err(CaptureError::Malformed(
            MalformedCapture::UnexplainedOrdinalGap {
                expected: 0,
                received: 1,
                reported_loss: 0,
            },
        ));
        assert_eq!(source.poll_next(), expected);
        assert_eq!(source.poll_next(), expected);
    }

    #[test]
    fn inconsistent_truncation_marker_is_rejected() {
        let mut malformed = packet(0, 0);
        malformed.packet.captured_len = 60;
        let mut source = replay(vec![malformed]);
        assert_eq!(
            source.poll_next(),
            Err(CaptureError::Malformed(
                MalformedCapture::TruncationMarkerMismatch
            ))
        );
    }

    #[test]
    fn bounded_buffer_returns_unconsumed_packet_on_backpressure() {
        let first = packet(0, 0);
        let second = packet(1, 1);
        let mut buffer = BoundedCaptureBuffer::new(NonZeroUsize::new(1).expect("non-zero"));
        buffer.try_push(first.clone()).expect("first fits");
        let error = buffer.try_push(second.clone()).expect_err("buffer full");
        assert_eq!(*error.packet, second);
        assert_eq!(buffer.pop(), Some(first));
        assert!(buffer.is_empty());
    }

    #[test]
    fn stop_is_idempotent_and_discards_pending_replay() {
        let mut source = replay(vec![packet(0, 0)]);
        source.stop().expect("stop");
        source.stop().expect("second stop");
        assert_eq!(source.poll_next().expect("poll"), CapturePoll::Stopped);
    }

    #[test]
    fn replay_input_is_capacity_bounded() {
        let records = vec![packet(0, 0), packet(1, 1)];
        assert!(matches!(
            ReplaySource::new(metadata(), records, NonZeroUsize::new(1).expect("non-zero")),
            Err(CaptureError::ReplayCapacityExceeded {
                capacity: 1,
                supplied: 2
            })
        ));
    }

    #[test]
    fn replay_packet_feeds_core_temporal_memory_without_conversion() {
        use argus_core::{CacheVersion, TemporalMemory};

        let version = CacheVersion {
            feature_schema: 1,
            model: 1,
        };
        let mut source = replay(vec![packet(0, 0)]);
        let CapturePoll::Packet(captured) = source.poll_next().expect("poll") else {
            panic!("replay must return its packet");
        };
        let mut memory = TemporalMemory::new(
            captured.flow,
            version,
            NonZeroUsize::new(4).expect("non-zero"),
            NonZeroUsize::new(2).expect("non-zero"),
            NonZeroUsize::new(4).expect("non-zero"),
        );
        memory
            .ingest(captured.packet, version)
            .expect("core accepts normalized packet");
        assert_eq!(memory.recent_packets().len(), 1);
    }
}

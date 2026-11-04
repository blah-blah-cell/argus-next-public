//! Causal, deterministic packet and flow features.

use std::fmt;

use argus_capture::CapturedPacket;
pub use argus_capture::PacketProtocolMetadata;
use argus_core::{Direction, FlowInstanceId};

pub mod ahem_v34;
pub mod entropy;

pub use ahem_v34::{
    AHEM_V34_FLOW_EVENTS, AHEM_V34_FLOW_IDLE_TIMEOUT_NS, AHEM_V34_LOCAL_PACKETS,
    AHEM_V34_PACKET_FEATURES, AHEM_V34_RELATIONAL_FEATURES, AHEM_V34_SCHEMA_ID, AhemV34Cache,
    AhemV34CacheConfig, AhemV34CacheError, AhemV34CacheStats, AhemV34FlowIdentity, AhemV34Ingest,
    AhemV34Inputs, AhemV34Reset,
};
pub use entropy::{calculate_byte_entropy, calculate_entropy_from_counts, calculate_jitter_ms};

/// Immutable identity of the feature layout emitted by this crate.
pub const FEATURE_SCHEMA_ID: &str = "argus.causal-flow";
/// Incremented only when the meaning, order, or encoding of a field changes.
pub const FEATURE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeatureField {
    pub name: &'static str,
    pub unit: &'static str,
}

/// Ordered field contract. Consumers must reject unknown IDs or versions rather than guessing.
pub const FEATURE_FIELDS: [FeatureField; 12] = [
    FeatureField {
        name: "packet_length_bytes",
        unit: "bytes",
    },
    FeatureField {
        name: "captured_length_bytes",
        unit: "bytes",
    },
    FeatureField {
        name: "direction",
        unit: "0=initiator_to_responder,1=responder_to_initiator",
    },
    FeatureField {
        name: "interarrival_ns",
        unit: "nanoseconds",
    },
    FeatureField {
        name: "flow_packet_count",
        unit: "packets",
    },
    FeatureField {
        name: "flow_byte_count",
        unit: "bytes",
    },
    FeatureField {
        name: "direction_packet_count",
        unit: "packets",
    },
    FeatureField {
        name: "direction_byte_count",
        unit: "bytes",
    },
    FeatureField {
        name: "packets_lost_before",
        unit: "packets",
    },
    FeatureField {
        name: "is_truncated",
        unit: "boolean",
    },
    FeatureField {
        name: "is_reordered",
        unit: "boolean",
    },
    FeatureField {
        name: "capture_ordinal",
        unit: "ordinal",
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeatureSchema {
    pub id: &'static str,
    pub version: u32,
    pub fields: &'static [FeatureField],
}

pub const FEATURE_SCHEMA: FeatureSchema = FeatureSchema {
    id: FEATURE_SCHEMA_ID,
    version: FEATURE_SCHEMA_VERSION,
    fields: &FEATURE_FIELDS,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeatureVector {
    pub schema_id: &'static str,
    pub schema_version: u32,
    pub flow: FlowInstanceId,
    pub monotonic_timestamp_ns: u64,
    pub protocol: PacketProtocolMetadata,
    pub values: [u64; FEATURE_FIELDS.len()],
}

impl FeatureVector {
    #[must_use]
    pub fn value(&self, field: FeatureIndex) -> u64 {
        self.values[field as usize]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum FeatureIndex {
    PacketLengthBytes = 0,
    CapturedLengthBytes = 1,
    Direction = 2,
    InterarrivalNs = 3,
    FlowPacketCount = 4,
    FlowByteCount = 5,
    DirectionPacketCount = 6,
    DirectionByteCount = 7,
    PacketsLostBefore = 8,
    IsTruncated = 9,
    IsReordered = 10,
    CaptureOrdinal = 11,
}

/// Stateful extractor for one flow instance. Every output uses only the current and prior packets.
#[derive(Clone, Debug)]
pub struct PerFlowExtractor {
    flow: FlowInstanceId,
    last_timestamp_ns: Option<u64>,
    packets: u64,
    bytes: u64,
    direction_packets: [u64; 2],
    direction_bytes: [u64; 2],
}

impl PerFlowExtractor {
    #[must_use]
    pub const fn new(flow: FlowInstanceId) -> Self {
        Self {
            flow,
            last_timestamp_ns: None,
            packets: 0,
            bytes: 0,
            direction_packets: [0; 2],
            direction_bytes: [0; 2],
        }
    }

    /// Emits features after incorporating this packet into cumulative counters.
    ///
    /// # Errors
    /// Returns an error for another flow, time moving backwards, or counter overflow.
    pub fn ingest(&mut self, captured: &CapturedPacket) -> Result<FeatureVector, FeatureError> {
        if captured.flow != self.flow {
            return Err(FeatureError::FlowMismatch);
        }
        let interarrival_ns = match self.last_timestamp_ns {
            Some(previous) => captured
                .packet
                .monotonic_timestamp_ns
                .checked_sub(previous)
                .ok_or(FeatureError::NonMonotonicTimestamp)?,
            None => 0,
        };
        let direction = match captured.packet.direction {
            Direction::InitiatorToResponder => 0,
            Direction::ResponderToInitiator => 1,
        };
        let packet_bytes = u64::from(captured.packet.original_len);
        let packets = self
            .packets
            .checked_add(1)
            .ok_or(FeatureError::CounterOverflow)?;
        let bytes = self
            .bytes
            .checked_add(packet_bytes)
            .ok_or(FeatureError::CounterOverflow)?;
        let direction_packets = self.direction_packets[direction]
            .checked_add(1)
            .ok_or(FeatureError::CounterOverflow)?;
        let direction_bytes = self.direction_bytes[direction]
            .checked_add(packet_bytes)
            .ok_or(FeatureError::CounterOverflow)?;

        self.last_timestamp_ns = Some(captured.packet.monotonic_timestamp_ns);
        self.packets = packets;
        self.bytes = bytes;
        self.direction_packets[direction] = direction_packets;
        self.direction_bytes[direction] = direction_bytes;

        Ok(FeatureVector {
            schema_id: FEATURE_SCHEMA_ID,
            schema_version: FEATURE_SCHEMA_VERSION,
            flow: self.flow.clone(),
            monotonic_timestamp_ns: captured.packet.monotonic_timestamp_ns,
            protocol: captured.protocol,
            values: [
                packet_bytes,
                u64::from(captured.packet.captured_len),
                direction as u64,
                interarrival_ns,
                packets,
                bytes,
                direction_packets,
                direction_bytes,
                captured.quality.packets_lost_before,
                u64::from(captured.quality.truncated),
                u64::from(captured.quality.reordered),
                captured.capture_ordinal,
            ],
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeatureError {
    FlowMismatch,
    NonMonotonicTimestamp,
    CounterOverflow,
}

impl fmt::Display for FeatureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "feature extraction failed: {self:?}")
    }
}

impl std::error::Error for FeatureError {}

#[cfg(test)]
mod tests {
    use super::*;
    use argus_capture::QualityMarkers;
    use argus_core::{Endpoint, FlowKey, PacketEvent, TransportProtocol};
    use std::net::{IpAddr, Ipv4Addr};

    fn flow() -> FlowInstanceId {
        FlowInstanceId {
            key: FlowKey {
                initiator: Endpoint {
                    address: IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1)),
                    port: 40_000,
                },
                responder: Endpoint {
                    address: IpAddr::V4(Ipv4Addr::new(198, 51, 100, 2)),
                    port: 443,
                },
                protocol: TransportProtocol::Tcp,
            },
            generation: 7,
        }
    }

    fn packet(ordinal: u64, timestamp: u64, length: u32, direction: Direction) -> CapturedPacket {
        CapturedPacket {
            capture_ordinal: ordinal,
            flow: flow(),
            packet: PacketEvent {
                sequence: ordinal,
                monotonic_timestamp_ns: timestamp,
                captured_len: length,
                original_len: length,
                direction,
            },
            protocol: PacketProtocolMetadata {
                ip_version: 4,
                tcp_flags: Some(0),
            },
            observations: argus_capture::PacketObservationMetadata::default(),
            quality: QualityMarkers::default(),
        }
    }

    #[test]
    fn schema_contract_is_golden() {
        assert_eq!(FEATURE_SCHEMA.id, "argus.causal-flow");
        assert_eq!(FEATURE_SCHEMA.version, 1);
        assert_eq!(
            FEATURE_FIELDS.map(|field| field.name),
            [
                "packet_length_bytes",
                "captured_length_bytes",
                "direction",
                "interarrival_ns",
                "flow_packet_count",
                "flow_byte_count",
                "direction_packet_count",
                "direction_byte_count",
                "packets_lost_before",
                "is_truncated",
                "is_reordered",
                "capture_ordinal",
            ]
        );
    }

    #[test]
    fn causal_vectors_match_golden_values() {
        let mut extractor = PerFlowExtractor::new(flow());
        let first = extractor
            .ingest(&packet(0, 1_000, 60, Direction::InitiatorToResponder))
            .expect("first");
        let second = extractor
            .ingest(&packet(1, 1_025, 100, Direction::ResponderToInitiator))
            .expect("second");
        assert_eq!(first.values, [60, 60, 0, 0, 1, 60, 1, 60, 0, 0, 0, 0]);
        assert_eq!(second.values, [100, 100, 1, 25, 2, 160, 1, 100, 0, 0, 0, 1]);
    }

    #[test]
    fn failed_time_validation_does_not_mutate_state() {
        let mut extractor = PerFlowExtractor::new(flow());
        extractor
            .ingest(&packet(0, 10, 60, Direction::InitiatorToResponder))
            .expect("first");
        assert_eq!(
            extractor.ingest(&packet(1, 9, 70, Direction::InitiatorToResponder)),
            Err(FeatureError::NonMonotonicTimestamp)
        );
        let next = extractor
            .ingest(&packet(2, 12, 80, Direction::InitiatorToResponder))
            .expect("next");
        assert_eq!(next.value(FeatureIndex::InterarrivalNs), 2);
        assert_eq!(next.value(FeatureIndex::FlowPacketCount), 2);
    }
}

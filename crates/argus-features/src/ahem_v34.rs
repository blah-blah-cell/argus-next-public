//! Exact, bounded capture-to-tensor contract for cache-native AHEM v3.4.
//!
//! This module does not run a model. It constructs the five tensors consumed by
//! `AhemV34FlowIndexedNextPacket.forward_from_indexed_lane` without retaining a
//! full packet history or inventing flow identifiers as neural features.

use std::{
    collections::{HashMap, VecDeque},
    fmt,
    num::{NonZeroU64, NonZeroUsize},
};

use argus_capture::CapturedPacket;
use argus_core::{CacheVersion, Direction, FlowKey, TransportProtocol};

pub const AHEM_V34_PACKET_FEATURES: usize = 12;
pub const AHEM_V34_RELATIONAL_FEATURES: usize = 13;
pub const AHEM_V34_LOCAL_PACKETS: usize = 128;
pub const AHEM_V34_FLOW_EVENTS: usize = 17;
pub const AHEM_V34_SCHEMA_ID: &str = "argus.packet-causal-flow-v34";
pub const AHEM_V34_FLOW_IDLE_TIMEOUT_NS: u64 = 120 * 1_000_000_000;

/// Directional, cache-local generation identity matching the ARGUSLR2 journal
/// converter. The generation token is unique within one capture session.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AhemV34FlowIdentity {
    pub key: FlowKey,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AhemV34CacheConfig {
    pub flow_capacity: NonZeroUsize,
    pub history_span_packets: NonZeroU64,
}

impl Default for AhemV34CacheConfig {
    fn default() -> Self {
        Self {
            flow_capacity: NonZeroUsize::new(4_096).expect("constant is non-zero"),
            history_span_packets: NonZeroU64::new(30_000).expect("constant is non-zero"),
        }
    }
}

/// Fixed-shape batch-one inputs for a cache-native AHEM v3.4 graph.
#[derive(Clone, Debug, PartialEq)]
pub struct AhemV34Inputs {
    pub local_packets: [[f32; AHEM_V34_RELATIONAL_FEATURES]; AHEM_V34_LOCAL_PACKETS],
    pub flow_packets: [[f32; AHEM_V34_PACKET_FEATURES]; AHEM_V34_FLOW_EVENTS],
    pub flow_ordinals: [i64; AHEM_V34_FLOW_EVENTS],
    pub flow_valid: [bool; AHEM_V34_FLOW_EVENTS],
    pub context_spans: [f32; 1],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AhemV34Reset {
    CaptureChanged,
    VersionChanged,
    ClockRollback,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AhemV34Ingest {
    pub flow: AhemV34FlowIdentity,
    pub packet_features: [f32; AHEM_V34_PACKET_FEATURES],
    pub inputs: Option<AhemV34Inputs>,
    pub reset: Option<AhemV34Reset>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AhemV34CacheStats {
    pub local_packets: usize,
    pub flow_lanes: usize,
    pub flow_events: usize,
    pub flow_capacity: usize,
    pub maximum_flow_events: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AhemV34CacheError {
    EmptyCaptureId,
    HistorySpanTooLarge,
    NonIncreasingCaptureOrdinal { previous: u64, received: u64 },
    NonMonotonicFlowTimestamp,
    OrdinalOverflow,
    StateInvariantViolated,
}

impl fmt::Display for AhemV34CacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "AHEM v3.4 cache rejected input: {self:?}")
    }
}

impl std::error::Error for AhemV34CacheError {}

#[derive(Clone, Debug, PartialEq)]
struct IndexedPacket {
    flow: AhemV34FlowIdentity,
    ordinal: u64,
    features: [f32; AHEM_V34_PACKET_FEATURES],
}

#[derive(Clone, Debug)]
struct FlowLane {
    events: VecDeque<IndexedPacket>,
    last_timestamp_ns: u64,
    previous: Option<AhemV34FlowIdentity>,
    next: Option<AhemV34FlowIdentity>,
}

#[derive(Clone, Debug)]
struct FlowLifecycle {
    identity: AhemV34FlowIdentity,
    source_generation: u64,
    last_timestamp_ns: u64,
    closed: bool,
}

/// Bounded global ring plus an exact flow-generation LRU index.
///
/// Appending and LRU maintenance are O(1). Building a model request is O(128 +
/// 17), independent of the configured history span.
#[derive(Clone, Debug)]
pub struct AhemV34Cache {
    config: AhemV34CacheConfig,
    capture_id: String,
    version: CacheVersion,
    first_ordinal: Option<u64>,
    last_ordinal: Option<u64>,
    next_generation_token: u64,
    local: VecDeque<IndexedPacket>,
    lanes: HashMap<AhemV34FlowIdentity, FlowLane>,
    lifecycles: HashMap<FlowKey, FlowLifecycle>,
    lru_head: Option<AhemV34FlowIdentity>,
    lru_tail: Option<AhemV34FlowIdentity>,
}

impl AhemV34Cache {
    /// Creates an empty cache scoped to one capture and model/schema version.
    ///
    /// # Errors
    /// Returns an error when the capture identity is empty.
    pub fn new(
        capture_id: impl Into<String>,
        version: CacheVersion,
        config: AhemV34CacheConfig,
    ) -> Result<Self, AhemV34CacheError> {
        let capture_id = capture_id.into();
        if capture_id.trim().is_empty() {
            return Err(AhemV34CacheError::EmptyCaptureId);
        }
        if config.history_span_packets.get() > u64::from(u16::MAX) {
            return Err(AhemV34CacheError::HistorySpanTooLarge);
        }
        Ok(Self {
            config,
            capture_id,
            version,
            first_ordinal: None,
            last_ordinal: None,
            next_generation_token: 0,
            local: VecDeque::with_capacity(AHEM_V34_LOCAL_PACKETS),
            lanes: HashMap::with_capacity(config.flow_capacity.get()),
            lifecycles: HashMap::with_capacity(config.flow_capacity.get()),
            lru_head: None,
            lru_tail: None,
        })
    }

    #[must_use]
    pub fn stats(&self) -> AhemV34CacheStats {
        AhemV34CacheStats {
            local_packets: self.local.len(),
            flow_lanes: self.lanes.len(),
            flow_events: self.lanes.values().map(|lane| lane.events.len()).sum(),
            flow_capacity: self.config.flow_capacity.get(),
            maximum_flow_events: self
                .config
                .flow_capacity
                .get()
                .saturating_mul(AHEM_V34_FLOW_EVENTS),
        }
    }

    /// Encodes and appends one normalized packet, returning a model request once
    /// the exact 128-packet local lane is warm.
    ///
    /// A capture, version, or declared clock-epoch change clears all prior
    /// representation state before accepting the packet.
    ///
    /// # Errors
    /// Rejects ambiguous capture identity, ordinal rollback/duplication, an
    /// unmarked per-flow clock rollback, overflow, or an internal LRU invariant.
    pub fn ingest(
        &mut self,
        capture_id: &str,
        version: CacheVersion,
        packet: &CapturedPacket,
    ) -> Result<AhemV34Ingest, AhemV34CacheError> {
        if capture_id.trim().is_empty() {
            return Err(AhemV34CacheError::EmptyCaptureId);
        }

        let reset = if packet.quality.clock_rollback {
            self.reset(capture_id, version);
            Some(AhemV34Reset::ClockRollback)
        } else if capture_id != self.capture_id {
            self.reset(capture_id, version);
            Some(AhemV34Reset::CaptureChanged)
        } else if version != self.version {
            self.reset(capture_id, version);
            Some(AhemV34Reset::VersionChanged)
        } else {
            None
        };

        if let Some(previous) = self.last_ordinal
            && packet.capture_ordinal <= previous
        {
            return Err(AhemV34CacheError::NonIncreasingCaptureOrdinal {
                previous,
                received: packet.capture_ordinal,
            });
        }

        let flow = self.resolve_flow(packet)?;
        let previous_timestamp = self.lanes.get(&flow).map(|lane| lane.last_timestamp_ns);
        let features = encode_packet(packet, previous_timestamp)?;
        let indexed = IndexedPacket {
            flow: flow.clone(),
            ordinal: packet.capture_ordinal,
            features,
        };

        if self.local.len() == AHEM_V34_LOCAL_PACKETS {
            self.local.pop_front();
        }
        self.local.push_back(indexed.clone());
        self.append_flow(indexed, packet.packet.monotonic_timestamp_ns)?;
        self.first_ordinal.get_or_insert(packet.capture_ordinal);
        self.last_ordinal = Some(packet.capture_ordinal);

        let inputs = (self.local.len() == AHEM_V34_LOCAL_PACKETS)
            .then(|| self.build_inputs(&flow))
            .transpose()?;
        Ok(AhemV34Ingest {
            flow,
            packet_features: features,
            inputs,
            reset,
        })
    }

    fn reset(&mut self, capture_id: &str, version: CacheVersion) {
        self.capture_id.clear();
        self.capture_id.push_str(capture_id);
        self.version = version;
        self.first_ordinal = None;
        self.last_ordinal = None;
        self.next_generation_token = 0;
        self.local.clear();
        self.lanes.clear();
        self.lifecycles.clear();
        self.lru_head = None;
        self.lru_tail = None;
    }

    fn append_flow(
        &mut self,
        packet: IndexedPacket,
        timestamp_ns: u64,
    ) -> Result<(), AhemV34CacheError> {
        let flow = packet.flow.clone();
        if self.lanes.contains_key(&flow) {
            self.touch(&flow)?;
        } else {
            if self.lanes.len() == self.config.flow_capacity.get() {
                self.evict_lru()?;
            }
            let previous = self.lru_tail.clone();
            self.lanes.insert(
                flow.clone(),
                FlowLane {
                    events: VecDeque::with_capacity(AHEM_V34_FLOW_EVENTS),
                    last_timestamp_ns: timestamp_ns,
                    previous: previous.clone(),
                    next: None,
                },
            );
            if let Some(previous) = previous {
                self.lanes
                    .get_mut(&previous)
                    .ok_or(AhemV34CacheError::StateInvariantViolated)?
                    .next = Some(flow.clone());
            } else {
                self.lru_head = Some(flow.clone());
            }
            self.lru_tail = Some(flow.clone());
        }

        let lane = self
            .lanes
            .get_mut(&flow)
            .ok_or(AhemV34CacheError::StateInvariantViolated)?;
        if lane.events.len() == AHEM_V34_FLOW_EVENTS {
            lane.events.pop_front();
        }
        lane.events.push_back(packet);
        lane.last_timestamp_ns = timestamp_ns;
        Ok(())
    }

    fn resolve_flow(
        &mut self,
        packet: &CapturedPacket,
    ) -> Result<AhemV34FlowIdentity, AhemV34CacheError> {
        let key = directional_key(packet);
        let timestamp_ns = packet.packet.monotonic_timestamp_ns;
        let existing = self.lifecycles.get(&key);
        if let Some(state) = existing
            && timestamp_ns < state.last_timestamp_ns
        {
            return Err(AhemV34CacheError::NonMonotonicFlowTimestamp);
        }
        let needs_generation = existing.is_none_or(|state| {
            state.closed
                || state.source_generation != packet.flow.generation
                || timestamp_ns.saturating_sub(state.last_timestamp_ns)
                    > AHEM_V34_FLOW_IDLE_TIMEOUT_NS
        });
        let identity = if needs_generation {
            let generation = self.next_generation_token;
            self.next_generation_token = self
                .next_generation_token
                .checked_add(1)
                .ok_or(AhemV34CacheError::OrdinalOverflow)?;
            AhemV34FlowIdentity {
                key: key.clone(),
                generation,
            }
        } else {
            existing
                .ok_or(AhemV34CacheError::StateInvariantViolated)?
                .identity
                .clone()
        };
        let flags = packet.protocol.tcp_flags.unwrap_or(0);
        let closes = packet.flow.key.protocol == TransportProtocol::Tcp && flags & 0x05 != 0;
        self.lifecycles.insert(
            key,
            FlowLifecycle {
                identity: identity.clone(),
                source_generation: packet.flow.generation,
                last_timestamp_ns: timestamp_ns,
                closed: closes,
            },
        );
        Ok(identity)
    }

    fn touch(&mut self, flow: &AhemV34FlowIdentity) -> Result<(), AhemV34CacheError> {
        if self.lru_tail.as_ref() == Some(flow) {
            return Ok(());
        }
        let lane = self
            .lanes
            .get(flow)
            .ok_or(AhemV34CacheError::StateInvariantViolated)?;
        let previous = lane.previous.clone();
        let next = lane.next.clone();
        if let Some(previous) = &previous {
            self.lanes
                .get_mut(previous)
                .ok_or(AhemV34CacheError::StateInvariantViolated)?
                .next
                .clone_from(&next);
        } else {
            self.lru_head.clone_from(&next);
        }
        if let Some(next) = &next {
            self.lanes
                .get_mut(next)
                .ok_or(AhemV34CacheError::StateInvariantViolated)?
                .previous = previous;
        }
        let old_tail = self.lru_tail.clone();
        if let Some(old_tail) = &old_tail {
            self.lanes
                .get_mut(old_tail)
                .ok_or(AhemV34CacheError::StateInvariantViolated)?
                .next = Some(flow.clone());
        }
        let lane = self
            .lanes
            .get_mut(flow)
            .ok_or(AhemV34CacheError::StateInvariantViolated)?;
        lane.previous = old_tail;
        lane.next = None;
        self.lru_tail = Some(flow.clone());
        Ok(())
    }

    fn evict_lru(&mut self) -> Result<(), AhemV34CacheError> {
        let head = self
            .lru_head
            .take()
            .ok_or(AhemV34CacheError::StateInvariantViolated)?;
        let removed = self
            .lanes
            .remove(&head)
            .ok_or(AhemV34CacheError::StateInvariantViolated)?;
        if self
            .lifecycles
            .get(&head.key)
            .is_some_and(|state| state.identity == head)
        {
            self.lifecycles.remove(&head.key);
        }
        self.lru_head.clone_from(&removed.next);
        if let Some(next) = removed.next {
            self.lanes
                .get_mut(&next)
                .ok_or(AhemV34CacheError::StateInvariantViolated)?
                .previous = None;
        } else {
            self.lru_tail = None;
        }
        Ok(())
    }

    fn build_inputs(
        &self,
        query: &AhemV34FlowIdentity,
    ) -> Result<AhemV34Inputs, AhemV34CacheError> {
        let anchor = self
            .last_ordinal
            .ok_or(AhemV34CacheError::StateInvariantViolated)?;
        let first = self
            .first_ordinal
            .ok_or(AhemV34CacheError::StateInvariantViolated)?;
        let observed_span = anchor
            .checked_sub(first)
            .and_then(|value| value.checked_add(1))
            .ok_or(AhemV34CacheError::OrdinalOverflow)?;
        let span = observed_span.min(self.config.history_span_packets.get());
        let context_start = anchor
            .checked_add(1)
            .and_then(|value| value.checked_sub(span))
            .ok_or(AhemV34CacheError::OrdinalOverflow)?;

        let mut local_packets = [[0.0; AHEM_V34_RELATIONAL_FEATURES]; AHEM_V34_LOCAL_PACKETS];
        for (output, packet) in local_packets.iter_mut().zip(&self.local) {
            output[..AHEM_V34_PACKET_FEATURES].copy_from_slice(&packet.features);
            output[AHEM_V34_PACKET_FEATURES] = f32::from(packet.flow == *query);
        }

        let lane = self
            .lanes
            .get(query)
            .ok_or(AhemV34CacheError::StateInvariantViolated)?;
        let mut flow_packets = [[0.0; AHEM_V34_PACKET_FEATURES]; AHEM_V34_FLOW_EVENTS];
        let mut flow_ordinals = [0_i64; AHEM_V34_FLOW_EVENTS];
        let mut flow_valid = [false; AHEM_V34_FLOW_EVENTS];
        for (index, event) in lane
            .events
            .iter()
            .rev()
            .filter(|event| event.ordinal >= context_start)
            .take(AHEM_V34_FLOW_EVENTS)
            .enumerate()
        {
            flow_packets[index] = event.features;
            let relative = event
                .ordinal
                .checked_sub(context_start)
                .ok_or(AhemV34CacheError::OrdinalOverflow)?;
            flow_ordinals[index] =
                i64::try_from(relative).map_err(|_| AhemV34CacheError::OrdinalOverflow)?;
            flow_valid[index] = true;
        }
        if !flow_valid[0] || flow_ordinals[0] != i64::try_from(span - 1).unwrap_or(i64::MAX) {
            return Err(AhemV34CacheError::StateInvariantViolated);
        }

        Ok(AhemV34Inputs {
            local_packets,
            flow_packets,
            flow_ordinals,
            flow_valid,
            context_spans: [f32::from(
                u16::try_from(span).map_err(|_| AhemV34CacheError::HistorySpanTooLarge)?,
            )],
        })
    }
}

fn directional_key(packet: &CapturedPacket) -> FlowKey {
    match packet.packet.direction {
        Direction::InitiatorToResponder => packet.flow.key.clone(),
        Direction::ResponderToInitiator => FlowKey {
            initiator: packet.flow.key.responder.clone(),
            responder: packet.flow.key.initiator.clone(),
            protocol: packet.flow.key.protocol,
        },
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "the frozen training encoder computes in f64 and stores IEEE float32 model inputs"
)]
fn encode_packet(
    packet: &CapturedPacket,
    previous_timestamp_ns: Option<u64>,
) -> Result<[f32; AHEM_V34_PACKET_FEATURES], AhemV34CacheError> {
    let interarrival_ns = match previous_timestamp_ns {
        Some(previous) => packet
            .packet
            .monotonic_timestamp_ns
            .checked_sub(previous)
            .ok_or(AhemV34CacheError::NonMonotonicFlowTimestamp)?,
        None => 0,
    };
    let interarrival_seconds = (interarrival_ns as f64 / 1_000_000_000.0).clamp(0.0, 60.0);
    let flags = packet.protocol.tcp_flags.unwrap_or(0);
    let is_tcp = packet.flow.key.protocol == TransportProtocol::Tcp;
    let is_udp = packet.flow.key.protocol == TransportProtocol::Udp;
    Ok([
        f64::from(packet.packet.original_len).ln_1p() as f32,
        interarrival_seconds.ln_1p() as f32,
        f32::from(packet.protocol.ip_version == 4),
        f32::from(packet.protocol.ip_version == 6),
        f32::from(is_tcp),
        f32::from(is_udp),
        f32::from(is_tcp && flags & 0x02 != 0),
        f32::from(is_tcp && flags & 0x10 != 0),
        f32::from(is_tcp && flags & 0x01 != 0),
        f32::from(is_tcp && flags & 0x04 != 0),
        f32::from(is_tcp && flags & 0x08 != 0),
        f32::from(is_tcp && flags & 0x20 != 0),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use argus_capture::{PacketProtocolMetadata, QualityMarkers};
    use argus_core::{Direction, Endpoint, FlowInstanceId, FlowKey, PacketEvent};
    use std::net::{IpAddr, Ipv4Addr};

    fn flow(generation: u64, port: u16) -> FlowInstanceId {
        FlowInstanceId {
            key: FlowKey {
                initiator: Endpoint {
                    address: IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1)),
                    port,
                },
                responder: Endpoint {
                    address: IpAddr::V4(Ipv4Addr::new(198, 51, 100, 2)),
                    port: 443,
                },
                protocol: TransportProtocol::Tcp,
            },
            generation,
        }
    }

    fn packet(ordinal: u64, flow: FlowInstanceId) -> CapturedPacket {
        CapturedPacket {
            capture_ordinal: ordinal,
            flow,
            packet: PacketEvent {
                sequence: ordinal,
                monotonic_timestamp_ns: ordinal.saturating_mul(1_000_000_000),
                captured_len: 60,
                original_len: 64,
                direction: Direction::InitiatorToResponder,
            },
            protocol: PacketProtocolMetadata {
                ip_version: 4,
                tcp_flags: Some(0x12),
            },
            observations: argus_capture::PacketObservationMetadata::default(),
            quality: QualityMarkers::default(),
        }
    }

    fn cache(flow_capacity: usize, history_span: u64) -> AhemV34Cache {
        AhemV34Cache::new(
            "capture-a",
            CacheVersion {
                feature_schema: 34,
                model: 1,
            },
            AhemV34CacheConfig {
                flow_capacity: NonZeroUsize::new(flow_capacity).expect("non-zero"),
                history_span_packets: NonZeroU64::new(history_span).expect("non-zero"),
            },
        )
        .expect("cache")
    }

    #[test]
    fn exact_feature_contract_matches_training_encoder() {
        let mut value = packet(3, flow(0, 50_000));
        value.packet.original_len = 1_499;
        value.packet.monotonic_timestamp_ns = 3_500_000_000;
        value.protocol.tcp_flags = Some(0x3b);
        let encoded = encode_packet(&value, Some(1_000_000_000)).expect("encode");
        assert!((encoded[0] - 1_500_f32.ln()).abs() < 1e-6);
        assert!((encoded[1] - 2.5_f32.ln_1p()).abs() < 1e-6);
        assert_eq!(&encoded[2..6], &[1.0, 0.0, 1.0, 0.0]);
        assert_eq!(&encoded[6..], &[1.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn warms_exactly_at_128_and_builds_descending_flow_lane() {
        let query = flow(0, 50_000);
        let other = flow(0, 50_001);
        let mut cache = cache(8, 30_000);
        for ordinal in 0..127 {
            let selected = if ordinal % 10 == 0 {
                query.clone()
            } else {
                other.clone()
            };
            assert!(
                cache
                    .ingest("capture-a", cache.version, &packet(ordinal, selected))
                    .expect("ingest")
                    .inputs
                    .is_none()
            );
        }
        let result = cache
            .ingest("capture-a", cache.version, &packet(127, query.clone()))
            .expect("warm")
            .inputs
            .expect("inputs");
        assert!((result.context_spans[0] - 128.0).abs() < f32::EPSILON);
        assert_eq!(result.flow_ordinals[0], 127);
        assert_eq!(result.flow_ordinals[1], 120);
        assert!(result.flow_valid[0]);
        assert!((result.local_packets[127][12] - 1.0).abs() < f32::EPSILON);
        assert!(result.local_packets[126][12].abs() < f32::EPSILON);
    }

    #[test]
    fn flow_generation_prevents_reused_tuple_leakage() {
        let old = flow(0, 50_000);
        let new = flow(1, 50_000);
        let filler = flow(0, 50_001);
        let mut cache = cache(8, 30_000);
        for ordinal in 0..127 {
            let selected = if ordinal == 10 {
                old.clone()
            } else {
                filler.clone()
            };
            cache
                .ingest("capture-a", cache.version, &packet(ordinal, selected))
                .expect("ingest");
        }
        let inputs = cache
            .ingest("capture-a", cache.version, &packet(127, new))
            .expect("new generation")
            .inputs
            .expect("warm");
        assert_eq!(inputs.flow_valid.iter().filter(|value| **value).count(), 1);
        assert!(inputs.local_packets[10][12].abs() < f32::EPSILON);
    }

    #[test]
    fn reverse_direction_is_not_same_flow_for_trained_relation_contract() {
        let query = flow(0, 50_000);
        let filler = flow(0, 50_001);
        let mut cache = cache(8, 30_000);
        for ordinal in 0..127 {
            let mut value = packet(ordinal, filler.clone());
            if ordinal == 10 {
                value = packet(ordinal, query.clone());
                value.packet.direction = Direction::ResponderToInitiator;
            }
            cache
                .ingest("capture-a", cache.version, &value)
                .expect("ingest");
        }
        let inputs = cache
            .ingest("capture-a", cache.version, &packet(127, query))
            .expect("query")
            .inputs
            .expect("warm");
        assert!(inputs.local_packets[10][12].abs() < f32::EPSILON);
        assert_eq!(inputs.flow_valid.iter().filter(|value| **value).count(), 1);
    }

    #[test]
    fn fin_rst_and_idle_timeout_start_new_directional_generations() {
        let query = flow(0, 50_000);
        let mut cache = cache(8, 30_000);
        let mut fin = packet(0, query.clone());
        fin.protocol.tcp_flags = Some(0x11);
        let first = cache
            .ingest("capture-a", cache.version, &fin)
            .expect("FIN")
            .flow;
        let after_fin = cache
            .ingest("capture-a", cache.version, &packet(1, query.clone()))
            .expect("after FIN")
            .flow;
        assert_ne!(first, after_fin);

        let after_idle = cache
            .ingest("capture-a", cache.version, &packet(122, query))
            .expect("after idle")
            .flow;
        assert_ne!(after_fin, after_idle);
    }

    #[test]
    fn lru_and_every_packet_store_remain_bounded_in_long_stream() {
        let mut cache = cache(4, 30_000);
        for ordinal in 0..100_000 {
            let selected = flow(0, 50_000 + (ordinal % 11) as u16);
            cache
                .ingest("capture-a", cache.version, &packet(ordinal, selected))
                .expect("bounded stream");
            let stats = cache.stats();
            assert!(stats.local_packets <= AHEM_V34_LOCAL_PACKETS);
            assert!(stats.flow_lanes <= 4);
            assert!(stats.flow_events <= 4 * AHEM_V34_FLOW_EVENTS);
        }
    }

    #[test]
    fn capture_version_and_clock_changes_clear_all_old_state() {
        let query = flow(0, 50_000);
        let mut cache = cache(8, 30_000);
        cache
            .ingest("capture-a", cache.version, &packet(0, query.clone()))
            .expect("first");
        let version = CacheVersion {
            feature_schema: 34,
            model: 2,
        };
        let changed = cache
            .ingest("capture-a", version, &packet(1, query.clone()))
            .expect("version reset");
        assert_eq!(changed.reset, Some(AhemV34Reset::VersionChanged));
        assert_eq!(cache.stats().local_packets, 1);
        let changed = cache
            .ingest("capture-b", version, &packet(0, query.clone()))
            .expect("capture reset");
        assert_eq!(changed.reset, Some(AhemV34Reset::CaptureChanged));
        let mut rollback = packet(0, query);
        rollback.quality.clock_rollback = true;
        let changed = cache
            .ingest("capture-b", version, &rollback)
            .expect("clock reset");
        assert_eq!(changed.reset, Some(AhemV34Reset::ClockRollback));
        assert_eq!(cache.stats().local_packets, 1);
    }

    #[test]
    fn duplicate_or_rollback_without_marker_fails_closed() {
        let query = flow(0, 50_000);
        let mut cache = cache(8, 30_000);
        cache
            .ingest("capture-a", cache.version, &packet(5, query.clone()))
            .expect("first");
        assert_eq!(
            cache.ingest("capture-a", cache.version, &packet(5, query)),
            Err(AhemV34CacheError::NonIncreasingCaptureOrdinal {
                previous: 5,
                received: 5,
            })
        );
    }

    #[test]
    fn context_horizon_excludes_stale_flow_events() {
        let query = flow(0, 50_000);
        let filler = flow(0, 50_001);
        let mut cache = cache(8, 128);
        for ordinal in 0..255 {
            let selected = if ordinal == 0 || ordinal == 100 {
                query.clone()
            } else {
                filler.clone()
            };
            cache
                .ingest("capture-a", cache.version, &packet(ordinal, selected))
                .expect("ingest");
        }
        let inputs = cache
            .ingest("capture-a", cache.version, &packet(255, query))
            .expect("query")
            .inputs
            .expect("warm");
        assert!((inputs.context_spans[0] - 128.0).abs() < f32::EPSILON);
        assert_eq!(inputs.flow_valid.iter().filter(|value| **value).count(), 1);
    }
}

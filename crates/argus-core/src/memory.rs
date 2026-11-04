use std::{collections::VecDeque, fmt, num::NonZeroUsize};

use crate::FlowInstanceId;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CacheVersion {
    pub feature_schema: u64,
    pub model: u64,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ChunkId {
    pub flow: FlowInstanceId,
    pub version: CacheVersion,
    pub index: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Direction {
    InitiatorToResponder,
    ResponderToInitiator,
}

/// Causally available packet metadata retained by the exact short-term ring.
///
/// `sequence` is the observed packet ordinal within an Argus flow instance. It is not the TCP
/// header sequence number and does not imply TCP stream reassembly or retransmission tracking.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PacketEvent {
    pub sequence: u64,
    pub monotonic_timestamp_ns: u64,
    pub captured_len: u32,
    pub original_len: u32,
    pub direction: Direction,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkSummary {
    pub packet_count: u64,
    pub captured_bytes: u64,
    pub original_bytes: u64,
    pub initiator_packets: u64,
    pub responder_packets: u64,
    pub first_timestamp_ns: u64,
    pub last_timestamp_ns: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkSnapshot {
    pub id: ChunkId,
    pub summary: ChunkSummary,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IngestOutcome {
    pub invalidated_previous_version: Option<CacheVersion>,
    pub closed_chunk: Option<ChunkSnapshot>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IngestError {
    UnexpectedSequence { expected: u64, received: u64 },
    CapturedLengthExceedsOriginal,
    CounterOverflow,
    StateInvariantViolated,
}

impl fmt::Display for IngestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedSequence { expected, received } => {
                write!(
                    formatter,
                    "expected packet sequence {expected}, received {received}"
                )
            }
            Self::CapturedLengthExceedsOriginal => {
                formatter.write_str("captured length exceeds original packet length")
            }
            Self::CounterOverflow => formatter.write_str("temporal-memory counter overflow"),
            Self::StateInvariantViolated => {
                formatter.write_str("temporal-memory state invariant violated")
            }
        }
    }
}

impl std::error::Error for IngestError {}

/// Per-flow bounded memory. A cache version change atomically discards all old state.
#[derive(Clone, Debug)]
pub struct TemporalMemory {
    flow: FlowInstanceId,
    version: CacheVersion,
    short_term_capacity: NonZeroUsize,
    chunk_size: NonZeroUsize,
    max_closed_chunks: NonZeroUsize,
    next_sequence: u64,
    next_chunk_index: u64,
    short_term: VecDeque<PacketEvent>,
    closed_chunks: VecDeque<ChunkSnapshot>,
    partial: Option<ChunkSummary>,
}

impl TemporalMemory {
    #[must_use]
    pub fn new(
        flow: FlowInstanceId,
        version: CacheVersion,
        short_term_capacity: NonZeroUsize,
        chunk_size: NonZeroUsize,
        max_closed_chunks: NonZeroUsize,
    ) -> Self {
        Self::new_at_sequence(
            flow,
            version,
            short_term_capacity,
            chunk_size,
            max_closed_chunks,
            0,
        )
    }

    /// Creates empty temporal memory beginning at an explicit observed flow-packet ordinal.
    /// This is used after a capture-clock reset, where representation state must be discarded but
    /// the adapter's flow-local ordinal remains monotonic. It is not a TCP sequence number.
    #[must_use]
    pub fn new_at_sequence(
        flow: FlowInstanceId,
        version: CacheVersion,
        short_term_capacity: NonZeroUsize,
        chunk_size: NonZeroUsize,
        max_closed_chunks: NonZeroUsize,
        next_sequence: u64,
    ) -> Self {
        Self {
            flow,
            version,
            short_term_capacity,
            chunk_size,
            max_closed_chunks,
            next_sequence,
            next_chunk_index: 0,
            short_term: VecDeque::with_capacity(short_term_capacity.get()),
            closed_chunks: VecDeque::with_capacity(max_closed_chunks.get()),
            partial: None,
        }
    }

    #[must_use]
    pub const fn version(&self) -> CacheVersion {
        self.version
    }

    #[must_use]
    pub fn recent_packets(&self) -> &VecDeque<PacketEvent> {
        &self.short_term
    }

    #[must_use]
    pub fn closed_chunks(&self) -> &VecDeque<ChunkSnapshot> {
        &self.closed_chunks
    }

    /// Accepts exactly one next packet. A valid version change invalidates before insert.
    ///
    /// # Errors
    ///
    /// Returns [`IngestError`] for a sequence gap, invalid lengths, exhausted counters, or an
    /// internal state-invariant violation. Rejected packets are not inserted.
    pub fn ingest(
        &mut self,
        packet: PacketEvent,
        version: CacheVersion,
    ) -> Result<IngestOutcome, IngestError> {
        let version_changed = version != self.version;
        if packet.sequence != self.next_sequence {
            return Err(IngestError::UnexpectedSequence {
                expected: self.next_sequence,
                received: packet.sequence,
            });
        }
        if packet.captured_len > packet.original_len {
            return Err(IngestError::CapturedLengthExceedsOriginal);
        }
        let invalidated_previous_version = version_changed.then(|| {
            let previous = self.version;
            self.invalidate(version);
            previous
        });

        let summary = self.partial.get_or_insert(ChunkSummary {
            packet_count: 0,
            captured_bytes: 0,
            original_bytes: 0,
            initiator_packets: 0,
            responder_packets: 0,
            first_timestamp_ns: packet.monotonic_timestamp_ns,
            last_timestamp_ns: packet.monotonic_timestamp_ns,
        });
        summary.packet_count = summary
            .packet_count
            .checked_add(1)
            .ok_or(IngestError::CounterOverflow)?;
        summary.captured_bytes = summary
            .captured_bytes
            .checked_add(u64::from(packet.captured_len))
            .ok_or(IngestError::CounterOverflow)?;
        summary.original_bytes = summary
            .original_bytes
            .checked_add(u64::from(packet.original_len))
            .ok_or(IngestError::CounterOverflow)?;
        match packet.direction {
            Direction::InitiatorToResponder => {
                summary.initiator_packets = summary
                    .initiator_packets
                    .checked_add(1)
                    .ok_or(IngestError::CounterOverflow)?;
            }
            Direction::ResponderToInitiator => {
                summary.responder_packets = summary
                    .responder_packets
                    .checked_add(1)
                    .ok_or(IngestError::CounterOverflow)?;
            }
        }
        summary.last_timestamp_ns = packet.monotonic_timestamp_ns;

        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(IngestError::CounterOverflow)?;
        if self.short_term.len() == self.short_term_capacity.get() {
            self.short_term.pop_front();
        }
        self.short_term.push_back(packet);

        let closed_chunk = if self
            .partial
            .as_ref()
            .is_some_and(|summary| summary.packet_count == self.chunk_size.get() as u64)
        {
            let summary = self
                .partial
                .take()
                .ok_or(IngestError::StateInvariantViolated)?;
            let snapshot = ChunkSnapshot {
                id: ChunkId {
                    flow: self.flow.clone(),
                    version: self.version,
                    index: self.next_chunk_index,
                },
                summary,
            };
            self.next_chunk_index = self
                .next_chunk_index
                .checked_add(1)
                .ok_or(IngestError::CounterOverflow)?;
            if self.closed_chunks.len() == self.max_closed_chunks.get() {
                self.closed_chunks.pop_front();
            }
            self.closed_chunks.push_back(snapshot.clone());
            Some(snapshot)
        } else {
            None
        };

        Ok(IngestOutcome {
            invalidated_previous_version,
            closed_chunk,
        })
    }

    fn invalidate(&mut self, new_version: CacheVersion) {
        self.version = new_version;
        self.next_chunk_index = 0;
        self.short_term.clear();
        self.closed_chunks.clear();
        self.partial = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Endpoint, FlowKey, TransportProtocol};
    use std::{net::IpAddr, num::NonZeroUsize};

    fn flow() -> FlowInstanceId {
        FlowInstanceId {
            key: FlowKey {
                initiator: Endpoint {
                    address: "192.0.2.1".parse::<IpAddr>().expect("valid IP"),
                    port: 1234,
                },
                responder: Endpoint {
                    address: "198.51.100.1".parse::<IpAddr>().expect("valid IP"),
                    port: 443,
                },
                protocol: TransportProtocol::Tcp,
            },
            generation: 7,
        }
    }

    fn packet(sequence: u64) -> PacketEvent {
        PacketEvent {
            sequence,
            monotonic_timestamp_ns: 1_000 + sequence,
            captured_len: 60,
            original_len: 64,
            direction: Direction::InitiatorToResponder,
        }
    }

    fn memory(version: CacheVersion) -> TemporalMemory {
        TemporalMemory::new(
            flow(),
            version,
            NonZeroUsize::new(2).expect("non-zero"),
            NonZeroUsize::new(2).expect("non-zero"),
            NonZeroUsize::new(2).expect("non-zero"),
        )
    }

    #[test]
    fn closes_deterministically_sized_chunks() {
        let version = CacheVersion {
            feature_schema: 1,
            model: 1,
        };
        let mut memory = memory(version);
        assert!(
            memory
                .ingest(packet(0), version)
                .expect("ingest")
                .closed_chunk
                .is_none()
        );
        let snapshot = memory
            .ingest(packet(1), version)
            .expect("ingest")
            .closed_chunk
            .expect("chunk closes");
        assert_eq!(snapshot.id.index, 0);
        assert_eq!(snapshot.summary.packet_count, 2);
        assert_eq!(snapshot.summary.original_bytes, 128);
    }

    #[test]
    fn exact_ring_and_closed_chunks_are_bounded() {
        let version = CacheVersion {
            feature_schema: 1,
            model: 1,
        };
        let mut memory = memory(version);
        for sequence in 0..6 {
            memory.ingest(packet(sequence), version).expect("ingest");
        }
        assert_eq!(memory.recent_packets().len(), 2);
        assert_eq!(memory.recent_packets()[0].sequence, 4);
        assert_eq!(memory.closed_chunks().len(), 2);
        assert_eq!(memory.closed_chunks()[0].id.index, 1);
    }

    #[test]
    fn version_change_clears_every_old_state_before_accepting_new_sequence() {
        let old = CacheVersion {
            feature_schema: 1,
            model: 1,
        };
        let new = CacheVersion {
            feature_schema: 2,
            model: 1,
        };
        let mut memory = memory(old);
        memory.ingest(packet(0), old).expect("old packet");
        memory.ingest(packet(1), old).expect("old packet");

        let outcome = memory.ingest(packet(2), new).expect("new packet");
        assert_eq!(outcome.invalidated_previous_version, Some(old));
        assert_eq!(memory.version(), new);
        assert_eq!(memory.recent_packets().len(), 1);
        assert!(memory.closed_chunks().is_empty());
        assert!(
            memory
                .ingest(packet(3), new)
                .expect("new packet")
                .closed_chunk
                .is_some()
        );
    }

    #[test]
    fn rejects_packet_gaps() {
        let version = CacheVersion {
            feature_schema: 1,
            model: 1,
        };
        let mut memory = memory(version);
        assert_eq!(
            memory.ingest(packet(2), version),
            Err(IngestError::UnexpectedSequence {
                expected: 0,
                received: 2,
            })
        );
    }

    #[test]
    fn rejected_version_change_does_not_destroy_valid_state() {
        let old = CacheVersion {
            feature_schema: 1,
            model: 1,
        };
        let new = CacheVersion {
            feature_schema: 2,
            model: 1,
        };
        let mut memory = memory(old);
        memory.ingest(packet(0), old).expect("old packet");

        assert!(matches!(
            memory.ingest(packet(2), new),
            Err(IngestError::UnexpectedSequence { .. })
        ));
        assert_eq!(memory.version(), old);
        assert_eq!(memory.recent_packets().len(), 1);
    }

    #[test]
    fn version_epochs_cannot_collide_chunk_ids() {
        let old = CacheVersion {
            feature_schema: 1,
            model: 1,
        };
        let new = CacheVersion {
            feature_schema: 1,
            model: 2,
        };
        let mut memory = memory(old);
        memory.ingest(packet(0), old).expect("old packet");
        let old_id = memory
            .ingest(packet(1), old)
            .expect("old packet")
            .closed_chunk
            .expect("old chunk")
            .id;
        memory.ingest(packet(2), new).expect("new packet");
        let new_id = memory
            .ingest(packet(3), new)
            .expect("new packet")
            .closed_chunk
            .expect("new chunk")
            .id;
        assert_eq!(old_id.index, new_id.index);
        assert_ne!(old_id, new_id);
    }

    #[test]
    fn long_versioned_stream_preserves_all_resource_bounds() {
        let mut version = CacheVersion {
            feature_schema: 1,
            model: 1,
        };
        let mut memory = memory(version);
        for sequence in 0..500_u64 {
            if sequence > 0 && sequence.is_multiple_of(37) {
                version.model += 1;
            }
            memory
                .ingest(packet(sequence), version)
                .expect("bounded valid stream");
            assert!(memory.recent_packets().len() <= 2);
            assert!(memory.closed_chunks().len() <= 2);
            assert!(
                memory
                    .closed_chunks()
                    .iter()
                    .all(|snapshot| snapshot.id.version == version)
            );
        }
    }

    #[test]
    fn reset_memory_can_resume_at_observed_flow_ordinal() {
        let version = CacheVersion {
            feature_schema: 1,
            model: 1,
        };
        let mut memory = TemporalMemory::new_at_sequence(
            flow(),
            version,
            NonZeroUsize::new(2).expect("non-zero"),
            NonZeroUsize::new(2).expect("non-zero"),
            NonZeroUsize::new(2).expect("non-zero"),
            42,
        );
        memory.ingest(packet(42), version).expect("resume packet");
        assert_eq!(memory.recent_packets()[0].sequence, 42);
    }
}

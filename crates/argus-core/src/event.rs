use crate::{CacheVersion, ChunkId, FlowInstanceId};

/// Structured runtime event boundary. Concrete logging/telemetry belongs outside core.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeEvent {
    pub monotonic_timestamp_ns: u64,
    pub flow: FlowInstanceId,
    pub kind: RuntimeEventKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeEventKind {
    ChunkClosed(ChunkId),
    CacheInvalidated {
        previous: CacheVersion,
        current: CacheVersion,
    },
    PacketRejected,
}

/// Non-blocking implementations should enqueue; core never assumes a global logger.
pub trait EventSink {
    fn emit(&mut self, event: &RuntimeEvent);
}

impl EventSink for () {
    fn emit(&mut self, _event: &RuntimeEvent) {}
}

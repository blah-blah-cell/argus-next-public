//! Platform-neutral contracts for the Argus Next runtime.

pub mod config;
pub mod event;
pub mod identity;
pub mod memory;
pub mod persistence;

pub use config::{ConfigError, RuntimeConfig};
pub use event::{EventSink, RuntimeEvent, RuntimeEventKind};
pub use identity::{
    Endpoint, FlowGenerationError, FlowGenerationTracker, FlowInstanceId, FlowKey,
    TransportProtocol,
};
pub use memory::{
    CacheVersion, ChunkId, ChunkSnapshot, ChunkSummary, Direction, IngestError, IngestOutcome,
    PacketEvent, TemporalMemory,
};
pub use persistence::{
    DurableSnapshotStore, InMemorySnapshotStore, SnapshotStore, SnapshotStoreError,
};

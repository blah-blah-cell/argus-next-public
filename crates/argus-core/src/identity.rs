use std::{
    collections::{HashMap, VecDeque},
    fmt,
    net::IpAddr,
};

/// Default maximum number of distinct directional flow keys retained by
/// [`FlowGenerationTracker`] before least-recently-used entries are evicted.
pub const DEFAULT_MAX_TRACKED_FLOWS: usize = 65_536;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Endpoint {
    pub address: IpAddr,
    pub port: u16,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TransportProtocol {
    Tcp,
    Udp,
    Other(u8),
}

/// Directional flow identity. Initiator/responder ordering is significant.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FlowKey {
    pub initiator: Endpoint,
    pub responder: Endpoint,
    pub protocol: TransportProtocol,
}

/// A flow key plus lifecycle generation, preventing stale 5-tuple reuse.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FlowInstanceId {
    pub key: FlowKey,
    pub generation: u64,
}

#[derive(Clone, Debug)]
pub struct FlowGenerationTracker {
    latest: HashMap<FlowKey, (u64, u64)>,
    order: VecDeque<(FlowKey, u64)>,
    max_entries: usize,
    access_tick: u64,
    evicted_count: u64,
}

impl Default for FlowGenerationTracker {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_MAX_TRACKED_FLOWS)
    }
}

impl FlowGenerationTracker {
    /// Creates a bounded generation tracker retaining at most `max_entries`
    /// distinct flow keys (clamped to a minimum of `1`). When full, the
    /// least-recently-used flow key is deterministically evicted.
    #[must_use]
    pub fn with_capacity(max_entries: usize) -> Self {
        let max_entries = max_entries.max(1);
        Self {
            latest: HashMap::new(),
            order: VecDeque::new(),
            max_entries,
            access_tick: 0,
            evicted_count: 0,
        }
    }

    /// Number of currently tracked flow keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.latest.len()
    }

    /// Returns `true` when no flow keys are currently tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.latest.is_empty()
    }

    /// Configured maximum number of tracked flow keys.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.max_entries
    }

    /// Total number of flow keys evicted due to capacity limits.
    #[must_use]
    pub const fn evicted_count(&self) -> u64 {
        self.evicted_count
    }

    /// Begins a distinct lifecycle for a directional flow key.
    ///
    /// # Errors
    ///
    /// Returns [`FlowGenerationError::Exhausted`] if a key has used every `u64` generation.
    pub fn begin(&mut self, key: FlowKey) -> Result<FlowInstanceId, FlowGenerationError> {
        let generation = if let Some((previous, _)) = self.latest.get(&key) {
            previous
                .checked_add(1)
                .ok_or(FlowGenerationError::Exhausted)?
        } else {
            self.evict_if_full();
            0
        };
        self.access_tick = self.access_tick.wrapping_add(1);
        let tick = self.access_tick;
        self.latest.insert(key.clone(), (generation, tick));
        self.order.push_back((key.clone(), tick));
        if self.order.len() > self.max_entries.saturating_mul(2) {
            self.compact_order();
        }
        Ok(FlowInstanceId { key, generation })
    }

    fn evict_if_full(&mut self) {
        while self.latest.len() >= self.max_entries {
            let Some((candidate, recorded_tick)) = self.order.pop_front() else {
                break;
            };
            if self
                .latest
                .get(&candidate)
                .is_some_and(|(_, current_tick)| *current_tick == recorded_tick)
            {
                self.latest.remove(&candidate);
                self.evicted_count = self.evicted_count.saturating_add(1);
            }
        }
    }

    fn compact_order(&mut self) {
        let latest = &self.latest;
        self.order.retain(|(key, tick)| {
            latest
                .get(key)
                .is_some_and(|(_, current_tick)| current_tick == tick)
        });
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FlowGenerationError {
    Exhausted,
}

impl fmt::Display for FlowGenerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("flow generation counter exhausted")
    }
}

impl std::error::Error for FlowGenerationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn key() -> FlowKey {
        FlowKey {
            initiator: Endpoint {
                address: IpAddr::V4(Ipv4Addr::LOCALHOST),
                port: 50000,
            },
            responder: Endpoint {
                address: IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1)),
                port: 443,
            },
            protocol: TransportProtocol::Tcp,
        }
    }

    #[test]
    fn reused_tuple_gets_a_new_flow_instance() {
        let mut tracker = FlowGenerationTracker::default();
        let first = tracker.begin(key()).expect("first generation");
        let second = tracker.begin(key()).expect("second generation");
        assert_eq!(first.generation, 0);
        assert_eq!(second.generation, 1);
        assert_ne!(first, second);
    }

    #[test]
    fn endpoint_direction_is_part_of_identity() {
        let forward = key();
        let reverse = FlowKey {
            initiator: forward.responder.clone(),
            responder: forward.initiator.clone(),
            protocol: forward.protocol,
        };
        assert_ne!(forward, reverse);
    }

    fn key_with_port(port: u16) -> FlowKey {
        FlowKey {
            initiator: Endpoint {
                address: IpAddr::V4(Ipv4Addr::LOCALHOST),
                port,
            },
            responder: Endpoint {
                address: IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1)),
                port: 443,
            },
            protocol: TransportProtocol::Udp,
        }
    }

    #[test]
    fn tracker_bounds_capacity_and_evicts_lru_deterministically() {
        let mut tracker = FlowGenerationTracker::with_capacity(2);
        assert_eq!(tracker.capacity(), 2);
        assert!(tracker.is_empty());

        let k1 = key_with_port(10_001);
        let k2 = key_with_port(10_002);
        let k3 = key_with_port(10_003);

        assert_eq!(tracker.begin(k1.clone()).expect("k1").generation, 0);
        assert_eq!(tracker.begin(k2.clone()).expect("k2").generation, 0);
        assert_eq!(tracker.len(), 2);
        assert_eq!(tracker.evicted_count(), 0);

        // Touch k1 so k2 becomes the least-recently-used key.
        assert_eq!(tracker.begin(k1.clone()).expect("k1 gen 1").generation, 1);
        assert_eq!(tracker.len(), 2);
        assert_eq!(tracker.evicted_count(), 0);

        // Inserting k3 must evict k2 (not k1).
        assert_eq!(tracker.begin(k3).expect("k3").generation, 0);
        assert_eq!(tracker.len(), 2);
        assert_eq!(tracker.evicted_count(), 1);

        // k1 was retained (next generation is 2), whereas k2 was evicted and restarts at 0.
        assert_eq!(tracker.begin(k1).expect("k1 gen 2").generation, 2);
        assert_eq!(tracker.begin(k2).expect("k2 restarted").generation, 0);
        assert_eq!(tracker.len(), 2);
        assert_eq!(tracker.evicted_count(), 2);
    }

    #[test]
    fn tracker_handles_scan_flood_above_capacity() {
        let mut tracker = FlowGenerationTracker::with_capacity(32);
        for offset in 0..256_u16 {
            let instance = tracker
                .begin(key_with_port(20_000 + offset))
                .expect("scan flow");
            assert_eq!(instance.generation, 0);
        }
        assert_eq!(tracker.len(), 32);
        assert_eq!(tracker.evicted_count(), 224);
    }
}

use std::{collections::HashMap, fmt, net::IpAddr};

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

#[derive(Clone, Debug, Default)]
pub struct FlowGenerationTracker {
    latest: HashMap<FlowKey, u64>,
}

impl FlowGenerationTracker {
    /// Begins a distinct lifecycle for a directional flow key.
    ///
    /// # Errors
    ///
    /// Returns [`FlowGenerationError::Exhausted`] if a key has used every `u64` generation.
    pub fn begin(&mut self, key: FlowKey) -> Result<FlowInstanceId, FlowGenerationError> {
        let generation = match self.latest.get(&key) {
            Some(previous) => previous
                .checked_add(1)
                .ok_or(FlowGenerationError::Exhausted)?,
            None => 0,
        };
        self.latest.insert(key.clone(), generation);
        Ok(FlowInstanceId { key, generation })
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
}

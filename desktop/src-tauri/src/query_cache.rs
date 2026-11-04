//! In-memory query cache for Tauri IPC commands.
//!
//! Provides bounded TTL caching for expensive IPC endpoints (`recent_captures`,
//! `get_network_topology`, and `generate_network_dossier`) using standard library
//! primitives (`Instant`, `Duration`, `Mutex`, `HashMap`) with zero unsafe code.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock, Weak},
    time::{Duration, Instant},
};

use argus_capture::{AssetFingerprint, NetworkTopology};

use crate::DossierReportDto;

static ACTIVE_CACHE: OnceLock<Mutex<Option<Weak<IpcQueryCache>>>> = OnceLock::new();

fn active_cache_registry() -> &'static Mutex<Option<Weak<IpcQueryCache>>> {
    ACTIVE_CACHE.get_or_init(|| Mutex::new(None))
}

/// Traffic state fingerprint used to key the topology cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TopologyStateFingerprint {
    pub segments: u64,
    pub packets: u64,
    pub bytes: u64,
    pub flows_len: usize,
    pub evicted_flows: u64,
}

/// Composite state fingerprint used to key the dossier cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DossierStateFingerprint {
    pub topology: TopologyStateFingerprint,
    pub alert_count: usize,
    pub total_critical: u64,
    pub total_suspicious: u64,
}

/// In-memory cache for Tauri IPC query responses.
pub struct IpcQueryCache {
    recent_captures: Mutex<Option<(Instant, Vec<String>)>>,
    topology: Mutex<Option<(Instant, TopologyStateFingerprint, NetworkTopology)>>,
    dossier: Mutex<Option<(Instant, DossierStateFingerprint, DossierReportDto)>>,
    #[allow(dead_code)]
    ip_assets: Mutex<HashMap<String, AssetFingerprint>>,
}

impl Default for IpcQueryCache {
    fn default() -> Self {
        Self {
            recent_captures: Mutex::new(None),
            topology: Mutex::new(None),
            dossier: Mutex::new(None),
            ip_assets: Mutex::new(HashMap::new()),
        }
    }
}

impl IpcQueryCache {
    /// Creates a new, empty query cache.
    #[allow(dead_code)]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers this cache instance as the globally active instance for invalidation hooks.
    pub fn register_active(self: &Arc<Self>) {
        if let Ok(mut guard) = active_cache_registry().lock() {
            *guard = Some(Arc::downgrade(self));
        }
    }

    /// Invalidates recent captures on the globally active cache instance, if any.
    pub fn invalidate_global_recent_captures() {
        if let Ok(guard) = active_cache_registry().lock()
            && let Some(weak) = guard.as_ref()
            && let Some(arc) = weak.upgrade()
        {
            arc.invalidate_recent_captures();
        }
    }

    /// Invalidates recent captures cached entry.
    pub fn invalidate_recent_captures(&self) {
        if let Ok(mut guard) = self.recent_captures.lock() {
            *guard = None;
        }
    }

    /// Retrieves cached recent captures if the entry exists and has not expired.
    #[must_use]
    pub fn get_recent_captures(&self, ttl: Duration) -> Option<Vec<String>> {
        let guard = self.recent_captures.lock().ok()?;
        let (ts, captures) = guard.as_ref()?;
        if ts.elapsed() < ttl {
            Some(captures.clone())
        } else {
            None
        }
    }

    /// Updates the cached recent captures entry with the current timestamp.
    pub fn set_recent_captures(&self, captures: Vec<String>) {
        if let Ok(mut guard) = self.recent_captures.lock() {
            *guard = Some((Instant::now(), captures));
        }
    }

    /// Invalidates cached network topology and network dossier entries.
    #[allow(dead_code)]
    pub fn invalidate_topology(&self) {
        if let Ok(mut guard) = self.topology.lock() {
            *guard = None;
        }
        if let Ok(mut guard) = self.dossier.lock() {
            *guard = None;
        }
    }

    /// Retrieves cached network topology if fingerprint matches and entry has not expired.
    #[must_use]
    pub fn get_topology(
        &self,
        fingerprint: TopologyStateFingerprint,
        ttl: Duration,
    ) -> Option<NetworkTopology> {
        let guard = self.topology.lock().ok()?;
        let (ts, cached_fp, topo) = guard.as_ref()?;
        if *cached_fp == fingerprint && ts.elapsed() < ttl {
            Some(topo.clone())
        } else {
            None
        }
    }

    /// Updates the cached network topology entry with the given fingerprint and current timestamp.
    pub fn set_topology(&self, fingerprint: TopologyStateFingerprint, topology: NetworkTopology) {
        if let Ok(mut guard) = self.topology.lock() {
            *guard = Some((Instant::now(), fingerprint, topology));
        }
    }

    /// Retrieves cached network dossier if fingerprint matches and entry has not expired.
    #[must_use]
    pub fn get_dossier(
        &self,
        fingerprint: DossierStateFingerprint,
        ttl: Duration,
    ) -> Option<DossierReportDto> {
        let guard = self.dossier.lock().ok()?;
        let (ts, cached_fp, dossier) = guard.as_ref()?;
        if *cached_fp == fingerprint && ts.elapsed() < ttl {
            Some(dossier.clone())
        } else {
            None
        }
    }

    /// Updates the cached network dossier entry with the given fingerprint and current timestamp.
    pub fn set_dossier(&self, fingerprint: DossierStateFingerprint, dossier: DossierReportDto) {
        if let Ok(mut guard) = self.dossier.lock() {
            *guard = Some((Instant::now(), fingerprint, dossier));
        }
    }

    /// Looks up a cached asset fingerprint for the specified IP address.
    #[allow(dead_code)]
    #[must_use]
    pub fn get_asset(&self, ip: &str) -> Option<AssetFingerprint> {
        self.ip_assets.lock().ok()?.get(ip).cloned()
    }

    /// Caches an asset fingerprint for the specified IP address.
    #[allow(dead_code)]
    pub fn set_asset(&self, ip: String, asset: AssetFingerprint) {
        if let Ok(mut guard) = self.ip_assets.lock() {
            guard.insert(ip, asset);
        }
    }

    /// Returns a copy of the current IP assets map.
    #[allow(dead_code)]
    #[must_use]
    pub fn clone_assets(&self) -> HashMap<String, AssetFingerprint> {
        self.ip_assets.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recent_captures_cache_hit_and_expiration() {
        let cache = IpcQueryCache::default();
        let ttl = Duration::from_millis(1500);

        assert_eq!(cache.get_recent_captures(ttl), None);

        let files = vec!["capture1.pcap".to_string(), "capture2.pcap".to_string()];
        cache.set_recent_captures(files.clone());

        let start = Instant::now();
        let hit = cache.get_recent_captures(ttl);
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(1),
            "Cache hit must respond in < 1ms, took {elapsed:?}"
        );
        assert_eq!(hit, Some(files));

        // Invalidate
        cache.invalidate_recent_captures();
        assert_eq!(cache.get_recent_captures(ttl), None);
    }

    #[test]
    fn test_topology_cache_hit_under_one_millisecond() {
        let cache = IpcQueryCache::default();
        let ttl = Duration::from_secs(2);

        let fp1 = TopologyStateFingerprint {
            segments: 10,
            packets: 1000,
            bytes: 50000,
            flows_len: 25,
            evicted_flows: 0,
        };
        let fp2 = TopologyStateFingerprint {
            segments: 11,
            packets: 1100,
            bytes: 55000,
            flows_len: 26,
            evicted_flows: 0,
        };

        let dummy_topo = NetworkTopology {
            nodes: Vec::new(),
            edges: Vec::new(),
            protocol_distribution: argus_capture::ProtocolDistribution::default(),
            lan_hosts_count: 0,
            external_hosts_count: 0,
            tor_nodes_count: 0,
            malicious_nodes_count: 0,
            total_bytes: 50000,
            total_packets: 1000,
        };

        cache.set_topology(fp1, dummy_topo.clone());

        let start = Instant::now();
        let hit = cache.get_topology(fp1, ttl);
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(1),
            "Topology cache hit must respond in < 1ms, took {elapsed:?}"
        );
        assert_eq!(hit, Some(dummy_topo));

        // Different fingerprint must miss
        assert_eq!(cache.get_topology(fp2, ttl), None);

        // Invalidate
        cache.invalidate_topology();
        assert_eq!(cache.get_topology(fp1, ttl), None);
    }

    #[test]
    fn test_global_invalidation_registration() {
        let cache = Arc::new(IpcQueryCache::default());
        cache.register_active();

        cache.set_recent_captures(vec!["test.pcap".to_string()]);
        assert!(cache.get_recent_captures(Duration::from_secs(10)).is_some());

        IpcQueryCache::invalidate_global_recent_captures();
        assert_eq!(cache.get_recent_captures(Duration::from_secs(10)), None);
    }
}

//! Network asset profiling, protocol distribution, and interactive topology modeling.
//!
//! Inspired by `PcapXray` (srinivas11789/pcapxray), providing structured node-link
//! representations of observed communications with device classification,
//! OUI resolution, Tor detection, and ML anomaly annotations.

use std::{
    collections::{BTreeSet, HashMap},
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
};

use serde::{Deserialize, Serialize};

use crate::{
    asset_db::AssetFingerprint,
    dissector::is_tor_traffic,
    oui::{format_mac, lookup_vendor},
};

/// Role classification for a network endpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeRole {
    Gateway,
    LanHost,
    ExternalServer,
    TorNode,
    Broadcast,
    Multicast,
}

/// A node in the network topology graph.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TopologyNode {
    pub id: String,
    pub label: String,
    pub ip: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mac: Option<String>,
    pub role: NodeRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vendor: Option<String>,
    pub total_in_bytes: u64,
    pub total_out_bytes: u64,
    pub packet_count: u64,
    pub is_tor: bool,
    pub threat_score: f32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub active_ports: Vec<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub organization: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asn: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<String>,
}

/// A directed edge in the network topology graph.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TopologyEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub protocol: String,
    pub total_bytes: u64,
    pub packet_count: u64,
    pub is_tor: bool,
    pub is_malicious: bool,
    pub threat_score: f32,
}

/// Layer 3-7 protocol distribution telemetry across all observed traffic.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ProtocolDistribution {
    pub tcp_bytes: u64,
    pub udp_bytes: u64,
    pub icmp_bytes: u64,
    pub other_bytes: u64,
    pub dns_queries: u64,
    pub tls_sessions: u64,
    pub http_requests: u64,
    pub tor_packets: u64,
}

/// The complete aggregated network topology model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NetworkTopology {
    pub nodes: Vec<TopologyNode>,
    pub edges: Vec<TopologyEdge>,
    pub protocol_distribution: ProtocolDistribution,
    pub lan_hosts_count: usize,
    pub external_hosts_count: usize,
    pub tor_nodes_count: usize,
    pub malicious_nodes_count: usize,
    pub total_bytes: u64,
    pub total_packets: u64,
}

/// Normalized input record for constructing the network topology.
#[derive(Clone, Debug)]
pub struct FlowTopologyRecord {
    pub src_ip: String,
    pub dst_ip: String,
    pub src_port: u16,
    pub dst_port: u16,
    pub protocol: String,
    pub src_mac: Option<[u8; 6]>,
    pub dst_mac: Option<[u8; 6]>,
    pub bytes: u64,
    pub initiator_to_responder_bytes: u64,
    pub responder_to_initiator_bytes: u64,
    pub packet_count: u64,
    pub dns_queries: u64,
    pub http_requests: u64,
    pub tor_port_matches: u64,
    pub threat_score: f32,
    pub is_malicious: bool,
    pub sni: Option<String>,
    pub dns_qname: Option<String>,
}

/// Determines if an IP address belongs to a private local area network.
#[must_use]
pub fn is_private_or_local_ip(ip_str: &str) -> bool {
    let Ok(ip) = ip_str.parse::<IpAddr>() else {
        return false;
    };

    match ip {
        IpAddr::V4(v4) => {
            v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_multicast()
        }
        IpAddr::V6(v6) => {
            v6.is_loopback() || v6.is_multicast() || is_ipv6_unique_local_or_link_local(v6)
        }
    }
}

const fn is_ipv6_unique_local_or_link_local(v6: Ipv6Addr) -> bool {
    let octets = v6.octets();
    // Unique local (fc00::/7) or link-local (fe80::/10)
    (octets[0] & 0xfe) == 0xfc || ((octets[0] == 0xfe) && ((octets[1] & 0xc0) == 0x80))
}

/// Classifies the role of an endpoint based on its IP and MAC addresses.
#[must_use]
pub fn classify_endpoint_role(ip_str: &str, is_tor: bool) -> NodeRole {
    let Ok(ip) = ip_str.parse::<IpAddr>() else {
        return if is_tor {
            NodeRole::TorNode
        } else {
            NodeRole::ExternalServer
        };
    };

    match ip {
        IpAddr::V4(v4) => {
            if v4.is_broadcast() || v4 == Ipv4Addr::BROADCAST {
                NodeRole::Broadcast
            } else if v4.is_multicast() {
                NodeRole::Multicast
            } else if v4.is_private() || v4.is_loopback() || v4.is_link_local() {
                // Heuristic: default gateways frequently use .1 or .254 in private subnets
                let last_octet = v4.octets()[3];
                if (last_octet == 1 || last_octet == 254) && v4.is_private() {
                    NodeRole::Gateway
                } else {
                    NodeRole::LanHost
                }
            } else if is_tor {
                NodeRole::TorNode
            } else {
                NodeRole::ExternalServer
            }
        }
        IpAddr::V6(v6) => {
            if v6.is_multicast() {
                NodeRole::Multicast
            } else if v6.is_loopback() || is_ipv6_unique_local_or_link_local(v6) {
                NodeRole::LanHost
            } else if is_tor {
                NodeRole::TorNode
            } else {
                NodeRole::ExternalServer
            }
        }
    }
}

struct NodeAccumulator {
    ip: String,
    mac: Option<String>,
    vendor: Option<String>,
    total_in_bytes: u64,
    total_out_bytes: u64,
    packet_count: u64,
    is_tor: bool,
    highest_threat_score: f32,
    active_ports: BTreeSet<u16>,
    override_label: Option<String>,
    sni: Option<String>,
    dns_qname: Option<String>,
}

/// Builds a complete `NetworkTopology` graph from a collection of flow records.
#[must_use]
pub fn build_network_topology(records: &[FlowTopologyRecord]) -> NetworkTopology {
    build_network_topology_with_asset_cache(records, &HashMap::new())
}

/// Builds a complete `NetworkTopology` graph from a collection of flow records,
/// querying `asset_cache` before falling back to asset identification heuristics.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn build_network_topology_with_asset_cache<S: std::hash::BuildHasher>(
    records: &[FlowTopologyRecord],
    asset_cache: &HashMap<String, AssetFingerprint, S>,
) -> NetworkTopology {
    let mut node_accs: HashMap<String, NodeAccumulator> = HashMap::new();
    let mut edge_map: HashMap<(String, String, String), TopologyEdge> = HashMap::new();
    let mut distribution = ProtocolDistribution::default();

    let mut total_bytes = 0_u64;
    let mut total_packets = 0_u64;

    for record in records {
        total_bytes = total_bytes.saturating_add(record.bytes);
        total_packets = total_packets.saturating_add(record.packet_count);

        let flow_is_tor =
            record.tor_port_matches > 0 || is_tor_traffic(record.src_port, record.dst_port);

        // Protocol breakdown
        match record.protocol.to_uppercase().as_str() {
            "TCP" => {
                distribution.tcp_bytes = distribution.tcp_bytes.saturating_add(record.bytes);
            }
            "UDP" => {
                distribution.udp_bytes = distribution.udp_bytes.saturating_add(record.bytes);
            }
            "ICMP" => {
                distribution.icmp_bytes = distribution.icmp_bytes.saturating_add(record.bytes);
            }
            _ => {
                distribution.other_bytes = distribution.other_bytes.saturating_add(record.bytes);
            }
        }

        distribution.tor_packets = distribution
            .tor_packets
            .saturating_add(record.tor_port_matches);
        distribution.dns_queries = distribution.dns_queries.saturating_add(record.dns_queries);

        if record.sni.is_some() {
            distribution.tls_sessions = distribution.tls_sessions.saturating_add(1);
        }
        distribution.http_requests = distribution
            .http_requests
            .saturating_add(record.http_requests);

        // 1. Source Node Accumulation
        let src_vendor = record.src_mac.and_then(lookup_vendor).map(str::to_owned);
        let src_mac_str = record.src_mac.map(format_mac);

        let src_entry = node_accs
            .entry(record.src_ip.clone())
            .or_insert_with(|| NodeAccumulator {
                ip: record.src_ip.clone(),
                mac: src_mac_str.clone(),
                vendor: src_vendor.clone(),
                total_in_bytes: 0,
                total_out_bytes: 0,
                packet_count: 0,
                is_tor: false,
                highest_threat_score: 0.0,
                active_ports: BTreeSet::new(),
                override_label: None,
                sni: None,
                dns_qname: None,
            });

        src_entry.total_out_bytes = src_entry
            .total_out_bytes
            .saturating_add(record.initiator_to_responder_bytes);
        src_entry.total_in_bytes = src_entry
            .total_in_bytes
            .saturating_add(record.responder_to_initiator_bytes);
        src_entry.packet_count = src_entry.packet_count.saturating_add(record.packet_count);
        src_entry.active_ports.insert(record.src_port);
        if flow_is_tor {
            src_entry.is_tor = true;
        }
        if record.threat_score > src_entry.highest_threat_score {
            src_entry.highest_threat_score = record.threat_score;
        }
        if src_entry.mac.is_none() && src_mac_str.is_some() {
            src_entry.mac = src_mac_str;
            src_entry.vendor = src_vendor;
        }
        if src_entry.sni.is_none() && record.sni.is_some() {
            src_entry.sni.clone_from(&record.sni);
        }
        if src_entry.dns_qname.is_none() && record.dns_qname.is_some() {
            src_entry.dns_qname.clone_from(&record.dns_qname);
        }

        // 2. Destination Node Accumulation
        let dst_vendor = record.dst_mac.and_then(lookup_vendor).map(str::to_owned);
        let dst_mac_str = record.dst_mac.map(format_mac);

        let dst_entry = node_accs
            .entry(record.dst_ip.clone())
            .or_insert_with(|| NodeAccumulator {
                ip: record.dst_ip.clone(),
                mac: dst_mac_str.clone(),
                vendor: dst_vendor.clone(),
                total_in_bytes: 0,
                total_out_bytes: 0,
                packet_count: 0,
                is_tor: false,
                highest_threat_score: 0.0,
                active_ports: BTreeSet::new(),
                override_label: None,
                sni: None,
                dns_qname: None,
            });

        dst_entry.total_in_bytes = dst_entry
            .total_in_bytes
            .saturating_add(record.initiator_to_responder_bytes);
        dst_entry.total_out_bytes = dst_entry
            .total_out_bytes
            .saturating_add(record.responder_to_initiator_bytes);
        dst_entry.packet_count = dst_entry.packet_count.saturating_add(record.packet_count);
        dst_entry.active_ports.insert(record.dst_port);
        if flow_is_tor {
            dst_entry.is_tor = true;
        }
        if record.threat_score > dst_entry.highest_threat_score {
            dst_entry.highest_threat_score = record.threat_score;
        }
        if dst_entry.mac.is_none() && dst_mac_str.is_some() {
            dst_entry.mac = dst_mac_str;
            dst_entry.vendor = dst_vendor;
        }
        if dst_entry.sni.is_none() && record.sni.is_some() {
            dst_entry.sni.clone_from(&record.sni);
        }
        if dst_entry.dns_qname.is_none() && record.dns_qname.is_some() {
            dst_entry.dns_qname.clone_from(&record.dns_qname);
        }
        if let Some(sni) = &record.sni {
            dst_entry.override_label = Some(sni.clone());
        } else if record.dns_qname.is_some() && dst_entry.override_label.is_none() {
            dst_entry.override_label.clone_from(&record.dns_qname);
        }

        // 3. Edge Accumulation
        let edge_key = (
            record.src_ip.clone(),
            record.dst_ip.clone(),
            record.protocol.clone(),
        );

        let edge = edge_map.entry(edge_key).or_insert_with(|| TopologyEdge {
            id: format!("{}-{}-{}", record.src_ip, record.dst_ip, record.protocol),
            source: record.src_ip.clone(),
            target: record.dst_ip.clone(),
            protocol: record.protocol.clone(),
            total_bytes: 0,
            packet_count: 0,
            is_tor: false,
            is_malicious: false,
            threat_score: 0.0,
        });

        edge.total_bytes = edge.total_bytes.saturating_add(record.bytes);
        edge.packet_count = edge.packet_count.saturating_add(record.packet_count);
        if flow_is_tor {
            edge.is_tor = true;
        }
        if record.is_malicious || record.threat_score >= 0.7 {
            edge.is_malicious = true;
        }
        if record.threat_score > edge.threat_score {
            edge.threat_score = record.threat_score;
        }
    }

    let mut lan_hosts_count = 0_usize;
    let mut external_hosts_count = 0_usize;
    let mut tor_nodes_count = 0_usize;
    let mut malicious_nodes_count = 0_usize;

    let nodes: Vec<TopologyNode> = node_accs
        .into_values()
        .map(|acc| {
            let role = classify_endpoint_role(&acc.ip, acc.is_tor);
            match role {
                NodeRole::LanHost | NodeRole::Gateway => lan_hosts_count += 1,
                NodeRole::ExternalServer => external_hosts_count += 1,
                NodeRole::TorNode => tor_nodes_count += 1,
                _ => {}
            }

            if acc.highest_threat_score >= 0.7 {
                malicious_nodes_count += 1;
            }

            let active_ports: Vec<u16> = acc.active_ports.into_iter().take(32).collect();

            let fp = if let Some(cached) = asset_cache.get(&acc.ip) {
                cached.clone()
            } else {
                crate::asset_db::identify_asset(
                    &acc.ip,
                    acc.mac.as_deref(),
                    acc.vendor.as_deref(),
                    &active_ports,
                    acc.sni.as_deref().or(acc.override_label.as_deref()),
                    acc.dns_qname.as_deref(),
                    acc.is_tor,
                )
            };

            let label = if let Some(override_label) = acc.override_label {
                override_label
            } else if !fp.entity_name.is_empty()
                && !fp.entity_name.starts_with("External WAN Server")
                && !fp.entity_name.starts_with("LAN Workstation Endpoint")
            {
                fp.entity_name.clone()
            } else if let Some(vendor) = &acc.vendor {
                format!("{} ({})", acc.ip, vendor)
            } else {
                acc.ip.clone()
            };

            TopologyNode {
                id: acc.ip.clone(),
                label,
                ip: acc.ip,
                mac: acc.mac,
                role,
                vendor: acc.vendor,
                total_in_bytes: acc.total_in_bytes,
                total_out_bytes: acc.total_out_bytes,
                packet_count: acc.packet_count,
                is_tor: acc.is_tor,
                threat_score: acc.highest_threat_score,
                active_ports,
                entity_name: Some(fp.entity_name),
                organization: Some(fp.organization),
                asn: fp.asn,
                category: Some(fp.category),
                device_type: Some(fp.device_type),
                description: Some(fp.description),
                confidence: Some(fp.confidence),
            }
        })
        .collect();

    let edges: Vec<TopologyEdge> = edge_map.into_values().collect();

    NetworkTopology {
        nodes,
        edges,
        protocol_distribution: distribution,
        lan_hosts_count,
        external_hosts_count,
        tor_nodes_count,
        malicious_nodes_count,
        total_bytes,
        total_packets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_endpoint_role() {
        assert_eq!(
            classify_endpoint_role("192.168.1.1", false),
            NodeRole::Gateway
        );
        assert_eq!(
            classify_endpoint_role("192.168.1.150", false),
            NodeRole::LanHost
        );
        assert_eq!(classify_endpoint_role("10.0.0.1", false), NodeRole::Gateway);
        assert_eq!(
            classify_endpoint_role("10.0.0.42", false),
            NodeRole::LanHost
        );
        assert_eq!(
            classify_endpoint_role("8.8.8.8", false),
            NodeRole::ExternalServer
        );
        assert_eq!(
            classify_endpoint_role("255.255.255.255", false),
            NodeRole::Broadcast
        );
        assert_eq!(
            classify_endpoint_role("224.0.0.251", false),
            NodeRole::Multicast
        );
        assert_eq!(
            classify_endpoint_role("198.51.100.20", true),
            NodeRole::TorNode
        );
        assert_eq!(
            classify_endpoint_role("192.168.1.50", true),
            NodeRole::LanHost
        );
    }

    #[test]
    fn test_build_network_topology_aggregation() {
        let records = vec![
            FlowTopologyRecord {
                src_ip: "192.168.1.100".to_string(),
                dst_ip: "192.168.1.1".to_string(),
                src_port: 54321,
                dst_port: 53,
                protocol: "UDP".to_string(),
                src_mac: Some([0x00, 0x15, 0x5D, 0x11, 0x22, 0x33]), // Hyper-V
                dst_mac: Some([0x00, 0x09, 0x5B, 0x44, 0x55, 0x66]), // Netgear
                bytes: 128,
                initiator_to_responder_bytes: 64,
                responder_to_initiator_bytes: 64,
                packet_count: 2,
                dns_queries: 1,
                http_requests: 0,
                tor_port_matches: 0,
                threat_score: 0.1,
                is_malicious: false,
                sni: None,
                dns_qname: Some("internal.router".to_string()),
            },
            FlowTopologyRecord {
                src_ip: "192.168.1.100".to_string(),
                dst_ip: "198.51.100.20".to_string(),
                src_port: 49152,
                dst_port: 9001, // Tor ORPort
                protocol: "TCP".to_string(),
                src_mac: Some([0x00, 0x15, 0x5D, 0x11, 0x22, 0x33]),
                dst_mac: None,
                bytes: 4096,
                initiator_to_responder_bytes: 3072,
                responder_to_initiator_bytes: 1024,
                packet_count: 10,
                dns_queries: 0,
                http_requests: 0,
                tor_port_matches: 10,
                threat_score: 0.85,
                is_malicious: true,
                sni: None,
                dns_qname: None,
            },
        ];

        let topology = build_network_topology(&records);

        assert_eq!(topology.nodes.len(), 3);
        assert_eq!(topology.edges.len(), 2);
        assert_eq!(topology.total_bytes, 4224);
        assert_eq!(topology.total_packets, 12);
        assert_eq!(topology.protocol_distribution.udp_bytes, 128);
        assert_eq!(topology.protocol_distribution.tcp_bytes, 4096);
        assert_eq!(topology.protocol_distribution.dns_queries, 1);
        assert_eq!(topology.protocol_distribution.tor_packets, 10);

        // Find Hyper-V client
        let client = topology
            .nodes
            .iter()
            .find(|n| n.ip == "192.168.1.100")
            .expect("client exists");
        assert_eq!(client.vendor.as_deref(), Some("Microsoft Hyper-V"));
        assert_eq!(client.role, NodeRole::LanHost);
        assert_eq!(
            client.entity_name.as_deref(),
            Some("Microsoft Hyper-V Personal Computer")
        );
        assert_eq!(client.category.as_deref(), Some("Client Workstation"));

        // Find Gateway
        let gateway = topology
            .nodes
            .iter()
            .find(|n| n.ip == "192.168.1.1")
            .expect("gateway exists");
        assert_eq!(gateway.vendor.as_deref(), Some("Netgear"));
        assert_eq!(gateway.role, NodeRole::Gateway);
        assert_eq!(
            gateway.entity_name.as_deref(),
            Some("LAN Default Gateway Router")
        );
        assert_eq!(gateway.category.as_deref(), Some("Gateway / Router"));

        // Find Tor destination
        let tor_node = topology
            .nodes
            .iter()
            .find(|n| n.ip == "198.51.100.20")
            .expect("tor node exists");
        assert_eq!(tor_node.role, NodeRole::TorNode);
        assert!(tor_node.is_tor);
        assert!(tor_node.threat_score >= 0.8);
        assert_eq!(
            tor_node.entity_name.as_deref(),
            Some("Tor-Associated Endpoint")
        );
        assert_eq!(tor_node.category.as_deref(), Some("Tor Port Heuristic"));
    }

    #[test]
    fn test_build_network_topology_with_asset_cache_prepass() {
        let records = vec![FlowTopologyRecord {
            src_ip: "10.0.0.50".to_string(),
            dst_ip: "198.51.100.99".to_string(),
            src_port: 49500,
            dst_port: 443,
            protocol: "TCP".to_string(),
            src_mac: None,
            dst_mac: None,
            bytes: 1500,
            initiator_to_responder_bytes: 1000,
            responder_to_initiator_bytes: 500,
            packet_count: 5,
            dns_queries: 0,
            http_requests: 0,
            tor_port_matches: 0,
            threat_score: 0.0,
            is_malicious: false,
            sni: Some("custom.example.com".to_string()),
            dns_qname: None,
        }];

        let mut custom_cache = HashMap::new();
        custom_cache.insert(
            "198.51.100.99".to_string(),
            AssetFingerprint {
                entity_name: "Pre-resolved Custom Asset".to_string(),
                organization: "Acme Cloud Services".to_string(),
                asn: Some("AS64500".to_string()),
                category: "Custom Cloud Infrastructure".to_string(),
                device_type: "Edge Proxy".to_string(),
                description: "Fingerprint injected from pre-pass cache".to_string(),
                confidence: "High (Synthetic Prepass)".to_string(),
            },
        );

        let topology = build_network_topology_with_asset_cache(&records, &custom_cache);
        assert_eq!(topology.nodes.len(), 2);

        let dst_node = topology
            .nodes
            .iter()
            .find(|n| n.ip == "198.51.100.99")
            .expect("destination node exists");

        assert_eq!(
            dst_node.entity_name.as_deref(),
            Some("Pre-resolved Custom Asset")
        );
        assert_eq!(
            dst_node.organization.as_deref(),
            Some("Acme Cloud Services")
        );
        assert_eq!(dst_node.asn.as_deref(), Some("AS64500"));
        assert_eq!(
            dst_node.category.as_deref(),
            Some("Custom Cloud Infrastructure")
        );
    }
}

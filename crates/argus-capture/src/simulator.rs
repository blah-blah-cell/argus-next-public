//! Local Adversarial Traffic Simulator and Attack Vector Harness.
//!
//! Generates deterministic, structurally valid libpcap byte streams
//! simulating specific MITRE ATT&CK adversarial techniques in self-contained local memory.
//! Used exclusively for non-destructive detection engineering and model validation.

use serde::{Deserialize, Serialize};

/// MITRE ATT&CK technique metadata associated with an adversarial simulation scenario.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AttackTechnique {
    pub technique_id: String,
    pub technique_name: String,
    pub tactic: String,
}

/// Adversarial simulation scenario descriptor.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SimulationScenario {
    pub id: String,
    pub title: String,
    pub description: String,
    pub category: String,
    pub technique: AttackTechnique,
    pub default_packet_count: usize,
}

/// Returns the registry of available self-owned adversarial simulation scenarios.
#[must_use]
pub fn list_available_scenarios() -> Vec<SimulationScenario> {
    vec![
        SimulationScenario {
            id: "tcp_recon_evasion".to_owned(),
            title: "TCP Reconnaissance & Evasion Probe".to_owned(),
            description: "Rapid TCP port sweep containing SYN, Xmas (FIN+PSH+URG), and NULL flag anomalies targeting multiple listening services.".to_owned(),
            category: "Reconnaissance".to_owned(),
            technique: AttackTechnique {
                technique_id: "T1046".to_owned(),
                technique_name: "Network Service Discovery".to_owned(),
                tactic: "Discovery".to_owned(),
            },
            default_packet_count: 64,
        },
        SimulationScenario {
            id: "dns_tunnel_exfil".to_owned(),
            title: "DNS Tunneling & C2 Query Flood".to_owned(),
            description: "Repeated DNS TXT queries with encoded-looking subdomains to exercise DNS parsing; this is not a validated tunnel detector.".to_owned(),
            category: "Command & Control".to_owned(),
            technique: AttackTechnique {
                technique_id: "T1071.004".to_owned(),
                technique_name: "DNS Application Layer Protocol".to_owned(),
                tactic: "Command and Control".to_owned(),
            },
            default_packet_count: 160,
        },
        SimulationScenario {
            id: "high_entropy_c2_exfil".to_owned(),
            title: "Encrypted Data Exfiltration Burst".to_owned(),
            description: "Pseudo-random 256-byte UDP payloads on one flow to exercise payload inspection; high entropy alone does not establish exfiltration.".to_owned(),
            category: "Exfiltration".to_owned(),
            technique: AttackTechnique {
                technique_id: "T1048".to_owned(),
                technique_name: "Exfiltration Over Alternative Protocol".to_owned(),
                tactic: "Exfiltration".to_owned(),
            },
            default_packet_count: 160,
        },
        SimulationScenario {
            id: "traffic_volume_spike".to_owned(),
            title: "Volumetric Rate Anomaly & Syn Flood".to_owned(),
            description: "Rapid SYN packets across changing client ports to stress rate and flow handling; a rule match is not guaranteed.".to_owned(),
            category: "Impact / Denial of Service".to_owned(),
            technique: AttackTechnique {
                technique_id: "T1498".to_owned(),
                technique_name: "Network Denial of Service".to_owned(),
                tactic: "Impact".to_owned(),
            },
            default_packet_count: 150,
        },
        SimulationScenario {
            id: "nominal_baseline_traffic".to_owned(),
            title: "Nominal Baseline Web & TLS Traffic".to_owned(),
            description: "Alternating TCP packets carrying HTTP-like text at port 443; not a real TLS handshake or HTTP/2 exchange.".to_owned(),
            category: "Baseline / Benign".to_owned(),
            technique: AttackTechnique {
                technique_id: "T1071.001".to_owned(),
                technique_name: "Web Protocols (Benign Baseline)".to_owned(),
                tactic: "Normal Operations".to_owned(),
            },
            default_packet_count: 160,
        },
    ]
}

/// Generates the raw PCAP bytes for the specified simulation scenario.
///
/// # Errors
///
/// Returns an error message if the scenario ID is unknown or PCAP generation fails.
pub fn generate_simulation_pcap(scenario_id: &str, packet_count: usize) -> Result<Vec<u8>, String> {
    let count = packet_count.clamp(10, 500);
    match scenario_id {
        "tcp_recon_evasion" => Ok(build_tcp_recon_pcap(count)),
        "dns_tunnel_exfil" => Ok(build_dns_tunnel_pcap(count)),
        "high_entropy_c2_exfil" => Ok(build_high_entropy_pcap(count)),
        "traffic_volume_spike" => Ok(build_traffic_spike_pcap(count)),
        "nominal_baseline_traffic" => Ok(build_baseline_pcap(count)),
        other => Err(format!("Unknown simulation scenario '{other}'")),
    }
}

fn build_tcp_recon_pcap(count: usize) -> Vec<u8> {
    let mut frames = Vec::new();
    let base_ts = 1_700_000_000_000_000_000_u64; // Unix nanoseconds

    for i in 0..count {
        let ts = base_ts + (i as u64) * 20_000_000; // 20ms apart
        let port_index = u16::try_from(i).expect("packet index fits u16");
        let dst_port = 20_u16 + port_index % 1000;
        let src_port = 45000 + port_index % 50;

        let tcp_flag = match i % 4 {
            0 => 0x02, // SYN
            1 => 0x29, // Xmas: FIN (0x01) + PSH (0x08) + URG (0x20)
            2 => 0x00, // NULL scan
            _ => 0x01, // FIN scan
        };

        let frame = build_ipv4_tcp_frame(
            [192, 168, 1, 150],
            [10, 0, 0, 5],
            src_port,
            dst_port,
            tcp_flag,
            b"SCAN_PROBE",
        );
        frames.push((ts, frame));
    }

    assemble_pcap(&frames)
}

fn build_dns_tunnel_pcap(count: usize) -> Vec<u8> {
    let mut frames = Vec::new();
    let base_ts = 1_700_000_000_000_000_000_u64;

    for i in 0..count {
        let ts = base_ts + (i as u64) * 15_000_000;
        let query_id = u16::try_from(0x1000 + i).expect("query id fits u16");
        let mut dns_payload = Vec::new();
        // Transaction ID
        dns_payload.extend_from_slice(&query_id.to_be_bytes());
        // Standard Query Flags (0x0100)
        dns_payload.extend_from_slice(&[0x01, 0x00]);
        // Questions: 1, Answer RRs: 0, Authority RRs: 0, Additional RRs: 0
        dns_payload.extend_from_slice(&[0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);

        // C2 Subdomain label: e.g. "x9a2f4c1e8b7.tunnel.c2.lab"
        let hex_prefix = format!("x{:08x}payload{:04x}", i * 98765, i);
        dns_payload.push(u8::try_from(hex_prefix.len()).expect("hex prefix fits u8"));
        dns_payload.extend_from_slice(hex_prefix.as_bytes());
        dns_payload.extend_from_slice(b"\x06tunnel\x02c2\x03lab\x00");
        // QTYPE: TXT (16 = 0x0010)
        dns_payload.extend_from_slice(&[0x00, 0x10]);
        // QCLASS: IN (1 = 0x0001)
        dns_payload.extend_from_slice(&[0x00, 0x01]);

        let frame = build_ipv4_udp_frame([192, 168, 1, 105], [8, 8, 8, 8], 53211, 53, &dns_payload);
        frames.push((ts, frame));
    }

    assemble_pcap(&frames)
}

fn build_high_entropy_pcap(count: usize) -> Vec<u8> {
    let mut frames = Vec::new();
    let base_ts = 1_700_000_000_000_000_000_u64;

    for i in 0..count {
        let ts = base_ts + (i as u64) * 5_000_000; // 5ms apart
        // Generate pseudo-random bytes with maximal Shannon entropy
        let mut pseudo_random = Vec::with_capacity(256);
        let mut seed = u32::try_from(i)
            .expect("packet index fits u32")
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        for _ in 0..256 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            pseudo_random.push(u8::try_from((seed >> 16) & 0xff).expect("masked byte"));
        }

        let frame = build_ipv4_udp_frame(
            [192, 168, 1, 88],
            [198, 51, 100, 42],
            48912,
            9001,
            &pseudo_random,
        );
        frames.push((ts, frame));
    }

    assemble_pcap(&frames)
}

fn build_traffic_spike_pcap(count: usize) -> Vec<u8> {
    let mut frames = Vec::new();
    let base_ts = 1_700_000_000_000_000_000_u64;

    for i in 0..count {
        // High-density sub-microsecond spacing (500 nanoseconds apart)
        let ts = base_ts + (i as u64) * 500;
        let frame = build_ipv4_tcp_frame(
            [192, 168, 1, 200],
            [10, 0, 0, 1],
            55000 + u16::try_from(i).expect("packet index fits u16") % 100,
            80,
            0x02, // SYN flood burst
            b"DOS_BURST",
        );
        frames.push((ts, frame));
    }

    assemble_pcap(&frames)
}

fn build_baseline_pcap(count: usize) -> Vec<u8> {
    let mut frames = Vec::new();
    let base_ts = 1_700_000_000_000_000_000_u64;

    for i in 0..count {
        let ts = base_ts + (i as u64) * 100_000_000; // 100ms apart
        let (client_ip, server_ip, client_port, server_port) = if i % 2 == 0 {
            ([192, 168, 1, 50], [142, 250, 190, 46], 51234, 443)
        } else {
            ([142, 250, 190, 46], [192, 168, 1, 50], 443, 51234)
        };

        let frame = build_ipv4_tcp_frame(
            client_ip,
            server_ip,
            client_port,
            server_port,
            0x18, // PSH + ACK
            b"GET / HTTP/1.1\r\nHost: example.org\r\nUser-Agent: Mozilla/5.0\r\n\r\n",
        );
        frames.push((ts, frame));
    }

    assemble_pcap(&frames)
}

fn build_ipv4_tcp_frame(
    src_ip: [u8; 4],
    dst_ip: [u8; 4],
    src_port: u16,
    dst_port: u16,
    tcp_flags: u8,
    payload: &[u8],
) -> Vec<u8> {
    let mut frame = Vec::new();
    // Ethernet Header (14 bytes)
    frame.extend_from_slice(&[0x00, 0x11, 0x22, 0x33, 0x44, 0x55]); // Dst MAC
    frame.extend_from_slice(&[0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff]); // Src MAC
    frame.extend_from_slice(&[0x08, 0x00]); // EtherType: IPv4

    // IPv4 Header (20 bytes)
    let ip_total_len = u16::try_from(20 + 20 + payload.len()).expect("frame length fits u16");
    frame.push(0x45); // Version 4, IHL 5
    frame.push(0x00); // DSCP/ECN
    frame.extend_from_slice(&ip_total_len.to_be_bytes());
    frame.extend_from_slice(&[0x12, 0x34]); // Identification
    frame.extend_from_slice(&[0x40, 0x00]); // Flags: Don't Fragment
    frame.push(64); // TTL
    frame.push(6); // Protocol: TCP
    frame.extend_from_slice(&[0x00, 0x00]); // Checksum (zero for test)
    frame.extend_from_slice(&src_ip);
    frame.extend_from_slice(&dst_ip);

    // TCP Header (20 bytes)
    frame.extend_from_slice(&src_port.to_be_bytes());
    frame.extend_from_slice(&dst_port.to_be_bytes());
    frame.extend_from_slice(&[0x00, 0x00, 0x10, 0x00]); // Sequence Number
    frame.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Ack Number
    frame.push(0x50); // Data Offset: 5 (20 bytes)
    frame.push(tcp_flags);
    frame.extend_from_slice(&[0xfa, 0xf0]); // Window Size
    frame.extend_from_slice(&[0x00, 0x00]); // Checksum
    frame.extend_from_slice(&[0x00, 0x00]); // Urgent Pointer

    // Payload
    frame.extend_from_slice(payload);

    frame
}

fn build_ipv4_udp_frame(
    src_ip: [u8; 4],
    dst_ip: [u8; 4],
    src_port: u16,
    dst_port: u16,
    payload: &[u8],
) -> Vec<u8> {
    let mut frame = Vec::new();
    // Ethernet Header (14 bytes)
    frame.extend_from_slice(&[0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
    frame.extend_from_slice(&[0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff]);
    frame.extend_from_slice(&[0x08, 0x00]);

    // IPv4 Header (20 bytes)
    let ip_total_len = u16::try_from(20 + 8 + payload.len()).expect("frame length fits u16");
    frame.push(0x45);
    frame.push(0x00);
    frame.extend_from_slice(&ip_total_len.to_be_bytes());
    frame.extend_from_slice(&[0x56, 0x78]);
    frame.extend_from_slice(&[0x00, 0x00]);
    frame.push(64);
    frame.push(17); // UDP
    frame.extend_from_slice(&[0x00, 0x00]);
    frame.extend_from_slice(&src_ip);
    frame.extend_from_slice(&dst_ip);

    // UDP Header (8 bytes)
    let udp_len = u16::try_from(8 + payload.len()).expect("frame length fits u16");
    frame.extend_from_slice(&src_port.to_be_bytes());
    frame.extend_from_slice(&dst_port.to_be_bytes());
    frame.extend_from_slice(&udp_len.to_be_bytes());
    frame.extend_from_slice(&[0x00, 0x00]); // Checksum

    // Payload
    frame.extend_from_slice(payload);

    frame
}

fn assemble_pcap(frames: &[(u64, Vec<u8>)]) -> Vec<u8> {
    let mut file_buf = Vec::new();
    let packet_refs: Vec<(u64, &[u8], usize)> = frames
        .iter()
        .map(|(ts, f)| (*ts, f.as_slice(), f.len()))
        .collect();
    crate::writer::write_pcap_stream(&mut file_buf, packet_refs).expect("in-memory pcap write");
    file_buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_registered_scenarios() {
        let list = list_available_scenarios();
        assert_eq!(list.len(), 5);
        assert!(list.iter().any(|s| s.id == "tcp_recon_evasion"));
        assert!(list.iter().any(|s| s.id == "dns_tunnel_exfil"));
    }

    #[test]
    fn generates_valid_pcap_for_all_scenarios() {
        for scenario in list_available_scenarios() {
            let pcap_bytes =
                generate_simulation_pcap(&scenario.id, 20).expect("simulation generation succeeds");
            assert!(pcap_bytes.len() > 24, "pcap must include header + packets");
            // Check PCAP magic number 0xa1b2c3d4 (little-endian: d4 c3 b2 a1)
            assert_eq!(&pcap_bytes[0..4], &[0xd4, 0xc3, 0xb2, 0xa1]);
        }
    }
}

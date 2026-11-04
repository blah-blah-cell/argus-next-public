//! Structured packet and flow filtering engine with BPF compilation support.
//!
//! Supports compound queries combining key-value predicates, CIDR masks, port ranges,
//! TCP flag combinations, byte/packet thresholds, L7 protocol observations (TLS SNI, DNS QNAME),
//! and `!`/`-` negation, while also compiling network/transport predicates into standard
//! libpcap BPF syntax for live capture or filtered PCAP slice export.

use std::net::IpAddr;
#[cfg(test)]
use std::net::Ipv4Addr;

use argus_core::TransportProtocol;

use crate::{CaptureError, CapturedPacket};

/// Numeric comparison operator for packet/flow metrics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThresholdOp {
    GreaterThan,
    GreaterOrEqual,
    LessThan,
    LessOrEqual,
    Equal,
}

impl ThresholdOp {
    #[must_use]
    pub const fn evaluate(self, actual: u64, target: u64) -> bool {
        match self {
            Self::GreaterThan => actual > target,
            Self::GreaterOrEqual => actual >= target,
            Self::LessThan => actual < target,
            Self::LessOrEqual => actual <= target,
            Self::Equal => actual == target,
        }
    }
}

/// IPv4 or IPv6 CIDR prefix matcher.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IpCidr {
    pub network: IpAddr,
    pub prefix_len: u8,
}

impl IpCidr {
    /// Parses `ip/prefix` or a single IP address (`prefix = 32` or `128`).
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError::InvalidMetadata`] if the IP or prefix length is invalid.
    pub fn parse(input: &str) -> Result<Self, CaptureError> {
        if let Some((ip_part, prefix_part)) = input.split_once('/') {
            let network: IpAddr = ip_part
                .parse()
                .map_err(|_| CaptureError::InvalidMetadata("invalid CIDR IP address"))?;
            let prefix_len: u8 = prefix_part
                .parse()
                .map_err(|_| CaptureError::InvalidMetadata("invalid CIDR prefix length"))?;
            let max_prefix = if network.is_ipv4() { 32 } else { 128 };
            if prefix_len > max_prefix {
                return Err(CaptureError::InvalidMetadata(
                    "CIDR prefix length exceeds address family width",
                ));
            }
            Ok(Self {
                network,
                prefix_len,
            })
        } else {
            let network: IpAddr = input
                .parse()
                .map_err(|_| CaptureError::InvalidMetadata("invalid IP address"))?;
            let prefix_len = if network.is_ipv4() { 32 } else { 128 };
            Ok(Self {
                network,
                prefix_len,
            })
        }
    }

    #[must_use]
    pub fn contains(&self, candidate: IpAddr) -> bool {
        match (self.network, candidate) {
            (IpAddr::V4(net), IpAddr::V4(ip)) => {
                if self.prefix_len == 0 {
                    return true;
                }
                let shift = 32_u32.saturating_sub(u32::from(self.prefix_len));
                let mask = u32::MAX.checked_shl(shift).unwrap_or(0);
                (u32::from(net) & mask) == (u32::from(ip) & mask)
            }
            (IpAddr::V6(net), IpAddr::V6(ip)) => {
                if self.prefix_len == 0 {
                    return true;
                }
                let shift = 128_u32.saturating_sub(u32::from(self.prefix_len));
                let mask = u128::MAX.checked_shl(shift).unwrap_or(0);
                (u128::from(net) & mask) == (u128::from(ip) & mask)
            }
            _ => false,
        }
    }
}

/// Atomic predicate in a [`PacketFilter`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FilterPredicate {
    Protocol(TransportProtocol),
    IpVersion(u8),
    AnyHost(IpCidr),
    SourceHost(IpCidr),
    DestinationHost(IpCidr),
    AnyPort { min: u16, max: u16 },
    SourcePort { min: u16, max: u16 },
    DestinationPort { min: u16, max: u16 },
    TcpFlagsMask { mask: u8, expected: u8 },
    PacketBytes { op: ThresholdOp, bytes: u64 },
    TlsSniContains(String),
    DnsQnameContains(String),
    HasHttp,
    HasTor,
    HasTls,
    HasDns,
    ProcessContains(String),
    FreeText(String),
}

/// Potentially negated filter clause.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FilterClause {
    pub negated: bool,
    pub predicate: FilterPredicate,
}

/// Compiled packet and flow filter with conjunctive clauses.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PacketFilter {
    pub clauses: Vec<FilterClause>,
}

/// Flow-level summary view used for filtering aggregated flows.
#[derive(Clone, Debug)]
pub struct FlowFilterView<'a> {
    pub protocol: &'a str,
    pub initiator_addr: IpAddr,
    pub initiator_port: u16,
    pub responder_addr: IpAddr,
    pub responder_port: u16,
    pub packets: u64,
    pub bytes: u64,
    pub tcp_flags: Option<u8>,
    pub tls_sni: Option<&'a str>,
    pub dns_qname: Option<&'a str>,
    pub http_request: bool,
    pub tor_port_match: bool,
    pub process_name: Option<&'a str>,
}

impl PacketFilter {
    /// Parses a structured filter query string into a [`PacketFilter`].
    ///
    /// Supported tokens:
    /// - `proto:tcp`, `proto:udp`, `ipv4`, `ipv6`
    /// - `ip:192.168.1.0/24`, `src:10.0.0.1`, `dst:8.8.8.8`
    /// - `port:443`, `port:8000-9000`, `src_port:53`, `dst_port:443`
    /// - `flags:syn`, `flags:rst`, `flags:fin`, `flags:xmas`, `flags:null`
    /// - `bytes:>1000`, `len:>=512`
    /// - `sni:github`, `dns:example.com`, `proc:node`, `http`, `tls`, `dns`, `tor`
    /// - Prefixing any token with `!` or `-` negates the clause.
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError::InvalidMetadata`] if an explicit key-value token has malformed syntax.
    pub fn parse(query: &str) -> Result<Self, CaptureError> {
        let mut clauses = Vec::new();
        for raw_token in query.split_whitespace() {
            let (negated, token) = if let Some(rest) = raw_token.strip_prefix('!') {
                (true, rest)
            } else if let Some(rest) = raw_token.strip_prefix('-') {
                (true, rest)
            } else {
                (false, raw_token)
            };
            if token.is_empty() {
                continue;
            }
            let predicate = parse_predicate(token)?;
            clauses.push(FilterClause { negated, predicate });
        }
        Ok(Self { clauses })
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }

    /// Evaluates whether a normalized [`CapturedPacket`] satisfies all clauses in the filter.
    #[must_use]
    pub fn matches_packet(&self, packet: &CapturedPacket) -> bool {
        self.clauses.iter().all(|clause| {
            let matched = match &clause.predicate {
                FilterPredicate::Protocol(proto) => packet.flow.key.protocol == *proto,
                FilterPredicate::IpVersion(ver) => packet.protocol.ip_version == *ver,
                FilterPredicate::AnyHost(cidr) => {
                    cidr.contains(packet.flow.key.initiator.address)
                        || cidr.contains(packet.flow.key.responder.address)
                }
                FilterPredicate::SourceHost(cidr) => {
                    cidr.contains(packet.flow.key.initiator.address)
                }
                FilterPredicate::DestinationHost(cidr) => {
                    cidr.contains(packet.flow.key.responder.address)
                }
                FilterPredicate::AnyPort { min, max } => {
                    let sp = packet.flow.key.initiator.port;
                    let dp = packet.flow.key.responder.port;
                    (sp >= *min && sp <= *max) || (dp >= *min && dp <= *max)
                }
                FilterPredicate::SourcePort { min, max } => {
                    let sp = packet.flow.key.initiator.port;
                    sp >= *min && sp <= *max
                }
                FilterPredicate::DestinationPort { min, max } => {
                    let dp = packet.flow.key.responder.port;
                    dp >= *min && dp <= *max
                }
                FilterPredicate::TcpFlagsMask { mask, expected } => packet
                    .protocol
                    .tcp_flags
                    .is_some_and(|flags| (flags & *mask) == *expected),
                FilterPredicate::PacketBytes { op, bytes } => {
                    op.evaluate(u64::from(packet.packet.original_len), *bytes)
                }
                FilterPredicate::TlsSniContains(needle) => packet
                    .observations
                    .tls_sni
                    .as_ref()
                    .is_some_and(|sni| sni.to_ascii_lowercase().contains(needle)),
                FilterPredicate::DnsQnameContains(needle) => packet
                    .observations
                    .dns_qname
                    .as_ref()
                    .is_some_and(|qname| qname.to_ascii_lowercase().contains(needle)),
                FilterPredicate::HasHttp => packet.observations.http_request,
                FilterPredicate::HasTor => packet.observations.tor_port_match,
                FilterPredicate::HasTls => packet.observations.tls_sni.is_some(),
                FilterPredicate::HasDns => packet.observations.dns_message,
                FilterPredicate::ProcessContains(_) => false,
                FilterPredicate::FreeText(needle) => {
                    let init_ip = packet.flow.key.initiator.address.to_string();
                    let resp_ip = packet.flow.key.responder.address.to_string();
                    let init_port = packet.flow.key.initiator.port.to_string();
                    let resp_port = packet.flow.key.responder.port.to_string();
                    let proto = match packet.flow.key.protocol {
                        TransportProtocol::Tcp => "tcp",
                        TransportProtocol::Udp => "udp",
                        TransportProtocol::Other(_) => "other",
                    };
                    init_ip.contains(needle)
                        || resp_ip.contains(needle)
                        || init_port == *needle
                        || resp_port == *needle
                        || proto.contains(needle)
                        || packet
                            .observations
                            .tls_sni
                            .as_ref()
                            .is_some_and(|s| s.to_ascii_lowercase().contains(needle))
                        || packet
                            .observations
                            .dns_qname
                            .as_ref()
                            .is_some_and(|d| d.to_ascii_lowercase().contains(needle))
                }
            };
            if clause.negated { !matched } else { matched }
        })
    }

    /// Evaluates whether an aggregated flow satisfies all clauses in the filter.
    #[must_use]
    pub fn matches_flow(&self, flow: &FlowFilterView<'_>) -> bool {
        self.clauses.iter().all(|clause| {
            let matched = match &clause.predicate {
                FilterPredicate::Protocol(proto) => {
                    let expected = match proto {
                        TransportProtocol::Tcp => "TCP",
                        TransportProtocol::Udp => "UDP",
                        TransportProtocol::Other(_) => "OTHER",
                    };
                    flow.protocol.eq_ignore_ascii_case(expected)
                }
                FilterPredicate::IpVersion(ver) => {
                    (*ver == 4 && flow.initiator_addr.is_ipv4())
                        || (*ver == 6 && flow.initiator_addr.is_ipv6())
                }
                FilterPredicate::AnyHost(cidr) => {
                    cidr.contains(flow.initiator_addr) || cidr.contains(flow.responder_addr)
                }
                FilterPredicate::SourceHost(cidr) => cidr.contains(flow.initiator_addr),
                FilterPredicate::DestinationHost(cidr) => cidr.contains(flow.responder_addr),
                FilterPredicate::AnyPort { min, max } => {
                    (flow.initiator_port >= *min && flow.initiator_port <= *max)
                        || (flow.responder_port >= *min && flow.responder_port <= *max)
                }
                FilterPredicate::SourcePort { min, max } => {
                    flow.initiator_port >= *min && flow.initiator_port <= *max
                }
                FilterPredicate::DestinationPort { min, max } => {
                    flow.responder_port >= *min && flow.responder_port <= *max
                }
                FilterPredicate::TcpFlagsMask { mask, expected } => flow
                    .tcp_flags
                    .is_some_and(|flags| (flags & *mask) == *expected),
                FilterPredicate::PacketBytes { op, bytes } => op.evaluate(flow.bytes, *bytes),
                FilterPredicate::TlsSniContains(needle) => flow
                    .tls_sni
                    .is_some_and(|s| s.to_ascii_lowercase().contains(needle)),
                FilterPredicate::DnsQnameContains(needle) => flow
                    .dns_qname
                    .is_some_and(|d| d.to_ascii_lowercase().contains(needle)),
                FilterPredicate::HasHttp => flow.http_request,
                FilterPredicate::HasTor => flow.tor_port_match,
                FilterPredicate::HasTls => flow.tls_sni.is_some(),
                FilterPredicate::HasDns => flow.dns_qname.is_some(),
                FilterPredicate::ProcessContains(needle) => flow
                    .process_name
                    .is_some_and(|p| p.to_ascii_lowercase().contains(needle)),
                FilterPredicate::FreeText(needle) => {
                    flow.initiator_addr.to_string().contains(needle)
                        || flow.responder_addr.to_string().contains(needle)
                        || flow.initiator_port.to_string() == *needle
                        || flow.responder_port.to_string() == *needle
                        || flow.protocol.to_ascii_lowercase().contains(needle)
                        || flow
                            .tls_sni
                            .is_some_and(|s| s.to_ascii_lowercase().contains(needle))
                        || flow
                            .dns_qname
                            .is_some_and(|d| d.to_ascii_lowercase().contains(needle))
                        || flow
                            .process_name
                            .is_some_and(|p| p.to_ascii_lowercase().contains(needle))
                }
            };
            if clause.negated { !matched } else { matched }
        })
    }

    /// Compiles exactly representable clauses into a libpcap BPF expression.
    ///
    /// # Errors
    /// Returns an error for application-only or approximate predicates rather than omitting them.
    pub fn to_bpf_expression(&self) -> Result<String, CaptureError> {
        let mut parts = Vec::new();
        for clause in &self.clauses {
            if !matches!(
                clause.predicate,
                FilterPredicate::Protocol(TransportProtocol::Tcp | TransportProtocol::Udp)
                    | FilterPredicate::IpVersion(4 | 6)
                    | FilterPredicate::AnyHost(_)
                    | FilterPredicate::SourceHost(_)
                    | FilterPredicate::DestinationHost(_)
                    | FilterPredicate::AnyPort { .. }
                    | FilterPredicate::SourcePort { .. }
                    | FilterPredicate::DestinationPort { .. }
            ) {
                return Err(CaptureError::InvalidMetadata(
                    "filter cannot be represented exactly in BPF; use application filtering",
                ));
            }
            let expr = match &clause.predicate {
                FilterPredicate::Protocol(TransportProtocol::Tcp) => Some("tcp".to_owned()),
                FilterPredicate::Protocol(TransportProtocol::Udp) => Some("udp".to_owned()),
                FilterPredicate::IpVersion(4) => Some("ip".to_owned()),
                FilterPredicate::IpVersion(6) => Some("ip6".to_owned()),
                FilterPredicate::AnyHost(cidr) => Some(cidr_to_bpf("host", "net", *cidr)),
                FilterPredicate::SourceHost(cidr) => {
                    Some(cidr_to_bpf("src host", "src net", *cidr))
                }
                FilterPredicate::DestinationHost(cidr) => {
                    Some(cidr_to_bpf("dst host", "dst net", *cidr))
                }
                FilterPredicate::AnyPort { min, max } => {
                    if min == max {
                        Some(format!("port {min}"))
                    } else {
                        Some(format!("portrange {min}-{max}"))
                    }
                }
                FilterPredicate::SourcePort { min, max } => {
                    if min == max {
                        Some(format!("src port {min}"))
                    } else {
                        Some(format!("src portrange {min}-{max}"))
                    }
                }
                FilterPredicate::DestinationPort { min, max } => {
                    if min == max {
                        Some(format!("dst port {min}"))
                    } else {
                        Some(format!("dst portrange {min}-{max}"))
                    }
                }
                FilterPredicate::HasDns => Some("port 53".to_owned()),
                FilterPredicate::HasTls => Some("tcp port 443".to_owned()),
                FilterPredicate::HasHttp => Some("tcp port 80 or tcp port 8080".to_owned()),
                FilterPredicate::HasTor => {
                    Some("port 9001 or port 9030 or port 9050 or port 9150".to_owned())
                }
                _ => None,
            };
            if let Some(inner) = expr {
                if clause.negated {
                    parts.push(format!("not ({inner})"));
                } else {
                    parts.push(format!("({inner})"));
                }
            }
        }
        Ok(parts.join(" and "))
    }
}

fn cidr_to_bpf(host_kw: &str, net_kw: &str, cidr: IpCidr) -> String {
    let max_prefix = if cidr.network.is_ipv4() { 32 } else { 128 };
    if cidr.prefix_len == max_prefix {
        format!("{host_kw} {}", cidr.network)
    } else {
        format!("{net_kw} {}/{}", cidr.network, cidr.prefix_len)
    }
}

fn parse_predicate(token: &str) -> Result<FilterPredicate, CaptureError> {
    let lower = token.to_ascii_lowercase();
    match lower.as_str() {
        "tcp" => return Ok(FilterPredicate::Protocol(TransportProtocol::Tcp)),
        "udp" => return Ok(FilterPredicate::Protocol(TransportProtocol::Udp)),
        "ipv4" | "ip4" => return Ok(FilterPredicate::IpVersion(4)),
        "ipv6" | "ip6" => return Ok(FilterPredicate::IpVersion(6)),
        "http" => return Ok(FilterPredicate::HasHttp),
        "tls" | "ssl" | "sni" => return Ok(FilterPredicate::HasTls),
        "dns" => return Ok(FilterPredicate::HasDns),
        "tor" => return Ok(FilterPredicate::HasTor),
        _ => {}
    }

    let Some((key, val)) = lower.split_once(':') else {
        // Check if bare token is an IP/CIDR
        if (lower.contains('.') || lower.contains('/'))
            && let Ok(cidr) = IpCidr::parse(&lower)
        {
            return Ok(FilterPredicate::AnyHost(cidr));
        }
        return Ok(FilterPredicate::FreeText(lower));
    };

    if val.is_empty() {
        return Err(CaptureError::InvalidMetadata(
            "filter predicate value cannot be empty",
        ));
    }

    match key {
        "proto" | "protocol" => match val {
            "tcp" => Ok(FilterPredicate::Protocol(TransportProtocol::Tcp)),
            "udp" => Ok(FilterPredicate::Protocol(TransportProtocol::Udp)),
            _ => Err(CaptureError::InvalidMetadata(
                "unsupported protocol filter (expected tcp or udp)",
            )),
        },
        "ip" | "host" | "cidr" | "net" => Ok(FilterPredicate::AnyHost(IpCidr::parse(val)?)),
        "src" | "src_ip" | "initiator" => Ok(FilterPredicate::SourceHost(IpCidr::parse(val)?)),
        "dst" | "dst_ip" | "responder" => Ok(FilterPredicate::DestinationHost(IpCidr::parse(val)?)),
        "port" => {
            let (min, max) = parse_port_range(val)?;
            Ok(FilterPredicate::AnyPort { min, max })
        }
        "src_port" | "sport" => {
            let (min, max) = parse_port_range(val)?;
            Ok(FilterPredicate::SourcePort { min, max })
        }
        "dst_port" | "dport" => {
            let (min, max) = parse_port_range(val)?;
            Ok(FilterPredicate::DestinationPort { min, max })
        }
        "flags" | "tcp_flags" => parse_tcp_flags_predicate(val),
        "bytes" | "len" | "size" => {
            let (op, bytes) = parse_threshold(val)?;
            Ok(FilterPredicate::PacketBytes { op, bytes })
        }
        "sni" | "tls" => Ok(FilterPredicate::TlsSniContains(val.to_owned())),
        "dns" | "qname" => Ok(FilterPredicate::DnsQnameContains(val.to_owned())),
        "proc" | "process" => Ok(FilterPredicate::ProcessContains(val.to_owned())),
        _ => Ok(FilterPredicate::FreeText(lower)),
    }
}

fn parse_port_range(val: &str) -> Result<(u16, u16), CaptureError> {
    if let Some((start_s, end_s)) = val.split_once('-') {
        let min: u16 = start_s
            .parse()
            .map_err(|_| CaptureError::InvalidMetadata("invalid start port in port range"))?;
        let max: u16 = end_s
            .parse()
            .map_err(|_| CaptureError::InvalidMetadata("invalid end port in port range"))?;
        if min > max {
            return Err(CaptureError::InvalidMetadata(
                "port range start exceeds end port",
            ));
        }
        Ok((min, max))
    } else {
        let port: u16 = val
            .parse()
            .map_err(|_| CaptureError::InvalidMetadata("invalid port number"))?;
        Ok((port, port))
    }
}

fn parse_tcp_flags_predicate(val: &str) -> Result<FilterPredicate, CaptureError> {
    match val {
        "syn" => Ok(FilterPredicate::TcpFlagsMask {
            mask: 0x02,
            expected: 0x02,
        }),
        "synonly" | "syn_only" => Ok(FilterPredicate::TcpFlagsMask {
            mask: 0x12,
            expected: 0x02,
        }),
        "synack" | "syn_ack" => Ok(FilterPredicate::TcpFlagsMask {
            mask: 0x12,
            expected: 0x12,
        }),
        "ack" => Ok(FilterPredicate::TcpFlagsMask {
            mask: 0x10,
            expected: 0x10,
        }),
        "fin" => Ok(FilterPredicate::TcpFlagsMask {
            mask: 0x01,
            expected: 0x01,
        }),
        "rst" | "reset" => Ok(FilterPredicate::TcpFlagsMask {
            mask: 0x04,
            expected: 0x04,
        }),
        "psh" | "push" => Ok(FilterPredicate::TcpFlagsMask {
            mask: 0x08,
            expected: 0x08,
        }),
        "urg" => Ok(FilterPredicate::TcpFlagsMask {
            mask: 0x20,
            expected: 0x20,
        }),
        "xmas" => Ok(FilterPredicate::TcpFlagsMask {
            mask: 0x29,
            expected: 0x29,
        }),
        "null" => Ok(FilterPredicate::TcpFlagsMask {
            mask: 0x3f,
            expected: 0x00,
        }),
        _ => Err(CaptureError::InvalidMetadata(
            "unsupported TCP flag token (expected syn, synack, ack, fin, rst, psh, urg, xmas, or null)",
        )),
    }
}

fn parse_threshold(val: &str) -> Result<(ThresholdOp, u64), CaptureError> {
    let (op, rest) = if let Some(r) = val.strip_prefix(">=") {
        (ThresholdOp::GreaterOrEqual, r)
    } else if let Some(r) = val.strip_prefix("<=") {
        (ThresholdOp::LessOrEqual, r)
    } else if let Some(r) = val.strip_prefix('>') {
        (ThresholdOp::GreaterThan, r)
    } else if let Some(r) = val.strip_prefix('<') {
        (ThresholdOp::LessThan, r)
    } else if let Some(r) = val.strip_prefix('=') {
        (ThresholdOp::Equal, r)
    } else {
        (ThresholdOp::GreaterOrEqual, val)
    };
    let number: u64 = rest
        .parse()
        .map_err(|_| CaptureError::InvalidMetadata("invalid numeric threshold"))?;
    Ok((op, number))
}

#[cfg(test)]
mod tests {
    use super::*;
    use argus_core::{Direction, Endpoint, FlowInstanceId, FlowKey, PacketEvent};

    use crate::{PacketObservationMetadata, PacketProtocolMetadata, QualityMarkers};

    fn sample_packet(
        src_ip: IpAddr,
        src_port: u16,
        dst_ip: IpAddr,
        dst_port: u16,
        protocol: TransportProtocol,
        tcp_flags: Option<u8>,
        sni: Option<&str>,
    ) -> CapturedPacket {
        CapturedPacket {
            capture_ordinal: 1,
            flow: FlowInstanceId {
                key: FlowKey {
                    initiator: Endpoint {
                        address: src_ip,
                        port: src_port,
                    },
                    responder: Endpoint {
                        address: dst_ip,
                        port: dst_port,
                    },
                    protocol,
                },
                generation: 1,
            },
            packet: PacketEvent {
                sequence: 0,
                monotonic_timestamp_ns: 1_000_000,
                captured_len: 512,
                original_len: 512,
                direction: Direction::InitiatorToResponder,
            },
            protocol: PacketProtocolMetadata {
                ip_version: if src_ip.is_ipv4() { 4 } else { 6 },
                tcp_flags,
            },
            observations: PacketObservationMetadata {
                source_mac: None,
                destination_mac: None,
                tls_sni: sni.map(str::to_owned),
                dns_qname: None,
                dns_message: false,
                dns_query: false,
                http_request: false,
                tor_port_match: false,
            },
            quality: QualityMarkers::default(),
        }
    }

    #[test]
    fn matches_cidr_ports_flags_sni_and_negation() {
        let pkt = sample_packet(
            IpAddr::V4(Ipv4Addr::new(192, 168, 1, 42)),
            51234,
            IpAddr::V4(Ipv4Addr::new(140, 82, 121, 4)),
            443,
            TransportProtocol::Tcp,
            Some(0x02),
            Some("api.github.com"),
        );

        let filter = PacketFilter::parse(
            "proto:tcp src:192.168.1.0/24 dst_port:400-500 flags:syn sni:github !port:22",
        )
        .expect("valid filter");
        assert!(filter.matches_packet(&pkt));

        let negated = PacketFilter::parse("proto:tcp !sni:github").expect("valid filter");
        assert!(!negated.matches_packet(&pkt));
    }

    #[test]
    fn compiles_network_and_port_predicates_to_bpf() {
        let filter =
            PacketFilter::parse("proto:tcp src:10.0.0.0/8 dst_port:443 !port:22").expect("parse");
        assert_eq!(
            filter.to_bpf_expression().expect("exact BPF"),
            "(tcp) and (src net 10.0.0.0/8) and (dst port 443) and not (port 22)"
        );
    }

    #[test]
    fn rejects_partial_bpf_instead_of_silently_broadening() {
        for query in [
            "proto:tcp sni:example.com",
            "!dns",
            "proc:browser",
            "bytes:>100",
        ] {
            let filter = PacketFilter::parse(query).expect("valid application filter");
            assert!(filter.to_bpf_expression().is_err(), "{query}");
        }
    }
}

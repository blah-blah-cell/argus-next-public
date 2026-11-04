//! Zero-mock, bounded application-layer protocol dissectors.
//!
//! Extracts essential security telemetry (DNS QNAME/RCODE, TLS SNI, and ICMP types)
//! from packet payload slices with strict length checks, zero recursion, and bounded memory.

/// Decoded DNS transaction summary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DnsSummary {
    pub is_response: bool,
    pub opcode: u8,
    pub rcode: u8,
    pub qname: Option<String>,
    pub qtype: Option<u16>,
}

/// Decoded ICMP message summary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IcmpSummary {
    pub icmp_type: u8,
    pub code: u8,
}

/// Decoded HTTP transaction summary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpSummary {
    pub is_response: bool,
    pub method: Option<String>,
    pub path: Option<String>,
    pub host: Option<String>,
    pub status_code: Option<u16>,
}

/// Checks whether a TCP/UDP port is a standard Tor port.
///
/// Ports flagged (inspired by `PcapXray` `tor_traffic_handle`):
/// - 9001: Default Tor relay `ORPort`
/// - 9030: Default Tor directory mirror `DirPort`
/// - 9050: Standard Tor SOCKS proxy
/// - 9051: Tor `ControlPort`
/// - 9150: Tor Browser bundle SOCKS proxy
#[must_use]
pub const fn is_known_tor_port(port: u16) -> bool {
    matches!(port, 9001 | 9030 | 9050 | 9051 | 9150)
}

/// Identifies if communication is likely Tor relay or proxy traffic based on ports.
#[must_use]
pub const fn is_tor_traffic(src_port: u16, dst_port: u16) -> bool {
    is_known_tor_port(src_port) || is_known_tor_port(dst_port)
}

impl IcmpSummary {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match (self.icmp_type, self.code) {
            (0, _) => "Echo Reply",
            (3, 0) => "Dest Unreachable (Net)",
            (3, 1) => "Dest Unreachable (Host)",
            (3, 3) => "Dest Unreachable (Port)",
            (3, _) => "Dest Unreachable",
            (8, _) => "Echo Request (Ping)",
            (11, 0) => "TTL Expired in Transit",
            (11, _) => "Time Exceeded",
            (12, _) => "Parameter Problem",
            _ => "ICMP Control",
        }
    }
}

/// Safely inspects UDP payload for DNS protocol headers and query questions.
#[must_use]
pub fn parse_dns_summary(payload: &[u8]) -> Option<DnsSummary> {
    if payload.len() < 12 {
        return None;
    }
    // DNS Header: ID(2), Flags(2), QDCOUNT(2), ANCOUNT(2), NSCOUNT(2), ARCOUNT(2)
    let flags = u16::from_be_bytes([payload[2], payload[3]]);
    let is_response = (flags & 0x8000) != 0;
    let opcode = ((flags >> 11) & 0x0f) as u8;
    let rcode = (flags & 0x000f) as u8;
    let qdcount = u16::from_be_bytes([payload[4], payload[5]]);

    let mut qname = None;
    let mut qtype = None;

    if qdcount > 0
        && payload.len() > 12
        && let Some((name, question_tail)) = parse_dns_name(payload, 12)
    {
        if !name.is_empty() {
            qname = Some(name);
        }
        if question_tail + 4 <= payload.len() {
            qtype = Some(u16::from_be_bytes([
                payload[question_tail],
                payload[question_tail + 1],
            ]));
        }
    }

    Some(DnsSummary {
        is_response,
        opcode,
        rcode,
        qname,
        qtype,
    })
}

fn parse_dns_name(payload: &[u8], start: usize) -> Option<(String, usize)> {
    let mut cursor = start;
    let mut question_tail = None;
    let mut labels = Vec::new();
    let mut total_chars = 0_usize;
    let mut jumps = 0_u8;

    loop {
        let length = *payload.get(cursor)? as usize;
        if length == 0 {
            cursor += 1;
            return Some((labels.join("."), question_tail.unwrap_or(cursor)));
        }
        if length & 0xc0 == 0xc0 {
            let low = usize::from(*payload.get(cursor + 1)?);
            let pointer = ((length & 0x3f) << 8) | low;
            if pointer >= payload.len() || jumps >= 10 {
                return None;
            }
            question_tail.get_or_insert(cursor + 2);
            cursor = pointer;
            jumps += 1;
            continue;
        }
        if length & 0xc0 != 0 || length > 63 {
            return None;
        }
        cursor += 1;
        let end = cursor.checked_add(length)?;
        if end > payload.len() || total_chars.saturating_add(length) > 253 {
            return None;
        }
        let label = std::str::from_utf8(&payload[cursor..end]).ok()?;
        labels.push(label);
        total_chars = total_chars.saturating_add(length + 1);
        cursor = end;
    }
}

/// Safely inspects TCP payload for TLS `ClientHello` records and extracts Server Name Indication (SNI).
#[must_use]
pub fn parse_tls_sni(payload: &[u8]) -> Option<String> {
    // Minimum TLS record + handshake header length
    if payload.len() < 5 + 4 + 2 + 32 {
        return None;
    }
    // ContentType == Handshake (0x16)
    if payload[0] != 0x16 {
        return None;
    }
    // Version major == 3 (SSL 3.0 / TLS 1.0..1.3)
    if payload[1] != 0x03 {
        return None;
    }
    let record_len = u16::from_be_bytes([payload[3], payload[4]]) as usize;
    let record_end = 5_usize.checked_add(record_len)?;
    if record_len == 0 || payload.len() < record_end {
        return None;
    }

    let mut cursor = 5;
    // Handshake Type == ClientHello (0x01)
    if payload[cursor] != 0x01 {
        return None;
    }
    let handshake_len = (usize::from(payload[cursor + 1]) << 16)
        | (usize::from(payload[cursor + 2]) << 8)
        | usize::from(payload[cursor + 3]);
    let handshake_end = cursor.checked_add(4)?.checked_add(handshake_len)?;
    if handshake_end > record_end {
        return None;
    }
    cursor += 4; // Skip type (1) + length (3)
    cursor += 2 + 32; // Skip client version (2) + random (32)

    if cursor >= payload.len() {
        return None;
    }
    // Session ID length
    let session_id_len = payload[cursor] as usize;
    cursor += 1 + session_id_len;

    if cursor + 2 > payload.len() {
        return None;
    }
    // Cipher Suites length
    let cipher_suites_len = u16::from_be_bytes([payload[cursor], payload[cursor + 1]]) as usize;
    cursor += 2 + cipher_suites_len;

    if cursor + 1 > payload.len() {
        return None;
    }
    // Compression Methods length
    let comp_methods_len = payload[cursor] as usize;
    cursor += 1 + comp_methods_len;

    if cursor + 2 > payload.len() {
        return None;
    }
    // Extensions length
    let extensions_len = u16::from_be_bytes([payload[cursor], payload[cursor + 1]]) as usize;
    cursor += 2;
    let extensions_end = cursor.checked_add(extensions_len)?;
    if extensions_end > handshake_end {
        return None;
    }

    while cursor + 4 <= extensions_end {
        let ext_type = u16::from_be_bytes([payload[cursor], payload[cursor + 1]]);
        let ext_len = u16::from_be_bytes([payload[cursor + 2], payload[cursor + 3]]) as usize;
        cursor += 4;

        if ext_type == 0x0000 {
            // Server Name Indication extension
            if cursor + 2 > extensions_end {
                return None;
            }
            let server_name_list_len =
                usize::from(u16::from_be_bytes([payload[cursor], payload[cursor + 1]]));
            if cursor + 2 + server_name_list_len > cursor + ext_len {
                return None;
            }
            let name_type = payload.get(cursor + 2)?;
            if *name_type != 0x00 {
                // host_name type
                return None;
            }
            let name_len =
                u16::from_be_bytes([*payload.get(cursor + 3)?, *payload.get(cursor + 4)?]) as usize;
            let start = cursor + 5;
            let end = start + name_len;
            if end <= cursor + ext_len && end <= extensions_end {
                let name_bytes = &payload[start..end];
                if let Ok(hostname) = std::str::from_utf8(name_bytes)
                    && valid_tls_hostname(hostname)
                {
                    return Some(hostname.to_owned());
                }
            }
            return None;
        }

        cursor += ext_len;
    }

    None
}

fn valid_tls_hostname(hostname: &str) -> bool {
    !hostname.is_empty()
        && hostname.len() <= 253
        && hostname.split('.').all(|label| {
            !label.is_empty()
                && label.len() <= 63
                && !label.starts_with('-')
                && !label.ends_with('-')
                && label
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        })
}

/// Safely inspects ICMP packet payload.
#[must_use]
pub fn parse_icmp_summary(payload: &[u8]) -> Option<IcmpSummary> {
    if payload.len() < 2 {
        return None;
    }
    Some(IcmpSummary {
        icmp_type: payload[0],
        code: payload[1],
    })
}

/// Safely inspects TCP payload for HTTP/1.x headers.
#[must_use]
pub fn parse_http_summary(payload: &[u8]) -> Option<HttpSummary> {
    if payload.len() < 10 {
        return None;
    }

    let inspect_len = payload.len().min(1024);
    let slice = &payload[..inspect_len];
    let text = std::str::from_utf8(slice).ok()?;

    let mut lines = text.split("\r\n");
    let first_line = lines.next()?;

    if let Some(rest) = first_line.strip_prefix("HTTP/1.") {
        let mut parts = rest.split_whitespace();
        let _minor_version = parts.next();
        let status_code = parts.next().and_then(|c| c.parse::<u16>().ok());
        return Some(HttpSummary {
            is_response: true,
            method: None,
            path: None,
            host: None,
            status_code,
        });
    }

    let methods = [
        "GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH", "CONNECT",
    ];
    for method in methods {
        if first_line.starts_with(method) && first_line.len() > method.len() + 1 {
            let mut parts = first_line.split_whitespace();
            let req_method = parts.next()?;
            let path = parts.next().map(str::to_owned);

            let mut host = None;
            for line in lines {
                if line.is_empty() {
                    break;
                }
                if let Some(h) = line
                    .strip_prefix("Host: ")
                    .or_else(|| line.strip_prefix("host: "))
                {
                    host = Some(h.trim().to_owned());
                    break;
                }
            }

            return Some(HttpSummary {
                is_response: false,
                method: Some(req_method.to_owned()),
                path,
                host,
                status_code: None,
            });
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_http_request_and_response() {
        let req = b"GET /api/v1/health HTTP/1.1\r\nHost: argus.local\r\nUser-Agent: test\r\n\r\n";
        let summary = parse_http_summary(req).expect("http request parsed");
        assert!(!summary.is_response);
        assert_eq!(summary.method.as_deref(), Some("GET"));
        assert_eq!(summary.path.as_deref(), Some("/api/v1/health"));
        assert_eq!(summary.host.as_deref(), Some("argus.local"));

        let resp = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nOK";
        let resp_summary = parse_http_summary(resp).expect("http response parsed");
        assert!(resp_summary.is_response);
        assert_eq!(resp_summary.status_code, Some(200));
    }

    #[test]
    fn detects_tor_ports_and_traffic() {
        assert!(is_known_tor_port(9001));
        assert!(is_known_tor_port(9050));
        assert!(is_known_tor_port(9150));
        assert!(!is_known_tor_port(80));
        assert!(!is_known_tor_port(443));

        assert!(is_tor_traffic(54321, 9050));
        assert!(is_tor_traffic(9001, 12345));
        assert!(!is_tor_traffic(80, 443));
    }

    #[test]
    fn parses_dns_standard_query() {
        // DNS query for example.com (type A)
        let dns_pkt = [
            0x12, 0x34, // ID
            0x01, 0x00, // Flags: Standard query
            0x00, 0x01, // QDCOUNT: 1
            0x00, 0x00, // ANCOUNT: 0
            0x00, 0x00, // NSCOUNT: 0
            0x00, 0x00, // ARCOUNT: 0
            // QNAME: 7'example' 3'com' 0
            0x07, b'e', b'x', b'a', b'm', b'p', b'l', b'e', 0x03, b'c', b'o', b'm', 0x00, 0x00,
            0x01, // QTYPE: A (1)
            0x00, 0x01, // QCLASS: IN (1)
        ];

        let summary = parse_dns_summary(&dns_pkt).expect("dns parsed");
        assert!(!summary.is_response);
        assert_eq!(summary.opcode, 0);
        assert_eq!(summary.rcode, 0);
        assert_eq!(summary.qname.as_deref(), Some("example.com"));
        assert_eq!(summary.qtype, Some(1));
    }

    #[test]
    fn compressed_question_keeps_qtype_at_original_tail() {
        let dns = [
            0x12, 0x34, 0x01, 0x00, 0x00, 0x01, 0, 0, 0, 0, 0, 0, 0xc0, 0x12, 0x00, 0x01, 0x00,
            0x01, 0x07, b'e', b'x', b'a', b'm', b'p', b'l', b'e', 0x03, b'c', b'o', b'm', 0,
        ];
        let summary = parse_dns_summary(&dns).expect("compressed query parsed");
        assert_eq!(summary.qname.as_deref(), Some("example.com"));
        assert_eq!(summary.qtype, Some(1));
    }

    #[test]
    fn rejects_truncated_declared_tls_record() {
        let mut payload = vec![0x16, 0x03, 0x03, 0x10, 0x00];
        payload.extend_from_slice(&[0; 64]);
        assert_eq!(parse_tls_sni(&payload), None);
    }

    #[test]
    fn parses_tls_client_hello_sni() {
        // Minimal TLS 1.2 ClientHello with SNI "api.github.com"
        let host = b"api.github.com";
        let host_len = u16::try_from(host.len()).expect("host length fits u16");
        let mut pkt = Vec::new();
        // Record Header (5 bytes)
        pkt.extend_from_slice(&[0x16, 0x03, 0x01, 0x00, 0x00]); // Length updated later
        // Handshake Header
        pkt.push(0x01); // ClientHello
        pkt.extend_from_slice(&[0x00, 0x00, 0x00]); // Length
        pkt.extend_from_slice(&[0x03, 0x03]); // TLS 1.2
        pkt.extend_from_slice(&[0xaa; 32]); // Random
        pkt.push(0x00); // Session ID len = 0
        pkt.extend_from_slice(&[0x00, 0x02, 0x13, 0x01]); // Ciphers (2 bytes len + 1 cipher)
        pkt.extend_from_slice(&[0x01, 0x00]); // Compressions (1 byte len + null comp)

        // Extensions
        let ext_data_len = 2 + 1 + 2 + host.len();
        let extensions_total_len = 4 + ext_data_len;
        pkt.extend_from_slice(
            &u16::try_from(extensions_total_len)
                .expect("extensions fit u16")
                .to_be_bytes(),
        );
        // Extension: SNI (0x0000)
        pkt.extend_from_slice(&[0x00, 0x00]);
        pkt.extend_from_slice(
            &u16::try_from(ext_data_len)
                .expect("extension data fits u16")
                .to_be_bytes(),
        );
        pkt.extend_from_slice(
            &u16::try_from(1 + 2 + host.len())
                .expect("SNI name fits u16")
                .to_be_bytes(),
        );
        pkt.push(0x00); // NameType: host_name
        pkt.extend_from_slice(&host_len.to_be_bytes());
        pkt.extend_from_slice(host);

        let handshake_body = u32::try_from(pkt.len() - 9).expect("handshake body fits u32");
        let handshake_bytes = handshake_body.to_be_bytes();
        pkt[6..9].copy_from_slice(&handshake_bytes[1..4]);
        // Fix record length
        let total_body = u16::try_from(pkt.len() - 5).expect("handshake body fits u16");
        pkt[3..5].copy_from_slice(&total_body.to_be_bytes());

        let sni = parse_tls_sni(&pkt).expect("sni parsed");
        assert_eq!(sni, "api.github.com");
    }

    #[test]
    fn parses_icmp_types() {
        let ping_req = [0x08, 0x00, 0x12, 0x34];
        let summary = parse_icmp_summary(&ping_req).expect("icmp parsed");
        assert_eq!(summary.icmp_type, 8);
        assert_eq!(summary.code, 0);
        assert_eq!(summary.name(), "Echo Request (Ping)");
    }
}

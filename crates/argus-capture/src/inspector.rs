//! Raw packet payload inspector and layer demarcation engine.
//!
//! Parses raw frames to extract structured Hex + ASCII representations,
//! exact layer boundaries (Ethernet, IPv4/IPv6, TCP/UDP/ICMP, Payload),
//! and byte-level Shannon entropy.

use serde::{Deserialize, Serialize};

/// Detailed layer segmentation for one captured packet.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PacketInspection {
    pub ordinal: usize,
    pub timestamp_ns: u64,
    pub captured_length: usize,
    pub original_length: usize,
    pub ethernet_range: (usize, usize),
    pub ip_range: Option<(usize, usize)>,
    pub transport_range: Option<(usize, usize)>,
    pub payload_range: Option<(usize, usize)>,
    pub entropy: f64,
    pub summary: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub hex_bytes: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ascii_chars: Vec<String>,
    pub hex_dump_formatted: String,
}

/// Inspects a raw packet frame and breaks it down into constituent protocol layers.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn inspect_frame(
    raw_frame: &[u8],
    ordinal: usize,
    timestamp_ns: u64,
    original_len: usize,
) -> PacketInspection {
    let captured_len = raw_frame.len();
    let entropy = calculate_slice_entropy(raw_frame);

    let mut ethernet_end = 14.min(captured_len);
    let mut ip_start = None;
    let mut ip_end = None;
    let mut transport_start = None;
    let mut transport_end = None;
    let mut payload_start = None;
    let mut summary = String::from("Ethernet frame");

    // Check Ethernet Header
    if captured_len >= 14 {
        let mut ethertype = u16::from_be_bytes([raw_frame[12], raw_frame[13]]);
        let mut cursor = 14;

        // Skip 802.1Q VLAN tags if present
        while (ethertype == 0x8100 || ethertype == 0x88a8) && cursor + 4 <= captured_len {
            ethertype = u16::from_be_bytes([raw_frame[cursor + 2], raw_frame[cursor + 3]]);
            cursor += 4;
            ethernet_end = cursor;
        }

        if ethertype == 0x0800 && cursor + 20 <= captured_len {
            // IPv4
            let ihl = (raw_frame[cursor] & 0x0f) as usize * 4;
            let _ip_total_len =
                u16::from_be_bytes([raw_frame[cursor + 2], raw_frame[cursor + 3]]) as usize;
            let protocol = raw_frame[cursor + 9];

            ip_start = Some(cursor);
            let actual_ip_end = (cursor + ihl).min(captured_len);
            ip_end = Some(actual_ip_end);
            cursor = actual_ip_end;

            match protocol {
                6 => {
                    // TCP
                    if cursor + 20 <= captured_len {
                        let data_offset = ((raw_frame[cursor + 12] >> 4) as usize) * 4;
                        transport_start = Some(cursor);
                        let actual_tcp_end = (cursor + data_offset).min(captured_len);
                        transport_end = Some(actual_tcp_end);
                        cursor = actual_tcp_end;

                        let src_port = u16::from_be_bytes([
                            raw_frame[cursor - data_offset],
                            raw_frame[cursor - data_offset + 1],
                        ]);
                        let dst_port = u16::from_be_bytes([
                            raw_frame[cursor - data_offset + 2],
                            raw_frame[cursor - data_offset + 3],
                        ]);
                        summary = format!("TCP {src_port} → {dst_port}");

                        if cursor < captured_len {
                            payload_start = Some(cursor);
                        }
                    }
                }
                17 => {
                    // UDP
                    if cursor + 8 <= captured_len {
                        let src_port =
                            u16::from_be_bytes([raw_frame[cursor], raw_frame[cursor + 1]]);
                        let dst_port =
                            u16::from_be_bytes([raw_frame[cursor + 2], raw_frame[cursor + 3]]);
                        transport_start = Some(cursor);
                        transport_end = Some(cursor + 8);
                        cursor += 8;
                        summary = format!("UDP {src_port} → {dst_port}");

                        if cursor < captured_len {
                            payload_start = Some(cursor);
                        }
                    }
                }
                1 => {
                    // ICMP
                    if cursor + 4 <= captured_len {
                        transport_start = Some(cursor);
                        transport_end = Some(cursor + 4);
                        cursor += 4;
                        summary = format!("ICMP type {}", raw_frame[cursor - 4]);
                        if cursor < captured_len {
                            payload_start = Some(cursor);
                        }
                    }
                }
                _ => {
                    summary = format!("IPv4 Protocol {protocol}");
                }
            }
        } else if ethertype == 0x86dd && cursor + 40 <= captured_len {
            // IPv6
            let next_header = raw_frame[cursor + 6];
            ip_start = Some(cursor);
            cursor += 40;
            ip_end = Some(cursor);

            if next_header == 6 && cursor + 20 <= captured_len {
                // TCP
                let data_offset = ((raw_frame[cursor + 12] >> 4) as usize) * 4;
                transport_start = Some(cursor);
                cursor = (cursor + data_offset).min(captured_len);
                transport_end = Some(cursor);
                "IPv6 TCP".clone_into(&mut summary);
                if cursor < captured_len {
                    payload_start = Some(cursor);
                }
            } else if next_header == 17 && cursor + 8 <= captured_len {
                // UDP
                transport_start = Some(cursor);
                cursor += 8;
                transport_end = Some(cursor);
                "IPv6 UDP".clone_into(&mut summary);
                if cursor < captured_len {
                    payload_start = Some(cursor);
                }
            }
        }
    }

    // The frontend exclusively reads hex_dump_formatted; avoid allocating
    // hundreds of thousands of ephemeral Strings for unread hex_bytes/ascii_chars.
    let hex_bytes = Vec::new();
    let ascii_chars = Vec::new();
    let hex_dump_formatted = format_hexdump(raw_frame);

    PacketInspection {
        ordinal,
        timestamp_ns,
        captured_length: captured_len,
        original_length: original_len,
        ethernet_range: (0, ethernet_end),
        ip_range: ip_start.zip(ip_end),
        transport_range: transport_start.zip(transport_end),
        payload_range: payload_start.map(|s| (s, captured_len)),
        entropy,
        summary,
        hex_bytes,
        ascii_chars,
        hex_dump_formatted,
    }
}

/// Formats byte slice into classic 16-bytes-per-line hexdump text.
#[must_use]
pub fn format_hexdump(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    for (chunk_idx, chunk) in bytes.chunks(16).enumerate() {
        let offset = chunk_idx * 16;
        let _ = write!(out, "{offset:04x}: ");

        // Hex section
        for (i, byte) in chunk.iter().enumerate() {
            let _ = write!(out, "{byte:02x} ");
            if i == 7 {
                out.push(' ');
            }
        }

        // Padding for incomplete lines
        if chunk.len() < 16 {
            let missing = 16 - chunk.len();
            for i in 0..missing {
                out.push_str("   ");
                if chunk.len() + i == 7 {
                    out.push(' ');
                }
            }
        }

        out.push_str(" |");
        // ASCII section
        for &byte in chunk {
            if byte.is_ascii_graphic() || byte == b' ' {
                out.push(byte as char);
            } else {
                out.push('.');
            }
        }
        out.push_str("|\n");
    }
    out
}

// Byte counts are bounded by the captured frame length, so converting them to
// `f64` for the Shannon ratio cannot lose precision in practice.
#[allow(clippy::cast_precision_loss)]
fn calculate_slice_entropy(bytes: &[u8]) -> f64 {
    if bytes.is_empty() {
        return 0.0;
    }
    let mut freq = [0u64; 256];
    for &b in bytes {
        freq[b as usize] += 1;
    }
    let total = bytes.len() as f64;
    let mut entropy = 0.0;
    for &count in &freq {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }
    entropy.clamp(0.0, 8.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inspects_ipv4_tcp_frame_correctly() {
        // Minimal Ethernet + IPv4 + TCP packet
        let mut frame = Vec::new();
        // Ethernet (14 bytes)
        frame.extend_from_slice(&[0xff; 6]);
        frame.extend_from_slice(&[0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        frame.extend_from_slice(&[0x08, 0x00]); // IPv4
        // IPv4 (20 bytes)
        frame.push(0x45); // IPv4, IHL=5 (20 bytes)
        frame.push(0x00);
        frame.extend_from_slice(&[0x00, 0x3c]); // Total len = 60
        frame.extend_from_slice(&[0x00, 0x01, 0x00, 0x00]);
        frame.push(64); // TTL
        frame.push(6); // TCP
        frame.extend_from_slice(&[0x00, 0x00]); // Checksum
        frame.extend_from_slice(&[192, 168, 1, 10]); // Src IP
        frame.extend_from_slice(&[192, 168, 1, 1]); // Dst IP
        // TCP (20 bytes)
        frame.extend_from_slice(&[0x1f, 0x90]); // Src port 8080
        frame.extend_from_slice(&[0x01, 0xbb]); // Dst port 443
        frame.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]); // Seq
        frame.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Ack
        frame.push(0x50); // Data offset = 5 (20 bytes)
        frame.push(0x02); // SYN
        frame.extend_from_slice(&[0xfa, 0xf0]); // Window
        frame.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        // Payload (6 bytes)
        frame.extend_from_slice(b"ARGUS!");

        let inspection = inspect_frame(&frame, 1, 1000, frame.len());
        assert_eq!(inspection.ethernet_range, (0, 14));
        assert_eq!(inspection.ip_range, Some((14, 34)));
        assert_eq!(inspection.transport_range, Some((34, 54)));
        assert_eq!(inspection.payload_range, Some((54, 60)));
        assert_eq!(inspection.summary, "TCP 8080 → 443");
        assert!(inspection.entropy > 0.0);
        assert!(inspection.hex_dump_formatted.contains("ARGUS!"));
    }
}

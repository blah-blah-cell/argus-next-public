use std::{
    collections::{HashMap, VecDeque},
    fs,
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    num::NonZeroUsize,
    path::Path,
};

use argus_core::{
    Direction, Endpoint, FlowGenerationTracker, FlowInstanceId, FlowKey, PacketEvent,
    TransportProtocol,
};

use crate::{
    AdapterOutcome, CaptureCounters, CaptureError, CapturePoll, CaptureSessionMetadata,
    CaptureSource, CapturedPacket, LinkType, NormalizationStatus, PacketObservationMetadata,
    PacketProtocolMetadata, QualityMarkers, RawCapturePoll, RawCaptureSource, RawPacket,
    RawPacketClassification, RawPacketQuality, RawTimestamp, ReplaySource, SourceKind,
    SourceMetadata, UnsupportedReason, is_tor_traffic, parse_dns_summary, parse_http_summary,
    parse_tls_sni,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PcapLimits {
    pub max_file_bytes: NonZeroUsize,
    pub max_records: NonZeroUsize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PcapLoadReport {
    pub records_seen: usize,
    pub packets_emitted: usize,
    pub unsupported_link_packets: usize,
    pub unsupported_network_packets: usize,
    /// Aggregate retained for consumers that do not distinguish IP versions.
    pub fragmented_packets: usize,
    /// IPv4 packets skipped because fragment reassembly is intentionally unsupported.
    pub unsupported_ipv4_fragment_packets: usize,
    /// IPv6 packets skipped because fragment reassembly is intentionally unsupported.
    pub unsupported_ipv6_fragment_packets: usize,
}

/// Strict PCAP/PCAPNG reader for bounded deterministic replay.
#[derive(Clone, Debug)]
pub struct PcapReplaySource {
    replay: ReplaySource,
    raw_records: VecDeque<RawPacket>,
    session: CaptureSessionMetadata,
    raw_stopped: bool,
    report: PcapLoadReport,
}

impl PcapReplaySource {
    /// Loads a bounded capture and normalizes supported Ethernet IPv4/IPv6 TCP/UDP packets.
    ///
    /// # Errors
    ///
    /// Returns a hard [`CaptureError`] for I/O, invalid/truncated PCAP structures, unsupported
    /// file/link formats, limit violations, or invalid normalized records. Non-IP and unsupported
    /// IP packets are counted in [`PcapLoadReport`] instead of being silently represented.
    pub fn from_path(path: &Path, limits: PcapLimits) -> Result<Self, CaptureError> {
        let metadata = fs::metadata(path)
            .map_err(|error| CaptureError::SourceFailure(format!("PCAP metadata: {error}")))?;
        if metadata.len() > limits.max_file_bytes.get() as u64 {
            return Err(CaptureError::SourceFailure(format!(
                "PCAP is {} bytes, exceeding configured limit {}",
                metadata.len(),
                limits.max_file_bytes
            )));
        }
        let bytes = fs::read(path)
            .map_err(|error| CaptureError::SourceFailure(format!("PCAP read: {error}")))?;
        let source_id = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("pcap-replay");
        Self::from_bytes(source_id, &bytes, limits)
    }

    /// Parses bounded PCAP bytes, primarily for deterministic tests.
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError`] under the same conditions as [`Self::from_path`].
    pub fn from_bytes(
        source_id: &str,
        bytes: &[u8],
        limits: PcapLimits,
    ) -> Result<Self, CaptureError> {
        if bytes.len() > limits.max_file_bytes.get() {
            return Err(CaptureError::SourceFailure(format!(
                "PCAP is {} bytes, exceeding configured limit {}",
                bytes.len(),
                limits.max_file_bytes
            )));
        }
        let converted;
        let raw_timestamps;
        let (parse_bytes, timestamps) = if bytes.starts_with(&[0x0a, 0x0d, 0x0d, 0x0a]) {
            let conversion = pcapng_to_classic(bytes, limits)?;
            converted = conversion.bytes;
            raw_timestamps = conversion.raw_timestamps;
            (converted.as_slice(), Some(raw_timestamps))
        } else {
            (bytes, None)
        };
        let parsed = Parser::new(parse_bytes, limits.max_records.get(), timestamps)?.parse()?;
        let replay_capacity = NonZeroUsize::new(parsed.packets.len().max(1))
            .ok_or_else(|| CaptureError::SourceFailure("invalid replay capacity".to_owned()))?;
        let metadata = SourceMetadata::new(
            source_id,
            SourceKind::Replay,
            LinkType::Ethernet,
            parsed.snapshot_length,
        )?;
        let session = CaptureSessionMetadata::fresh("pcap-replay")?;
        let raw_records = parsed
            .raw_records
            .into_iter()
            .map(|record| record.with_metadata(metadata.clone(), session.clone()))
            .collect();
        let replay = ReplaySource::new_with_session(
            metadata,
            session.clone(),
            parsed.packets,
            replay_capacity,
        )?;
        Ok(Self {
            replay,
            raw_records,
            session,
            raw_stopped: false,
            report: parsed.report,
        })
    }

    /// Polls one raw record together with its normalized packet, when supported.
    ///
    /// This preserves the exact source-record-to-flow relationship needed by
    /// forensic consumers. Call it only on a fresh source; mixing it with
    /// [`CaptureSource::poll_next`] or [`RawCaptureSource::poll_next_raw`] would
    /// consume one side of the paired stream independently.
    ///
    /// # Errors
    ///
    /// Returns a capture error if a raw record marked as normalized has no
    /// corresponding normalized packet.
    pub fn poll_next_paired(&mut self) -> Result<Option<AdapterOutcome>, CaptureError> {
        let Some(raw) = self.raw_records.pop_front() else {
            return Ok(None);
        };
        if raw.normalization == NormalizationStatus::Normalized {
            match self.replay.poll_next()? {
                CapturePoll::Packet(packet) => Ok(Some(AdapterOutcome::Normalized { raw, packet })),
                CapturePoll::EndOfStream | CapturePoll::Stopped => {
                    Err(CaptureError::SourceFailure(
                        "normalized PCAP stream ended before its raw record stream".to_owned(),
                    ))
                }
            }
        } else {
            Ok(Some(AdapterOutcome::Unnormalized(raw)))
        }
    }

    #[must_use]
    pub const fn report(&self) -> PcapLoadReport {
        self.report
    }
}

// Block validation is kept in one pass so offsets cannot escape their section bounds.
struct PcapngConversion {
    bytes: Vec<u8>,
    raw_timestamps: Vec<RawTimestamp>,
}

#[allow(clippy::too_many_lines)]
fn pcapng_to_classic(bytes: &[u8], limits: PcapLimits) -> Result<PcapngConversion, CaptureError> {
    if bytes.len() < 28 {
        return Err(pcap_error("PCAPNG section header is truncated"));
    }
    let order = match bytes[8..12] {
        [0x4d, 0x3c, 0x2b, 0x1a] => ByteOrder::Little,
        [0x1a, 0x2b, 0x3c, 0x4d] => ByteOrder::Big,
        _ => return Err(pcap_error("PCAPNG byte-order magic is invalid")),
    };
    let mut output = Vec::with_capacity(bytes.len());
    output.extend_from_slice(&[0x4d, 0x3c, 0xb2, 0xa1]);
    output.extend_from_slice(&2_u16.to_le_bytes());
    output.extend_from_slice(&4_u16.to_le_bytes());
    output.extend_from_slice(&0_i32.to_le_bytes());
    output.extend_from_slice(&0_u32.to_le_bytes());
    output.extend_from_slice(&65_535_u32.to_le_bytes());
    output.extend_from_slice(&1_u32.to_le_bytes());

    let mut offset = 0_usize;
    let mut interfaces = Vec::<PcapngInterface>::new();
    let mut raw_timestamps = Vec::new();
    let mut records = 0_usize;
    while offset < bytes.len() {
        if bytes.len() - offset < 12 {
            return Err(pcap_error("PCAPNG block header is truncated"));
        }
        let block_type = if offset == 0 {
            0x0a0d_0d0a
        } else {
            order.u32(&bytes[offset..offset + 4])
        };
        let total_length = order.u32(&bytes[offset + 4..offset + 8]) as usize;
        if total_length < 12 || !total_length.is_multiple_of(4) {
            return Err(pcap_error("PCAPNG block length is invalid"));
        }
        let end = offset
            .checked_add(total_length)
            .ok_or_else(|| pcap_error("PCAPNG block offset overflow"))?;
        if end > bytes.len() {
            return Err(pcap_error("PCAPNG block is truncated"));
        }
        if order.u32(&bytes[end - 4..end]) as usize != total_length {
            return Err(pcap_error("PCAPNG block trailing length does not match"));
        }
        match block_type {
            0x0a0d_0d0a => {
                if offset != 0 {
                    return Err(pcap_error("multiple PCAPNG sections are not supported"));
                }
                if total_length < 28
                    || order.u16(&bytes[offset + 12..offset + 14]) != 1
                    || order.u16(&bytes[offset + 14..offset + 16]) != 0
                {
                    return Err(pcap_error("unsupported PCAPNG section version"));
                }
            }
            1 => {
                if total_length < 20 {
                    return Err(pcap_error("PCAPNG interface block is truncated"));
                }
                if order.u16(&bytes[offset + 8..offset + 10]) != 1 {
                    return Err(pcap_error("only Ethernet PCAPNG interfaces are supported"));
                }
                let snapshot_length = order.u32(&bytes[offset + 12..offset + 16]);
                let timestamp_exponent =
                    parse_timestamp_resolution(&bytes[offset + 16..end - 4], order)?;
                interfaces.push(PcapngInterface {
                    snapshot_length,
                    timestamp_exponent,
                });
            }
            6 => {
                if records == limits.max_records.get() {
                    return Err(pcap_error("record count exceeds configured limit"));
                }
                if total_length < 32 {
                    return Err(pcap_error("PCAPNG enhanced packet block is truncated"));
                }
                let interface_index = order.u32(&bytes[offset + 8..offset + 12]) as usize;
                let interface = interfaces
                    .get(interface_index)
                    .ok_or_else(|| pcap_error("PCAPNG packet references an unknown interface"))?;
                let timestamp = (u64::from(order.u32(&bytes[offset + 12..offset + 16])) << 32)
                    | u64::from(order.u32(&bytes[offset + 16..offset + 20]));
                let ticks_per_second = 10_u64
                    .checked_pow(u32::from(interface.timestamp_exponent))
                    .and_then(std::num::NonZeroU64::new)
                    .ok_or_else(|| pcap_error("PCAPNG timestamp resolution exceeds u64"))?;
                let captured_length = order.u32(&bytes[offset + 20..offset + 24]);
                let original_length = order.u32(&bytes[offset + 24..offset + 28]);
                if captured_length == 0
                    || captured_length > original_length
                    // PCAPNG permits zero snaplen to mean that no capture
                    // length limit is specified. Pktmon emits that form.
                    || (interface.snapshot_length != 0
                        && captured_length > interface.snapshot_length)
                {
                    return Err(pcap_error("invalid PCAPNG packet lengths"));
                }
                let padded_length = (captured_length as usize)
                    .checked_add(3)
                    .map(|length| length & !3)
                    .ok_or_else(|| pcap_error("PCAPNG packet padding overflow"))?;
                let packet_end = (offset + 28)
                    .checked_add(padded_length)
                    .ok_or_else(|| pcap_error("PCAPNG packet offset overflow"))?;
                if packet_end > end - 4 {
                    return Err(pcap_error("PCAPNG packet data is truncated"));
                }
                let timestamp_ns =
                    timestamp_to_nanoseconds(timestamp, interface.timestamp_exponent)?;
                let seconds = u32::try_from(timestamp_ns / 1_000_000_000).map_err(|_| {
                    pcap_error("PCAPNG timestamp seconds exceed classic PCAP range")
                })?;
                let nanoseconds = u32::try_from(timestamp_ns % 1_000_000_000)
                    .map_err(|_| pcap_error("PCAPNG timestamp fraction is invalid"))?;
                output.extend_from_slice(&seconds.to_le_bytes());
                output.extend_from_slice(&nanoseconds.to_le_bytes());
                output.extend_from_slice(&captured_length.to_le_bytes());
                output.extend_from_slice(&original_length.to_le_bytes());
                output
                    .extend_from_slice(&bytes[offset + 28..offset + 28 + captured_length as usize]);
                raw_timestamps.push(RawTimestamp {
                    ticks: timestamp,
                    ticks_per_second,
                });
                records += 1;
            }
            3 => {
                return Err(pcap_error(
                    "PCAPNG simple packet blocks lack timestamps and are not supported",
                ));
            }
            _ => {}
        }
        offset = end;
    }
    if interfaces.is_empty() {
        return Err(pcap_error("PCAPNG contains no interface description"));
    }
    if output.len() > limits.max_file_bytes.get() {
        return Err(pcap_error("converted PCAPNG exceeds configured file limit"));
    }
    Ok(PcapngConversion {
        bytes: output,
        raw_timestamps,
    })
}

#[derive(Clone, Copy)]
struct PcapngInterface {
    snapshot_length: u32,
    timestamp_exponent: u8,
}

fn parse_timestamp_resolution(options: &[u8], order: ByteOrder) -> Result<u8, CaptureError> {
    let mut offset = 0_usize;
    let mut exponent = 6_u8;
    while offset < options.len() {
        if options.len() - offset < 4 {
            return Err(pcap_error("PCAPNG interface option is truncated"));
        }
        let code = order.u16(&options[offset..offset + 2]);
        let length = order.u16(&options[offset + 2..offset + 4]) as usize;
        offset += 4;
        if code == 0 {
            if length != 0 {
                return Err(pcap_error("PCAPNG end option has a nonzero length"));
            }
            break;
        }
        let padded = length
            .checked_add(3)
            .map(|value| value & !3)
            .ok_or_else(|| pcap_error("PCAPNG option length overflow"))?;
        if offset + padded > options.len() {
            return Err(pcap_error("PCAPNG interface option value is truncated"));
        }
        if code == 9 {
            if length != 1 {
                return Err(pcap_error("PCAPNG timestamp resolution length is invalid"));
            }
            let value = options[offset];
            if value & 0x80 != 0 {
                return Err(pcap_error(
                    "binary PCAPNG timestamp resolution is unsupported",
                ));
            }
            exponent = value;
        }
        offset += padded;
    }
    Ok(exponent)
}

fn timestamp_to_nanoseconds(timestamp: u64, exponent: u8) -> Result<u64, CaptureError> {
    if exponent <= 9 {
        let multiplier = 10_u64
            .checked_pow(u32::from(9 - exponent))
            .ok_or_else(|| pcap_error("PCAPNG timestamp multiplier overflow"))?;
        timestamp
            .checked_mul(multiplier)
            .ok_or_else(|| pcap_error("PCAPNG timestamp overflow"))
    } else {
        let divisor = 10_u64
            .checked_pow(u32::from(exponent - 9))
            .ok_or_else(|| pcap_error("PCAPNG timestamp resolution is too fine"))?;
        Ok(timestamp / divisor)
    }
}

impl CaptureSource for PcapReplaySource {
    fn metadata(&self) -> &SourceMetadata {
        self.replay.metadata()
    }

    fn session_metadata(&self) -> &CaptureSessionMetadata {
        &self.session
    }

    fn counters(&self) -> CaptureCounters {
        let replay = self.replay.counters();
        CaptureCounters {
            source_records: u64::try_from(self.report.records_seen).unwrap_or(u64::MAX),
            normalized_packets: replay.normalized_packets,
            non_ip_records: u64::try_from(self.report.unsupported_link_packets).unwrap_or(u64::MAX),
            unsupported_records: u64::try_from(
                self.report
                    .unsupported_network_packets
                    .saturating_add(self.report.fragmented_packets),
            )
            .unwrap_or(u64::MAX),
            malformed_records: 0,
            packets_lost: replay.packets_lost,
            truncated_packets: replay.truncated_packets,
            reordered_packets: replay.reordered_packets,
            clock_rollbacks: replay.clock_rollbacks,
        }
    }

    fn poll_next(&mut self) -> Result<CapturePoll, CaptureError> {
        self.replay.poll_next()
    }

    fn stop(&mut self) -> Result<(), CaptureError> {
        self.replay.stop()
    }
}

impl RawCaptureSource for PcapReplaySource {
    fn metadata(&self) -> &SourceMetadata {
        self.replay.metadata()
    }

    fn session_metadata(&self) -> &CaptureSessionMetadata {
        &self.session
    }

    fn poll_next_raw(&mut self) -> Result<RawCapturePoll, CaptureError> {
        if self.raw_stopped {
            return Ok(RawCapturePoll::Stopped);
        }
        Ok(self
            .raw_records
            .pop_front()
            .map_or(RawCapturePoll::EndOfStream, RawCapturePoll::Packet))
    }

    fn stop_raw(&mut self) -> Result<(), CaptureError> {
        self.raw_stopped = true;
        self.raw_records.clear();
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum ByteOrder {
    Little,
    Big,
}

impl ByteOrder {
    fn u16(self, bytes: &[u8]) -> u16 {
        let bytes = [bytes[0], bytes[1]];
        match self {
            Self::Little => u16::from_le_bytes(bytes),
            Self::Big => u16::from_be_bytes(bytes),
        }
    }

    fn u32(self, bytes: &[u8]) -> u32 {
        let bytes = [bytes[0], bytes[1], bytes[2], bytes[3]];
        match self {
            Self::Little => u32::from_le_bytes(bytes),
            Self::Big => u32::from_be_bytes(bytes),
        }
    }
}

struct ParsedPcap {
    snapshot_length: NonZeroUsize,
    packets: Vec<CapturedPacket>,
    raw_records: Vec<RawRecordData>,
    report: PcapLoadReport,
}

#[derive(Clone, Debug)]
struct RawRecordData {
    source_ordinal: u64,
    raw_timestamp: RawTimestamp,
    monotonic_timestamp_ns: u64,
    captured_len: u32,
    original_len: u32,
    bytes: Vec<u8>,
    quality: RawPacketQuality,
    classification: RawPacketClassification,
    normalization: NormalizationStatus,
}

impl RawRecordData {
    fn with_metadata(self, source: SourceMetadata, session: CaptureSessionMetadata) -> RawPacket {
        RawPacket {
            source,
            session,
            source_ordinal: self.source_ordinal,
            raw_timestamp: self.raw_timestamp,
            monotonic_timestamp_ns: self.monotonic_timestamp_ns,
            captured_len: self.captured_len,
            original_len: self.original_len,
            bytes: self.bytes,
            quality: self.quality,
            classification: self.classification,
            normalization: self.normalization,
        }
    }
}

type ParsedNetwork<'a> = (IpAddr, IpAddr, u8, &'a [u8]);

struct Parser<'a> {
    bytes: &'a [u8],
    order: ByteOrder,
    nanosecond_timestamps: bool,
    snapshot_length: NonZeroUsize,
    max_records: usize,
    offset: usize,
    report: PcapLoadReport,
    generation: FlowGenerationTracker,
    active: HashMap<WireFlow, ActiveFlow>,
    last_timestamp_ns: u64,
    last_raw_timestamp_ns: u64,
    raw_timestamps: Option<Vec<RawTimestamp>>,
}

impl<'a> Parser<'a> {
    fn new(
        bytes: &'a [u8],
        max_records: usize,
        raw_timestamps: Option<Vec<RawTimestamp>>,
    ) -> Result<Self, CaptureError> {
        if bytes.len() < 4 {
            return Err(pcap_error("global header is truncated"));
        }
        if bytes[0..4] == [0x0a, 0x0d, 0x0d, 0x0a] {
            return Err(pcap_error(
                "PCAPNG is detected but not implemented; convert it to classic PCAP",
            ));
        }
        if bytes.len() < 24 {
            return Err(pcap_error("global header is truncated"));
        }
        let (order, nanosecond_timestamps) = match bytes[0..4] {
            [0xd4, 0xc3, 0xb2, 0xa1] => (ByteOrder::Little, false),
            [0xa1, 0xb2, 0xc3, 0xd4] => (ByteOrder::Big, false),
            [0x4d, 0x3c, 0xb2, 0xa1] => (ByteOrder::Little, true),
            [0xa1, 0xb2, 0x3c, 0x4d] => (ByteOrder::Big, true),
            _ => return Err(pcap_error("unknown capture magic")),
        };
        if order.u16(&bytes[4..6]) != 2 || order.u16(&bytes[6..8]) != 4 {
            return Err(pcap_error("only PCAP version 2.4 is supported"));
        }
        let snapshot_length = NonZeroUsize::new(order.u32(&bytes[16..20]) as usize)
            .ok_or_else(|| pcap_error("snapshot length is zero"))?;
        if order.u32(&bytes[20..24]) != 1 {
            return Err(pcap_error("only Ethernet PCAP link type is supported"));
        }
        Ok(Self {
            bytes,
            order,
            nanosecond_timestamps,
            snapshot_length,
            max_records,
            offset: 24,
            report: PcapLoadReport::default(),
            generation: FlowGenerationTracker::default(),
            active: HashMap::new(),
            last_timestamp_ns: 0,
            last_raw_timestamp_ns: 0,
            raw_timestamps,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn parse(mut self) -> Result<ParsedPcap, CaptureError> {
        let mut packets = Vec::new();
        let mut raw_records = Vec::new();
        while self.offset < self.bytes.len() {
            if self.report.records_seen == self.max_records {
                return Err(pcap_error("record count exceeds configured limit"));
            }
            if self.bytes.len() - self.offset < 16 {
                return Err(pcap_error("packet record header is truncated"));
            }
            let header = &self.bytes[self.offset..self.offset + 16];
            let seconds = u64::from(self.order.u32(&header[0..4]));
            let fraction = u64::from(self.order.u32(&header[4..8]));
            let included = self.order.u32(&header[8..12]) as usize;
            let original = self.order.u32(&header[12..16]) as usize;
            self.offset += 16;
            self.report.records_seen += 1;
            if included == 0 || included > original || included > self.snapshot_length.get() {
                return Err(pcap_error("invalid included/original packet lengths"));
            }
            let end = self
                .offset
                .checked_add(included)
                .ok_or_else(|| pcap_error("packet offset overflow"))?;
            if end > self.bytes.len() {
                return Err(pcap_error("packet bytes are truncated"));
            }
            let frame = &self.bytes[self.offset..end];
            self.offset = end;
            let divisor = if self.nanosecond_timestamps {
                1_000_000_000
            } else {
                1_000_000
            };
            if fraction >= divisor {
                return Err(pcap_error("timestamp fraction is out of range"));
            }
            let multiplier = if self.nanosecond_timestamps { 1 } else { 1_000 };
            let timestamp_ns = seconds
                .checked_mul(1_000_000_000)
                .and_then(|value| value.checked_add(fraction * multiplier))
                .ok_or_else(|| pcap_error("timestamp overflow"))?;
            let raw_reordered = timestamp_ns < self.last_raw_timestamp_ns;
            let raw_monotonic_timestamp_ns = timestamp_ns.max(self.last_raw_timestamp_ns);
            self.last_raw_timestamp_ns = raw_monotonic_timestamp_ns;
            let before = self.report;
            let normalized = self.parse_frame(frame, included, original, timestamp_ns)?;
            let (classification, normalization) = if normalized.is_some() {
                (
                    RawPacketClassification::SupportedIp,
                    NormalizationStatus::Normalized,
                )
            } else if self.report.unsupported_link_packets > before.unsupported_link_packets {
                (
                    RawPacketClassification::NonIp,
                    NormalizationStatus::Unsupported,
                )
            } else if self.report.unsupported_ipv4_fragment_packets
                > before.unsupported_ipv4_fragment_packets
            {
                (
                    RawPacketClassification::Unsupported(UnsupportedReason::FragmentedIpv4),
                    NormalizationStatus::Unsupported,
                )
            } else if self.report.unsupported_ipv6_fragment_packets
                > before.unsupported_ipv6_fragment_packets
            {
                (
                    RawPacketClassification::Unsupported(UnsupportedReason::FragmentedIpv6),
                    NormalizationStatus::Unsupported,
                )
            } else {
                (
                    RawPacketClassification::Unsupported(UnsupportedReason::NetworkProtocol),
                    NormalizationStatus::Unsupported,
                )
            };
            let raw_timestamp = if let Some(timestamps) = &self.raw_timestamps {
                *timestamps
                    .get(self.report.records_seen - 1)
                    .ok_or_else(|| pcap_error("missing PCAPNG raw timestamp"))?
            } else {
                let ticks_per_second = if self.nanosecond_timestamps {
                    1_000_000_000
                } else {
                    1_000_000
                };
                let ticks = seconds
                    .checked_mul(ticks_per_second)
                    .and_then(|value| value.checked_add(fraction))
                    .ok_or_else(|| pcap_error("raw timestamp overflow"))?;
                RawTimestamp {
                    ticks,
                    ticks_per_second: std::num::NonZeroU64::new(ticks_per_second)
                        .expect("PCAP timestamp resolution is nonzero"),
                }
            };
            raw_records.push(RawRecordData {
                source_ordinal: u64::try_from(self.report.records_seen - 1)
                    .map_err(|_| pcap_error("source ordinal exceeds u64"))?,
                raw_timestamp,
                monotonic_timestamp_ns: raw_monotonic_timestamp_ns,
                captured_len: u32::try_from(included)
                    .map_err(|_| pcap_error("included length exceeds u32"))?,
                original_len: u32::try_from(original)
                    .map_err(|_| pcap_error("original length exceeds u32"))?,
                bytes: frame.to_vec(),
                quality: RawPacketQuality {
                    packets_lost_before: 0,
                    reordered: raw_reordered,
                    truncated: included < original,
                    clock_rollback: raw_reordered,
                },
                classification,
                normalization,
            });
            if let Some(packet) = normalized {
                packets.push(packet);
            }
        }
        self.report.packets_emitted = packets.len();
        Ok(ParsedPcap {
            snapshot_length: self.snapshot_length,
            packets,
            raw_records,
            report: self.report,
        })
    }

    // Ethernet/IP/transport validation stays together so no partially validated view escapes.
    #[allow(clippy::too_many_lines)]
    fn parse_frame(
        &mut self,
        frame: &[u8],
        included: usize,
        original: usize,
        timestamp_ns: u64,
    ) -> Result<Option<CapturedPacket>, CaptureError> {
        if frame.len() < 14 {
            return Err(pcap_error("Ethernet frame is shorter than its header"));
        }
        let (network_offset, ether_type) = ethernet_or_pktmon_wifi_network_offset(frame)?;
        let network = &frame[network_offset..];
        let original_network_length = original
            .checked_sub(network_offset)
            .ok_or_else(|| pcap_error("original frame is shorter than its Ethernet header"))?;
        let (source, destination, next_header, transport) = match ether_type {
            0x0800 => match self.parse_ipv4(network, original_network_length)? {
                Some(parsed) => parsed,
                None => return Ok(None),
            },
            0x86dd => match self.parse_ipv6(network, original_network_length)? {
                Some(parsed) => parsed,
                None => return Ok(None),
            },
            _ => {
                self.report.unsupported_link_packets += 1;
                return Ok(None);
            }
        };
        let protocol = match next_header {
            6 => TransportProtocol::Tcp,
            17 => TransportProtocol::Udp,
            _ => {
                self.report.unsupported_network_packets += 1;
                return Ok(None);
            }
        };
        let minimum = if protocol == TransportProtocol::Tcp {
            20
        } else {
            8
        };
        if transport.len() < minimum {
            return Err(pcap_error("transport header is truncated"));
        }
        let transport_header_length = if protocol == TransportProtocol::Tcp {
            let header_length = usize::from(transport[12] >> 4) * 4;
            if header_length < 20 || header_length > transport.len() {
                return Err(pcap_error("invalid TCP header length"));
            }
            header_length
        } else {
            let udp_length = usize::from(u16::from_be_bytes([transport[4], transport[5]]));
            if udp_length < 8 || (udp_length > transport.len() && included == original) {
                return Err(pcap_error("invalid UDP length"));
            }
            8
        };
        let source_port = u16::from_be_bytes([transport[0], transport[1]]);
        let destination_port = u16::from_be_bytes([transport[2], transport[3]]);
        let application_payload = &transport[transport_header_length..];
        let dns_payload = if protocol == TransportProtocol::Tcp {
            // DNS over TCP prefixes each message with a two-byte length. Only
            // decode a complete message present in this packet; no reassembly
            // is claimed here.
            application_payload.get(..2).and_then(|prefix| {
                let length = usize::from(u16::from_be_bytes([prefix[0], prefix[1]]));
                (length >= 12)
                    .then(|| application_payload.get(2..2 + length))
                    .flatten()
            })
        } else {
            Some(application_payload)
        };
        let dns = (source_port == 53 || destination_port == 53)
            .then(|| dns_payload.and_then(parse_dns_summary))
            .flatten();
        let tls_sni = (protocol == TransportProtocol::Tcp)
            .then(|| parse_tls_sni(application_payload))
            .flatten();
        let http_request = (protocol == TransportProtocol::Tcp
            && (matches!(source_port, 80 | 8080) || matches!(destination_port, 80 | 8080)))
            && parse_http_summary(application_payload)
                .is_some_and(|summary| !summary.is_response && summary.method.is_some());
        let (source_mac, destination_mac) = ethernet_mac_addresses(frame, network_offset);
        let wire = WireFlow {
            source,
            source_port,
            destination,
            destination_port,
            protocol,
        };
        let reverse = wire.reverse();
        let tcp_flags = (protocol == TransportProtocol::Tcp).then(|| transport[13]);
        let starts_new_connection =
            tcp_flags.is_some_and(|flags| flags & 0x02 != 0 && flags & 0x10 == 0);
        if starts_new_connection && self.active.contains_key(&wire) {
            self.active.remove(&wire);
            self.active.remove(&reverse);
        }
        let active = if let Some(active) = self.active.get_mut(&wire) {
            active
        } else {
            let flow = self
                .generation
                .begin(FlowKey {
                    initiator: Endpoint {
                        address: source,
                        port: source_port,
                    },
                    responder: Endpoint {
                        address: destination,
                        port: destination_port,
                    },
                    protocol,
                })
                .map_err(|error| CaptureError::SourceFailure(error.to_string()))?;
            self.active.insert(
                wire,
                ActiveFlow {
                    flow: flow.clone(),
                    direction: Direction::InitiatorToResponder,
                    next_sequence: 0,
                },
            );
            self.active.insert(
                reverse,
                ActiveFlow {
                    flow,
                    direction: Direction::ResponderToInitiator,
                    next_sequence: 0,
                },
            );
            self.active.get_mut(&wire).ok_or_else(|| {
                CaptureError::SourceFailure("flow index insertion failed".to_owned())
            })?
        };
        let flow = active.flow.clone();
        let direction = active.direction;
        let sequence = active.next_sequence;
        let next_sequence = sequence
            .checked_add(1)
            .ok_or_else(|| CaptureError::SourceFailure("flow sequence overflow".to_owned()))?;
        if let Some(entry) = self.active.get_mut(&wire) {
            entry.next_sequence = next_sequence;
        }
        if let Some(entry) = self.active.get_mut(&reverse) {
            entry.next_sequence = next_sequence;
        }

        let reordered = timestamp_ns < self.last_timestamp_ns;
        let monotonic_timestamp_ns = timestamp_ns.max(self.last_timestamp_ns);
        self.last_timestamp_ns = monotonic_timestamp_ns;
        let capture_ordinal = self.report.packets_emitted as u64;
        self.report.packets_emitted += 1;
        let close_flow = tcp_flags.is_some_and(|flags| flags & 0x04 != 0);
        if close_flow {
            self.active.remove(&wire);
            self.active.remove(&reverse);
        }
        Ok(Some(CapturedPacket {
            capture_ordinal,
            flow,
            packet: PacketEvent {
                sequence,
                monotonic_timestamp_ns,
                captured_len: u32::try_from(included)
                    .map_err(|_| pcap_error("included length exceeds u32"))?,
                original_len: u32::try_from(original)
                    .map_err(|_| pcap_error("original length exceeds u32"))?,
                direction,
            },
            protocol: PacketProtocolMetadata {
                ip_version: if source.is_ipv4() { 4 } else { 6 },
                tcp_flags,
            },
            observations: PacketObservationMetadata {
                source_mac,
                destination_mac,
                tls_sni,
                dns_qname: dns.as_ref().and_then(|summary| summary.qname.clone()),
                dns_message: dns.is_some(),
                dns_query: dns.as_ref().is_some_and(|summary| !summary.is_response),
                http_request,
                tor_port_match: is_tor_traffic(source_port, destination_port),
            },
            quality: QualityMarkers {
                packets_lost_before: 0,
                truncated: included < original,
                reordered,
                clock_rollback: reordered,
            },
        }))
    }

    fn parse_ipv4<'b>(
        &mut self,
        ip: &'b [u8],
        original_length: usize,
    ) -> Result<Option<ParsedNetwork<'b>>, CaptureError> {
        if ip.len() < 20 {
            return Err(pcap_error("IPv4 header is truncated"));
        }
        if ip[0] >> 4 != 4 {
            return Err(pcap_error("EtherType IPv4 contains a non-IPv4 header"));
        }
        let header_length = usize::from(ip[0] & 0x0f) * 4;
        if header_length < 20 || ip.len() < header_length {
            return Err(pcap_error("invalid IPv4 header length"));
        }
        let total_length = usize::from(u16::from_be_bytes([ip[2], ip[3]]));
        if total_length < header_length {
            return Err(pcap_error("invalid IPv4 total length"));
        }
        if total_length > original_length {
            return Err(pcap_error(
                "IPv4 total length exceeds original frame length",
            ));
        }
        let fragment = u16::from_be_bytes([ip[6], ip[7]]);
        if fragment & 0x3fff != 0 {
            self.report.fragmented_packets += 1;
            self.report.unsupported_ipv4_fragment_packets += 1;
            return Ok(None);
        }
        let available_length = total_length.min(ip.len());
        let source = Ipv4Addr::new(ip[12], ip[13], ip[14], ip[15]);
        let destination = Ipv4Addr::new(ip[16], ip[17], ip[18], ip[19]);
        Ok(Some((
            IpAddr::V4(source),
            IpAddr::V4(destination),
            ip[9],
            &ip[header_length..available_length],
        )))
    }

    fn parse_ipv6<'b>(
        &mut self,
        ip: &'b [u8],
        original_length: usize,
    ) -> Result<Option<ParsedNetwork<'b>>, CaptureError> {
        if ip.len() < 40 {
            return Err(pcap_error("IPv6 header is truncated"));
        }
        if ip[0] >> 4 != 6 {
            return Err(pcap_error("EtherType IPv6 contains a non-IPv6 header"));
        }
        let payload_length = usize::from(u16::from_be_bytes([ip[4], ip[5]]));
        if payload_length == 0 {
            self.report.unsupported_network_packets += 1;
            return Ok(None);
        }
        let declared_length = 40_usize
            .checked_add(payload_length)
            .ok_or_else(|| pcap_error("IPv6 payload length overflow"))?;
        if declared_length > original_length {
            return Err(pcap_error(
                "IPv6 payload length exceeds original frame length",
            ));
        }
        let payload_end = declared_length.min(ip.len());
        let source = Ipv6Addr::from(<[u8; 16]>::try_from(&ip[8..24]).expect("fixed IPv6 source"));
        let destination =
            Ipv6Addr::from(<[u8; 16]>::try_from(&ip[24..40]).expect("fixed IPv6 destination"));
        let mut next_header = ip[6];
        let mut offset = 40_usize;
        let mut extensions = 0_u8;
        loop {
            let extension_length = match next_header {
                0 | 43 | 60 => {
                    if payload_end.saturating_sub(offset) < 2 {
                        return Err(pcap_error("IPv6 extension header is truncated"));
                    }
                    usize::from(ip[offset + 1])
                        .checked_add(1)
                        .and_then(|units| units.checked_mul(8))
                        .ok_or_else(|| pcap_error("IPv6 extension header length overflow"))?
                }
                51 => {
                    if payload_end.saturating_sub(offset) < 2 {
                        return Err(pcap_error("IPv6 AH header is truncated"));
                    }
                    usize::from(ip[offset + 1])
                        .checked_add(2)
                        .and_then(|words| words.checked_mul(4))
                        .ok_or_else(|| pcap_error("IPv6 AH header length overflow"))?
                }
                44 => {
                    if payload_end.saturating_sub(offset) < 8 {
                        return Err(pcap_error("IPv6 fragment header is truncated"));
                    }
                    self.report.fragmented_packets += 1;
                    self.report.unsupported_ipv6_fragment_packets += 1;
                    return Ok(None);
                }
                _ => break,
            };
            if extension_length > payload_end.saturating_sub(offset) {
                return Err(pcap_error("IPv6 extension header is truncated"));
            }
            next_header = ip[offset];
            offset += extension_length;
            extensions = extensions
                .checked_add(1)
                .ok_or_else(|| pcap_error("too many IPv6 extension headers"))?;
            if extensions > 16 {
                return Err(pcap_error("too many IPv6 extension headers"));
            }
        }
        Ok(Some((
            IpAddr::V6(source),
            IpAddr::V6(destination),
            next_header,
            &ip[offset..payload_end],
        )))
    }
}

fn ethernet_mac_addresses(
    frame: &[u8],
    network_offset: usize,
) -> (Option<[u8; 6]>, Option<[u8; 6]>) {
    // Ethernet plus the maximum four supported VLAN tags ends at byte 30.
    // Pktmon's 802.11 + LLC/SNAP representation starts its IP payload later,
    // and its address fields do not have Ethernet source/destination semantics.
    if frame.len() < 14 || network_offset > 30 {
        return (None, None);
    }
    let destination = frame[0..6].try_into().ok();
    let source = frame[6..12].try_into().ok();
    (source, destination)
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct WireFlow {
    source: IpAddr,
    source_port: u16,
    destination: IpAddr,
    destination_port: u16,
    protocol: TransportProtocol,
}

impl WireFlow {
    fn reverse(self) -> Self {
        Self {
            source: self.destination,
            source_port: self.destination_port,
            destination: self.source,
            destination_port: self.source_port,
            protocol: self.protocol,
        }
    }
}

#[derive(Clone, Debug)]
struct ActiveFlow {
    flow: FlowInstanceId,
    direction: Direction,
    next_sequence: u64,
}

fn pcap_error(message: &str) -> CaptureError {
    CaptureError::SourceFailure(format!("invalid PCAP: {message}"))
}

/// Returns the network payload offset for Ethernet or Pktmon's observed 802.11
/// data + LLC/SNAP framing. Pktmon may label a wireless adapter as Ethernet in
/// PCAPNG while emitting this latter frame form, so link type alone is not a
/// sufficient parser contract.
fn ethernet_or_pktmon_wifi_network_offset(frame: &[u8]) -> Result<(usize, u16), CaptureError> {
    const VLAN_TYPES: [u16; 3] = [0x8100, 0x88a8, 0x9100];
    const MAX_VLAN_DEPTH: usize = 4;

    let ethernet_type = u16::from_be_bytes([frame[12], frame[13]]);
    if matches!(ethernet_type, 0x0800 | 0x86dd) {
        return Ok((14, ethernet_type));
    }

    if VLAN_TYPES.contains(&ethernet_type) {
        let mut network_offset = 14_usize;
        let mut inner_type = ethernet_type;
        for _ in 0..MAX_VLAN_DEPTH {
            if !VLAN_TYPES.contains(&inner_type) {
                return Ok((network_offset, inner_type));
            }
            let inner_type_end = network_offset
                .checked_add(4)
                .ok_or_else(|| pcap_error("VLAN Ethernet offset overflow"))?;
            if frame.len() < inner_type_end {
                return Err(pcap_error("VLAN Ethernet frame is truncated"));
            }
            inner_type = u16::from_be_bytes([frame[network_offset + 2], frame[network_offset + 3]]);
            network_offset = inner_type_end;
        }
        if VLAN_TYPES.contains(&inner_type) {
            return Err(pcap_error("Ethernet frame exceeds the VLAN nesting limit"));
        }
        return Ok((network_offset, inner_type));
    }

    let frame_control = u16::from_le_bytes([frame[0], frame[1]]);
    let is_data = ((frame_control >> 2) & 0x03) == 2;
    if !is_data {
        return Ok((14, ethernet_type));
    }
    let to_distribution = frame_control & 0x0100 != 0;
    let from_distribution = frame_control & 0x0200 != 0;
    let quality_of_service = frame_control & 0x0080 != 0;
    let header_length = 24
        + usize::from(to_distribution && from_distribution) * 6
        + usize::from(quality_of_service) * 2;
    let llc_end = header_length
        .checked_add(8)
        .ok_or_else(|| pcap_error("802.11 LLC offset overflow"))?;
    if frame.len() < llc_end
        || frame[header_length..header_length + 6] != [0xaa, 0xaa, 0x03, 0, 0, 0]
    {
        return Ok((14, ethernet_type));
    }
    Ok((
        llc_end,
        u16::from_be_bytes([frame[header_length + 6], frame[header_length + 7]]),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> PcapLimits {
        PcapLimits {
            max_file_bytes: NonZeroUsize::new(1_000_000).expect("non-zero"),
            max_records: NonZeroUsize::new(100).expect("non-zero"),
        }
    }

    fn tcp_frame(source: [u8; 4], destination: [u8; 4], flags: u8) -> Vec<u8> {
        let mut frame = vec![0_u8; 54];
        frame[12..14].copy_from_slice(&0x0800_u16.to_be_bytes());
        frame[14] = 0x45;
        frame[16..18].copy_from_slice(&40_u16.to_be_bytes());
        frame[23] = 6;
        frame[26..30].copy_from_slice(&source);
        frame[30..34].copy_from_slice(&destination);
        frame[34..36].copy_from_slice(&50_000_u16.to_be_bytes());
        frame[36..38].copy_from_slice(&443_u16.to_be_bytes());
        frame[46] = 0x50;
        frame[47] = flags;
        frame
    }

    fn pktmon_wifi_tcp_frame() -> Vec<u8> {
        let ethernet = tcp_frame([192, 0, 2, 1], [198, 51, 100, 2], 0x02);
        let mut frame = vec![0_u8; 24];
        frame[0..2].copy_from_slice(&0x0108_u16.to_le_bytes());
        frame.extend_from_slice(&[0xaa, 0xaa, 0x03, 0, 0, 0, 0x08, 0x00]);
        frame.extend_from_slice(&ethernet[14..]);
        frame
    }

    fn vlan_frame(frame: &[u8], tag_types: &[u16]) -> Vec<u8> {
        assert!(!tag_types.is_empty());
        let mut tagged = frame[..12].to_vec();
        for tag_type in tag_types {
            tagged.extend_from_slice(&tag_type.to_be_bytes());
            tagged.extend_from_slice(&100_u16.to_be_bytes());
        }
        tagged.extend_from_slice(&frame[12..]);
        tagged
    }

    fn ipv6_frame(next_header: u8, payload: &[u8]) -> Vec<u8> {
        let mut frame = vec![0_u8; 54 + payload.len()];
        frame[12..14].copy_from_slice(&0x86dd_u16.to_be_bytes());
        frame[14] = 0x60;
        frame[18..20].copy_from_slice(
            &u16::try_from(payload.len())
                .expect("test payload length fits u16")
                .to_be_bytes(),
        );
        frame[20] = next_header;
        frame[21] = 64;
        frame[22..38]
            .copy_from_slice(&[0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
        frame[38..54]
            .copy_from_slice(&[0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]);
        frame[54..].copy_from_slice(payload);
        frame
    }

    fn tcp_header(source_port: u16, destination_port: u16, flags: u8) -> [u8; 20] {
        let mut header = [0_u8; 20];
        header[0..2].copy_from_slice(&source_port.to_be_bytes());
        header[2..4].copy_from_slice(&destination_port.to_be_bytes());
        header[12] = 0x50;
        header[13] = flags;
        header
    }

    fn udp_header(source_port: u16, destination_port: u16) -> [u8; 8] {
        let mut header = [0_u8; 8];
        header[0..2].copy_from_slice(&source_port.to_be_bytes());
        header[2..4].copy_from_slice(&destination_port.to_be_bytes());
        header[4..6].copy_from_slice(&8_u16.to_be_bytes());
        header
    }

    fn pcap(records: &[(u32, Vec<u8>)]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&[0xd4, 0xc3, 0xb2, 0xa1]);
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&4_u16.to_le_bytes());
        bytes.extend_from_slice(&0_i32.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&65_535_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        for (microseconds, frame) in records {
            let frame_length = u32::try_from(frame.len()).expect("test frame length fits u32");
            bytes.extend_from_slice(&1_u32.to_le_bytes());
            bytes.extend_from_slice(&microseconds.to_le_bytes());
            bytes.extend_from_slice(&frame_length.to_le_bytes());
            bytes.extend_from_slice(&frame_length.to_le_bytes());
            bytes.extend_from_slice(frame);
        }
        bytes
    }

    fn pcapng(frame: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x0a0d_0d0a_u32.to_le_bytes());
        bytes.extend_from_slice(&28_u32.to_le_bytes());
        bytes.extend_from_slice(&0x1a2b_3c4d_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(&28_u32.to_le_bytes());

        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&20_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&65_535_u32.to_le_bytes());
        bytes.extend_from_slice(&20_u32.to_le_bytes());

        let padded_length = (frame.len() + 3) & !3;
        let block_length = u32::try_from(32 + padded_length).expect("fixture length fits");
        let frame_length = u32::try_from(frame.len()).expect("fixture length fits");
        bytes.extend_from_slice(&6_u32.to_le_bytes());
        bytes.extend_from_slice(&block_length.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&1_000_000_u32.to_le_bytes());
        bytes.extend_from_slice(&frame_length.to_le_bytes());
        bytes.extend_from_slice(&frame_length.to_le_bytes());
        bytes.extend_from_slice(frame);
        bytes.resize(bytes.len() + padded_length - frame.len(), 0);
        bytes.extend_from_slice(&block_length.to_le_bytes());
        bytes
    }

    #[test]
    fn parses_bidirectional_flow_with_deterministic_sequences() {
        let forward = tcp_frame([192, 0, 2, 1], [198, 51, 100, 2], 0x02);
        let mut reverse = tcp_frame([198, 51, 100, 2], [192, 0, 2, 1], 0x12);
        reverse[34..36].copy_from_slice(&443_u16.to_be_bytes());
        reverse[36..38].copy_from_slice(&50_000_u16.to_be_bytes());
        let mut source =
            PcapReplaySource::from_bytes("fixture", &pcap(&[(0, forward), (1, reverse)]), limits())
                .expect("parse");
        let CapturePoll::Packet(first) = source.poll_next().expect("first") else {
            panic!("expected packet");
        };
        let CapturePoll::Packet(second) = source.poll_next().expect("second") else {
            panic!("expected packet");
        };
        assert_eq!(first.flow, second.flow);
        assert_eq!(first.packet.sequence, 0);
        assert_eq!(first.protocol.ip_version, 4);
        assert_eq!(first.protocol.tcp_flags, Some(0x02));
        assert_eq!(second.packet.sequence, 1);
        assert_eq!(second.protocol.tcp_flags, Some(0x12));
        assert_eq!(second.packet.direction, Direction::ResponderToInitiator);
    }

    #[test]
    fn preserves_ethernet_and_decoded_application_observations() {
        let payload = b"GET /status HTTP/1.1\r\nHost: example.test\r\n\r\n";
        let mut frame = tcp_frame([192, 0, 2, 1], [198, 51, 100, 2], 0x18);
        frame[0..6].copy_from_slice(&[0x00, 0x09, 0x5b, 0x44, 0x55, 0x66]);
        frame[6..12].copy_from_slice(&[0x00, 0x15, 0x5d, 0x11, 0x22, 0x33]);
        frame[36..38].copy_from_slice(&80_u16.to_be_bytes());
        frame.extend_from_slice(payload);
        frame[16..18].copy_from_slice(
            &u16::try_from(40 + payload.len())
                .expect("fixture length fits u16")
                .to_be_bytes(),
        );

        let mut source =
            PcapReplaySource::from_bytes("observations", &pcap(&[(0, frame)]), limits())
                .expect("parse");
        let CapturePoll::Packet(packet) = source.poll_next().expect("packet") else {
            panic!("expected packet");
        };
        assert_eq!(
            packet.observations.source_mac,
            Some([0x00, 0x15, 0x5d, 0x11, 0x22, 0x33])
        );
        assert_eq!(
            packet.observations.destination_mac,
            Some([0x00, 0x09, 0x5b, 0x44, 0x55, 0x66])
        );
        assert!(packet.observations.http_request);
        assert!(!packet.observations.dns_message);
        assert!(!packet.observations.dns_query);
        assert!(!packet.observations.tor_port_match);
    }

    #[test]
    fn dns_response_is_a_message_but_not_a_query() {
        fn dns_frame(response: bool) -> Vec<u8> {
            let dns_header = [
                0x12,
                0x34,
                if response { 0x80 } else { 0x01 },
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ];
            let mut frame = vec![0_u8; 14 + 20];
            frame[12..14].copy_from_slice(&0x0800_u16.to_be_bytes());
            frame[14] = 0x45;
            frame[16..18].copy_from_slice(&40_u16.to_be_bytes());
            frame[23] = 17;
            frame[26..30].copy_from_slice(&[192, 0, 2, 1]);
            frame[30..34].copy_from_slice(&[198, 51, 100, 2]);
            let mut udp = udp_header(
                if response { 53 } else { 50_000 },
                if response { 50_000 } else { 53 },
            );
            udp[4..6].copy_from_slice(&20_u16.to_be_bytes());
            frame.extend_from_slice(&udp);
            frame.extend_from_slice(&dns_header);
            frame
        }

        let mut source = PcapReplaySource::from_bytes(
            "dns-query-response",
            &pcap(&[(0, dns_frame(false)), (1, dns_frame(true))]),
            limits(),
        )
        .expect("parse");
        let CapturePoll::Packet(query) = source.poll_next().expect("query") else {
            panic!("expected query packet");
        };
        let CapturePoll::Packet(response) = source.poll_next().expect("response") else {
            panic!("expected response packet");
        };
        assert!(query.observations.dns_message);
        assert!(query.observations.dns_query);
        assert!(response.observations.dns_message);
        assert!(!response.observations.dns_query);
    }

    #[test]
    fn dns_over_tcp_requires_a_complete_length_prefixed_message() {
        let dns_header = [0x12, 0x34, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut complete = tcp_frame([192, 0, 2, 1], [198, 51, 100, 2], 0x18);
        complete[36..38].copy_from_slice(&53_u16.to_be_bytes());
        complete.extend_from_slice(&12_u16.to_be_bytes());
        complete.extend_from_slice(&dns_header);
        complete[16..18].copy_from_slice(&54_u16.to_be_bytes());

        let mut incomplete = complete.clone();
        incomplete.truncate(incomplete.len() - 1);
        incomplete[16..18].copy_from_slice(&53_u16.to_be_bytes());

        let mut source = PcapReplaySource::from_bytes(
            "dns-tcp",
            &pcap(&[(0, complete), (1, incomplete)]),
            limits(),
        )
        .expect("parse");
        let CapturePoll::Packet(query) = source.poll_next().expect("complete") else {
            panic!("expected packet");
        };
        let CapturePoll::Packet(partial) = source.poll_next().expect("partial") else {
            panic!("expected packet");
        };
        assert!(query.observations.dns_query);
        assert!(!partial.observations.dns_message);
        assert!(!partial.observations.dns_query);
    }

    #[test]
    fn parses_single_vlan_and_qinq_tcp_frames() {
        let base = tcp_frame([192, 0, 2, 1], [198, 51, 100, 2], 0x02);
        let single = vlan_frame(&base, &[0x8100]);
        let qinq = vlan_frame(&base, &[0x88a8, 0x8100]);
        let replay =
            PcapReplaySource::from_bytes("tagged", &pcap(&[(0, single), (1, qinq)]), limits())
                .expect("parse tagged Ethernet frames");
        assert_eq!(replay.report().packets_emitted, 2);
        assert_eq!(replay.report().unsupported_link_packets, 0);
    }

    #[test]
    fn rejects_truncated_and_excessively_nested_vlan_frames() {
        let mut truncated = vec![0_u8; 16];
        truncated[12..14].copy_from_slice(&0x8100_u16.to_be_bytes());
        let error =
            PcapReplaySource::from_bytes("truncated-vlan", &pcap(&[(0, truncated)]), limits())
                .expect_err("truncated VLAN tag must fail closed");
        assert!(
            error
                .to_string()
                .contains("VLAN Ethernet frame is truncated")
        );

        let base = tcp_frame([192, 0, 2, 1], [198, 51, 100, 2], 0x02);
        let nested = vlan_frame(&base, &[0x88a8, 0x8100, 0x9100, 0x8100, 0x8100]);
        let error = PcapReplaySource::from_bytes("nested-vlan", &pcap(&[(0, nested)]), limits())
            .expect_err("VLAN nesting beyond the parser contract must fail closed");
        assert!(error.to_string().contains("VLAN nesting limit"));
    }

    #[test]
    fn parses_ipv6_tcp_golden_packet() {
        let frame = ipv6_frame(6, &tcp_header(50_000, 443, 0x02));
        let mut source =
            PcapReplaySource::from_bytes("ipv6", &pcap(&[(7, frame)]), limits()).expect("parse");
        let CapturePoll::Packet(packet) = source.poll_next().expect("packet") else {
            panic!("expected packet");
        };
        assert_eq!(
            packet.flow.key,
            FlowKey {
                initiator: Endpoint {
                    address: "2001:db8::1".parse().expect("IPv6 address"),
                    port: 50_000,
                },
                responder: Endpoint {
                    address: "2001:db8::2".parse().expect("IPv6 address"),
                    port: 443,
                },
                protocol: TransportProtocol::Tcp,
            }
        );
        assert_eq!(packet.capture_ordinal, 0);
        assert_eq!(packet.protocol.ip_version, 6);
        assert_eq!(packet.protocol.tcp_flags, Some(0x02));
        assert_eq!(packet.packet.sequence, 0);
        assert_eq!(packet.packet.monotonic_timestamp_ns, 1_000_007_000);
        assert_eq!(packet.packet.direction, Direction::InitiatorToResponder);
        assert_eq!(source.report().packets_emitted, 1);
    }

    #[test]
    fn parses_ipv6_udp_after_destination_options() {
        let mut payload = vec![0_u8; 8];
        payload[0] = 17;
        payload.extend_from_slice(&udp_header(53_000, 53));
        let frame = ipv6_frame(60, &payload);
        let mut source = PcapReplaySource::from_bytes("ipv6-ext", &pcap(&[(0, frame)]), limits())
            .expect("parse");
        let CapturePoll::Packet(packet) = source.poll_next().expect("packet") else {
            panic!("expected packet");
        };
        assert_eq!(packet.flow.key.protocol, TransportProtocol::Udp);
        assert_eq!(packet.flow.key.initiator.port, 53_000);
        assert_eq!(packet.flow.key.responder.port, 53);
    }

    #[test]
    fn classifies_ipv4_and_ipv6_fragments_separately() {
        let mut ipv4 = tcp_frame([192, 0, 2, 1], [198, 51, 100, 2], 0);
        ipv4[20..22].copy_from_slice(&0x2000_u16.to_be_bytes());
        let ipv6 = ipv6_frame(44, &[6, 0, 0, 0, 0, 0, 0, 1]);
        let source =
            PcapReplaySource::from_bytes("fragments", &pcap(&[(0, ipv4), (1, ipv6)]), limits())
                .expect("classified capture");
        assert_eq!(source.report().records_seen, 2);
        assert_eq!(source.report().packets_emitted, 0);
        assert_eq!(source.report().fragmented_packets, 2);
        assert_eq!(source.report().unsupported_ipv4_fragment_packets, 1);
        assert_eq!(source.report().unsupported_ipv6_fragment_packets, 1);
    }

    #[test]
    fn classifies_unsupported_link_and_network_packets() {
        let mut arp = vec![0_u8; 14];
        arp[12..14].copy_from_slice(&0x0806_u16.to_be_bytes());
        let unsupported_ipv6 = ipv6_frame(58, &[0_u8; 8]);
        let source = PcapReplaySource::from_bytes(
            "unsupported",
            &pcap(&[(0, arp), (1, unsupported_ipv6)]),
            limits(),
        )
        .expect("classified capture");
        assert_eq!(source.report().unsupported_link_packets, 1);
        assert_eq!(source.report().unsupported_network_packets, 1);
        assert_eq!(source.report().packets_emitted, 0);
        let counters = source.counters();
        assert_eq!(counters.source_records, 2);
        assert_eq!(counters.non_ip_records, 1);
        assert_eq!(counters.unsupported_records, 1);
    }

    #[test]
    fn raw_replay_emits_unsupported_records_in_source_order() {
        let mut arp = vec![0_u8; 14];
        arp[12..14].copy_from_slice(&0x0806_u16.to_be_bytes());
        let unsupported_ipv6 = ipv6_frame(58, &[0_u8; 8]);
        let mut source = PcapReplaySource::from_bytes(
            "raw-unsupported",
            &pcap(&[(2, arp.clone()), (1, unsupported_ipv6.clone())]),
            limits(),
        )
        .expect("classified capture");

        let RawCapturePoll::Packet(first) = source.poll_next_raw().expect("first raw") else {
            panic!("expected raw packet");
        };
        assert_eq!(first.source.source_id, "raw-unsupported");
        assert!(first.session.session_id.starts_with("pcap-replay:"));
        assert_eq!(first.session, *RawCaptureSource::session_metadata(&source));
        assert_eq!(first.source_ordinal, 0);
        assert_eq!(first.bytes, arp);
        assert_eq!(first.raw_timestamp.ticks, 1_000_002);
        assert_eq!(first.raw_timestamp.ticks_per_second.get(), 1_000_000);
        assert_eq!(first.classification, RawPacketClassification::NonIp);
        assert_eq!(first.normalization, NormalizationStatus::Unsupported);
        assert!(!first.quality.reordered);

        let RawCapturePoll::Packet(second) = source.poll_next_raw().expect("second raw") else {
            panic!("expected raw packet");
        };
        assert_eq!(second.source_ordinal, 1);
        assert_eq!(second.bytes, unsupported_ipv6);
        assert_eq!(
            second.classification,
            RawPacketClassification::Unsupported(UnsupportedReason::NetworkProtocol)
        );
        assert!(second.quality.reordered);
        assert!(second.quality.clock_rollback);
        assert_eq!(second.monotonic_timestamp_ns, first.monotonic_timestamp_ns);
        assert_eq!(
            source.poll_next_raw().expect("raw end"),
            RawCapturePoll::EndOfStream
        );
    }

    #[test]
    fn raw_and_normalized_replay_cursors_are_independent() {
        let frame = tcp_frame([192, 0, 2, 1], [198, 51, 100, 2], 0x02);
        let mut source =
            PcapReplaySource::from_bytes("dual", &pcap(&[(0, frame)]), limits()).expect("parse");
        assert!(matches!(
            source.poll_next_raw().expect("raw"),
            RawCapturePoll::Packet(RawPacket {
                normalization: NormalizationStatus::Normalized,
                classification: RawPacketClassification::SupportedIp,
                ..
            })
        ));
        assert!(matches!(
            source.poll_next().expect("normalized"),
            CapturePoll::Packet(_)
        ));
    }

    #[test]
    fn paired_replay_preserves_raw_to_normalized_flow_identity() {
        let mut arp = vec![0_u8; 14];
        arp[12..14].copy_from_slice(&0x0806_u16.to_be_bytes());
        let tcp = tcp_frame([192, 0, 2, 1], [198, 51, 100, 2], 0x02);
        let mut source =
            PcapReplaySource::from_bytes("paired", &pcap(&[(0, arp), (1, tcp.clone())]), limits())
                .expect("parse paired capture");

        assert!(matches!(
            source.poll_next_paired().expect("unsupported pair"),
            Some(AdapterOutcome::Unnormalized(_))
        ));
        let Some(AdapterOutcome::Normalized { raw, packet }) =
            source.poll_next_paired().expect("normalized pair")
        else {
            panic!("expected normalized pair");
        };
        assert_eq!(raw.bytes, tcp);
        assert_eq!(raw.source_ordinal, 1);
        assert_eq!(packet.capture_ordinal, 0);
        assert_eq!(
            packet.flow.key.initiator.address,
            IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1))
        );
        assert_eq!(source.poll_next_paired().expect("paired end"), None);
    }

    #[test]
    fn rejects_truncated_ipv6_extension_header() {
        let mut extension = [0_u8; 8];
        extension[0] = 6;
        extension[1] = 1;
        let frame = ipv6_frame(60, &extension);
        let error = PcapReplaySource::from_bytes("bad-ipv6", &pcap(&[(0, frame)]), limits())
            .expect_err("declared extension is truncated");
        assert!(
            error
                .to_string()
                .contains("IPv6 extension header is truncated")
        );
    }

    #[test]
    fn rejects_ether_type_version_mismatch() {
        let mut frame = ipv6_frame(6, &tcp_header(50_000, 443, 0));
        frame[14] = 0x40;
        let error = PcapReplaySource::from_bytes("bad-version", &pcap(&[(0, frame)]), limits())
            .expect_err("version mismatch");
        assert!(
            error
                .to_string()
                .contains("EtherType IPv6 contains a non-IPv6 header")
        );
    }

    #[test]
    fn rejects_truncated_record_instead_of_partial_success() {
        let mut bytes = pcap(&[(0, tcp_frame([1, 1, 1, 1], [2, 2, 2, 2], 0))]);
        bytes.pop();
        assert!(PcapReplaySource::from_bytes("bad", &bytes, limits()).is_err());
    }

    #[test]
    fn parses_little_endian_pcapng_enhanced_packet() {
        let frame = tcp_frame([192, 0, 2, 1], [198, 51, 100, 2], 0x02);
        let mut source =
            PcapReplaySource::from_bytes("pcapng", &pcapng(&frame), limits()).expect("parse");
        let CapturePoll::Packet(packet) = source.poll_next().expect("packet") else {
            panic!("expected packet");
        };
        assert_eq!(packet.capture_ordinal, 0);
        assert_eq!(packet.packet.monotonic_timestamp_ns, 1_000_000_000);
        assert_eq!(source.report().packets_emitted, 1);
        let RawCapturePoll::Packet(raw) = source.poll_next_raw().expect("raw packet") else {
            panic!("expected raw packet");
        };
        assert_eq!(raw.raw_timestamp.ticks, 1_000_000);
        assert_eq!(raw.raw_timestamp.ticks_per_second.get(), 1_000_000);
    }

    #[test]
    fn parses_pktmon_wifi_data_with_llc_snap_network_payload() {
        let replay = PcapReplaySource::from_bytes(
            "pktmon-wifi",
            &pcap(&[(0, pktmon_wifi_tcp_frame())]),
            limits(),
        )
        .expect("parse");
        assert_eq!(replay.report().packets_emitted, 1);
    }

    #[test]
    fn accepts_pktmon_style_zero_pcapng_snapshot_length() {
        let frame = tcp_frame([192, 0, 2, 1], [198, 51, 100, 2], 0x02);
        let mut bytes = pcapng(&frame);
        // Section header is 28 bytes; the IDB snaplen starts 12 bytes into it.
        bytes[40..44].copy_from_slice(&0_u32.to_le_bytes());
        let replay = PcapReplaySource::from_bytes("pktmon-zero-snaplen", &bytes, limits())
            .expect("zero PCAPNG snaplen is an unlimited capture length");
        assert_eq!(replay.report().packets_emitted, 1);
    }

    #[test]
    fn rejects_truncated_pcapng_section() {
        let bytes = [0x0a, 0x0d, 0x0d, 0x0a, 0, 0, 0, 0];
        assert!(PcapReplaySource::from_bytes("pcapng", &bytes, limits()).is_err());
    }

    #[test]
    fn marks_normalized_timestamp_reordering() {
        let frame = tcp_frame([192, 0, 2, 1], [198, 51, 100, 2], 0);
        let mut source = PcapReplaySource::from_bytes(
            "fixture",
            &pcap(&[(2, frame.clone()), (1, frame)]),
            limits(),
        )
        .expect("parse");
        let CapturePoll::Packet(first) = source.poll_next().expect("first") else {
            panic!("packet");
        };
        let CapturePoll::Packet(second) = source.poll_next().expect("second") else {
            panic!("packet");
        };
        assert!(second.quality.reordered);
        assert!(second.quality.clock_rollback);
        assert_eq!(
            first.packet.monotonic_timestamp_ns,
            second.packet.monotonic_timestamp_ns
        );
    }

    #[test]
    fn deterministic_mutation_corpus_never_panics() {
        let base = pcap(&[(0, tcp_frame([192, 0, 2, 1], [198, 51, 100, 2], 0x02))]);
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for case in 0..512_usize {
            let mut mutated = base.clone();
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mutations = usize::try_from(state % 8).expect("value is at most seven") + 1;
            for _ in 0..mutations {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let length = u64::try_from(mutated.len()).expect("fixture length fits u64");
                let index = usize::try_from(state % length).expect("index fits usize");
                let mutation = u8::try_from((state >> 32) & 0xff).expect("masked byte");
                mutated[index] ^= mutation;
            }
            if case % 7 == 0 {
                mutated.truncate(case % mutated.len());
            }
            let outcome = std::panic::catch_unwind(|| {
                PcapReplaySource::from_bytes("mutation", &mutated, limits())
            });
            assert!(outcome.is_ok(), "parser panicked for mutation case {case}");
        }
    }
}

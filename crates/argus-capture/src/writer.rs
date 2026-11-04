//! Bounded standard PCAP file writer for forensic exports.
//!
//! Writes deterministic, standard libpcap format (`LinkType` = Ethernet) with microsecond precision.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

const PCAP_MAGIC_MICROSECONDS: u32 = 0xa1b2_c3d4;
const PCAP_VERSION_MAJOR: u16 = 2;
const PCAP_VERSION_MINOR: u16 = 4;
const LINKTYPE_ETHERNET: u32 = 1;

/// Writes a sequence of captured frames into any [`std::io::Write`] stream.
///
/// # Errors
///
/// Returns [`std::io::Error`] if stream write fails.
///
/// # Panics
///
/// Panics if a timestamp seconds value or frame length exceeds the `u32`
/// record limits of the standard pcap format.
pub fn write_pcap_stream<'a, W, I>(writer: &mut W, packets: I) -> Result<usize, std::io::Error>
where
    W: Write,
    I: IntoIterator<Item = (u64, &'a [u8], usize)>,
{
    // Global Header (24 bytes)
    writer.write_all(&PCAP_MAGIC_MICROSECONDS.to_le_bytes())?;
    writer.write_all(&PCAP_VERSION_MAJOR.to_le_bytes())?;
    writer.write_all(&PCAP_VERSION_MINOR.to_le_bytes())?;
    writer.write_all(&0i32.to_le_bytes())?; // thiszone = 0 (GMT)
    writer.write_all(&0u32.to_le_bytes())?; // sigfigs = 0
    writer.write_all(&65535u32.to_le_bytes())?; // snaplen = 65535
    writer.write_all(&LINKTYPE_ETHERNET.to_le_bytes())?; // network = Ethernet

    let mut count = 0;
    for (timestamp_ns, frame_data, original_wire_len) in packets {
        let seconds =
            u32::try_from(timestamp_ns / 1_000_000_000).expect("timestamp seconds fit u32");
        let microseconds = u32::try_from((timestamp_ns % 1_000_000_000) / 1_000)
            .expect("timestamp microseconds fit u32");
        let incl_len = u32::try_from(frame_data.len()).expect("frame length fits u32");
        let orig_len = u32::try_from(original_wire_len).expect("frame length fits u32");

        writer.write_all(&seconds.to_le_bytes())?;
        writer.write_all(&microseconds.to_le_bytes())?;
        writer.write_all(&incl_len.to_le_bytes())?;
        writer.write_all(&orig_len.to_le_bytes())?;
        writer.write_all(frame_data)?;

        count += 1;
    }

    writer.flush()?;
    Ok(count)
}

/// Writes a sequence of captured frames into a valid standard PCAP file.
///
/// # Errors
///
/// Returns [`std::io::Error`] if file creation or buffered writes fail.
pub fn write_pcap_file<'a, I>(destination: &Path, packets: I) -> Result<usize, std::io::Error>
where
    I: IntoIterator<Item = (u64, &'a [u8], usize)>, // (timestamp_ns, frame_data, original_wire_len)
{
    let file = File::create(destination)?;
    let mut writer = BufWriter::new(file);
    write_pcap_stream(&mut writer, packets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn writes_valid_pcap_header_and_packet_records() {
        let temp_dir = std::env::temp_dir().join(format!("argus_pcap_test_{}", std::process::id()));
        fs::create_dir_all(&temp_dir).expect("temp dir");
        let pcap_path = temp_dir.join("test_out.pcap");

        let dummy_frame = [
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x08, 0x00,
        ];
        let packets = vec![(1_700_000_000_500_000_000u64, &dummy_frame[..], 60usize)];

        let written = write_pcap_file(&pcap_path, packets).expect("write pcap");
        assert_eq!(written, 1);

        let data = fs::read(&pcap_path).expect("read written pcap");
        assert_eq!(data.len(), 24 + 16 + dummy_frame.len());
        // Verify Magic
        assert_eq!(&data[0..4], &0xa1b2_c3d4u32.to_le_bytes());
        // Clean up
        let _ = fs::remove_dir_all(temp_dir);
    }
}

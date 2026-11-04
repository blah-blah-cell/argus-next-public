use std::{
    collections::BTreeMap,
    convert::Infallible,
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{
    CacheVersion, ChunkId, ChunkSnapshot, ChunkSummary, Endpoint, FlowInstanceId, FlowKey,
    TransportProtocol,
};

const RECORD_MAGIC: &[u8; 8] = b"ARGUSNP\0";
const RECORD_FORMAT_VERSION: u16 = 1;
const MAX_RECORD_BYTES: u64 = 512;
const SNAPSHOT_SUFFIX: &str = ".snap";
const BACKUP_SUFFIX: &str = ".bak";
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Boundary for local chunk persistence. Implementations must filter exact versions.
pub trait SnapshotStore {
    type Error;

    /// Appends or replaces one complete chunk snapshot.
    ///
    /// # Errors
    ///
    /// Returns the implementation-specific persistence error.
    fn append(&mut self, snapshot: ChunkSnapshot) -> Result<(), Self::Error>;

    /// Loads snapshots matching the exact flow generation and cache version.
    ///
    /// # Errors
    ///
    /// Returns the implementation-specific persistence error.
    fn load_compatible(
        &self,
        flow: &FlowInstanceId,
        version: CacheVersion,
    ) -> Result<Vec<ChunkSnapshot>, Self::Error>;

    /// Removes every snapshot belonging to one flow instance.
    ///
    /// # Errors
    ///
    /// Returns the implementation-specific persistence error.
    fn invalidate_flow(&mut self, flow: &FlowInstanceId) -> Result<usize, Self::Error>;
}

/// Deterministic reference implementation used before the durable Phase 3 store.
#[derive(Clone, Debug, Default)]
pub struct InMemorySnapshotStore {
    snapshots: BTreeMap<ChunkId, ChunkSnapshot>,
}

impl SnapshotStore for InMemorySnapshotStore {
    type Error = Infallible;

    fn append(&mut self, snapshot: ChunkSnapshot) -> Result<(), Self::Error> {
        self.snapshots.insert(snapshot.id.clone(), snapshot);
        Ok(())
    }

    fn load_compatible(
        &self,
        flow: &FlowInstanceId,
        version: CacheVersion,
    ) -> Result<Vec<ChunkSnapshot>, Self::Error> {
        Ok(self
            .snapshots
            .values()
            .filter(|snapshot| snapshot.id.flow == *flow && snapshot.id.version == version)
            .cloned()
            .collect())
    }

    fn invalidate_flow(&mut self, flow: &FlowInstanceId) -> Result<usize, Self::Error> {
        let before = self.snapshots.len();
        self.snapshots.retain(|id, _| id.flow != *flow);
        Ok(before - self.snapshots.len())
    }
}

/// A crash-tolerant, local snapshot directory using only the standard library.
///
/// Each record is independently checksummed and atomically published from a synced temporary
/// file. A backup is retained across replacement so [`Self::open`] can recover if a process stops
/// between the two Windows-compatible rename operations.
#[derive(Debug)]
pub struct DurableSnapshotStore {
    directory: PathBuf,
}

impl DurableSnapshotStore {
    /// Opens (or creates) a snapshot directory and repairs interrupted replacements.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the directory cannot be created, inspected, or repaired.
    pub fn open(directory: impl AsRef<Path>) -> Result<Self, SnapshotStoreError> {
        let directory = directory.as_ref().to_path_buf();
        fs::create_dir_all(&directory)?;
        let store = Self { directory };
        store.recover_interrupted_writes()?;
        Ok(store)
    }

    /// Returns the directory containing the durable records.
    #[must_use]
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    fn recover_interrupted_writes(&self) -> Result<(), SnapshotStoreError> {
        let mut entries = fs::read_dir(&self.directory)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(fs::DirEntry::file_name);

        for entry in &entries {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(".argus-") && name.ends_with(".tmp") {
                remove_file_if_present(&entry.path())?;
            }
        }

        for entry in entries {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let Some(target_name) = name.strip_suffix(BACKUP_SUFFIX) else {
                continue;
            };
            if !is_snapshot_name(target_name) {
                continue;
            }
            let target = self.directory.join(target_name);
            if target.exists() {
                remove_file_if_present(&entry.path())?;
            } else {
                fs::rename(entry.path(), target)?;
            }
        }
        Ok(())
    }

    fn publish(&self, target: &Path, bytes: &[u8]) -> Result<(), SnapshotStoreError> {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary = self
            .directory
            .join(format!(".argus-{}-{sequence}.tmp", std::process::id()));
        let backup = path_with_extra_suffix(target, BACKUP_SUFFIX);

        let result = (|| -> Result<(), SnapshotStoreError> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            file.write_all(bytes)?;
            file.flush()?;
            file.sync_all()?;
            drop(file);

            remove_file_if_present(&backup)?;
            if target.exists() {
                fs::rename(target, &backup)?;
            }
            if let Err(error) = fs::rename(&temporary, target) {
                if backup.exists() && !target.exists() {
                    let _ = fs::rename(&backup, target);
                }
                return Err(error.into());
            }
            remove_file_if_present(&backup)?;
            sync_directory(&self.directory)?;
            Ok(())
        })();

        if result.is_err() {
            let _ = remove_file_if_present(&temporary);
        }
        result
    }
}

impl SnapshotStore for DurableSnapshotStore {
    type Error = SnapshotStoreError;

    fn append(&mut self, snapshot: ChunkSnapshot) -> Result<(), Self::Error> {
        validate_snapshot(&snapshot)?;
        let target = self.directory.join(snapshot_file_name(&snapshot.id));
        self.publish(&target, &encode_record(&snapshot))
    }

    fn load_compatible(
        &self,
        flow: &FlowInstanceId,
        version: CacheVersion,
    ) -> Result<Vec<ChunkSnapshot>, Self::Error> {
        let mut snapshots = BTreeMap::new();
        let mut entries = fs::read_dir(&self.directory)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(fs::DirEntry::file_name);
        for entry in entries {
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if !is_snapshot_name(&name) || entry.file_type()?.is_dir() {
                continue;
            }
            let Some(snapshot) = read_valid_record(&entry.path())? else {
                continue;
            };
            if snapshot_file_name(&snapshot.id) != name
                || snapshot.id.flow != *flow
                || snapshot.id.version != version
            {
                continue;
            }
            snapshots.insert(snapshot.id.clone(), snapshot);
        }
        Ok(snapshots.into_values().collect())
    }

    fn invalidate_flow(&mut self, flow: &FlowInstanceId) -> Result<usize, Self::Error> {
        let mut removed = 0;
        for entry in fs::read_dir(&self.directory)? {
            let entry = entry?;
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            let Some(id) = chunk_id_from_file_name(&name) else {
                continue;
            };
            if id.flow == *flow {
                remove_file_if_present(&entry.path())?;
                removed += 1;
            }
        }
        sync_directory(&self.directory)?;
        Ok(removed)
    }
}

/// Failure returned by [`DurableSnapshotStore`].
#[derive(Debug)]
pub enum SnapshotStoreError {
    Io(io::Error),
    InvalidSnapshot(&'static str),
}

impl fmt::Display for SnapshotStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "snapshot store I/O error: {error}"),
            Self::InvalidSnapshot(reason) => write!(formatter, "invalid snapshot: {reason}"),
        }
    }
}

impl std::error::Error for SnapshotStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::InvalidSnapshot(_) => None,
        }
    }
}

impl From<io::Error> for SnapshotStoreError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

fn validate_snapshot(snapshot: &ChunkSnapshot) -> Result<(), SnapshotStoreError> {
    let summary = &snapshot.summary;
    let directional = summary
        .initiator_packets
        .checked_add(summary.responder_packets);
    if summary.packet_count == 0 {
        return Err(SnapshotStoreError::InvalidSnapshot(
            "packet count must be nonzero",
        ));
    }
    if directional != Some(summary.packet_count) {
        return Err(SnapshotStoreError::InvalidSnapshot(
            "directional packet counts do not equal packet count",
        ));
    }
    if summary.captured_bytes > summary.original_bytes {
        return Err(SnapshotStoreError::InvalidSnapshot(
            "captured bytes exceed original bytes",
        ));
    }
    if summary.first_timestamp_ns > summary.last_timestamp_ns {
        return Err(SnapshotStoreError::InvalidSnapshot(
            "first timestamp follows last timestamp",
        ));
    }
    Ok(())
}

fn encode_record(snapshot: &ChunkSnapshot) -> Vec<u8> {
    let mut payload = encode_chunk_id(&snapshot.id);
    for value in [
        snapshot.summary.packet_count,
        snapshot.summary.captured_bytes,
        snapshot.summary.original_bytes,
        snapshot.summary.initiator_packets,
        snapshot.summary.responder_packets,
        snapshot.summary.first_timestamp_ns,
        snapshot.summary.last_timestamp_ns,
    ] {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    let mut record = Vec::with_capacity(8 + 2 + 4 + payload.len() + 4);
    record.extend_from_slice(RECORD_MAGIC);
    record.extend_from_slice(&RECORD_FORMAT_VERSION.to_le_bytes());
    let payload_len =
        u32::try_from(payload.len()).expect("snapshot payload has a fixed small size");
    record.extend_from_slice(&payload_len.to_le_bytes());
    record.extend_from_slice(&payload);
    record.extend_from_slice(&crc32(&record).to_le_bytes());
    record
}

fn read_valid_record(path: &Path) -> Result<Option<ChunkSnapshot>, SnapshotStoreError> {
    let metadata = fs::metadata(path)?;
    if metadata.len() > MAX_RECORD_BYTES {
        return Ok(None);
    }
    let Ok(capacity) = usize::try_from(metadata.len()) else {
        return Ok(None);
    };
    let mut bytes = Vec::with_capacity(capacity);
    File::open(path)?.read_to_end(&mut bytes)?;
    Ok(decode_record(&bytes))
}

fn decode_record(bytes: &[u8]) -> Option<ChunkSnapshot> {
    if bytes.len() < 18 || bytes.get(..8)? != RECORD_MAGIC {
        return None;
    }
    let format = u16::from_le_bytes(bytes.get(8..10)?.try_into().ok()?);
    if format != RECORD_FORMAT_VERSION {
        return None;
    }
    let payload_len = u32::from_le_bytes(bytes.get(10..14)?.try_into().ok()?) as usize;
    let checksum_offset = 14_usize.checked_add(payload_len)?;
    if bytes.len() != checksum_offset.checked_add(4)? {
        return None;
    }
    let expected = u32::from_le_bytes(bytes.get(checksum_offset..)?.try_into().ok()?);
    if crc32(bytes.get(..checksum_offset)?) != expected {
        return None;
    }
    let mut cursor = Cursor::new(bytes.get(14..checksum_offset)?);
    let id = decode_chunk_id(&mut cursor)?;
    let summary = ChunkSummary {
        packet_count: cursor.u64()?,
        captured_bytes: cursor.u64()?,
        original_bytes: cursor.u64()?,
        initiator_packets: cursor.u64()?,
        responder_packets: cursor.u64()?,
        first_timestamp_ns: cursor.u64()?,
        last_timestamp_ns: cursor.u64()?,
    };
    if !cursor.remaining().is_empty() {
        return None;
    }
    let snapshot = ChunkSnapshot { id, summary };
    validate_snapshot(&snapshot).ok()?;
    Some(snapshot)
}

fn encode_chunk_id(id: &ChunkId) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(80);
    encode_endpoint(&mut bytes, &id.flow.key.initiator);
    encode_endpoint(&mut bytes, &id.flow.key.responder);
    match id.flow.key.protocol {
        TransportProtocol::Tcp => bytes.extend_from_slice(&[0, 0]),
        TransportProtocol::Udp => bytes.extend_from_slice(&[1, 0]),
        TransportProtocol::Other(value) => bytes.extend_from_slice(&[2, value]),
    }
    for value in [
        id.flow.generation,
        id.version.feature_schema,
        id.version.model,
        id.index,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn encode_endpoint(bytes: &mut Vec<u8>, endpoint: &Endpoint) {
    match endpoint.address {
        IpAddr::V4(address) => {
            bytes.push(4);
            bytes.extend_from_slice(&address.octets());
        }
        IpAddr::V6(address) => {
            bytes.push(6);
            bytes.extend_from_slice(&address.octets());
        }
    }
    bytes.extend_from_slice(&endpoint.port.to_le_bytes());
}

fn decode_chunk_id(cursor: &mut Cursor<'_>) -> Option<ChunkId> {
    let initiator = decode_endpoint(cursor)?;
    let responder = decode_endpoint(cursor)?;
    let protocol = match (cursor.u8()?, cursor.u8()?) {
        (0, 0) => TransportProtocol::Tcp,
        (1, 0) => TransportProtocol::Udp,
        (2, value) => TransportProtocol::Other(value),
        _ => return None,
    };
    Some(ChunkId {
        flow: FlowInstanceId {
            key: FlowKey {
                initiator,
                responder,
                protocol,
            },
            generation: cursor.u64()?,
        },
        version: CacheVersion {
            feature_schema: cursor.u64()?,
            model: cursor.u64()?,
        },
        index: cursor.u64()?,
    })
}

fn decode_endpoint(cursor: &mut Cursor<'_>) -> Option<Endpoint> {
    let address = match cursor.u8()? {
        4 => IpAddr::V4(Ipv4Addr::from(cursor.array::<4>()?)),
        6 => IpAddr::V6(Ipv6Addr::from(cursor.array::<16>()?)),
        _ => return None,
    };
    Some(Endpoint {
        address,
        port: cursor.u16()?,
    })
}

struct Cursor<'a> {
    bytes: &'a [u8],
}

impl<'a> Cursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    const fn remaining(&self) -> &'a [u8] {
        self.bytes
    }

    fn array<const N: usize>(&mut self) -> Option<[u8; N]> {
        let (value, rest) = self.bytes.split_at_checked(N)?;
        self.bytes = rest;
        value.try_into().ok()
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.array::<1>()?[0])
    }

    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.array()?))
    }
}

fn snapshot_file_name(id: &ChunkId) -> String {
    let encoded = encode_chunk_id(id);
    let mut name = String::with_capacity(encoded.len() * 2 + SNAPSHOT_SUFFIX.len());
    for byte in encoded {
        use fmt::Write as _;
        write!(name, "{byte:02x}").expect("writing to String cannot fail");
    }
    name.push_str(SNAPSHOT_SUFFIX);
    name
}

fn chunk_id_from_file_name(name: &str) -> Option<ChunkId> {
    let encoded = name.strip_suffix(SNAPSHOT_SUFFIX)?;
    if encoded.is_empty() || encoded.len() % 2 != 0 || !encoded.is_ascii() {
        return None;
    }
    let mut bytes = Vec::with_capacity(encoded.len() / 2);
    for pair in encoded.as_bytes().chunks_exact(2) {
        let pair = std::str::from_utf8(pair).ok()?;
        bytes.push(u8::from_str_radix(pair, 16).ok()?);
    }
    let mut cursor = Cursor::new(&bytes);
    let id = decode_chunk_id(&mut cursor)?;
    cursor.remaining().is_empty().then_some(id)
}

fn is_snapshot_name(name: &str) -> bool {
    chunk_id_from_file_name(name).is_some()
}

fn path_with_extra_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_owned();
    value.push(suffix);
    PathBuf::from(value)
}

fn remove_file_if_present(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(unix)]
fn sync_directory(directory: &Path) -> io::Result<()> {
    File::open(directory)?.sync_all()
}

#[cfg(not(unix))]
#[allow(clippy::unnecessary_wraps)]
fn sync_directory(_directory: &Path) -> io::Result<()> {
    // Stable std does not expose Windows directory handles with backup semantics. Record contents
    // are still sync_all'd before rename, and the backup protocol makes either name recoverable.
    Ok(())
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & 0_u32.wrapping_sub(crc & 1));
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChunkSummary, Endpoint, FlowKey, TransportProtocol};
    use std::{
        net::IpAddr,
        sync::atomic::{AtomicU64, Ordering},
    };

    static TEST_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn unique(name: &str) -> Self {
            let sequence = TEST_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "argus-core-{name}-{}-{sequence}",
                std::process::id()
            ));
            if path.exists() {
                fs::remove_dir_all(&path).expect("remove stale test directory");
            }
            fs::create_dir(&path).expect("create test directory");
            Self(path)
        }
    }

    impl AsRef<Path> for TestDirectory {
        fn as_ref(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn flow(generation: u64) -> FlowInstanceId {
        FlowInstanceId {
            key: FlowKey {
                initiator: Endpoint {
                    address: "127.0.0.1".parse::<IpAddr>().expect("valid IP"),
                    port: 123,
                },
                responder: Endpoint {
                    address: "127.0.0.2".parse::<IpAddr>().expect("valid IP"),
                    port: 456,
                },
                protocol: TransportProtocol::Udp,
            },
            generation,
        }
    }

    fn snapshot(flow: FlowInstanceId, version: CacheVersion) -> ChunkSnapshot {
        ChunkSnapshot {
            id: ChunkId {
                flow,
                version,
                index: 0,
            },
            summary: ChunkSummary {
                packet_count: 1,
                captured_bytes: 60,
                original_bytes: 64,
                initiator_packets: 1,
                responder_packets: 0,
                first_timestamp_ns: 1,
                last_timestamp_ns: 1,
            },
        }
    }

    #[test]
    fn load_requires_exact_flow_generation_and_cache_version() {
        let version = CacheVersion {
            feature_schema: 1,
            model: 2,
        };
        let mut store = InMemorySnapshotStore::default();
        store.append(snapshot(flow(0), version)).expect("append");

        assert_eq!(
            store
                .load_compatible(&flow(0), version)
                .expect("load")
                .len(),
            1
        );
        assert!(
            store
                .load_compatible(
                    &flow(1),
                    CacheVersion {
                        feature_schema: 1,
                        model: 2,
                    }
                )
                .expect("load")
                .is_empty()
        );
        assert!(
            store
                .load_compatible(
                    &flow(0),
                    CacheVersion {
                        feature_schema: 1,
                        model: 3,
                    }
                )
                .expect("load")
                .is_empty()
        );
    }

    #[test]
    fn durable_store_survives_restart_and_orders_chunks() {
        let directory = TestDirectory::unique("restart");
        let version = CacheVersion {
            feature_schema: 3,
            model: 7,
        };
        let mut second = snapshot(flow(4), version);
        second.id.index = 2;
        let mut first = second.clone();
        first.id.index = 1;

        {
            let mut store = DurableSnapshotStore::open(&directory).expect("open");
            store.append(second.clone()).expect("append second");
            store.append(first.clone()).expect("append first");
        }

        let store = DurableSnapshotStore::open(&directory).expect("reopen");
        assert_eq!(
            store
                .load_compatible(&flow(4), version)
                .expect("load after restart"),
            vec![first, second]
        );
    }

    #[test]
    fn corrupt_partial_and_incompatible_records_are_never_loaded() {
        let directory = TestDirectory::unique("corruption");
        let version = CacheVersion {
            feature_schema: 1,
            model: 9,
        };
        let wanted = snapshot(flow(0), version);
        let incompatible_flow = snapshot(flow(1), version);
        let incompatible_version = snapshot(
            flow(0),
            CacheVersion {
                feature_schema: 1,
                model: 10,
            },
        );
        let mut corrupt = wanted.clone();
        corrupt.id.index = 8;
        let mut partial = wanted.clone();
        partial.id.index = 9;

        let mut store = DurableSnapshotStore::open(&directory).expect("open");
        for value in [
            wanted.clone(),
            incompatible_flow,
            incompatible_version,
            corrupt.clone(),
            partial.clone(),
        ] {
            store.append(value).expect("append");
        }
        let corrupt_path = directory.as_ref().join(snapshot_file_name(&corrupt.id));
        let mut bytes = fs::read(&corrupt_path).expect("read corrupt candidate");
        let payload_byte = bytes.get_mut(20).expect("record payload");
        *payload_byte ^= 0x80;
        fs::write(corrupt_path, bytes).expect("corrupt checksum");
        let partial_path = directory.as_ref().join(snapshot_file_name(&partial.id));
        let bytes = fs::read(&partial_path).expect("read partial candidate");
        fs::write(partial_path, &bytes[..bytes.len() / 2]).expect("truncate record");

        assert_eq!(
            store
                .load_compatible(&flow(0), version)
                .expect("load valid records only"),
            vec![wanted]
        );
    }

    #[test]
    fn open_cleans_abandoned_temp_and_restores_backup() {
        let directory = TestDirectory::unique("crash");
        let version = CacheVersion {
            feature_schema: 5,
            model: 6,
        };
        let value = snapshot(flow(2), version);
        let target = directory.as_ref().join(snapshot_file_name(&value.id));
        let backup = path_with_extra_suffix(&target, BACKUP_SUFFIX);
        fs::write(&backup, encode_record(&value)).expect("write simulated backup");
        let abandoned = directory.as_ref().join(".argus-999-1.tmp");
        fs::write(&abandoned, b"partial").expect("write simulated temporary file");

        let store = DurableSnapshotStore::open(&directory).expect("recover");
        assert!(!backup.exists());
        assert!(!abandoned.exists());
        assert_eq!(
            store
                .load_compatible(&flow(2), version)
                .expect("load recovered backup"),
            vec![value]
        );
    }

    #[test]
    fn replacement_is_durable_and_invalid_snapshots_are_rejected() {
        let directory = TestDirectory::unique("replace");
        let version = CacheVersion {
            feature_schema: 2,
            model: 3,
        };
        let mut original = snapshot(flow(3), version);
        let mut replacement = original.clone();
        replacement.summary.captured_bytes = 61;
        replacement.summary.original_bytes = 65;

        let mut store = DurableSnapshotStore::open(&directory).expect("open");
        store.append(original.clone()).expect("append original");
        store
            .append(replacement.clone())
            .expect("append replacement");
        original.summary.packet_count = 0;
        assert!(matches!(
            store.append(original),
            Err(SnapshotStoreError::InvalidSnapshot(_))
        ));

        drop(store);
        let store = DurableSnapshotStore::open(&directory).expect("reopen");
        assert_eq!(
            store
                .load_compatible(&flow(3), version)
                .expect("load replacement"),
            vec![replacement]
        );
    }

    #[test]
    fn filenames_are_deterministic_path_safe_and_drive_invalidation() {
        let directory = TestDirectory::unique("names");
        let version = CacheVersion {
            feature_schema: u64::MAX,
            model: 42,
        };
        let value = snapshot(flow(7), version);
        let name = snapshot_file_name(&value.id);
        assert!(name.ends_with(SNAPSHOT_SUFFIX));
        assert!(
            name.strip_suffix(SNAPSHOT_SUFFIX)
                .expect("suffix")
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        );
        assert_eq!(chunk_id_from_file_name(&name), Some(value.id.clone()));

        let mut store = DurableSnapshotStore::open(&directory).expect("open");
        store.append(value).expect("append");
        assert_eq!(store.invalidate_flow(&flow(7)).expect("invalidate"), 1);
        assert!(
            store
                .load_compatible(&flow(7), version)
                .expect("load")
                .is_empty()
        );
    }

    #[test]
    fn encoding_is_deterministic_and_detects_checksum_damage() {
        let version = CacheVersion {
            feature_schema: 11,
            model: 12,
        };
        let value = snapshot(flow(8), version);
        let first = encode_record(&value);
        let second = encode_record(&value);
        assert_eq!(first, second);
        assert_eq!(decode_record(&first), Some(value));

        let mut damaged = first;
        *damaged.last_mut().expect("checksum") ^= 1;
        assert_eq!(decode_record(&damaged), None);
    }
}

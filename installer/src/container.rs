//! Appended payload format for the setup executable.
//!
//! Layout: `[stub bytes][entry ...][footer]`, where each entry is
//! `u16 path length | path | u64 size | 32-byte SHA-256 | data` and the footer is
//! `"ARGUSPAY1\0" | u64 payload offset | u32 entry count`.

use crate::sha256::{hex, Sha256};
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const FOOTER_MAGIC: [u8; 10] = *b"ARGUSPAY1\0";
pub const FOOTER_LEN: u64 = 22;
const COPY_BUFFER: usize = 64 * 1024;
const MAX_PATH_BYTES: usize = 1024;
const MAX_ENTRIES: u32 = 65_536;
const MAX_PREALLOC_ENTRIES: usize = 4_096;
/// Minimum encoded size of an entry header: u16 path_len (2) + >=1 path byte + u64 size (8) + 32-byte hash.
const MIN_ENTRY_HEADER_BYTES: u64 = 43;

#[derive(Clone, Debug)]
pub struct Entry {
    pub path: String,
    pub size: u64,
    pub hash: [u8; 32],
    pub offset: u64,
}

pub struct Package {
    file: File,
    start: u64,
    pub entries: Vec<Entry>,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn read_exact_at(file: &mut File, offset: u64, buffer: &mut [u8]) -> io::Result<()> {
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buffer)
}

impl Package {
    /// Opens the running executable and parses its appended payload.
    ///
    /// # Errors
    ///
    /// Returns an error when the footer is missing, malformed, or points outside
    /// the executable.
    pub fn open(executable: &Path) -> io::Result<Self> {
        let mut file = File::open(executable)?;
        let length = file.metadata()?.len();
        if length < FOOTER_LEN {
            return Err(invalid(
                "this executable has no installer payload; build it with argus-setup-builder",
            ));
        }

        let mut footer = [0_u8; FOOTER_LEN as usize];
        read_exact_at(&mut file, length - FOOTER_LEN, &mut footer)?;
        if footer[..10] != FOOTER_MAGIC {
            return Err(invalid(
                "this executable has no installer payload; build it with argus-setup-builder",
            ));
        }

        let start = u64::from_le_bytes(footer[10..18].try_into().expect("footer slice"));
        let count = u32::from_le_bytes(footer[18..22].try_into().expect("footer slice"));
        let payload_end = length
            .checked_sub(FOOTER_LEN)
            .ok_or_else(|| invalid("installer payload length is shorter than footer"))?;
        if start == 0 || start.checked_add(FOOTER_LEN).is_none_or(|end| end > length) {
            return Err(invalid("installer payload offset is out of range"));
        }
        let max_entries_in_bytes = (payload_end - start) / MIN_ENTRY_HEADER_BYTES;
        if count > MAX_ENTRIES || u64::from(count) > max_entries_in_bytes {
            return Err(invalid("installer payload entry count is out of range"));
        }

        let mut entries = Vec::with_capacity((count as usize).min(MAX_PREALLOC_ENTRIES));
        let mut position = start;
        for _ in 0..count {
            let next_after_len = position
                .checked_add(2)
                .ok_or_else(|| invalid("installer payload offset overflow"))?;
            if next_after_len > payload_end {
                return Err(invalid(
                    "installer payload entry header extends past the archive",
                ));
            }
            let mut path_length = [0_u8; 2];
            read_exact_at(&mut file, position, &mut path_length)?;
            let path_length = u16::from_le_bytes(path_length) as usize;
            if path_length == 0 || path_length > MAX_PATH_BYTES {
                return Err(invalid("installer payload contains an invalid path length"));
            }
            position = next_after_len;

            let next_after_path = position
                .checked_add(path_length as u64)
                .ok_or_else(|| invalid("installer payload path offset overflow"))?;
            if next_after_path > payload_end {
                return Err(invalid(
                    "installer payload entry path extends past the archive",
                ));
            }
            let mut path_bytes = vec![0_u8; path_length];
            read_exact_at(&mut file, position, &mut path_bytes)?;
            position = next_after_path;
            let path = String::from_utf8(path_bytes)
                .map_err(|_| invalid("installer payload contains a non-UTF-8 path"))?;

            let next_after_size = position
                .checked_add(8)
                .ok_or_else(|| invalid("installer payload size offset overflow"))?;
            if next_after_size > payload_end {
                return Err(invalid(
                    "installer payload entry size extends past the archive",
                ));
            }
            let mut size_bytes = [0_u8; 8];
            read_exact_at(&mut file, position, &mut size_bytes)?;
            position = next_after_size;
            let size = u64::from_le_bytes(size_bytes);

            let next_after_hash = position
                .checked_add(32)
                .ok_or_else(|| invalid("installer payload hash offset overflow"))?;
            if next_after_hash > payload_end {
                return Err(invalid(
                    "installer payload entry hash extends past the archive",
                ));
            }
            let mut hash = [0_u8; 32];
            read_exact_at(&mut file, position, &mut hash)?;
            position = next_after_hash;

            let entry_end = position
                .checked_add(size)
                .ok_or_else(|| invalid(format!("installer payload entry {path} size overflows")))?;
            if entry_end > payload_end {
                return Err(invalid(format!(
                    "installer payload entry {path} extends past the archive"
                )));
            }
            entries.push(Entry {
                path,
                size,
                hash,
                offset: position,
            });
            position = entry_end;
        }

        Ok(Self {
            file,
            start,
            entries,
        })
    }

    /// Byte length of the stub executable before the appended payload.
    ///
    /// Copying exactly this prefix produces a payload-free uninstaller.
    #[must_use]
    pub const fn stub_length(&self) -> u64 {
        self.start
    }

    /// Re-hashes every entry and confirms it matches the recorded digest.
    ///
    /// # Errors
    ///
    /// Returns an error on the first entry whose bytes do not match.
    pub fn verify(&mut self) -> io::Result<()> {
        let mut buffer = vec![0_u8; COPY_BUFFER];
        for entry in &self.entries {
            self.file.seek(SeekFrom::Start(entry.offset))?;
            let mut remaining = entry.size;
            let mut hasher = Sha256::new();
            while remaining > 0 {
                let take = usize::try_from(remaining.min(COPY_BUFFER as u64))
                    .map_err(|_| invalid("payload entry is too large to verify"))?;
                self.file.read_exact(&mut buffer[..take])?;
                hasher.update(&buffer[..take]);
                remaining -= take as u64;
            }
            if hasher.finish() != entry.hash {
                return Err(invalid(format!(
                    "installer payload failed verification: {}",
                    entry.path
                )));
            }
        }
        Ok(())
    }

    /// Returns the UTF-8 content of one payload entry, if it exists.
    ///
    /// # Errors
    ///
    /// Returns an error when the entry cannot be read or is not valid UTF-8.
    pub fn read_text(&mut self, name: &str) -> io::Result<Option<String>> {
        let Some(entry) = self
            .entries
            .iter()
            .find(|entry| entry.path == name)
            .cloned()
        else {
            return Ok(None);
        };
        let size = usize::try_from(entry.size)
            .map_err(|_| invalid("payload text entry is too large to read"))?;
        let mut bytes = vec![0_u8; size];
        read_exact_at(&mut self.file, entry.offset, &mut bytes)?;
        String::from_utf8(bytes)
            .map(Some)
            .map_err(|_| invalid("payload text entry is not valid UTF-8"))
    }

    /// Extracts every entry, re-hashing as it writes.
    ///
    /// # Errors
    ///
    /// Returns an error when a path is rejected, a directory cannot be created,
    /// or written bytes do not match the recorded digest.
    pub fn extract<F>(&mut self, mut resolve: F) -> io::Result<()>
    where
        F: FnMut(&str) -> io::Result<PathBuf>,
    {
        let mut buffer = vec![0_u8; COPY_BUFFER];
        for entry in self.entries.clone() {
            let target = resolve(&entry.path)?;
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            self.file.seek(SeekFrom::Start(entry.offset))?;
            let mut output = File::create(&target)?;
            let mut remaining = entry.size;
            let mut hasher = Sha256::new();
            while remaining > 0 {
                let take = usize::try_from(remaining.min(COPY_BUFFER as u64))
                    .map_err(|_| invalid("payload entry is too large to extract"))?;
                self.file.read_exact(&mut buffer[..take])?;
                hasher.update(&buffer[..take]);
                output.write_all(&buffer[..take])?;
                remaining -= take as u64;
            }
            output.flush()?;
            if hasher.finish() != entry.hash {
                return Err(invalid(format!(
                    "written file failed verification: {}",
                    entry.path
                )));
            }
        }
        Ok(())
    }
}

/// Rejects absolute paths, parent traversal, drive letters, and empty components.
///
/// # Errors
///
/// Returns an error for any path that could escape the install directory.
pub fn sanitize_relative(path: &str) -> io::Result<PathBuf> {
    if path.is_empty() {
        return Err(invalid("installer payload contains an empty path"));
    }
    let normalized = path.replace('\\', "/");
    if normalized.starts_with('/') || normalized.contains(':') {
        return Err(invalid(format!(
            "installer payload contains an absolute path: {path}"
        )));
    }
    let mut relative = PathBuf::new();
    for part in normalized.split('/') {
        if part.is_empty() || part == "." || part == ".." {
            return Err(invalid(format!(
                "installer payload contains an unsafe path: {path}"
            )));
        }
        relative.push(part);
    }
    Ok(relative)
}

fn collect_files(directory: &Path, output: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            collect_files(&entry.path(), output)?;
        } else if file_type.is_file() {
            output.push(entry.path());
        }
    }
    Ok(())
}

fn write_entry<W: Write>(writer: &mut W, path: &str, data: &[u8]) -> io::Result<()> {
    let path_bytes = path.as_bytes();
    let length = u16::try_from(path_bytes.len())
        .map_err(|_| invalid(format!("payload path is too long: {path}")))?;
    writer.write_all(&length.to_le_bytes())?;
    writer.write_all(path_bytes)?;
    writer.write_all(&(data.len() as u64).to_le_bytes())?;
    writer.write_all(&Sha256::digest(data))?;
    writer.write_all(data)?;
    Ok(())
}

/// Builds the setup executable from a stub plus every file in `package_directory`.
///
/// `SHA256SUMS.txt` is skipped because the installer carries its own digests and
/// writes an `INSTALL.txt` record of what it installed.
///
/// # Errors
///
/// Returns an error when the stub or package cannot be read or the output cannot
/// be written.
pub fn build_payload(
    package_directory: &Path,
    version: &str,
    stub: &Path,
    output: &Path,
) -> io::Result<u64> {
    let stub_bytes = fs::read(stub)?;
    let mut files = Vec::new();
    collect_files(package_directory, &mut files)?;
    files.sort();

    let mut writer = io::BufWriter::new(File::create(output)?);
    writer.write_all(&stub_bytes)?;
    let offset = stub_bytes.len() as u64;
    let mut count = 0_u32;

    for file in &files {
        let relative = file
            .strip_prefix(package_directory)
            .map_err(|_| invalid("package file is outside the package directory"))?
            .to_string_lossy()
            .replace('\\', "/");
        if relative == "SHA256SUMS.txt" || relative == "VERSION.txt" {
            continue;
        }
        let data = fs::read(file)?;
        write_entry(&mut writer, &relative, &data)?;
        count += 1;
    }

    write_entry(
        &mut writer,
        "VERSION.txt",
        format!("{version}\n").as_bytes(),
    )?;
    count += 1;

    writer.write_all(&FOOTER_MAGIC)?;
    writer.write_all(&offset.to_le_bytes())?;
    writer.write_all(&count.to_le_bytes())?;
    writer.flush()?;
    Ok(u64::from(count))
}

/// Renders a payload entry name and digest for install records.
#[must_use]
pub fn describe(entry: &Entry) -> String {
    format!("{}  {}", hex(&entry.hash), entry.path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_file(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!(
            "argus-container-{label}-{}-{nanos}.bin",
            std::process::id()
        ))
    }

    #[test]
    fn open_rejects_u64_max_footer_start_overflow() {
        let path = unique_temp_file("start-overflow");
        let mut bytes = vec![0_u8; 64];
        bytes.extend_from_slice(&FOOTER_MAGIC);
        bytes.extend_from_slice(&(u64::MAX - 5).to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        fs::write(&path, &bytes).unwrap();

        let err = Package::open(&path)
            .err()
            .expect("must reject overflowing footer start offset");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn open_rejects_u32_max_entry_count_without_oom() {
        let path = unique_temp_file("count-overflow");
        let mut bytes = vec![0_u8; 64];
        bytes.extend_from_slice(&FOOTER_MAGIC);
        bytes.extend_from_slice(&16_u64.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        fs::write(&path, &bytes).unwrap();

        let err = Package::open(&path)
            .err()
            .expect("must reject absurd entry count before allocating");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn open_rejects_u64_max_entry_size_overflow() {
        let path = unique_temp_file("size-overflow");
        let mut bytes = vec![0_u8; 16]; // fake stub
        let start = bytes.len() as u64;
        // Entry header: path "a" (len 1), size = u64::MAX - 10, 32-byte zero hash
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.push(b'a');
        bytes.extend_from_slice(&(u64::MAX - 10).to_le_bytes());
        bytes.extend_from_slice(&[0_u8; 32]);
        // Footer
        bytes.extend_from_slice(&FOOTER_MAGIC);
        bytes.extend_from_slice(&start.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        fs::write(&path, &bytes).unwrap();

        let err = Package::open(&path)
            .err()
            .expect("must reject overflowing entry size");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let _ = fs::remove_file(&path);
    }
}

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
        if start == 0 || start + FOOTER_LEN > length {
            return Err(invalid("installer payload offset is out of range"));
        }

        let mut entries = Vec::with_capacity(count as usize);
        let mut position = start;
        for _ in 0..count {
            let mut path_length = [0_u8; 2];
            read_exact_at(&mut file, position, &mut path_length)?;
            let path_length = u16::from_le_bytes(path_length) as usize;
            if path_length == 0 || path_length > MAX_PATH_BYTES {
                return Err(invalid("installer payload contains an invalid path length"));
            }
            position += 2;

            let mut path_bytes = vec![0_u8; path_length];
            read_exact_at(&mut file, position, &mut path_bytes)?;
            position += path_length as u64;
            let path = String::from_utf8(path_bytes)
                .map_err(|_| invalid("installer payload contains a non-UTF-8 path"))?;

            let mut size_bytes = [0_u8; 8];
            read_exact_at(&mut file, position, &mut size_bytes)?;
            position += 8;
            let size = u64::from_le_bytes(size_bytes);

            let mut hash = [0_u8; 32];
            read_exact_at(&mut file, position, &mut hash)?;
            position += 32;

            if position + size > length - FOOTER_LEN {
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
            position += size;
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

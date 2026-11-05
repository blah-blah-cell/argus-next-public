//! Conservative, non-recursive removal of files recorded by this installer.

use crate::container::sanitize_relative;
use crate::sha256::{hex, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const RECORD: &str = "ARGUS-INSTALL.txt";
const MAGIC: &str = "ARGUS-NEXT-PUBLIC-INSTALL-V1";
const FOOTER: &[u8; 8] = b"ARGUSOWN";

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn reject_links(path: &Path) -> io::Result<()> {
    for component in path.ancestors() {
        let metadata = fs::symlink_metadata(component)?;
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            if metadata.file_attributes() & 0x400 != 0 {
                return Err(invalid("install path contains a reparse point"));
            }
        }
        if metadata.file_type().is_symlink() {
            return Err(invalid("install path contains a symbolic link"));
        }
    }
    Ok(())
}

/// Read-only preflight; installation never upgrades or overwrites an old target.
pub fn fresh_install_preflight(
    directory: &Path,
    shared_registration_exists: bool,
) -> io::Result<()> {
    if !directory.is_absolute() || directory.parent().is_none() {
        return Err(invalid(
            "installation requires an absolute, non-root directory",
        ));
    }
    if shared_registration_exists {
        return Err(invalid(
            "an Argus registration or shortcut already exists; uninstall it first",
        ));
    }
    match fs::symlink_metadata(directory) {
        Ok(_) => {
            return Err(invalid(
                "install directory already exists; choose a fresh directory",
            ))
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => (),
        Err(error) => return Err(error),
    }
    for parent in directory.ancestors().skip(1) {
        match fs::symlink_metadata(parent) {
            Ok(metadata) => {
                if !metadata.is_dir() {
                    return Err(invalid("install parent is not a directory"));
                }
                return reject_links(parent);
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => (),
            Err(error) => return Err(error),
        }
    }
    Err(invalid("install directory has no existing parent"))
}

pub fn hash_file(path: &Path) -> io::Result<String> {
    reject_links(path)?;
    let mut file = fs::File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err(invalid("owned file is not a regular file"));
    }
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hex(&hash.finish()))
}

pub fn write_record(directory: &Path, files: &[(String, String)]) -> io::Result<()> {
    reject_links(directory)?;
    let root = fs::canonicalize(directory)?;
    let root_text = root.to_string_lossy();
    if root_text.contains(['\n', '\r']) {
        return Err(invalid("install directory contains a line break"));
    }
    let mut text = format!("{MAGIC}\n{root_text}\n");
    for (name, hash) in files {
        validate_name(name)?;
        text.push_str(&format!("{hash}  {name}\n"));
    }
    // Bind the record to the installed executable too: a loose, editable text
    // marker must never independently authorize deletion.
    let mut uninstaller = fs::OpenOptions::new()
        .append(true)
        .open(directory.join("argus-uninstall.exe"))?;
    uninstaller.write_all(text.as_bytes())?;
    uninstaller.write_all(&(text.len() as u64).to_le_bytes())?;
    uninstaller.write_all(FOOTER)?;
    uninstaller.flush()?;
    fs::write(directory.join(RECORD), text)
}

fn bound_record(uninstaller: &Path) -> io::Result<String> {
    reject_links(uninstaller)?;
    let mut file = fs::File::open(uninstaller)?;
    let length = file.metadata()?.len();
    if length < 16 {
        return Err(invalid("uninstaller has no bound ownership record"));
    }
    file.seek(SeekFrom::End(-16))?;
    let mut footer = [0_u8; 16];
    file.read_exact(&mut footer)?;
    let count = u64::from_le_bytes(footer[..8].try_into().expect("footer length"));
    if &footer[8..] != FOOTER || count > 4 * 1024 * 1024 || count > length - 16 {
        return Err(invalid("invalid bound ownership record"));
    }
    file.seek(SeekFrom::Start(length - 16 - count))?;
    let mut text = String::new();
    file.take(count).read_to_string(&mut text)?;
    Ok(text)
}

fn validate_name(name: &str) -> io::Result<PathBuf> {
    let path = sanitize_relative(name)?;
    // Reject Windows aliases, alternate streams, and line-oriented record injection.
    if name.contains(['\n', '\r'])
        || name
            .split(['/', '\\'])
            .any(|part| part.ends_with(['.', ' ']) || part.chars().any(|c| c.is_control()))
        || name.eq_ignore_ascii_case(RECORD)
        || name.eq_ignore_ascii_case("argus-uninstall.exe")
    {
        return Err(invalid("unsafe ownership record path"));
    }
    Ok(path)
}

pub struct Removal {
    root: PathBuf,
    files: Vec<(PathBuf, String)>,
    record_hash: String,
}

impl Removal {
    pub fn validate(directory: &Path, uninstaller: &Path) -> io::Result<Self> {
        if !directory.is_absolute() {
            return Err(invalid("uninstall directory must be absolute"));
        }
        reject_links(directory)?;
        let root = fs::canonicalize(directory)?;
        if root.parent().is_none() {
            return Err(invalid("cannot uninstall a filesystem root"));
        }
        let record = root.join(RECORD);
        reject_links(&record)?;
        if fs::metadata(&record)?.len() > 4 * 1024 * 1024 {
            return Err(invalid("ownership record is oversized"));
        }
        let record_hash = hash_file(&record)?;
        let text = fs::read_to_string(record)?;
        if text != bound_record(uninstaller)? {
            return Err(invalid(
                "ownership record differs from the installed uninstaller",
            ));
        }
        let mut lines = text.lines();
        if lines.next() != Some(MAGIC) || lines.next() != Some(root.to_string_lossy().as_ref()) {
            return Err(invalid("directory has no matching Argus ownership record"));
        }
        let mut files = Vec::new();
        let mut names = BTreeSet::new();
        for line in lines {
            let (hash, name) = line
                .split_once("  ")
                .ok_or_else(|| invalid("invalid ownership entry"))?;
            if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(invalid("invalid ownership digest"));
            }
            let relative = validate_name(name)?;
            if !names.insert(name.replace('\\', "/").to_lowercase()) {
                return Err(invalid("duplicate ownership entry"));
            }
            let file = root.join(relative);
            if hash_file(&file)? != hash {
                return Err(invalid("installed file was changed; refusing removal"));
            }
            files.push((file, hash.to_owned()));
        }
        let uninstaller_hash = hash_file(uninstaller)?;
        let installed_uninstaller = root.join("argus-uninstall.exe");
        if !names.contains("argus-desktop.exe")
            || hash_file(&installed_uninstaller)? != uninstaller_hash
        {
            return Err(invalid("ownership record does not match this uninstaller"));
        }
        files.push((installed_uninstaller, uninstaller_hash));
        Ok(Self {
            root,
            files,
            record_hash,
        })
    }

    /// Returns false when user-added files remain. Never recursively deletes.
    pub fn remove(self) -> io::Result<bool> {
        let mut directories = BTreeSet::new();
        for (file, expected) in &self.files {
            // Recheck just before removal; this is not a hostile-process race lock.
            if hash_file(file)? != *expected {
                return Err(invalid("owned file changed during uninstall"));
            }
            fs::remove_file(file)?;
            for parent in file.parent().into_iter().flat_map(Path::ancestors) {
                if parent == self.root {
                    break;
                }
                directories.insert(parent.to_path_buf());
            }
        }
        if hash_file(&self.root.join(RECORD))? != self.record_hash {
            return Err(invalid("ownership record changed during uninstall"));
        }
        fs::remove_file(self.root.join(RECORD))?;
        let mut directories: Vec<_> = directories.into_iter().collect();
        directories.sort_by_key(|p| std::cmp::Reverse(p.components().count()));
        for directory in directories {
            reject_links(&directory)?;
            match fs::remove_dir(directory) {
                Ok(()) => (),
                Err(error) if error.kind() == io::ErrorKind::DirectoryNotEmpty => (),
                Err(error) => return Err(error),
            }
        }
        reject_links(&self.root)?;
        match fs::remove_dir(&self.root) {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == io::ErrorKind::DirectoryNotEmpty => Ok(false),
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "argus-ownership-test-{}-{stamp}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            // Never reuse or delete a pre-existing directory.
            fs::create_dir(&root).unwrap();
            Self(root)
        }

        fn install(&self, name: &str) -> (PathBuf, PathBuf) {
            let root = self.0.join(name);
            fs::create_dir(&root).unwrap();
            fs::create_dir(root.join("python")).unwrap();
            fs::write(root.join("argus-desktop.exe"), b"test app").unwrap();
            fs::write(root.join("python/runtime.py"), b"test runtime").unwrap();
            fs::write(root.join("argus-uninstall.exe"), b"test stub").unwrap();
            let entries = ["argus-desktop.exe", "python/runtime.py"]
                .map(|name| (name.to_owned(), hash_file(&root.join(name)).unwrap()));
            write_record(&root, &entries).unwrap();
            let runner = self.0.join(format!("{name}-runner.exe"));
            fs::copy(root.join("argus-uninstall.exe"), &runner).unwrap();
            (root, runner)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            // Only this test's exclusively-created, unique temporary fixture.
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn removes_only_owned_files_and_empty_directories() {
        let fixture = Fixture::new();
        let (root, runner) = fixture.install("custom install");
        assert!(Removal::validate(&root, &runner).unwrap().remove().unwrap());
        assert!(!root.exists());
        assert!(runner.exists());
    }

    #[test]
    fn preserves_user_added_files_and_directories() {
        let fixture = Fixture::new();
        let (root, runner) = fixture.install("install");
        fs::write(root.join("python/user-notes.txt"), b"keep me").unwrap();
        fs::write(root.join("personal.txt"), b"also keep").unwrap();
        assert!(!Removal::validate(&root, &runner).unwrap().remove().unwrap());
        assert_eq!(
            fs::read(root.join("python/user-notes.txt")).unwrap(),
            b"keep me"
        );
        assert_eq!(fs::read(root.join("personal.txt")).unwrap(), b"also keep");
        assert!(!root.join("argus-desktop.exe").exists());
    }

    #[test]
    fn missing_ownership_record_never_authorizes_removal() {
        let fixture = Fixture::new();
        fs::write(fixture.0.join("personal.txt"), b"keep").unwrap();
        assert!(Removal::validate(&fixture.0, &fixture.0.join("missing.exe")).is_err());
        assert!(fixture.0.join("personal.txt").exists());
    }

    #[test]
    fn changed_file_aborts_before_any_deletion() {
        let fixture = Fixture::new();
        let (root, runner) = fixture.install("install");
        fs::write(root.join("python/runtime.py"), b"user edited").unwrap();
        assert!(Removal::validate(&root, &runner).is_err());
        assert!(root.join("argus-desktop.exe").exists());
        assert!(root.join(RECORD).exists());
    }

    #[test]
    fn edited_record_and_wrong_runner_are_rejected() {
        let fixture = Fixture::new();
        let (root, runner) = fixture.install("install");
        let original = fs::read(root.join(RECORD)).unwrap();
        fs::write(root.join(RECORD), b"forged record").unwrap();
        assert!(Removal::validate(&root, &runner).is_err());
        fs::write(root.join(RECORD), original).unwrap();
        fs::write(&runner, b"different executable").unwrap();
        assert!(Removal::validate(&root, &runner).is_err());
        assert!(root.join("argus-desktop.exe").exists());
    }

    #[test]
    fn copied_install_cannot_authorize_another_root() {
        let fixture = Fixture::new();
        let (root, runner) = fixture.install("install");
        let other = fixture.0.join("other");
        fs::create_dir(&other).unwrap();
        fs::copy(root.join(RECORD), other.join(RECORD)).unwrap();
        fs::copy(
            root.join("argus-uninstall.exe"),
            other.join("argus-uninstall.exe"),
        )
        .unwrap();
        assert!(Removal::validate(&other, &runner).is_err());
        assert!(root.join("argus-desktop.exe").exists());
    }

    #[test]
    fn unsafe_names_are_rejected() {
        for name in [
            "../personal",
            "C:/personal",
            "/absolute",
            "dir/../file",
            "dir/file.",
            "dir/file ",
            "dir/a\nb",
            RECORD,
            "argus-uninstall.exe",
        ] {
            assert!(validate_name(name).is_err(), "accepted {name:?}");
        }
    }

    #[test]
    fn fresh_install_refuses_existing_target_and_shared_registration() {
        let fixture = Fixture::new();
        fs::write(fixture.0.join("personal.txt"), b"keep").unwrap();
        assert!(fresh_install_preflight(&fixture.0, false).is_err());
        let fresh = fixture.0.join("new install");
        assert!(fresh_install_preflight(&fresh, true).is_err());
        assert!(!fresh.exists());
        assert!(fresh_install_preflight(&fresh, false).is_ok());
        assert!(!fresh.exists());
        assert_eq!(fs::read(fixture.0.join("personal.txt")).unwrap(), b"keep");
    }

    #[test]
    fn rechecks_files_before_removal() {
        let fixture = Fixture::new();
        let (root, runner) = fixture.install("install");
        let removal = Removal::validate(&root, &runner).unwrap();
        fs::write(
            root.join("argus-desktop.exe"),
            b"user replaced after validation",
        )
        .unwrap();
        assert!(removal.remove().is_err());
        assert_eq!(
            fs::read(root.join("argus-desktop.exe")).unwrap(),
            b"user replaced after validation"
        );
        assert!(root.join(RECORD).exists());
    }

    #[cfg(windows)]
    #[test]
    fn rejects_junction_in_owned_path_without_touching_target() {
        let fixture = Fixture::new();
        let (root, runner) = fixture.install("install");
        let outside = fixture.0.join("outside");
        fs::rename(root.join("python"), &outside).unwrap();
        let status = std::process::Command::new("cmd.exe")
            .args(["/c", "mklink", "/J"])
            .arg(root.join("python"))
            .arg(&outside)
            .output()
            .unwrap();
        assert!(
            status.status.success(),
            "{}",
            String::from_utf8_lossy(&status.stderr)
        );
        assert!(Removal::validate(&root, &runner).is_err());
        assert!(outside.join("runtime.py").exists());
        assert!(fresh_install_preflight(&root.join("python/new install"), false).is_err());
        // Remove just the fixture junction, never its target recursively.
        fs::remove_dir(root.join("python")).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn rejects_junction_as_install_root() {
        let fixture = Fixture::new();
        let (root, runner) = fixture.install("install");
        let alias = fixture.0.join("alias");
        let status = std::process::Command::new("cmd.exe")
            .args(["/c", "mklink", "/J"])
            .arg(&alias)
            .arg(&root)
            .output()
            .unwrap();
        assert!(status.status.success());
        assert!(Removal::validate(&alias, &runner).is_err());
        assert!(root.join("argus-desktop.exe").exists());
        fs::remove_dir(alias).unwrap();
    }
}

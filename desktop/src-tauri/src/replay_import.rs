//! Read-only, bounded offline capture selection. No shell or live sensor actions.
use std::{fs, io::Read, path::Path};

fn validate_path(path: &Path, max_bytes: usize) -> Result<(), String> {
    if !path.extension().is_some_and(|extension| {
        extension.eq_ignore_ascii_case("pcap") || extension.eq_ignore_ascii_case("pcapng")
    }) {
        return Err("Choose a .pcap or .pcapng capture file".to_owned());
    }
    let metadata =
        fs::symlink_metadata(path).map_err(|error| format!("Capture metadata: {error}"))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || is_reparse_point(&metadata) {
        return Err("Capture must be a regular file, not a link or reparse point".to_owned());
    }
    if metadata.len() > max_bytes as u64 {
        return Err(format!("Capture exceeds the {max_bytes}-byte replay limit"));
    }
    Ok(())
}

#[cfg(windows)]
fn is_reparse_point(metadata: &fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    metadata.file_attributes() & 0x400 != 0
}

#[cfg(not(windows))]
fn is_reparse_point(_: &fs::Metadata) -> bool {
    false
}

#[cfg(any(windows, test))]
fn selected_capture(
    selected: Option<std::path::PathBuf>,
    max_bytes: usize,
) -> Result<Option<String>, String> {
    selected
        .map(|path| {
            validate_path(&path, max_bytes)?;
            let canonical =
                dunce::canonicalize(&path).map_err(|error| format!("Capture path: {error}"))?;
            canonical
                .into_os_string()
                .into_string()
                .map_err(|_| "Capture path is not valid Unicode".to_owned())
        })
        .transpose()
}

pub(super) fn pick_capture() -> Result<Option<String>, String> {
    #[cfg(windows)]
    {
        let selected = rfd::FileDialog::new()
            .set_title("Import offline capture — PCAP / PCAPNG")
            .add_filter("Packet captures", &["pcap", "pcapng"])
            .pick_file();
        selected_capture(selected, super::default_pcap_limits().max_file_bytes.get())
    }
    #[cfg(not(windows))]
    Err("Native capture import is currently supported on Windows only".to_owned())
}

pub(super) fn read_capture(path: &Path, max_bytes: usize) -> Result<Vec<u8>, String> {
    validate_path(path, max_bytes)?;
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // Open the reparse point itself rather than following a replaced final link.
        options.custom_flags(0x0020_0000);
    }
    let file = options
        .open(path)
        .map_err(|error| format!("Capture read: {error}"))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("Capture metadata: {error}"))?;
    if !metadata.is_file() || is_reparse_point(&metadata) {
        return Err("Capture must be a regular file, not a reparse point".to_owned());
    }
    let mut bytes = Vec::new();
    file.take(max_bytes as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("Capture read: {error}"))?;
    if bytes.len() > max_bytes {
        return Err(format!("Capture exceeds the {max_bytes}-byte replay limit"));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::{fs, read_capture, selected_capture};

    #[test]
    fn cancellation_does_not_require_a_path() {
        assert_eq!(selected_capture(None, 1), Ok(None));
    }

    #[test]
    fn picker_validation_rejects_wrong_extension_missing_and_oversized_files() {
        let directory =
            std::env::temp_dir().join(format!("argus-import-validation-{}", std::process::id()));
        fs::create_dir_all(&directory).expect("create fixture directory");
        let capture = directory.join("fixture.PCAPNG");
        fs::write(&capture, [0_u8; 17]).expect("write fixture");
        assert!(
            selected_capture(Some(capture.clone()), 16)
                .unwrap_err()
                .contains("limit")
        );
        assert!(
            selected_capture(Some(capture.clone()), 17)
                .unwrap()
                .is_some()
        );
        assert!(selected_capture(Some(directory.join("missing.pcap")), 17).is_err());
        assert!(selected_capture(Some(directory.join("wrong.txt")), 17).is_err());
        assert!(selected_capture(Some(directory.clone()), 17).is_err());
        assert!(read_capture(&capture, 16).is_err());
        assert_eq!(fs::read(&capture).unwrap(), [0_u8; 17]);
        fs::remove_file(capture).expect("remove fixture");
        fs::remove_dir(directory).expect("remove fixture directory");
    }
}

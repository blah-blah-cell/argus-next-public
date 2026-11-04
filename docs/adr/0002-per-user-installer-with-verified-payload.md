# 0002 — Per-user installer with a verified payload

- Status: Accepted
- Date: 2026-11-04

## Context

The first public artifact was a portable ZIP. That is a poor first experience for
the intended audience: extract, find the executable, create your own shortcut, and
remember where it went. A real installer needs to work on a machine with no
administrator rights, no PowerShell 7, no WiX, and no Inno Setup, and it must not
depend on a network download at install time.

## Decision

Ship a single self-contained setup executable built from a small Rust crate in
`installer/` with **no external crates**:

- The payload is appended to the stub binary in a documented container format with
  a per-entry SHA-256 digest and a footer.
- The setup program verifies the whole payload before writing any file, and again
  while extracting, then reports a clear failure on mismatch.
- Installation is per-user: `%LOCALAPPDATA%\Programs\ArgusNextPublic`, an HKCU
  uninstall entry, and Start menu and desktop shortcuts resolved through
  `SHGetKnownFolderPath` so redirected folders work.
- User data lives in `%LOCALAPPDATA%\ArgusNextPublic` and survives uninstall.
- The installed uninstaller is a copy of the stub with a rewritten payload, so it
  needs no network and no administrator rights.

## Consequences

- One download, one UAC-free install, and a normal Apps & Features entry.
- The container format is ours to maintain and must stay versioned; it is covered
  by unit tests and a smoke test in CI.
- The setup program is unsigned until a code-signing certificate is provisioned,
  so SmartScreen warnings are expected and checksums are published instead.
- A future migration to MSIX or WiX is possible without changing the app, but the
  payload format would need a compatibility path.

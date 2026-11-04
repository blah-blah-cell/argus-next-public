# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html), with pre-release
identifiers for prototype milestones.

## [Unreleased]

### Added

- Continuous integration: formatting, clippy, Rust and installer tests, UI
  validation, release-isolation tests, metadata consistency, and supply-chain
  checks.
- Pinned toolchain, formatter, editor, and Cargo alias configuration.
- Governance documents: security policy, contribution guide, code of conduct,
  changelog, architecture decision records, and issue/pull request templates.
- Dependency, license, and source policy (`deny.toml`) plus Dependabot updates.

## [0.1.0-alpha.1] - 2026-11-04

### Added

- Per-user Windows setup program with a hash-verified payload, Start menu and
  desktop shortcuts, an uninstall entry, and a small installed uninstaller.
- Windows install guide and a public roadmap covering trust, runtime, retention,
  alerting, multi-site operation, and platform support.
- Architecture decision records and an architecture overview.
- Tag-driven release automation with checksums, an SBOM, and signed build
  provenance attestations.

### Changed

- `SHA256SUMS.txt` is written with LF line endings so `sha256sum -c` works in Git
  Bash, WSL, and PowerShell.
- Portable ZIP entries use forward-slash separators under a single package folder.
- The packaging script runs on Windows PowerShell 5.1 as well as PowerShell 7.

### Security

- The setup program verifies every embedded file before writing anything and
  refuses altered or truncated downloads.
- Published artifacts carry checksums and GitHub build provenance attestations;
  the binaries remain unsigned with Authenticode until a certificate is
  provisioned.

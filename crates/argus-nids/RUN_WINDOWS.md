# Run Argus NIDS on Windows

From the repository root, probe Pktmon readiness without starting a capture:

```powershell
cargo run -p argus-nids -- --pktmon-probe
```

Machine-readable status and evidence use a versioned JSON envelope:

```powershell
cargo run -p argus-nids -- --json --pktmon-probe
cargo run -p argus-nids -- --json --pcap C:\path\to\capture.pcapng
cargo run -p argus-nids -- --json --pktmon-probe
```

Replay a bounded classic PCAP file or measure vertical-slice throughput:

```powershell
cargo run --release -p argus-nids -- --pcap C:\path\to\capture.pcap
cargo run --release -p argus-nids -- --benchmark C:\path\to\capture.pcap
```

The PCAP benchmark includes file parsing and normalization.

The executable does not open a live network adapter. Windows Pktmon's
truthful integration is stopped-session ETL conversion rather than this pipeline's live pull
contract. Npcap is not required for this workflow. Convert a completed Pktmon ETL recording with the
Windows-provided tool before replay:

```powershell
pktmon etl2pcap capture.etl --out capture.pcapng
```

The CLI can orchestrate the same bounded stopped-session workflow when run from an elevated
terminal. It refuses relative paths, existing outputs, sessions over 30 seconds, failed commands,
empty outputs, and ambiguous multi-file results:

```powershell
cargo run --release -p argus-nids -- --pktmon-record 10 `
  C:\ArgusData\capture.etl C:\ArgusData\capture.pcapng
```

This is explicitly a recorder followed by replay, not a live capture source. The source ETL is
retained for audit and recovery.

The desktop uses the native Pktmon streaming sidecar for regular monitoring.
Build that narrow Windows boundary with:

```powershell
./native/pktmon-stream/build.ps1 -Configuration Debug
cargo run -p argus-desktop
```

When capture is explicitly started the desktop requests elevation to register `ArgusNextPublicSensor` for
the current binary and data directory. Later starts are idempotent. One Pktmon
session stays active while the sidecar rotates bounded five-second classic-PCAP
segments, so conversion does not create a capture gap. The desktop analyzes
completed segments oldest-first and deletes each only after successful analysis;
the sidecar independently caps retained live segments at eight.

The strict reader supports classic PCAP and PCAPNG section/interface/enhanced-packet blocks for
Ethernet, bounded VLAN/QinQ stacks, IPv4, IPv6 extension traversal, and TCP/UDP. Unsupported link
types, excessive VLAN nesting, malformed blocks, simple packet blocks without timestamps, and
unsupported timestamp encodings are hard errors. The output identifies the fixed threshold baseline
as `trained_model=false`, reports source and pipeline health counts, reports every skipped unsupported
record category, and prints only evidence-backed alerts.

Build the source release with:

```powershell
./scripts/build-windows.ps1
```

The executables are written to `target/release`. This does not create an installer
or a portable ZIP. See [Windows setup](../../docs/BUILD_WINDOWS.md) for Python,
model, launch, and permission requirements. Runtime state defaults to
`%LOCALAPPDATA%\ArgusNextPublic`.

JSON status schema version 2 includes capture source/session identity, raw/normalized and quality
counters, explicit packet-ordinal-only transport tracking, exact threshold provenance, and separate
alert `health`, `score`, `reason`, and `evidence`. A `null` score is intentional for the fixed-rule
baseline; it is not an unavailable probability.

Validate the workspace with:

```powershell
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets
```

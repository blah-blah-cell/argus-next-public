# Argus Next public roadmap

This is the working plan for making Argus Next usable by small organizations
without a dedicated security team. It is deliberately conservative: nothing here
is claimed as done until it has been built, tested on a clean machine, and
shipped with an artifact you can verify.

The Windows setup program is already shipped, and the repository carries a
verified engineering baseline: a pinned toolchain, CI gates for formatting,
clippy, Rust and installer tests, clean-room procedural desktop UI validation
(`180` checks), release metadata consistency, dependency/license/source policy,
secret scanning, bounded `65,536`-flow capture/identity tracking with `FIN`/`RST`
and idle eviction, a unit-tested `argus-capture` IPv6/IPv4 CIDR filter library
module, a `6 s` topology cache TTL, and tag-driven releases with checksums, an
SBOM and build provenance. Those are recorded in `docs/adr/`. Everything else
below is still ahead.

## Guiding constraints

- Windows is the initial evaluation target, not yet approved for an
  organizational deployment; Linux and macOS are in progress.
- The app must stay useful with no cloud service and no internet connection.
- Capture is always an explicit action. Nothing starts recording on its own.
- **Goal, not current behavior:** metadata-first capture with an explicit,
  bounded choice for payload retention. Current live capture writes temporary
  PCAP segments that can contain payloads; see ADR 0003.
- Forecasts are research outputs, not validated attack detections.
- A prototype that quietly overstates itself is worse than a small honest tool.

## Stage 0 — Trust and supply chain (in progress)

Delivered: root `LICENSE` (Apache-2.0) with model weights covered by a separate
notice, `SECURITY.md` with scope and response windows, a `NOTICE` file, clean-room
procedural spatial UI assets (`Apache-2.0` / `CC0-1.0`), ADRs, and CI gates for
formatting, clippy, Rust and installer tests, UI validation, release isolation,
version/license consistency, dependency and license policy, and secret scanning.
The release workflow produces SHA-256 manifests, a CycloneDX SBOM and build
provenance, with a guarded Authenticode step.

Remaining before pilot conversations:

- Verify release provenance independently in addition to the published SBOM
  and checksum files; confirm fresh-system startup and capture boundaries.
- Provision a signing certificate so Authenticode replaces the current
  "publisher unknown" warning.
- Threat model covering the elevated Pktmon sensor, the scheduled task, and the
  loopback IPC boundary.
- Pin GitHub Actions to commit SHAs, with a reviewed update path.

Exit criteria: a stranger can download a release, verify hashes, read the
license, and report a vulnerability without asking a question.

## Stage 1 — Safe single-machine evaluation (in progress)

Start with an authorized, known capture or explicitly generated test traffic;
do not require an organization to expose normal network traffic. Make replay
results distinguish observed flows, unsupported or skipped data, and the
reasons behind any alert.

Delivered in source:
- Native PCAP/PCAPNG picker, offline source labeling, replay-local rules,
  capture-quality evidence and read-only flow inspection.
- Bounded `65,536`-flow `FlowGenerationTracker` and `PcapReplaySource` with
  TCP `FIN`/`RST` closure eviction and `60 s` idle-timeout eviction, plus the
  unit-tested `argus-capture` `PacketFilter` library module (`filter.rs`, bare
  and keyed IPv6/IPv4 CIDR parsing and exact BPF lowering).
- `get_network_topology` cache TTL (`6 s`) aligned above the `5,000 ms` UI poll
  interval, and strict `packet-causal-v1` vs `packet-causal-flow-v2` feature
  contract separation.

Remaining:
- Package the post-alpha.3 updates in a fresh signed installer and verify on a
  clean Windows machine.
- Wire `argus-capture::PacketFilter` into an operator-facing UI filter bar if
  pilot workflows require interactive query filtering.
- Measure setup, idle and monitoring CPU/RAM, storage growth, and interruption
  recovery on identified Windows machines. Add hard storage quotas and true
  metadata-only capture mode before claiming hard retention caps.

Exit criteria: a reviewer can replay an authorized capture without starting
live monitoring, see what was processed or omitted and why, and understand the
files created and the limitations.

## Stage 2 — Installed app without Python (native runtime delivered; neural runtime optional)

Delivered in source (`scripts/package-windows.ps1` and `scripts/build-windows-release.ps1`):
- **Separated release & weightless packaging:** The CI/GitHub Release bundle
  (`scripts/package-windows.ps1`) ships the native binaries alongside frozen
  `AHEM-v6.1` research weights and Python sources (no Python venv or
  `onnxruntime.dll` included), while `scripts/build-windows-release.ps1`
  produces an optional weightless portable build (binaries, sensor script,
  launcher, README, and Apache-2.0/Three.js notices only; `-IncludeModels` opts
  into bundling Python sources and `models/`).
- **Native zero-parameter predictor & right-sized router:** Rust `split_cell_v1`
  runs live per packet even without `onnxruntime.dll`, ONNX checkpoints, or
  Python (`25.6 ns` batch-mean p50 at 4,096 lanes; `55.1 ns` at 65,536 lanes
  under L3 spill; `ewma` `15.9 / 25.7 ns`; `earned_v1` `27.9 / 73.1 ns`) and
  beats frozen `AHEM-v6.1` (`3.225 ms` PyTorch CPU p50; `15.2 us` native ctypes)
  across all tested zero-shot datasets. `ahem_v61` in `NativeOnnxWorker` stays
  in fail-closed ONNX manifest/artifact validation standby (never fed
  `packet-causal-v1` features). A SHA-256-pinned `4,591 B` pure-JSON
  `compact-router-v1` (`research/evidence/compact-router-v1.json`, `4.3 us`
  head-only CPU median, `0.97 ms` end-to-end pure-Python `route_window()`,
  wired into `compact_router.py`) replaces the quarantined 421M fp16 `Laya v2`
  router (never evaluated on the locked test split).

Remaining if optional ONNX neural forecasting is shipped:
- Ship `onnxruntime.dll` and verified ONNX exports beside the executable only if
  a downstream task justifies the latency/complexity over `split_cell_v1`.
- Keep PyTorch and the Python worker as research-only tooling so end users never
  install `uv`, Python, or torch.

Exit criteria: a fresh install runs bounded telemetry and `split_cell_v1`
without Python, and explains model readiness truthfully.

## Stage 3 — Survivable unattended operation (planned)

- Tiered retention with hard quotas across raw captures, flows, alerts, and
  generated reports. Exported reports currently have no visible rotation.
- A real store for flows and alerts (SQLite with WAL is sufficient for a single
  site) with schema versioning and migration.
- Alert delivery: email digest plus webhook (Teams, Slack, or a generic URL),
  severity levels, deduplication, and a "why did this alert fire" explanation.
- Signed update manifest with stable and beta channels, rollback, and an offline
  update bundle for air-gapped sites.
- Redacted diagnostics bundle for support without shipping capture files.
- Guardrails on Active Defense: blocking stays off by default, requires explicit
  confirmation, has a visible undo, and is audited.

Exit criteria: a site runs for 30 days without filling a disk, without manual
restarts, and with alerts reaching someone who is not staring at the dashboard.

## Stage 4 — More than one site (gated on two successful pilots)

- `argus-sensor`: headless capture service sharing the current capture core.
- `argus-collector`: single binary plus database, metadata-first ingestion,
  enrollment tokens, mutual TLS, per-site tags, spool-and-retry, and backpressure.
- Central read-only search with viewer, analyst, and admin roles and an
  append-only audit log.
- Payload stays on site unless an operator explicitly opts in per capture.

Exit criteria: two sites report to one collector and one search view; a sensor
offline for an hour loses nothing beyond configured policy.

## Stage 5 — Scale and interoperability (ongoing)

- Columnar archives (Parquet/DuckDB) for long retention and fast queries.
- Ingestion of NetFlow/IPFIX and Zeek or Suricata logs for organizations that
  cannot capture packets at all.
- Export to syslog CEF and OpenTelemetry; import threat-intel lists offline.
- Published throughput and resource figures, with a benchmark harness and
  regression thresholds.
- Linux sensor first, then macOS.

## Cross-cutting tracks

### Local device context extension (implemented in source, not released)

- Read-only Windows socket ownership and Defender status/history, refreshed on
  demand with optional session-only refresh during explicit live monitoring.
- Time-bounded TCP socket-match candidates and captured-byte summaries for the
  latest live capture, with unknown/ambiguous cases retained separately.
- Exact executable-path links to historical Defender records, never automatic
  malware verdicts. Imported and simulated traffic is not joined to current-host
  context. See `docs/DEVICE_CONTEXT.md` for limits.

### Evaluation & detection boundaries

- **Current evidence:** On CTU IoT-23 (`8-1`, `21-1`, benign `4-1`), frozen
  `AHEM-v6.1` residuals did not outperform `base_logreg` (`0.998246` vs
  `0.999219` PR-AUC on `8-1 -> 21-1`), benign `4-1` threshold calibration failed
  closed (`0.0 FA/h`, `0.0` recall), and full 48 GB AWS CIC-IDS2017 and
  CSE-CIC-IDS2018 raw PCAP archives are not staged locally, leaving CIC-IDS2017
  bridge evaluation pending.
- **Next requirement:** Any future detection claim requires full multi-day
  labeled captures, published precision/recall at a calibrated false-positive
  budget, and comparison against simple packet-causal baselines (`base_logreg`,
  `ewma`, `split_cell_v1`).

## Sizing guidance

At 10 Mbps sustained, the traffic volume is roughly 4.5 GB per hour and 108 GB
per day before capture overhead and any truncation or filtering. Defaults should
favor metadata, short raw rings, and explicit opt-in for payload retention once
those controls exist.

## Explicit non-goals

No multi-tenant SaaS, no cloud dependency, no automatic blocking, no in-product
training, no LLM chat layer, no mobile client.

## Open decisions

1. Signing budget: Azure Trusted Signing versus an OV certificate.
2. Whether pilots are single-office or multi-site from the start.
3. Whether metadata-only is the permanent default for capture.
4. Licensing intent for model weights: permissive, research-only, or dual.
5. Whether replay and flow ingestion belong in scope for the first pilot.

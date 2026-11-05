# Argus Next public roadmap

This is the working plan for making Argus Next usable by small organizations
without a dedicated security team. It is deliberately conservative: nothing here
is claimed as done until it has been built, tested on a clean machine, and
shipped with an artifact you can verify.

The Windows setup program is already shipped, and the repository now carries an
engineering baseline: a pinned toolchain, CI gates for formatting, clippy, tests,
UI validation and release metadata, dependency/license/source policy, secret
scanning, and tag-driven releases with checksums, an SBOM and build provenance.
Those are recorded in `docs/adr/`. Everything else below is still ahead.

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

## Phase 0 — Trust and supply chain (in progress)

Delivered: root `LICENSE` (Apache-2.0) with model weights covered by a separate
notice, `SECURITY.md` with scope and response windows, a `NOTICE` file, ADRs,
and CI gates for formatting, clippy, Rust and installer tests, UI validation,
release isolation, version/license consistency, dependency and license policy,
and secret scanning. The release workflow now produces SHA-256 manifests, a
CycloneDX SBOM and build provenance, with a guarded Authenticode step.

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

## Phase 1 — Safe single-machine evaluation (next)

Start with an authorized, known capture or explicitly generated test traffic;
do not require an organization to expose normal network traffic. Make replay
results distinguish observed flows, unsupported or skipped data, and the
reasons behind any alert. Confirm what files are written, where they live, and
what a user must do to remove them. Add bounded retention and privacy controls
only with tests that verify actual behavior; do not label live capture
metadata-only until the implementation makes that true.

Development implementation: the desktop now has a native PCAP/PCAPNG picker,
offline source labeling, replay-local rules, capture-quality evidence and
read-only flow inspection. Replay does not run forecasting automatically.
This work has not been packaged in alpha.3. Native picker interaction and
clean-machine end-to-end usability still need verification; tests and browser
fixtures do not establish those outcomes. Low-disk protection is best effort,
not a complete hard quota or metadata-only capture mode.

Measure setup, idle and monitoring CPU/RAM, storage growth, and interruption
recovery on identified Windows machines. A repeated run on one PC is not an
independent site. Use labeled or controlled traffic when assessing alerts;
report misses, false alerts, and unsupported cases as well as successes. A
forecast-loss improvement is not an intrusion-detection result.

Exit criteria: a reviewer can replay an authorized capture without starting
live monitoring, see what was processed or omitted and why, and understand the
files created and the limitations. Privacy controls and resource measurements
must be verified, not inferred from defaults.

## Phase 2 — Installed app without Python (planned)

The installed app currently launches without a model runtime. Closing that gap
removes the largest remaining install friction.

- Ship `onnxruntime.dll` and the v6.1 ONNX export beside the executable; the
  `argus-inference` crate already fails closed when artifacts or runtime are
  absent.
- Verify ONNX parity in CI against the PyTorch checkpoint, including a
  significant-error bound, before the runtime is trusted.
- Keep PyTorch and the Python worker as research-only tooling, documented
  separately, so end users never install `uv`, Python, or torch.
- Add a model-readiness panel that states plainly which artifacts loaded and why
  forecasting is or is not available.

Exit criteria: a fresh install forecasts on a machine with no Python, and the
app explains its own readiness truthfully.

## Phase 3 — Survivable unattended operation (planned)

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

## Phase 4 — More than one site (gated on two successful pilots)

- `argus-sensor`: headless capture service sharing the current capture core.
- `argus-collector`: single binary plus database, metadata-first ingestion,
  enrollment tokens, mutual TLS, per-site tags, spool-and-retry, and backpressure.
- Central read-only search with viewer, analyst, and admin roles and an
  append-only audit log.
- Payload stays on site unless an operator explicitly opts in per capture.

Exit criteria: two sites report to one collector and one search view; a sensor
offline for an hour loses nothing beyond configured policy.

## Phase 5 — Scale and interoperability (ongoing)

- Columnar archives (Parquet/DuckDB) for long retention and fast queries.
- Ingestion of NetFlow/IPFIX and Zeek or Suricata logs for organizations that
  cannot capture packets at all.
- Export to syslog CEF and OpenTelemetry; import threat-intel lists offline.
- Published throughput and resource figures, with a benchmark harness and
  regression thresholds.
- Linux sensor first, then macOS.

## Cross-cutting workstreams

- **Accessibility and low-spec:** CPU-only inference, 4 GB RAM target, keyboard
  navigation, high contrast, reduced motion (already supported in the UI).
- **Privacy and compliance:** data-flow diagram, DPIA and DPA templates,
  redaction rules, hash-only evidence export with chain-of-custody notes.
- **Documentation:** a 15-minute administrator quickstart distinct from the
  developer build guide, a deployment checklist, and troubleshooting by symptom.
- **Evaluation:** replay harness with labeled captures, published precision and
  false-positive budget, and per-site baselines before any detection claim.
- **Localization:** defer until pilots request it; keep strings externalizable.

## Sizing guidance

At 10 Mbps sustained, the traffic volume is roughly 4.5 GB per hour and 108 GB
per day before capture overhead and any truncation or filtering. Actual PCAP and
flow-metadata storage depends on traffic, capture configuration, and schema;
measure it rather than assuming a fixed daily metadata volume. Defaults should
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

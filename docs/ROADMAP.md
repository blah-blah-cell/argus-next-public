# Argus Next public roadmap

This is the working plan for making Argus Next usable by small organizations
without a dedicated security team. It is deliberately conservative: nothing here
is claimed as done until it has been built, tested on a clean machine, and
shipped with an artifact you can verify.

The Windows setup program is already shipped. Everything below is still ahead.

## Guiding constraints

- Windows is the supported platform for pilots; Linux and macOS are in progress.
- The app must stay useful with no cloud service and no internet connection.
- Capture is always an explicit action. Nothing starts recording on its own.
- Metadata is the default; raw payload capture is an explicit, bounded choice.
- Forecasts are research outputs, not validated attack detections.
- A prototype that quietly overstates itself is worse than a small honest tool.

## Phase 0 — Trust and supply chain (next, ~1–2 weeks)

Small, cheap work that unblocks pilot conversations.

- Root `LICENSE` for source and a separate, clearly worded license for model
  weights. The workspace currently declares Apache-2.0 but ships no license text.
- `SECURITY.md` with a disclosure address, scope, and a stated response window.
- CycloneDX SBOM published beside each release, plus SHA-256 for every artifact.
- CI that builds, runs the UI and release-isolation suites, `cargo test`,
  `cargo clippy`, and refuses to publish on failure.
- SPDX headers or a NOTICE file listing third-party components (Three.js is
  already covered by `THIRD_PARTY/three-LICENSE.txt`).
- Threat model covering the elevated Pktmon sensor, the scheduled task, and the
  loopback IPC boundary.

Exit criteria: a stranger can download a release, verify hashes, read the
license, and report a vulnerability without asking a question.

## Phase 1 — Installed app without Python (~4–6 weeks)

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

## Phase 2 — Survivable unattended operation (~4–6 weeks)

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

## Phase 3 — More than one site (~8–12 weeks, gated on two successful pilots)

- `argus-sensor`: headless capture service sharing the current capture core.
- `argus-collector`: single binary plus database, metadata-first ingestion,
  enrollment tokens, mutual TLS, per-site tags, spool-and-retry, and backpressure.
- Central read-only search with viewer, analyst, and admin roles and an
  append-only audit log.
- Payload stays on site unless an operator explicitly opts in per capture.

Exit criteria: two sites report to one collector and one search view; a sensor
offline for an hour loses nothing beyond configured policy.

## Phase 4 — Scale and interoperability (ongoing)

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

At 10 Mbps sustained, raw PCAP is roughly 4.5 GB per hour and about 108 GB per
day. Raw retention therefore belongs in hours, not days. Flow metadata is a few
hundred megabytes per day at the same rate. Defaults should favor metadata,
short raw rings, and explicit opt-in for long payload retention.

## Explicit non-goals

No multi-tenant SaaS, no cloud dependency, no automatic blocking, no in-product
training, no LLM chat layer, no mobile client.

## Open decisions

1. Signing budget: Azure Trusted Signing versus an OV certificate.
2. Whether pilots are single-office or multi-site from the start.
3. Whether metadata-only is the permanent default for capture.
4. Licensing intent for model weights: permissive, research-only, or dual.
5. Whether replay and flow ingestion belong in scope for the first pilot.

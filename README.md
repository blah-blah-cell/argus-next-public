# Argus Next

[![CI](https://github.com/blah-blah-cell/argus-next-public/actions/workflows/ci.yml/badge.svg)](https://github.com/blah-blah-cell/argus-next-public/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Argus Next is a Windows prototype for inspecting network traffic. It captures
packets, groups them into bounded flows, and displays telemetry and alerts.
This is the public repository prepared for supervised pilots with NGOs that
don't have a dedicated security team.

It is a prototype, not a security product. A forecast is not an attack verdict,
and recent benchmarks show that frozen neural forecasters do not beat simple
online causal baselines zero-shot or improve intrusion detection over a basic
logistic packet probe on CTU IoT-23. Please don't rely on it as your only
security tool.

## Current status

The [Windows alpha.3 prerelease](https://github.com/blah-blah-cell/argus-next-public/releases/tag/v0.1.0-alpha.3)
has passed a fresh install/uninstall smoke test on a GitHub Windows runner. It is
unsigned and remains an early prototype; it is not yet ready for an NGO deployment.
**Do not install or run the uninstaller from the older alpha.1 release.**

The GitHub Release installer and portable ZIP (`.\scripts\package-windows.ps1`,
built and smoke-tested in CI) ship the native binaries alongside the frozen
`AHEM-v6.1` research weights, manifests, and Python research sources (no Python
environment or `onnxruntime.dll` is bundled; native Rust `split_cell_v1` runs
live per packet without them). For a minimal native-only runtime without Python
sources or research weights, `.\scripts\build-windows-release.ps1` produces an
optional weightless portable build (binaries, sensor script, launcher, README,
and Apache-2.0/Three.js notices; bundling Python and `models/` is opt-in via
`-IncludeModels`). Live capture is always opt-in and requires administrator
approval. See [docs/BUILD_WINDOWS.md](docs/BUILD_WINDOWS.md).

### Requirements

- Windows 10 or 11, x64, with the Microsoft Edge WebView2 Runtime.
- The Microsoft Visual C++ x64 runtime.
- Pktmon API support for live capture.

For a supervised evaluation, download the alpha.3 setup from the release page and
check its hash against `SHA256SUMS-RELEASE.txt`. A matching hash checks the download
against the published value, not the publisher's identity. Read the
[installation guide](docs/INSTALL_WINDOWS.md) before proceeding.

## What it does

The app captures traffic, imports saved PCAP/PCAPNG files via **Import capture & replay**,
and tracks flows in bounded memory (`65,536`-flow limits in `FlowGenerationTracker`
and `PcapReplaySource` with TCP `FIN`/`RST` closure eviction, `60 s` idle-timeout
eviction, and a `6 s` topology cache TTL aligned to `5,000 ms` UI polling). The
`argus-capture` crate also provides a unit-tested `PacketFilter` library module
(`crates/argus-capture/src/filter.rs`) supporting bare and keyed IPv6/IPv4 CIDR
predicates and exact BPF lowering (not wired as an active desktop UI filter bar).
Offline replay does not start live capture, update live alerts or model baselines,
or enable blocking actions.

Causal sequence inference strictly separates legacy global-interarrival models
(`packet-causal-v1/next_global_packet`) from per-flow models
(`packet-causal-flow-v2/next_same_flow_packet`). Native Rust `split_cell_v1` runs
live per packet with zero learned parameters (`25.6 ns` batch-mean p50 at 4,096
lanes; `55.1 ns` at 65,536 lanes under L3 cache spill) even when `onnxruntime.dll`
and ONNX checkpoints are absent, while `ahem_v61` in `NativeOnnxWorker` operates
in fail-closed ONNX manifest/artifact validation standby (never fed
`packet-causal-v1` features). In the optional Python runtime,
`python/src/argus_research/compact_router.py` evaluates the SHA-256-pinned
`4,591 B` pure-JSON `research/evidence/compact-router-v1.json` router
(`argus.compact-router/v1`, zero PyTorch/GPU, fail-closed fallback to
`lstm_cse`), replacing the quarantined 421M-parameter `Laya v2` transformer.

The public build uses its own settings and data directories under
`%LOCALAPPDATA%\ArgusNextPublic`, separate from any development installation.
Monitoring starts only when you choose to start it. All 3D desktop spatial
assets (`desktop/ui/spatial/assets/`) are clean-room procedurally generated under
`Apache-2.0` / `CC0-1.0` (see [desktop/ui/spatial/assets/NOTICE.md](desktop/ui/spatial/assets/NOTICE.md)).

## Privacy and pilots

The first step is always a supervised evaluation. Before capturing traffic,
we agree on what to monitor, who can access the data, how long it stays on
disk, and how to handle findings. Only capture traffic you have permission to
monitor.

Argus processes captured traffic locally. Live monitoring writes temporary PCAP
segments containing packet bytes, which can include payloads, and unlinks each
segment after analysis; the separate learning journal stores numeric features
rather than payloads. Treat captures and exports as sensitive.

## Research (honest summary)

Matched benchmarks with zero hidden fallback (`0%`) established clear boundaries
on what the neural components do and do not achieve (tracked in `research/evidence/`):

- **Zero-shot forecasting:** Zero-parameter online causal predictors
  (`split_cell_v1`, `earned_v1`, `ewma`) beat both frozen `AHEM-v6.1` (`40,851`
  params, `3.225 ms` PyTorch CPU p50; `15.2 us` native ctypes replay) and
  `dual_lane_lstm` (`40,855` params, `1.465 ms` p50) across all tested local and
  external captures (`T08`/`T09`, `bigFlows`, `UNSW-SOSR`, `GothamDataset2025`).
  In batched Rust microbenchmarks (`N=20,000`/batch, 50 batches, batch-mean
  quantiles), `observe_packet` runs in `15.9 ns` (`ewma`), `25.6 ns`
  (`split_cell_v1`), and `27.9 ns` (`earned_v1`) batch-mean p50 at 4,096
  L2-resident lanes, scaling to `25.7 ns`, `55.1 ns`, and `73.1 ns` at 65,536
  lanes under L3 spill.
- **IDS detection bridge:** On official CTU IoT-23 captures (`8-1`, `21-1`,
  benign `4-1`), frozen `AHEM-v6.1` residuals (`0.998246` PR-AUC on `8-1 -> 21-1`)
  did not beat the 14-feature `base_logreg` baseline (`0.999219` PR-AUC), and
  `ahem_v61_state` (`0.976940`) trailed `memory_disabled_state` (`0.992305`).
  Full 48 GB AWS CIC-IDS2017 and CSE-CIC-IDS2018 raw PCAP archives are not
  staged locally, so no CIC-IDS2017 detection results are claimed.
- **Router right-sizing & retirements:** A `4,591 B` (`4.483 KB`) pure-JSON
  tabular router (`research/evidence/compact-router-v1.json`, `4.3 us` head-only
  CPU median on pre-extracted features, `0.97 ms` end-to-end pure-Python
  `route_window()`, `67.0%` locked-test accuracy across $G=3$ disjoint days)
  replaces the quarantined 421M fp16 `Laya v2` router (`61.38 ms` GPU,
  `831.39 MiB` VRAM, never evaluated on the locked test split). Legacy
  fallback-masked checkpoints (`step-0003808.pt`, retired) have been removed
  from the public mirror.

See [research/README.md](research/README.md) for exact metrics and caveats.

## Development

Build and test on Windows with Rust, Node 22+, and the Visual C++ build tools:

```powershell
.\scripts\build-windows.ps1
```

To update an existing local installation and keep desktop shortcuts aligned:

```powershell
pwsh -File .\scripts\update-installed-desktop.ps1
pwsh -File .\scripts\update-installed-desktop.ps1 -Check
```

Useful checks include:

```powershell
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-targets
cargo test --locked --release --manifest-path installer/Cargo.toml
node scripts/validate-desktop-ui.cjs
node --test scripts/test-release-runtime.cjs
node --test scripts/test-release-workflow.cjs
node scripts/check-release-metadata.cjs
```

See the [CI workflow](.github/workflows/ci.yml) for the complete check set.
Packaging, installer and release steps are in
[docs/BUILD_WINDOWS.md](docs/BUILD_WINDOWS.md); the installer can be smoke-tested
without launching the app with `.\scripts\test-installer.ps1 -SetupPath <setup.exe>`.

Architecture is described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and
lasting decisions are recorded in [docs/adr/](docs/adr/README.md). Contributions
follow [CONTRIBUTING.md](CONTRIBUTING.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `desktop/` | Tauri app and embedded HTML/CSS/JS interface (clean-room spatial assets) |
| `crates/` | Rust workspace: core, capture, features, inference, detector, CLI |
| `native/pktmon-stream/` | Windows-only Pktmon stream sidecar |
| `installer/` | Per-user setup and uninstall program (separate workspace) |
| `python/` | Research, training, pure-Python `compact_router.py`, and model export |
| `models/` | Frozen `AHEM-v6.1` research weights and manifests; terms in [models/NOTICE.md](models/NOTICE.md) |
| `research/` | Research summary and tracked JSON evidence artifacts (`research/evidence/`) |
| `docs/` | Install, build, architecture, roadmap and ADRs |
| `scripts/` | Build, package, validation and release tooling |

## Roadmap

The public plan is in [docs/ROADMAP.md](docs/ROADMAP.md). It covers signing and
supply-chain trust, weightless Rust/tabular runtime defaults without Python,
retention and alert delivery, a multi-site sensor and collector, and Linux and
macOS work.

Windows is the current target. Linux and macOS support is in progress and those
platforms aren't ready for pilots yet.

## Security

Please report vulnerabilities privately, following [SECURITY.md](SECURITY.md).
Never attach packet captures, credentials or personal data to a public issue.

## License

Source code and documentation are licensed under Apache-2.0; see [LICENSE](LICENSE).
Model weights are distributed separately for research and evaluation; see
[models/NOTICE.md](models/NOTICE.md). Third-party notices are in [NOTICE](NOTICE).

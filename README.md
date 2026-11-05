# Argus Next

[![CI](https://github.com/blah-blah-cell/argus-next-public/actions/workflows/ci.yml/badge.svg)](https://github.com/blah-blah-cell/argus-next-public/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Argus Next is a Windows app for making network activity easier to inspect. It
brings packet capture, flow information and alerts into one place. I'm especially
interested in whether a tool like this could be useful to smaller organizations
that don't have a dedicated security team.

It is a prototype, not a security product. A forecast is not an attack verdict, and
I haven't yet established how well this detects threats in a real network or how
often it raises false alerts. Please don't rely on it as your only security tool.

## Current status

**Please do not install or run the uninstaller from the published alpha.** Its
installation and removal behavior needs further verification. Fixes are in the
source, but have not yet been validated as a replacement release. The repository
is available for code review; it is not yet ready for an NGO deployment.

The app currently targets Windows. Model forecasting is optional and requires a
separately configured Python/PyTorch runtime; the packaged weights alone do not
make forecasting available. See the [Windows build guide](docs/BUILD_WINDOWS.md).
Live capture is opt-in and may require administrator approval.

### Requirements

- Windows 10 or 11, x64, with the Microsoft Edge WebView2 Runtime.
- The Microsoft Visual C++ x64 runtime.
- Pktmon API support for live capture.

The [installation guide](docs/INSTALL_WINDOWS.md) documents intended behavior and
its limitations; it is not a recommendation to run the current alpha installer.

## What it does

The app can capture traffic, show network flows and replay packet captures. It also
includes models that forecast packet behavior. One of them, Adaptive Hierarchical
Event Memory (AHEM), is the focus of my research into how a small model can use a
flow's earlier history.

The public build uses its own settings and data directories under
`%LOCALAPPDATA%\ArgusNextPublic`, separate from any development installation.
Monitoring starts only when you choose to start it. The newer AHEM model is
experimental and disabled by default; it doesn't make security decisions.

## Privacy and pilots

I'm interested in working with organizations that would like to explore a pilot.
Before any capture, we'd agree on what traffic can be monitored, who can access the
data, how long it's kept and how findings are handled. Network metadata can be
sensitive; only capture traffic you have permission to monitor.

Argus processes captured traffic locally. Live monitoring writes temporary PCAP
segments containing packet bytes, which can include payloads; the separate learning
journal stores numeric features rather than payloads. Treat captures and exports as
sensitive. Settings and retained data stay on local disk, and uninstall keeps them
unless you choose otherwise. Installing dependencies requires internet access.

## Research

The question behind Argus is whether a small model can use a flow's own earlier,
causally observed transitions — only packets that have already arrived, never
future ones — to forecast that flow's next packet size and arrival delay. AHEM is
the attempt to answer it with a model small enough to run on an ordinary machine.

In a frozen experiment on a selected subset of CSE-CIC-IDS2018, AHEM v6.1 had
lower weighted forecasting loss than a similarly sized flow-lane LSTM. That is a
forecast result on returning flows with eligible history, not a detection result.
The two models did not receive identical recent global inputs, and the sample did
not cover every packet or every kind of network, so the comparison isolates the
effect of memory only partly. It says nothing yet about false-positive rates or
how the model behaves on a real network. A later, larger experimental model
remains disabled by default for the same reason.

The [research overview](research/README.md) sets out the method, the caveats, the
outcomes that did not reproduce, and what is and isn't published. The shipped
weights are inference-only and hash-verified; their terms are in
[models/NOTICE.md](models/NOTICE.md).

I would rather publish a small honest result than a large ambiguous one, so
negative and inconclusive findings are documented next to the positive ones. If
you are reviewing the method and something is missing, open an issue and ask.

## Development

Build and test on Windows with Rust, Node 22+, and the Visual C++ build tools:

```powershell
.\scripts\build-windows.ps1
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
| `desktop/` | Tauri app and embedded HTML/CSS/JS interface |
| `crates/` | Rust workspace: core, capture, features, inference, detector, CLI |
| `native/pktmon-stream/` | Windows-only Pktmon stream sidecar |
| `installer/` | Per-user setup and uninstall program (separate workspace) |
| `python/` | Research, training and model export |
| `models/` | Inference-only weights and manifests; terms in [models/NOTICE.md](models/NOTICE.md) |
| `docs/` | Install, build, architecture, roadmap and ADRs |
| `scripts/` | Build, package, validation and release tooling |

## Roadmap

The public plan is in [docs/ROADMAP.md](docs/ROADMAP.md). It covers signing and
supply-chain trust, removing the Python runtime from the installed app, retention
and alert delivery, a multi-site sensor and collector, and Linux and macOS work.

Windows is the current target. Linux and macOS support is in progress and those
platforms aren't ready for pilots yet.

## Security

Please report vulnerabilities privately, following [SECURITY.md](SECURITY.md).
Never attach packet captures, credentials or personal data to a public issue.

## License

Source code and documentation are licensed under Apache-2.0; see [LICENSE](LICENSE).
Model weights are distributed separately for research and evaluation; see
[models/NOTICE.md](models/NOTICE.md). Third-party notices are in [NOTICE](NOTICE).

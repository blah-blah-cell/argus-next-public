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

## Install on Windows

1. Open the [latest release](https://github.com/blah-blah-cell/argus-next-public/releases)
   and download `ArgusNextPublic-windows-x64-setup.exe`.
2. Run it. It installs for the current user, so there is no administrator prompt.
3. Windows may warn that the publisher is unknown, because the build is unsigned.
   Check the SHA-256 from the release notes first — see
   [Verify your download](#verify-your-download).
4. Launch **Argus Next Public** from the Start menu or the desktop shortcut.
5. Capture starts stopped. Start monitoring explicitly, and only on networks you
   are authorized to inspect.
6. To remove it, use **Settings → Apps → Argus Next Public**. Captured data and
   settings are kept by default and can be deleted during uninstall.

The setup program verifies every embedded file before writing anything, then adds
shortcuts and an uninstall entry. It does not install a capture driver, register a
scheduled task, request elevation, create firewall rules, or download anything.
The elevated sensor is registered only when you start live capture.

Model forecasting is optional: the setup program ships the app and its weights, not
a Python runtime. Without the optional PyTorch runtime the app still runs, captures
and shows flows; the interface reports model readiness instead of hiding it. See
the model runtime section of the [Windows build guide](docs/BUILD_WINDOWS.md).

Full detail, requirements and command-line flags: [Windows install guide](docs/INSTALL_WINDOWS.md).

### Requirements

- Windows 10 or 11, x64, with the Microsoft Edge WebView2 Runtime.
- The Microsoft Visual C++ x64 runtime.
- Pktmon API support for live capture.

### Unattended install

```powershell
.\ArgusNextPublic-windows-x64-setup.exe --silent
.\ArgusNextPublic-windows-x64-setup.exe --install-dir D:\Tools\Argus --no-launch
.\ArgusNextPublic-windows-x64-setup.exe --verify
```

## Verify your download

Release notes publish the SHA-256 of both artifacts:

```powershell
Get-FileHash .\ArgusNextPublic-windows-x64-setup.exe -Algorithm SHA256
```

`--verify` re-hashes every embedded file. That establishes integrity relative to the
publisher's manifest, which is not the same as publisher identity: the binaries are
not code-signed yet. Treat an unexpected hash as a reason not to run the file.

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

Nothing is sent off the machine. Captures, settings and exported evidence stay on
local disk, and uninstall keeps them unless you say otherwise.

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

The checks below are what CI enforces:

```powershell
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-targets
cargo test --locked --release --manifest-path installer/Cargo.toml
node scripts/validate-desktop-ui.cjs
node --test scripts/test-release-runtime.cjs
node scripts/check-release-metadata.cjs
```

`just check` runs the same set if you have [`just`](https://github.com/casey/just).
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

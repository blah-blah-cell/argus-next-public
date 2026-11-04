# Build and run on Windows

This is a source release, not a signed installer. The desktop is a Windows-only
prototype. Keep the checkout in place after building: these source-build scripts
use its Python sources and model files. Copying only the EXE is not a supported
installation method.

## Prerequisites

- Windows x64 with Microsoft Edge WebView2 Runtime.
- Rust 1.96 or newer, using the MSVC toolchain.
- Visual Studio 2022 Build Tools with **Desktop development with C++** and a
  Windows SDK. The native sensor script locates these through `vswhere.exe`.
- Node.js 22 or newer for static UI validation. There are no npm dependencies.
- `uv` for the optional Python environment setup below. It creates a Python 3.12
  environment and installs PyTorch; downloads can be large. GPU hardware is not
  required. No training run is needed.
- A Windows version exposing `PktmonApi.dll` for live capture. Replay and CLI
  checks do not establish live-capture compatibility on your Windows build.

Install prerequisites through their official distributions. Run the following in
PowerShell from the repository root. If local execution policy blocks scripts,
review them and use an execution policy appropriate to your organization; these
instructions do not change machine policy.

## Build

```powershell
.\scripts\build-windows.ps1
```

The script validates static UI assets, builds the native Pktmon sidecar, and builds
the desktop and CLI with `Cargo.lock`. Outputs are in `target\release`. For a debug
build use `-Configuration Debug`. No Tauri CLI, npm install, scheduled-task
registration, or administrator elevation is required to compile. The script
temporarily uses the checkout's `target` directory even if `CARGO_TARGET_DIR` is
configured, so all three executables remain together.

## Model runtime and launch

```powershell
.\scripts\setup-windows-model-runtime.ps1
.\scripts\run-desktop.ps1
```

Setup installs the local Python package into `python\.venv` and probes PyTorch.
It does not download model checkpoints, start training, or start capture. The
`models` directory contains the released forecast-only checkpoint and manifest;
a successful forecast is not validated threat detection. Extra research/export
dependencies are opt-in with `-ResearchExtras` and are not needed for this launch.

For a custom Python environment:

```powershell
.\scripts\setup-windows-model-runtime.ps1 -RuntimeDirectory 'C:\ArgusRuntime'
$env:ARGUS_PYTHON = 'C:\ArgusRuntime\Scripts\python.exe'
.\scripts\run-desktop.ps1
```

Use the matching `-Configuration Debug` on the launcher for a debug build. The
launcher supplies absolute checkout paths when unset, preserves explicit
`ARGUS_PYTHON`, `ARGUS_PYTHONPATH`, and `ARGUS_MODEL_DIR` overrides, and restores
the calling shell's environment after exit. Source-built binaries also discover
`python\.venv` and `models` from the executable's recognized checkout layout,
without embedding a developer checkout path. Extracted packages use resources
beside the executable and never probe the caller's working directory. No private sibling
`argus-dev` folder is needed.

## Permissions and data

Only monitor networks you are authorized to inspect. Live capture uses the native
sensor and can request UAC elevation to register the `ArgusNextPublicSensor` scheduled
task. That is separate from building or setting up Python. Do not approve capture
or task registration unless intended. The sensor task runs with elevated
privileges; keep its binary and checkout in a location you control.

Opening the app does not automatically start capture. Start it explicitly in the
UI when authorized. The public build uses a separate Tauri/WebView identity,
sensor task, and desktop shortcut from the private development app.

Runtime state defaults to `%LOCALAPPDATA%\ArgusNextPublic`; captures default to its
`captures` subdirectory. `ARGUS_DEV_ROOT` and `ARGUS_CAPTURE_ROOT` can override
these with writable absolute paths. Treat network captures and metadata as
sensitive. No captured traffic or personal runtime state is distributed here.

Wireshark is optional for opening captures externally. ONNX Runtime is optional
and not included; the released checkpoint uses the Python/PyTorch worker. Set
`ARGUS_ONNX_RUNTIME` only when providing a compatible native runtime and artifacts.

## Checks without live capture

```powershell
cargo test --locked --workspace --all-targets
cargo fmt --all --check
node scripts/validate-desktop-ui.cjs
node --test scripts/test-release-runtime.cjs
.\target\release\argus-nids.exe --json --pktmon-probe
```

The last command probes Pktmon readiness without starting packet capture; it is
not a trained-model security evaluation. See
[the CLI guide](../crates/argus-nids/RUN_WINDOWS.md) for capture-file replay.

Building successfully does not verify UAC behavior, sensor compatibility,
network permissions, model quality, or the UI on another machine. If Python or
the checkpoint is unavailable, model forecasting is unavailable; do not treat
missing forecasts as evidence of benign traffic.

## Optional distributable ZIP

After a successful Release build, use Windows PowerShell 5.1 or newer:

```powershell
.\scripts\package-windows.ps1
```

This creates `dist\ArgusNextPublic-windows-x64.zip` plus its staging directory.
Only explicitly listed executables, the released checkpoint/manifest, Python
source, setup/launch scripts, any supplied license and portable instructions are
included. If present, the sanitized latest experimental checkpoint and v6.1 ONNX
export are included with their manifests. Source and documentation are
Apache-2.0 (see `LICENSE`); model weights are distributed separately for research
and evaluation (see `models/NOTICE.md`).
The embedded UI's vendored Three.js MIT license is copied separately to
`THIRD_PARTY/three-LICENSE.txt` in every package.
Python environments, captures, runtime data, caches and private files are not
copied. Every included file has a SHA-256 entry in `SHA256SUMS.txt`. The script
also prints the ZIP hash. It refuses existing output; choose a fresh
`-OutputDirectory` when rebuilding. This is not a signed installer or a verified
clean-machine deployment.

To build the per-user setup program from that staging directory, run
`.\scripts\build-installer.ps1`. It appends a hash-verified payload to a small
stub executable; the result installs to `%LOCALAPPDATA%\Programs\ArgusNextPublic`
and registers an uninstall entry. See [the install guide](INSTALL_WINDOWS.md).

## Continuous integration

`.github/workflows/ci.yml` is the enforcement layer. It runs four jobs on every
push and pull request:

| Job | Checks |
| --- | --- |
| Lint, UI, and metadata | `cargo fmt --check`, clippy with `-D warnings`, UI validation, release-isolation tests, version/license consistency |
| Rust and installer tests | `cargo test --locked --workspace --all-targets`, setup-program unit tests |
| Research code hygiene | Ruff syntax/undefined-name checks, byte-compilation |
| Supply chain | `cargo-deny` advisories/bans/licenses/sources, secret scanning |

On `main` and on manual dispatch it also builds the release binaries, packages
them, builds the setup program, and runs
`scripts/test-installer.ps1` — verify, silent install into a temp directory,
assert the file set, silent uninstall, assert the directory is gone.

Tagging a commit `v*` drives `.github/workflows/release.yml`: tests, build,
package, setup program, installer smoke test, optional Authenticode signing when
certificate secrets exist, SHA-256 manifests, a CycloneDX SBOM, build provenance,
and the GitHub release itself.

Run `node scripts/check-release-metadata.cjs` after changing any version. It
asserts that the crates, `tauri.conf.json`, the installer crate, the pinned
toolchain, and `CHANGELOG.md` agree.

# Argus Next Public: Windows source-build distribution

This unsigned prototype ZIP is not an installer. Extract the whole folder to a
location you control. Keep all files together. Microsoft Edge WebView2 Runtime,
the Visual C++ x64 runtime and a compatible Windows x64 installation are required.
Live capture additionally needs Windows Pktmon API support. This package has not
been validated as a security detector and must not be your only security tool.

## First launch

Install `uv` from its official distribution. In PowerShell from the extracted
folder, review and run:

```powershell
.\scripts\setup-windows-model-runtime.ps1
.\scripts\run-desktop.ps1
```

Setup downloads Python 3.12 if necessary and installs PyTorch and NumPy into
`python\.venv`. These downloads can be large; GPU hardware is optional. The
v6.1 checkpoint is already included. When available in the source release, the
sanitized latest experimental checkpoint and v6.1 ONNX export are included with
their manifests too. They remain forecast-only research artifacts. ONNX inference
also requires a separately provided compatible native runtime; no ONNX runtime is
bundled. Setup does not download another model or train.
Research/export extras are optional and not needed for the included checkpoint.

The app starts with capture stopped. Start monitoring explicitly only on networks
you are authorized to inspect. Starting live capture can request elevation to
register the `ArgusNextPublicSensor` scheduled task. A desktop shortcut named
`Argus Next Public` can be created on launch. Closing the window can leave the app
in its tray; use the tray's Quit command to exit.

Temporary live PCAP segments can contain packet payloads and should be treated as
sensitive. The separate learning journal stores numeric features without payloads;
this does not make the raw capture metadata-only.

State and captures default to `%LOCALAPPDATA%\ArgusNextPublic`, separate from the
private developer app. Do not set `ARGUS_DEV_ROOT` or `ARGUS_CAPTURE_ROOT` to an
existing private app directory. These and other `ARGUS_*` environment overrides
are respected when explicitly provided. No cookies, user state, Python venv,
captures, ONNX runtime, or personal configuration is distributed in this ZIP.
Forecasts are research outputs, not validated attack detections.

`SHA256SUMS.txt` lists SHA-256 hashes for every packaged file except the manifest
itself. Hashes establish integrity relative to the manifest, not publisher
authenticity. There is no code signature or automatic updater.

Source code and documentation are licensed under Apache-2.0; the root `LICENSE`
file is authoritative. Model weights are distributed separately from that license
and remain research artifacts for evaluation use until a model license is
published: see `models/NOTICE.md`. Third-party components keep their own terms.
The embedded Three.js UI dependency retains its separate MIT notice at
`THIRD_PARTY/three-LICENSE.txt`.
The included hero asset provenance notice is at `THIRD_PARTY/hero-assets-NOTICE.md`.

For a Pktmon readiness probe (does not capture traffic):

```powershell
.\argus-nids.exe --json --pktmon-probe
```

The distribution does not silently uninstall its task or delete collected data.
Review your organization's retention policy before handling capture files.

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

State and captures default to `%LOCALAPPDATA%\ArgusNextPublic`, separate from the
private developer app. Do not set `ARGUS_DEV_ROOT` or `ARGUS_CAPTURE_ROOT` to an
existing private app directory. These and other `ARGUS_*` environment overrides
are respected when explicitly provided. No cookies, user state, Python venv,
captures, ONNX runtime, or personal configuration is distributed in this ZIP.
Forecasts are research outputs, not validated attack detections.

`SHA256SUMS.txt` lists SHA-256 hashes for every packaged file except the manifest
itself. Hashes establish integrity relative to the manifest, not publisher
authenticity. There is no code signature or automatic updater.

Licensing is pending the author's confirmation. The source workspace metadata
declares Apache-2.0, but no root license text or separate model-weight license was
provided at preparation time. Do not assume that declaration grants blanket
redistribution rights for all source, weights, or third-party dependencies.

For a model-free synthetic CLI check (does not capture traffic):

```powershell
.\argus-nids.exe --json
```

The distribution does not silently uninstall its task or delete collected data.
Review your organization's retention policy before handling capture files.

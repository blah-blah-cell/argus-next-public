# Installing Argus Next on Windows

**Pre-release safety notice:** Do not use the currently published alpha installer
or uninstaller. Fixes are in the source but a replacement release has not yet
been verified. The instructions below describe intended behavior, not a current
recommendation to install.

The release page provides a setup program for 64-bit Windows. It installs for the
current user, needs no administrator rights, and starts with capture stopped.

## Requirements

- Windows 10 or 11, x64, with Microsoft Edge WebView2 Runtime (present on most
  installations; the app will not start without it).
- The Microsoft Visual C++ x64 runtime.
- Windows Pktmon API support if you intend to capture live traffic.

## Install

1. Download `ArgusNextPublic-windows-x64-setup.exe` from the release page.
2. Run it. Windows may warn that the publisher is unknown because the build is
   unsigned; see [Verifying a download](#verifying-a-download) before choosing to
   run it.
3. Accept the install location, or pass `--install-dir <path>` for a custom one.
   The default is `%LOCALAPPDATA%\Programs\ArgusNextPublic`.
4. Setup verifies every embedded file, installs it, adds Start menu and desktop
   shortcuts, and registers an entry in **Settings → Apps** so it can be removed
   normally.

Setup verifies the embedded payload before extraction. An I/O or registration
failure during installation can still leave a partial installation behind.

The revised installer requires a new, non-existing installation directory and
will not overwrite an existing installation, shortcut or registration. In-place
upgrades are not yet supported. An interrupted installation may need manual review
and cleanup before retrying; do not remove files you do not recognize.

## First run

The app opens with capture stopped. Start monitoring explicitly, and only on
networks you are authorized to inspect. Settings and captures live in
`%LOCALAPPDATA%\ArgusNextPublic`, separate from any development installation.

**Model forecasting is optional.** The setup program ships the app and its
weights, not a Python runtime. Without the optional PyTorch runtime the app still
runs, captures, and shows flows; forecasting becomes available after following
the model runtime section of the [Windows build guide](BUILD_WINDOWS.md). The
interface reports model readiness rather than silently hiding it.

## Command line

```powershell
ArgusNextPublic-windows-x64-setup.exe --silent           # unattended install
ArgusNextPublic-windows-x64-setup.exe --no-launch        # install without offering to start
ArgusNextPublic-windows-x64-setup.exe --install-dir D:\Tools\Argus
ArgusNextPublic-windows-x64-setup.exe --verify           # re-check the embedded payload
```

## Uninstall

Use **Settings → Apps → Argus Next Public**, or run the installed
`argus-uninstall.exe --uninstall` (the installer itself is not kept). Uninstall
removes verified installer-owned program files, shortcuts and the uninstall entry.
Captured data, settings, user-added files and nonempty directories are retained.
Uninstall refuses to proceed if the ownership record is missing, files have changed,
or the path cannot be verified; older alpha, interrupted or moved installations
may require manual removal. These checks do not prevent a malicious process running
as the same user from racing filesystem operations.

## Verifying a download

The release notes publish the SHA-256 of the setup program:

```powershell
Get-FileHash .\ArgusNextPublic-windows-x64-setup.exe -Algorithm SHA256
```

Once downloaded, `--verify` re-hashes every embedded file. This establishes
integrity relative to the publisher's manifest, not publisher identity: there is
no code signature.

## What setup does not do

It does not install a packet capture driver, register a scheduled task, start a
capture, request elevation, create firewall rules, or download anything. The
elevated sensor task is registered only when you start live capture in the app.

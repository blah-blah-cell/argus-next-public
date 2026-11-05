# Architecture

Argus Next is a Windows desktop application with a native capture sidecar, a
research package, and a packaging pipeline. This document describes the pieces and
the boundaries between them. Decisions with lasting consequences are recorded in
[`docs/adr/`](adr/README.md).

## Processes

| Process | Built from | Role |
| --- | --- | --- |
| `argus-desktop.exe` | `desktop/` (Tauri) | UI, state, storage, inference, IPC host |
| `argus-pktmon-stream.exe` | `native/pktmon-stream/` | Elevated Pktmon stream reader, writes capture files |
| `argus-nids.exe` | `crates/argus-nids/` | Headless CLI for probes, replay, and diagnostics |
| `ArgusNextPublic-windows-x64-setup.exe` | `installer/` | Per-user install and uninstall |

The desktop process never captures packets itself. It starts the sidecar through
the elevation path and reads the files the sidecar produces. The sidecar holds no
UI state and no credentials.

## Crates

- `argus-core` — domain types, error taxonomy, shared contracts.
- `argus-capture` — capture file discovery, parsing, and flow reconstruction.
- `argus-features` — feature extraction over flows and windows.
- `argus-model-artifact` — artifact manifests, schema versions, hash validation.
- `argus-inference` — model loading and the inference path (CPU by default).
- `argus-moe` — mixture-of-experts routing used by the temporal ensemble.
- `argus-detector` — detection rules and alert assembly.
- `argus-nids` — CLI surface for the above.
- `argus-desktop` — Tauri commands, state, storage, and the embedded UI.

Dependencies point inward: `argus-core` depends on nothing in the workspace,
`argus-desktop` depends on the rest. The installer is a separate workspace so its
dependency graph and lockfile stay independent.

## UI

The interface is static HTML, CSS, and JavaScript under `desktop/ui/`, embedded in
the binary. There is no bundler and no runtime download. `scripts/validate-desktop-ui.cjs`
checks that every referenced asset exists and every inline handler resolves, and
`node --test scripts/test-release-runtime.cjs` asserts the release-specific
isolation rules.

The onboarding scene is part of the product's first impression. It is intentionally
untouched by this work and covered by its own validation.

## Data flow

1. The user starts capture. The app asks for elevation and launches the sidecar.
2. The sidecar writes capture segments to the user data directory.
3. `argus-capture` discovers and parses segments into flows.
4. `argus-features` builds windows; `argus-inference` scores them when a model is
   loaded; `argus-detector` turns scores and rules into alerts.
5. The UI reads state through Tauri commands. Nothing is sent off the machine.

## Storage

- Install: `%LOCALAPPDATA%\Programs\ArgusNextPublic` (per user, no admin).
- Data: `%LOCALAPPDATA%\ArgusNextPublic`, retained across uninstall.
- Settings and capture metadata live beside the data. Temporary live-capture
  PCAP segments can contain payloads; a separate payload opt-in and fully
  verified metadata-only mode are not yet implemented (ADR 0003).

## Trust boundaries

- **Elevation:** the sidecar runs elevated, the UI does not. Arguments crossing that
  boundary are validated; the sidecar must not execute arbitrary paths.
- **Loopback IPC:** treated as untrusted input even though it is local.
- **Capture files:** parsed as hostile input; parsers are the primary fuzzing target.
- **Model artifacts:** manifests and whole-artifact hashes are checked against
  their contracts. The manifests do not provide per-tensor digests; see the
  model notice for artifact and loading limitations.
- **Installer payload:** every entry is verified before it is written.

## Models

Two artifacts ship: a validated base ensemble and a shadow challenger. Both are
inference-only; no optimizer, RNG, or epoch state is published. Shadow output is
recorded for comparison and never changes a user-visible alert. Manifests carry a
schema version so an older app rejects a newer artifact instead of guessing.

## Extension points

- New capture formats belong behind the parser interface in `argus-capture`.
- New models must ship a manifest, a hash, and a model card; evaluation gates live
  with the research package.
- New alert destinations belong in `argus-detector`, behind an explicit opt-in.

## Where the checks live

`cargo xtest`, `cargo xlint`, `cargo xfmt`, and `cargo xinstaller` are the local
aliases; `.github/workflows/ci.yml` runs the same commands. Release artifacts are
produced only by `.github/workflows/release.yml` on a tag.

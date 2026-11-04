# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private
vulnerability reporting on this repository (Security → Advisories → Report a
vulnerability), or contact the maintainer listed in `CODEOWNERS`.

Include what you can: affected version or commit, Windows build, whether live
capture was running, reproduction steps, and the impact you believe it has. Do
not attach packet captures, credentials, or personal data.

## What to expect

- Acknowledgement within 7 days.
- An initial assessment within 14 days.
- A fix or a documented mitigation plan for confirmed issues, with credit unless
  you ask otherwise.

This is a prototype maintained by a small team. These are best-effort targets,
not a commercial support contract.

## Scope

In scope:

- The desktop application, CLI tools, native Pktmon sidecar, and setup program.
- Privilege boundaries: UAC elevation, the `ArgusNextPublicSensor` scheduled
  task, and the loopback IPC surface.
- Model artifact loading, manifest validation, and the inference path.
- Packaging and release artifacts (payload verification, path handling).

Out of scope:

- Vulnerabilities in Windows, WebView2, Pktmon, PyTorch, ONNX Runtime, or other
  third-party components; report those upstream.
- Findings that require an already-compromised administrator account.
- Missing hardening that has no demonstrated impact.
- The accuracy of forecasts. Model quality is a research question, not a
  security vulnerability; see `research/README.md`.

## Handling captured data

Argus can capture network traffic. Never send captures, cookies, credentials, or
personal data in a report. If a reproduction needs traffic, describe it or build
a synthetic capture.

# 0004 — Pinned toolchain and CI gates

- Status: Accepted
- Date: 2026-11-04

## Context

The workspace declares `rust-version = "1.96"`, but nothing forced a contributor
or a release build to use it, no formatter or linter configuration was committed,
and every check was run by hand. That makes "it builds on my machine" the only
guarantee, and it makes release artifacts depend on whoever ran the last build.

## Decision

- Pin the toolchain in `rust-toolchain.toml`; CI installs exactly that channel.
- Commit `rustfmt.toml`, `.editorconfig`, and Cargo aliases so local and CI
  commands are the same commands.
- Make CI the enforcement layer: formatting, clippy with `-D warnings`, workspace
  tests, installer tests, UI validation, release-isolation tests, and release
  metadata consistency, plus dependency/license/source policy and secret scanning.
- Keep the installer as its own workspace so its lockfile cannot perturb the
  application dependency graph.
- Generated artifacts are produced only by CI on a tag, with checksums, an SBOM,
  keyless signatures, and build provenance attached.

## Consequences

- A red check is a real blocker rather than a suggestion.
- Dependency upgrades arrive as reviewable pull requests instead of silent drift.
- CI cost and a small maintenance burden; the workflows must be updated when
  scripts change.
- Actions are currently referenced by version tag. Pinning them to commit SHAs is
  tracked as follow-up work, because it requires a reviewed update process.

# Contributing

Thanks for taking a look. This is a small public prototype, so the most useful
contributions are bug reports with clear reproduction steps, documentation
fixes, and narrowly scoped changes.

## Before you start

- Open an issue for anything larger than a typo or a small fix. It is cheaper to
  agree on the shape of a change than to review a surprise.
- Do not include packet captures, credentials, cookies, or personal data in an
  issue, pull request, or test fixture.
- Do not change the onboarding scene, its assets, or the `THIRD_PARTY` notices
  without discussing it first.

## Development setup

Follow [docs/BUILD_WINDOWS.md](docs/BUILD_WINDOWS.md). The short version:

```powershell
.\scripts\build-windows.ps1          # UI validation + native sidecar + Rust build
node --test scripts/test-release-runtime.cjs
```

Model work additionally needs `.\scripts\setup-windows-model-runtime.ps1`.

## Checks

CI runs these, and they should pass locally before you push:

```powershell
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-targets
node scripts/validate-desktop-ui.cjs
node --test scripts/test-release-runtime.cjs
cargo test --locked --release --manifest-path installer/Cargo.toml
node scripts/check-release-metadata.cjs
```

If you change packaging, also run `.\scripts\package-windows.ps1` into a fresh
`-OutputDirectory`, then `.\scripts\test-installer.ps1` against the result.

## Pull requests

- Keep the change focused. Separate refactors from behaviour changes.
- Add or update tests for behaviour you change.
- Update the relevant document in `docs/` when behaviour or a contract changes.
- Record a decision in `docs/adr/` when you make an architectural choice that a
  future reader would otherwise have to reverse-engineer.
- Describe what you actually ran. "Not tested" is acceptable; claiming a test you
  did not run is not.

## Commit messages

Use an imperative subject line, then explain why in the body when it is not
obvious. Generated code is welcome, but it needs the same tests and review as
anything else, and a human stays accountable for it.

## Licensing of contributions

By contributing you agree that your contribution is licensed under the same terms
as the repository (see `LICENSE`). Model weights are separate; see
`models/NOTICE.md`.

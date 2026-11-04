## What changed

<!-- One or two sentences. Explain why, not just what. -->

## How it was checked

<!-- List the commands you actually ran. "Not run" is fine; claiming a run you did not do is not. -->

- [ ] `cargo fmt --all --check`
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- [ ] `cargo test --locked --workspace --all-targets`
- [ ] `node scripts/validate-desktop-ui.cjs`
- [ ] `node --test scripts/test-release-runtime.cjs`
- [ ] `cargo test --locked --release --manifest-path installer/Cargo.toml`
- [ ] `node scripts/check-release-metadata.cjs`

## Impact

- [ ] Capture behaviour is unchanged (no automatic capture, no elevation prompt).
- [ ] Onboarding scene and assets are untouched.
- [ ] Model artifacts remain inference-only and unchanged, or the change is described above.
- [ ] No secrets, captures, credentials, or personal paths are included.
- [ ] Packaging changed, and `scripts/test-installer.ps1` was run.
- [ ] Documentation updated (README, docs/, ADR).

## Related issues

<!-- Closes #... -->

# Task runner. Optional: run the underlying command if `just` is not installed.
# Windows: `winget install --id Casey.Just`
set windows-shell := ["powershell.exe", "-NoProfile", "-Command"]

default:
    @just --list

# Format, lint, test, and check metadata: what CI requires.
check:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets --all-features -- -D warnings
    cargo test --locked --workspace --all-targets
    cargo test --locked --release --manifest-path installer/Cargo.toml
    node scripts/validate-desktop-ui.cjs
    node --test scripts/test-release-runtime.cjs
    node scripts/check-release-metadata.cjs

fmt:
    cargo fmt --all

lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
    cargo test --locked --workspace --all-targets

ui:
    node scripts/validate-desktop-ui.cjs
    node --test scripts/test-release-runtime.cjs

installer-test:
    cargo test --locked --release --manifest-path installer/Cargo.toml

metadata:
    node scripts/check-release-metadata.cjs

build:
    ./scripts/build-windows.ps1

package:
    ./scripts/package-windows.ps1 -OutputDirectory dist

# Build and smoke-test the setup program against a package in dist/.
installer: package
    ./scripts/build-installer.ps1 -PackageDirectory dist/ArgusNextPublic-windows-x64 -OutputPath dist/setup/ArgusNextPublic-windows-x64-setup.exe
    ./scripts/test-installer.ps1 -SetupPath dist/setup/ArgusNextPublic-windows-x64-setup.exe

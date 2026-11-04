[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:OS -ne 'Windows_NT') { throw 'Argus desktop builds require Windows.' }
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$Cargo = (Get-Command cargo -ErrorAction Stop).Source
$Node = (Get-Command node -ErrorAction Stop).Source

# Keep the sensor and Rust executables together, including with a custom target dir.
$previousTarget = $env:CARGO_TARGET_DIR
try {
    $env:CARGO_TARGET_DIR = Join-Path $RepositoryRoot 'target'
    Push-Location $RepositoryRoot
    try {
        & $Node (Join-Path $PSScriptRoot 'validate-desktop-ui.cjs')
        if ($LASTEXITCODE -ne 0) { throw 'Desktop UI validation failed.' }
        & (Join-Path $RepositoryRoot 'native/pktmon-stream/build.ps1') -Configuration $Configuration
        if ($LASTEXITCODE -ne 0) { throw 'The native Pktmon sidecar build failed.' }
        $arguments = @('build', '--locked', '-p', 'argus-desktop', '-p', 'argus-nids')
        if ($Configuration -eq 'Release') { $arguments += '--release' }
        & $Cargo @arguments
        if ($LASTEXITCODE -ne 0) { throw 'The Rust desktop/CLI build failed.' }
    } finally {
        Pop-Location
    }
} finally {
    $env:CARGO_TARGET_DIR = $previousTarget
}
Write-Host "Build complete: $(Join-Path $RepositoryRoot ('target/' + $Configuration.ToLowerInvariant()))"
Write-Host 'Launch with scripts/run-desktop.ps1 using the same -Configuration.'

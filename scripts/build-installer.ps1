[CmdletBinding()]
param(
    [string]$PackageDirectory,
    [string]$OutputPath,
    [string]$Version
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$RepositoryRoot = Split-Path -Parent $PSScriptRoot

if (-not $PackageDirectory) {
    $PackageDirectory = Join-Path $RepositoryRoot 'dist\ArgusNextPublic-windows-x64'
}
if (-not $Version) {
    $tauri = Get-Content -LiteralPath (Join-Path $RepositoryRoot 'desktop\src-tauri\tauri.conf.json') -Raw | ConvertFrom-Json
    $Version = $tauri.version
}
if (-not $OutputPath) {
    $OutputPath = Join-Path $RepositoryRoot 'dist\ArgusNextPublic-windows-x64-setup.exe'
}
$PackageDirectory = [IO.Path]::GetFullPath($PackageDirectory)
$OutputPath = [IO.Path]::GetFullPath($OutputPath)
if (-not (Test-Path -LiteralPath (Join-Path $PackageDirectory 'argus-desktop.exe') -PathType Leaf)) {
    throw "Package directory does not contain argus-desktop.exe: $PackageDirectory"
}
if (Test-Path -LiteralPath $OutputPath) {
    throw "Output already exists. Choose a fresh -OutputPath: $OutputPath"
}

$InstallerRoot = Join-Path $RepositoryRoot 'installer'
$Cargo = (Get-Command cargo -ErrorAction Stop).Source
& $Cargo build --release --manifest-path (Join-Path $InstallerRoot 'Cargo.toml')
if ($LASTEXITCODE -ne 0) { throw 'The setup program failed to compile.' }

$Stub = Join-Path $InstallerRoot 'target\release\argus-setup.exe'
$Builder = Join-Path $InstallerRoot 'target\release\argus-setup-builder.exe'
& $Builder --stub $Stub --dist $PackageDirectory --out $OutputPath --version $Version
if ($LASTEXITCODE -ne 0) { throw 'The setup executable could not be produced.' }

$hash = (Get-FileHash -LiteralPath $OutputPath -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "Setup: $OutputPath"
Write-Host "SHA256: $hash"
Write-Host 'The setup program installs per user and requests no elevation.'

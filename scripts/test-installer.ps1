#Requires -Version 5.1
<#
.SYNOPSIS
    Verifies, installs, and uninstalls the setup program without launching the app.

.DESCRIPTION
    Runs the published installer end to end in a temporary directory: payload
    verification, a silent install, an assertion that the expected files landed,
    then a silent uninstall and an assertion that the directory is gone.

    Captured data is left alone (uninstall never deletes it), and the desktop app
    is never started, so this is safe to run on a machine that already has Argus
    installed elsewhere.

.PARAMETER SetupPath
    Path to ArgusNextPublic-windows-x64-setup.exe.

.PARAMETER InstallDirectory
    Where to install. Defaults to a fresh directory under the temp folder.
#>
param(
    [Parameter(Mandatory = $true)][string]$SetupPath,
    [string]$InstallDirectory
)

$ErrorActionPreference = 'Stop'
$setup = (Resolve-Path -LiteralPath $SetupPath).Path

if (-not $InstallDirectory) {
    $InstallDirectory = Join-Path ([IO.Path]::GetTempPath()) ("argus-smoke-" + [Guid]::NewGuid().ToString('N').Substring(0, 8))
}

Write-Host "Verifying payload: $setup"
& $setup --verify
if ($LASTEXITCODE -ne 0) { throw "Payload verification failed with exit code $LASTEXITCODE." }

if (Test-Path -LiteralPath $InstallDirectory) {
    Remove-Item -LiteralPath $InstallDirectory -Recurse -Force
}

Write-Host "Installing to $InstallDirectory"
& $setup --silent --install-dir $InstallDirectory
if ($LASTEXITCODE -ne 0) { throw "Install failed with exit code $LASTEXITCODE." }

$desktop = Join-Path $InstallDirectory 'argus-desktop.exe'
$uninstaller = Join-Path $InstallDirectory 'argus-uninstall.exe'
foreach ($required in @($desktop, $uninstaller)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing installed file: $required" }
}

$fileCount = (Get-ChildItem -LiteralPath $InstallDirectory -Recurse -File).Count
if ($fileCount -lt 200) { throw "Only $fileCount files installed; expected the full package." }

Write-Host "Installed $fileCount files. Uninstalling."
& $uninstaller --uninstall --silent
if ($LASTEXITCODE -ne 0) { throw "Uninstall failed with exit code $LASTEXITCODE." }

$deadline = (Get-Date).AddSeconds(45)
while ((Test-Path -LiteralPath $InstallDirectory) -and (Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
}
if (Test-Path -LiteralPath $InstallDirectory) {
    throw "Install directory still present after uninstall: $InstallDirectory"
}

Write-Host "Installer smoke test passed ($fileCount files installed and removed)."

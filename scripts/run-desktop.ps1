[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:OS -ne 'Windows_NT') { throw 'Argus desktop requires Windows.' }
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$Executable = Join-Path $RepositoryRoot ('target/' + $Configuration.ToLowerInvariant() + '/argus-desktop.exe')
if (Test-Path -LiteralPath (Join-Path $RepositoryRoot 'argus-desktop.exe') -PathType Leaf) {
    $Executable = Join-Path $RepositoryRoot 'argus-desktop.exe'
}
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw 'Desktop executable not found. Run scripts/build-windows.ps1 first.'
}

# Scope overrides to the child process launch; never persist machine/user settings.
$defaults = @{
    ARGUS_PYTHON = Join-Path $RepositoryRoot 'python/.venv/Scripts/python.exe'
    ARGUS_PYTHONPATH = Join-Path $RepositoryRoot 'python/src'
    ARGUS_MODEL_DIR = Join-Path $RepositoryRoot 'models'
}
$previous = @{}
try {
    foreach ($name in $defaults.Keys) {
        $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        if (-not $previous[$name]) {
            [Environment]::SetEnvironmentVariable($name, $defaults[$name], 'Process')
        }
    }
    if (-not (Test-Path -LiteralPath $env:ARGUS_PYTHON -PathType Leaf)) {
        throw 'Model Python not found. Run scripts/setup-windows-model-runtime.ps1 or set ARGUS_PYTHON.'
    }
    & $Executable
    if ($LASTEXITCODE -ne 0) { throw "Argus exited with code $LASTEXITCODE." }
} finally {
    foreach ($name in $previous.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process')
    }
}

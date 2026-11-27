[CmdletBinding()]
param(
    [string]$OutputRoot,
    [string]$ModelDirectory,
    [switch]$IncludeModels
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if ($IncludeModels) {
    if (-not $ModelDirectory) {
        $ModelDirectory = Join-Path $projectRoot 'models'
    }
    $ModelDirectory = [IO.Path]::GetFullPath($ModelDirectory)
    if (-not (Test-Path -LiteralPath $ModelDirectory -PathType Container)) {
        throw "Model directory missing: $ModelDirectory. Pass -ModelDirectory to an approved model directory."
    }
}
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $projectRoot 'dist'
}
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$config = Get-Content (Join-Path $projectRoot 'desktop\src-tauri\tauri.conf.json') -Raw | ConvertFrom-Json
$version = [string]$config.version
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$packageName = "argus-next-$version-windows-x64-$stamp"
$packageDirectory = Join-Path $OutputRoot $packageName
$archive = "$packageDirectory.zip"

New-Item -ItemType Directory -Path $packageDirectory -Force | Out-Null
Push-Location $projectRoot
try {
    & (Join-Path $projectRoot 'native\pktmon-stream\build.ps1') -Configuration Release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    cargo build --locked --release -p argus-nids -p argus-desktop
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $files = [ordered]@{
        'target\release\argus-desktop.exe' = 'argus-desktop.exe'
        'target\release\argus-nids.exe' = 'argus-nids.exe'
        'target\release\argus-pktmon-stream.exe' = 'argus-pktmon-stream.exe'
        'native\pktmon-stream\install-task.ps1' = 'install-task.ps1'
        'NOTICE' = 'NOTICE'
        'desktop\ui\vendor\three\LICENSE' = 'THIRD_PARTY\three-LICENSE.txt'
    }
    if (Test-Path -LiteralPath (Join-Path $projectRoot 'LICENSE') -PathType Leaf) {
        $files['LICENSE'] = 'LICENSE'
    }
    foreach ($relative in $files.Keys) {
        $source = Join-Path $projectRoot $relative
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Release input is missing: $relative"
        }
        $destination = Join-Path $packageDirectory $files[$relative]
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination
    }

    if ($IncludeModels) {
        # Ship the actual model code and artifacts; Python itself is set up on the
        # destination machine rather than copying a non-portable developer venv.
        $pythonDestination = Join-Path $packageDirectory 'python'
        New-Item -ItemType Directory -Path $pythonDestination -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $projectRoot 'python\pyproject.toml') -Destination $pythonDestination
        $pythonSource = Join-Path $projectRoot 'python\src'
        Get-ChildItem -LiteralPath $pythonSource -File -Recurse -Filter '*.py' | ForEach-Object {
            $relative = $_.FullName.Substring($pythonSource.Length + 1)
            $destination = Join-Path (Join-Path $pythonDestination 'src') $relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
            Copy-Item -LiteralPath $_.FullName -Destination $destination
        }
        $scriptsDestination = Join-Path $packageDirectory 'scripts'
        New-Item -ItemType Directory -Path $scriptsDestination -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $projectRoot 'scripts\setup-windows-model-runtime.ps1') -Destination $scriptsDestination
        Copy-Item -LiteralPath $ModelDirectory -Destination (Join-Path $packageDirectory 'models') -Recurse
    }

    @'
@echo off
setlocal
start "Argus Next" "%~dp0argus-desktop.exe"
'@ | Set-Content -LiteralPath (Join-Path $packageDirectory 'START-ARGUS-NEXT.cmd') -Encoding Ascii

    if ($IncludeModels) {
        $readmeText = @"
Argus Next $version - Windows x64 portable release

1. Extract this entire folder to a stable location.
2. Install uv and Python 3.12, then run in PowerShell from this folder:
   .\scripts\setup-windows-model-runtime.ps1 -RuntimeDirectory "`$PWD\python\.venv"
3. Run START-ARGUS-NEXT.cmd.
4. Approve the administrator prompt the first time live monitoring starts.

Runtime state is stored under %LOCALAPPDATA%\ArgusNextPublic by default. Set
ARGUS_DEV_ROOT to an absolute writable folder before launch to override it.
Model weights are included but make no trained security verdict. The optional
Argus CERO checkpoint is not included; the built-in forecast router is used.
Wireshark is optional.
"@
    } else {
        $readmeText = @"
Argus Next $version - Windows x64 portable release (weightless runtime)

1. Extract this entire folder to a stable location.
2. Run START-ARGUS-NEXT.cmd.
3. Approve the administrator prompt the first time live monitoring starts.

Runtime state is stored under %LOCALAPPDATA%\ArgusNextPublic by default. Set
ARGUS_DEV_ROOT to an absolute writable folder before launch to override it.
No model weights or Python runtime are bundled in this portable package; the
built-in runtime is used without external weights. Wireshark is optional.
"@
    }
    $readmeText | Set-Content -LiteralPath (Join-Path $packageDirectory 'README.txt') -Encoding UTF8

    $manifest = Get-ChildItem -LiteralPath $packageDirectory -File -Recurse | Sort-Object FullName | ForEach-Object {
        [ordered]@{
            file = $_.FullName.Substring($packageDirectory.Length + 1).Replace('\', '/')
            bytes = $_.Length
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    [ordered]@{
        schema = 'argus.windows-portable-release'
        version = 1
        productVersion = $version
        target = 'x86_64-pc-windows-msvc'
        createdUtc = [DateTime]::UtcNow.ToString('o')
        modelWeightsIncluded = [bool]$IncludeModels
        files = @($manifest)
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $packageDirectory 'release-manifest.json') -Encoding UTF8

    Compress-Archive -LiteralPath $packageDirectory -DestinationPath $archive
    Write-Host "Portable release: $packageDirectory"
    Write-Host "Archive: $archive"
} finally {
    Pop-Location
}

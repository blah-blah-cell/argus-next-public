[CmdletBinding()]
param([string]$OutputDirectory)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $RepositoryRoot 'dist' }
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
$PackageRoot = Join-Path $OutputDirectory 'ArgusNextPublic-windows-x64'
$Archive = "$PackageRoot.zip"
if ((Test-Path -LiteralPath $PackageRoot) -or (Test-Path -LiteralPath $Archive)) {
    throw 'Package output already exists. Choose a fresh -OutputDirectory; existing files are never overwritten.'
}

# Explicit allowlist: never copy a venv, private runtime state, captures or caches.
$files = [ordered]@{
    'target/release/argus-desktop.exe' = 'argus-desktop.exe'
    'target/release/argus-nids.exe' = 'argus-nids.exe'
    'target/release/argus-pktmon-stream.exe' = 'argus-pktmon-stream.exe'
    'native/pktmon-stream/install-task.ps1' = 'install-task.ps1'
    'models/ahem_v61_forecast_only.pt' = 'models/ahem_v61_forecast_only.pt'
    'models/ahem_v61_forecast_only.pt.manifest.json' = 'models/ahem_v61_forecast_only.pt.manifest.json'
    'python/pyproject.toml' = 'python/pyproject.toml'
    'scripts/setup-windows-model-runtime.ps1' = 'scripts/setup-windows-model-runtime.ps1'
    'scripts/run-desktop.ps1' = 'scripts/run-desktop.ps1'
    'docs/PORTABLE_WINDOWS.md' = 'README.md'
    'desktop/ui/spatial/assets/NOTICE.md' = 'THIRD_PARTY/hero-assets-NOTICE.md'
    'desktop/ui/vendor/three/LICENSE' = 'THIRD_PARTY/three-LICENSE.txt'
}
$optionalModels = @(
    'models/ahem_v61.onnx',
    'models/challengers/ahem_e2e_experimental_step3808.pt'
)
foreach ($model in $optionalModels) {
    if (Test-Path -LiteralPath (Join-Path $RepositoryRoot $model) -PathType Leaf) {
        $files[$model] = $model
        $files["$model.manifest.json"] = "$model.manifest.json"
    }
}
if (Test-Path -LiteralPath (Join-Path $RepositoryRoot 'models/ahem_v61.manifest.json') -PathType Leaf) {
    $files['models/ahem_v61.manifest.json'] = 'models/ahem_v61.manifest.json'
}
if (Test-Path -LiteralPath (Join-Path $RepositoryRoot 'LICENSE') -PathType Leaf) {
    $files['LICENSE'] = 'LICENSE'
}
foreach ($source in $files.Keys) {
    if (-not (Test-Path -LiteralPath (Join-Path $RepositoryRoot $source) -PathType Leaf)) {
        throw "Required package input is missing: $source. Build Release first."
    }
}
$PythonSource = Join-Path $RepositoryRoot 'python/src'
$pythonFiles = @(Get-ChildItem -LiteralPath $PythonSource -Recurse -File -Filter '*.py')
if ($pythonFiles.Count -eq 0) { throw 'Python runtime sources are missing.' }
New-Item -ItemType Directory -Path $PackageRoot -Force | Out-Null
foreach ($source in $files.Keys) {
    $destination = Join-Path $PackageRoot $files[$source]
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $RepositoryRoot $source) -Destination $destination
}
foreach ($source in $pythonFiles) {
    $relative = [IO.Path]::GetRelativePath($PythonSource, $source.FullName)
    $destination = Join-Path $PackageRoot (Join-Path 'python/src' $relative)
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $source.FullName -Destination $destination
}
$manifest = @(Get-ChildItem -LiteralPath $PackageRoot -Recurse -File | Sort-Object FullName | ForEach-Object {
    $relative = [IO.Path]::GetRelativePath($PackageRoot, $_.FullName).Replace('\', '/')
    '{0}  {1}' -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(), $relative
})
[IO.File]::WriteAllLines((Join-Path $PackageRoot 'SHA256SUMS.txt'), $manifest, [Text.UTF8Encoding]::new($false))
Compress-Archive -LiteralPath $PackageRoot -DestinationPath $Archive -CompressionLevel Optimal
Write-Host "Package: $Archive"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host 'No Python environment, user data, packet captures or ONNX runtime is included.'

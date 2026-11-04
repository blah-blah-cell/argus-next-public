param(
    [string]$RuntimeDirectory,
    [switch]$ResearchExtras
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$PythonProject = Join-Path $RepositoryRoot "python"
if (-not $RuntimeDirectory) {
    $RuntimeDirectory = Join-Path $PythonProject ".venv"
}
$RuntimeDirectory = [IO.Path]::GetFullPath($RuntimeDirectory)

$Uv = (Get-Command uv -ErrorAction Stop).Source
$RuntimePython = Join-Path $RuntimeDirectory "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $RuntimePython -PathType Leaf)) {
    & $Uv venv --python 3.12 $RuntimeDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "uv could not create the Windows model runtime."
    }
}

$Package = if ($ResearchExtras) { "${PythonProject}[dev,export,laya-router]" } else { $PythonProject }
& $Uv pip install `
    --python $RuntimePython `
    --torch-backend auto `
    --editable $Package
if ($LASTEXITCODE -ne 0) {
    throw "uv could not synchronize the Windows model runtime."
}

& $RuntimePython -c @"
import torch
import argus_research.experimental_cnn

accelerator = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Argus model runtime ready: torch={torch.__version__} accelerator={accelerator}")
"@
if ($LASTEXITCODE -ne 0) {
    throw "The synchronized model runtime failed its PyTorch/Argus import probe."
}
Write-Host "Python: $RuntimePython"
Write-Host 'For a custom runtime location, set ARGUS_PYTHON to the Python path above before launching Argus.'

[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Debug'
)

$ErrorActionPreference = 'Stop'
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$installation = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'Visual C++ Build Tools are required' }
$developerShell = Join-Path $installation 'Common7\Tools\Launch-VsDevShell.ps1'
& $developerShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$profile = $Configuration.ToLowerInvariant()
$outputDirectory = Join-Path $PSScriptRoot "..\..\target\$profile"
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$output = Join-Path $outputDirectory 'argus-pktmon-stream.exe'
$object = Join-Path $outputDirectory 'argus-pktmon-stream.obj'
$optimization = if ($Configuration -eq 'Release') { '/O2' } else { '/Od' }
cl.exe /nologo /std:c++20 /EHsc /W4 /WX $optimization /DUNICODE /D_UNICODE "$PSScriptRoot\main.cpp" "/Fo$object" "/Fe$output" /link /SUBSYSTEM:WINDOWS Shell32.lib
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

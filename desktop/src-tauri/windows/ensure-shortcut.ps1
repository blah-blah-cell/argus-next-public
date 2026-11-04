param(
    [Parameter(Mandatory=$true)][string]$Executable,
    [Parameter(Mandatory=$true)][string]$IconPath,
    [string]$DesktopDirectory = [Environment]::GetFolderPath('Desktop')
)
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($DesktopDirectory)) { throw 'Desktop directory unavailable' }
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw 'Shortcut executable is missing' }
if (-not (Test-Path -LiteralPath $IconPath -PathType Leaf)) { throw 'Shortcut icon is missing' }
$path = Join-Path $DesktopDirectory 'Argus Next Public.lnk'
$existed = Test-Path -LiteralPath $path -PathType Leaf
$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut($path)
if ($existed -and $link.TargetPath -ne $Executable) {
    # Do not hijack a shortcut belonging to another install or a user's wrapper.
    return
}
if (-not $existed) {
    $link.TargetPath = $Executable
    $link.WorkingDirectory = Split-Path -Parent $Executable
    $link.Description = 'Argus Next network monitoring'
}
$iconLocation = "$IconPath,0"
if (-not $existed -or $link.IconLocation -ne $iconLocation) {
    $link.IconLocation = $iconLocation
    $link.Save()
}

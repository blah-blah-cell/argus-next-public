param(
    [ValidatePattern('^[A-Za-z0-9._-]{1,64}$')][string]$TaskName = 'ArgusNextPublicSensor',
    [Parameter(Mandatory)][string]$Sensor,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [Parameter(Mandatory)][string]$StopFile,
    [ValidateRange(1, 60)][int]$SegmentSeconds = 5
)

$ErrorActionPreference = 'Stop'
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Administrator elevation is required.'
}
$Sensor = [IO.Path]::GetFullPath($Sensor)
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
$StopFile = [IO.Path]::GetFullPath($StopFile)
if (-not [IO.File]::Exists($Sensor)) {
    throw 'The native sensor executable does not exist.'
}
if (-not [IO.Directory]::Exists($OutputDirectory)) {
    throw 'The live output directory does not exist.'
}
if ([IO.Path]::GetDirectoryName($StopFile) -ine $OutputDirectory -or [IO.Path]::GetFileName($StopFile) -ine 'sensor.stop') {
    throw 'The stop marker must be sensor.stop inside the live output directory.'
}
$action = New-ScheduledTaskAction -Execute $Sensor -Argument ('"{0}" "{1}" {2}' -f $OutputDirectory, $StopFile, $SegmentSeconds)
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings -Description 'Argus Next Pktmon streaming sensor. Starts only when the desktop requests it.' -Force | Out-Null

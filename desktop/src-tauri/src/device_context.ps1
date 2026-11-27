param([bool]$ReadDefender = $false)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$result = @{ schema = 'argus.device-context/v1'; observedAt = [DateTime]::UtcNow.ToString('o'); connections = @(); networkError = $null; defender = $null; defenderError = $null; detections = @(); detectionError = $null; limit = 512 }
$names = @{}
$paths = @{}
$starts = @{}
Get-Process -ErrorAction SilentlyContinue | ForEach-Object {
    $names[[int]$_.Id] = $_.ProcessName
    try { $paths[[int]$_.Id] = $_.Path; $starts[[int]$_.Id] = $_.StartTime.ToUniversalTime().ToString('o') } catch { }
}
$result.networkStartedAt = [DateTime]::UtcNow.ToString('o')
try {
    $tcp = @(Get-NetTCPConnection -ErrorAction Stop | Select-Object -First 512 | ForEach-Object {
        @{ protocol = 'TCP'; localAddress = $_.LocalAddress; localPort = $_.LocalPort; remoteAddress = $_.RemoteAddress; remotePort = $_.RemotePort; state = [string]$_.State; processId = $_.OwningProcess; processName = $names[[int]$_.OwningProcess]; processPath = $paths[[int]$_.OwningProcess]; processStartedAt = $starts[[int]$_.OwningProcess] }
    })
    $result.connections = $tcp
} catch { $result.networkError = 'TCP ownership unavailable: ' + $_.Exception.Message }
try {
    $udp = @(Get-NetUDPEndpoint -ErrorAction Stop | Select-Object -First 512 | ForEach-Object {
        @{ protocol = 'UDP'; localAddress = $_.LocalAddress; localPort = $_.LocalPort; remoteAddress = $null; remotePort = $null; state = 'Bound endpoint'; processId = $_.OwningProcess; processName = $names[[int]$_.OwningProcess]; processPath = $paths[[int]$_.OwningProcess]; processStartedAt = $starts[[int]$_.OwningProcess] }
    })
    $result.connections = @($result.connections) + $udp
} catch { $result.networkError = [string]$result.networkError + ' UDP ownership unavailable: ' + $_.Exception.Message }
$result.networkCompletedAt = [DateTime]::UtcNow.ToString('o')
if ($ReadDefender) {
try {
    $s = Get-MpComputerStatus -ErrorAction Stop
    $result.defender = @{ antivirusEnabled = $s.AntivirusEnabled; realTimeProtectionEnabled = $s.RealTimeProtectionEnabled; runningMode = $s.AMRunningMode; signatureVersion = $s.AntivirusSignatureVersion; signatureUpdated = if ($s.AntivirusSignatureLastUpdated) { $s.AntivirusSignatureLastUpdated.ToUniversalTime().ToString('o') } else { $null } }
} catch { $result.defenderError = $_.Exception.Message }
try {
    $result.detections = @(Get-MpThreatDetection -ErrorAction Stop | Sort-Object InitialDetectionTime -Descending | Select-Object -First 20 | ForEach-Object {
        @{ threatId = [string]$_.ThreatID; processName = $_.ProcessName; actionSuccess = $_.ActionSuccess; detectedAt = if ($_.InitialDetectionTime) { $_.InitialDetectionTime.ToUniversalTime().ToString('o') } else { $null } }
    })
} catch { $result.detectionError = $_.Exception.Message }
} else { $result.defenderError = 'Not connected in Argus'; $result.detectionError = 'Not connected in Argus' }
$result | ConvertTo-Json -Depth 5 -Compress

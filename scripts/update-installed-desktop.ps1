[CmdletBinding()]
param([switch]$Check, [switch]$NoLaunch, [switch]$Force)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$install = Join-Path $env:LOCALAPPDATA 'Programs/ArgusNextPublic'
$exe = Join-Path $install 'argus-desktop.exe'
$receiptPath = Join-Path $install 'LOCAL_BUILD.json'
$links = @((Join-Path ([Environment]::GetFolderPath('Desktop')) 'Argus Next Public.lnk')) | Select-Object -Unique
function SourceHash {
    $inputs = @('desktop/src-tauri/src','desktop/src-tauri/windows','desktop/src-tauri/icons','desktop/ui','crates')
    $lines = foreach($folder in $inputs) {
        Get-ChildItem -LiteralPath (Join-Path $repo $folder) -File -Recurse | Where-Object {$_.FullName -notmatch '[\\/](node_modules|target|\.git)[\\/]'} | ForEach-Object {
            $_.FullName.Substring($repo.Length) + ':' + (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
        }
    }
    foreach($file in @('Cargo.toml','Cargo.lock','desktop/src-tauri/Cargo.toml','desktop/src-tauri/build.rs','desktop/src-tauri/tauri.conf.json')) {$lines += $file + ':' + (Get-FileHash -LiteralPath (Join-Path $repo $file)).Hash}
    $digest = [Security.Cryptography.SHA256]::Create()
    try {return [Convert]::ToHexString($digest.ComputeHash([Text.Encoding]::UTF8.GetBytes((($lines | Sort-Object) -join "`n"))))} finally {$digest.Dispose()}
}
function VerifyInstall($expectedSource) {
    $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    if($receipt.sourceHash -ne $expectedSource){throw 'Installed build does not match current source. Run this script without -Check.'}
    if((Get-FileHash -LiteralPath $exe).Hash -ne $receipt.executableHash){throw 'Installed executable hash mismatch'}
    $shell = New-Object -ComObject WScript.Shell
    foreach($link in $links) {if(-not(Test-Path -LiteralPath $link)){throw "Missing shortcut: $link"}; $s=$shell.CreateShortcut($link); if($s.TargetPath -ne $exe -or $s.WorkingDirectory -ne $install){throw "Shortcut mismatch: $link"}}
    Write-Host 'VERIFIED: installed executable, source receipt and desktop shortcut agree.'
}
if($PSVersionTable.PSVersion.Major -lt 7){throw 'Use PowerShell 7 (pwsh) for this developer install command.'}
$sourceHash = SourceHash
if($Check){VerifyInstall $sourceHash; return}
if(-not(Test-Path -LiteralPath $exe)){throw 'Install the Windows alpha package first; this updater preserves its resources and user data.'}
$installerManaged = (Test-Path -LiteralPath (Join-Path $install 'ARGUS-INSTALL.txt')) -or (Test-Path -LiteralPath (Join-Path $install 'INSTALL.txt')) -or (Test-Path -LiteralPath (Join-Path $install 'argus-uninstall.exe'))
if($installerManaged -and -not $Force){
    throw 'Installer-managed ArgusNextPublic installation detected. Hot-swapping argus-desktop.exe invalidates argus-uninstall.exe''s hash-verified ownership record. Re-run the installer to upgrade cleanly, or pass -Force to override for local development.'
}
if($installerManaged -and $Force){
    Write-Warning 'Overwriting an installer-managed argus-desktop.exe with -Force invalidates argus-uninstall.exe''s ownership verification record.'
}
Push-Location $repo
try {
    & node scripts/validate-desktop-ui.cjs
    if($LASTEXITCODE -ne 0){throw 'UI tests failed; installed app untouched'}
    & cargo clippy --locked --release --bin argus-desktop -- -D warnings
    if($LASTEXITCODE -ne 0){throw 'Clippy failed; installed app untouched'}
    & cargo build --locked --release -p argus-desktop
    if($LASTEXITCODE -ne 0){throw 'Build failed; installed app untouched'}
} finally {Pop-Location}
if((SourceHash) -ne $sourceHash){throw 'Source changed during build; not installing a mismatched executable'}
$built = Join-Path $repo 'target/release/argus-desktop.exe'
$hash = (Get-FileHash -LiteralPath $built).Hash
$backup = Join-Path $env:LOCALAPPDATA ('ArgusNextPublic/install-backups/' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $backup | Out-Null
Copy-Item -LiteralPath $exe -Destination (Join-Path $backup 'argus-desktop.exe')
foreach($link in $links){if(Test-Path -LiteralPath $link){Copy-Item -LiteralPath $link -Destination (Join-Path $backup ([IO.Path]::GetFileName($link)))}}
$staged = Join-Path $install ('argus-' + [Guid]::NewGuid().ToString('N') + '.new.exe')
Copy-Item -LiteralPath $built -Destination $staged
if((Get-FileHash -LiteralPath $staged).Hash -ne $hash){throw 'Staged hash mismatch'}
Get-Process argus-desktop -ErrorAction SilentlyContinue | Where-Object {$_.Path -eq $exe} | ForEach-Object { $null=$_.CloseMainWindow(); if(-not $_.WaitForExit(1500)){Stop-Process -Id $_.Id; $_.WaitForExit()} }
Move-Item -LiteralPath $staged -Destination $exe -Force
$shell = New-Object -ComObject WScript.Shell
foreach($link in $links) { $s=$shell.CreateShortcut($link); $s.TargetPath=$exe; $s.Arguments=''; $s.WorkingDirectory=$install; $s.Save() }
@{sourceHash=$sourceHash;executableHash=$hash;builtAtUtc=[DateTime]::UtcNow.ToString('o');backup=$backup} | ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding utf8
VerifyInstall $sourceHash
if(-not $NoLaunch){Start-Process -FilePath $exe -WorkingDirectory $install -WindowStyle Hidden}

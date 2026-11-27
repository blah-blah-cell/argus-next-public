# Local application and Defender context

In the source build, open **Diagnostics → Applications and Windows Defender**
and select **Refresh device context**. Nothing is collected at app startup.
Onboarding's existing Device checks stage and Diagnostics offer a saved
**Defender connection** choice: Off (default), Read-only, or Read-only +
scan/update controls. Off skips Defender status/history queries, without changing
Windows protection. The choice is an Argus preference, not a Windows permission.
Refreshing shows the actual provider result; unavailable or protection-off states
never block completing setup. No scan is triggered by choosing a mode.
By default this is an on-demand socket snapshot. An unchecked-by-default option
refreshes context every 10 seconds while live monitoring, serially with no
overlapping requests. It is not saved across app sessions and pauses when the
app document is hidden. It never starts capture itself.

Windows PowerShell's built-in `Get-NetTCPConnection` and `Get-NetUDPEndpoint`
provide socket ownership. Process names come from a contemporaneous process
snapshot: exits, shared endpoints and PID reuse can make attribution incomplete.
UDP rows have no remote peer. Results are capped at 512 entries per protocol;
short-lived connections may be missed. No byte totals are inferred from sockets.
The latest live capture is matched separately against full TCP endpoint pairs in
either direction. A whole flow must fall within 10 seconds of the network
snapshot start, its packets cannot be future-dated, and its candidate process
must have a known start time preceding the flow. Snapshots expire after 10 seconds.
UDP, missing identities, long-lived flow totals and competing owners remain
unattributed. At most 5,000 displayed flows are inspected; duplicate flow UIDs
are counted once. Byte sums use captured bytes, not wire bytes or machine-wide
bandwidth. Nothing accumulates across capture updates. These are socket-match
candidates, not definitive packet-to-process attribution: races still exist.

`Get-MpComputerStatus` reports Defender status; `Get-MpThreatDetection` returns
up to 20 records, newest first. A detection record is not a verdict on a current
connection. A candidate app summary mentions history only when the full Windows
executable path equals the detection's process path (case-insensitive with slash
normalization); basenames and timestamps alone never establish a match. Argus
does not classify an app as safe when no history exists. Provider errors, including access restrictions,
remain visible separately. Another antivirus or an unavailable Defender provider
must not be presented as a successful clean scan.

The fixed read-only script uses no user-provided command text, has a 30-second
subprocess deadline and a 1 MiB output bound. It does not change settings, start
scans, quarantine files, request elevation or upload data. Results live only in
the panel, not replay evidence, reports or localStorage. No current-host context
is joined onto imported captures. Linux and macOS return an unsupported message.

## Optional elevated actions

Controls mode exposes only quick scan, full scan and signature updates. Each
requires an Argus confirmation followed by Windows UAC. A fixed enum allowlist is
passed to the current executable's one-shot elevated helper, which checks admin
membership and runs a fixed Defender cmdlet. No arbitrary command, path, exclusion,
protection-disable, restore or offline-reboot action is accepted. No credentials,
permanent admin service or scheduled control task are created. Only one action may
await approval/completion at a time. Scans can remediate files under Defender's
existing policy; signature updates can access configured update servers. Full
scans may run for a long time. Switching Off blocks new actions, not a previously
approved scan. Command success is never reported as a clean-scan verdict. Cancelling
UAC or command failure is reported as not confirmed successful, not silently retried.

Delivered locally: socket provider, time-bound live TCP matching, per-application
captured-byte summaries and exact-path historical Defender context. Not shipped
in the published installer. Manual packaged-app checks on a fresh machine remain
required before claiming deployment readiness.

Engineering decision: compact planning for read-only context; explicit user
approval and allowlist review for elevated actions; impact-level verification,
no delegation. The cinematic onboarding structure and capture policy are unchanged.

## Keeping the installed app aligned

Use `pwsh -File scripts/update-installed-desktop.ps1` from the checkout to validate,
build and update the existing per-user install. It checks that source did not
change during the build, stages and hashes the executable, backs up the previous
binary and shortcuts, and repairs both shortcuts to the stable install directory.
It preserves model resources, cookies and operator data. A `LOCAL_BUILD.json`
receipt records source and executable hashes. Run the same command with `-Check`
to detect source/build/shortcut drift without changing anything. This is a local
developer update, not a published release. Build-cache cleanup cannot remove the
installed executable or invalidate its shortcuts.

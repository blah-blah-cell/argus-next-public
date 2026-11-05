# Native Pktmon packet stream

This Windows-only sidecar dynamically loads the documented `PktmonApi.dll`
packet-streaming API. It keeps one live session active while rotating completed
classic-PCAP files for the Rust pipeline. Rotation does not stop capture.

The adapter deduplicates repeated Windows-stack appearances by Pktmon packet
identity, buffers one bounded five-second segment, orders it causally, and uses
precise callback receipt time. The task is registered on first app launch; it
starts only when Argus opens and stops when Argus exits. Registration and start
are idempotent across desktop restarts. The desktop accepts persisted ownership
only when the process ID, exact executable, live output directory, and fixed
`sensor.stop` path all match. Packaged and release-side sensor binaries are
preferred; `ARGUS_PKTMON_SENSOR` is available for an explicit development path.

Completed live segments are ephemeral. The sidecar attempts to retain the newest
eight after each publication; deletion failures can leave a backlog reported in
sensor health. The desktop releases each segment after analysis. Imported
or manually selected captures outside the generated `live` directory are never
deleted by this policy.

Before opening a new segment, the writer checks filesystem space available to
the process against the serialized PCAP size (24-byte file header, 16 bytes per
record, and packet bytes), plus a 256 MiB reserve. Insufficient space or a failed
space query stops monitoring through the existing failure state; the guard does
not delete captures to make room. This is a best-effort low-space guard, not a
hard quota: concurrent disk use can race the check, and existing reparse-point
and output-path handling is unchanged. The boundary calculation is covered by
`storage_budget_test.cpp` (compile as a C++20 console executable and run it).

The callback buffer is also bounded to 1,048,576 packets or 256 MiB of packet
bytes per segment; serialized PCAP record headers add up to another 16 MiB.
When either limit is reached, the sensor records an application drop instead of
growing memory without limit. `sensor-health.json` is replaced atomically after
each completed segment and exposes Pktmon missed-write/read counts, application
drops, malformed/unsupported records, duplicate stack appearances, segment
retention, and failure state to the desktop.

It requires administrator privileges and a supported Windows Pktmon API. The
source intentionally has no SDK header/import-library dependency because
Microsoft does not ship either for this API.

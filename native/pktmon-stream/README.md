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

Completed live segments are ephemeral. The sidecar retains at most eight as a
hard safety cap, and the desktop releases each segment after analysis. Imported
or manually selected captures outside the generated `live` directory are never
deleted by this policy.

The callback buffer is also bounded to 262,144 packets or 256 MiB per segment.
When either limit is reached, the sensor records an application drop instead of
growing memory without limit. `sensor-health.json` is replaced atomically after
each completed segment and exposes Pktmon missed-write/read counts, application
drops, malformed/unsupported records, duplicate stack appearances, segment
retention, and failure state to the desktop.

It requires administrator privileges and a supported Windows Pktmon API. The
source intentionally has no SDK header/import-library dependency because
Microsoft does not ship either for this API.

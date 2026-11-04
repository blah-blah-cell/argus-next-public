"""Convert official CSE-CIC-IDS2018 PCAP ZIPs into causal Argus journals."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from array import array
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

import dpkt

from .continual_learning import FEATURE_DIM, MAGIC_V2
from .flow_v2_features import (
    DATASET_SOURCES,
    FEATURE_CONTRACT,
    FEATURE_CONTRACT_SHA256,
    FEATURE_SCHEMA,
)

PCAP_MAGIC = {
    b"\xd4\xc3\xb2\xa1",
    b"\xa1\xb2\xc3\xd4",
    b"\x4d\x3c\xb2\xa1",
    b"\xa1\xb2\x3c\x4d",
}
PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"
FLOW_IDLE_TIMEOUT_NS = 120 * 1_000_000_000
MAX_I64 = (1 << 63) - 1
FlowKey = tuple[bytes, bytes, int, int, int]


@dataclass
class ConversionStats:
    source_records: int = 0
    accepted_packets: int = 0
    unsupported_packets: int = 0
    malformed_packets: int = 0
    truncated_capture_members: int = 0
    clock_rollbacks: int = 0
    journal_segments: int = 0


@dataclass(frozen=True)
class ParsedPacket:
    frame_length: int
    ip_version: int
    is_tcp: bool
    flags: int
    flow_key: FlowKey
    closes_flow: bool


@dataclass
class _FlowState:
    generation: int
    last_timestamp_ns: int
    closed: bool


class FlowGenerationTracker:
    """Assign deterministic generations to directional endpoint tuples."""

    def __init__(self, idle_timeout_ns: int = FLOW_IDLE_TIMEOUT_NS) -> None:
        if idle_timeout_ns <= 0:
            raise ValueError("flow idle timeout must be positive")
        self.idle_timeout_ns = idle_timeout_ns
        self._states: dict[FlowKey, _FlowState] = {}

    def observe(
        self, key: FlowKey, timestamp_ns: int, *, closes_flow: bool
    ) -> tuple[FlowKey, int]:
        state = self._states.get(key)
        if state is None:
            state = _FlowState(0, timestamp_ns, False)
        elif (
            state.closed
            or timestamp_ns - state.last_timestamp_ns > self.idle_timeout_ns
        ):
            state = _FlowState(state.generation + 1, timestamp_ns, False)
        state.last_timestamp_ns = timestamp_ns
        state.closed = closes_flow
        self._states[key] = state
        return key, state.generation


class TimestampNormalizer:
    """Map accepted capture timestamps to monotonic integer nanoseconds."""

    def __init__(self) -> None:
        self._last_raw_ns: int | None = None
        self._last_normalized_ns: int | None = None
        self._offset_ns = 0

    def observe(self, timestamp: float) -> tuple[int, bool]:
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("capture timestamp must be finite and non-negative")
        raw_ns = round(timestamp * 1_000_000_000)
        if raw_ns > MAX_I64:
            raise ValueError("capture timestamp exceeds the journal contract")
        rollback = self._last_raw_ns is not None and raw_ns < self._last_raw_ns
        if rollback:
            self._offset_ns += self._last_raw_ns - raw_ns
        normalized_ns = raw_ns + self._offset_ns
        if normalized_ns > MAX_I64:
            raise ValueError("normalized timestamp exceeds the journal contract")
        if self._last_normalized_ns is not None:
            normalized_ns = max(normalized_ns, self._last_normalized_ns)
        self._last_raw_ns = raw_ns
        self._last_normalized_ns = normalized_ns
        return normalized_ns, rollback


class SegmentBuffer:
    """Keep aligned v2 arrays and collision-free record-local flow IDs."""

    def __init__(self) -> None:
        self.features = array("f")
        self.flow_ids = array("Q")
        self.timestamps_ns = array("Q")
        self.source_ordinals = array("Q")
        self._flow_ids: dict[tuple[FlowKey, int], int] = {}

    @property
    def count(self) -> int:
        return len(self.flow_ids)

    def append(
        self,
        features: list[float],
        flow_instance: tuple[FlowKey, int],
        timestamp_ns: int,
        source_ordinal: int,
    ) -> None:
        flow_id = self._flow_ids.get(flow_instance)
        if flow_id is None:
            flow_id = len(self._flow_ids) + 1
            self._flow_ids[flow_instance] = flow_id
        self.features.extend(features)
        self.flow_ids.append(flow_id)
        self.timestamps_ns.append(timestamp_ns)
        self.source_ordinals.append(source_ordinal)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _CaptureStream:
    """Replay sniffed bytes without seeking or buffering an entire RAR member."""

    def __init__(self, source: BinaryIO, prefix: bytes = b"") -> None:
        self.source = source
        self.prefix = prefix

    def read(self, size: int) -> bytes:
        if size < 0:
            raise ValueError("capture reads must be bounded")
        result = bytearray(self.prefix[:size])
        self.prefix = self.prefix[size:]
        # Pipes can return short reads before EOF; dpkt expects complete headers.
        while len(result) < size:
            block = self.source.read(size - len(result))
            if not block:
                break
            result.extend(block)
        return bytes(result)


def _reader(source: BinaryIO) -> Iterator[tuple[float, bytes]]:
    stream = _CaptureStream(source)
    magic = stream.read(4)
    stream.prefix = magic
    if magic in PCAP_MAGIC:
        return iter(dpkt.pcap.Reader(stream))
    if magic == PCAPNG_MAGIC:
        return iter(dpkt.pcapng.Reader(stream))
    raise ValueError(f"unsupported capture magic {magic.hex()}")


def _parse_packet(frame: bytes) -> ParsedPacket | None:
    try:
        ethernet = dpkt.ethernet.Ethernet(frame)
        network = ethernet.data
        if isinstance(network, dpkt.ip.IP):
            ip_version = 4
        elif isinstance(network, dpkt.ip6.IP6):
            ip_version = 6
        else:
            return None
        transport = network.data
        is_tcp = isinstance(transport, dpkt.tcp.TCP)
        is_udp = isinstance(transport, dpkt.udp.UDP)
        if not is_tcp and not is_udp:
            return None
        flags = transport.flags if is_tcp else 0
        return ParsedPacket(
            frame_length=len(frame),
            ip_version=ip_version,
            is_tcp=is_tcp,
            flags=flags,
            flow_key=(
                bytes(network.src),
                bytes(network.dst),
                int(network.p if ip_version == 4 else network.nxt),
                int(transport.sport),
                int(transport.dport),
            ),
            closes_flow=is_tcp and bool(flags & (dpkt.tcp.TH_FIN | dpkt.tcp.TH_RST)),
        )
    except (AttributeError, dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, ValueError):
        raise ValueError("malformed supported packet") from None


def _features(packet: ParsedPacket, delta_seconds: float) -> list[float]:
    flags = packet.flags
    return [
        math.log1p(packet.frame_length),
        math.log1p(max(0.0, min(delta_seconds, 60.0))),
        float(packet.ip_version == 4),
        float(packet.ip_version == 6),
        float(packet.is_tcp),
        float(not packet.is_tcp),
        float(bool(flags & dpkt.tcp.TH_SYN)),
        float(bool(flags & dpkt.tcp.TH_ACK)),
        float(bool(flags & dpkt.tcp.TH_FIN)),
        float(bool(flags & dpkt.tcp.TH_RST)),
        float(bool(flags & dpkt.tcp.TH_PUSH)),
        float(bool(flags & dpkt.tcp.TH_URG)),
    ]


class JournalWriter:
    def __init__(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(f"journal output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".partial")
        if self.temporary.exists():
            raise FileExistsError(
                f"partial journal output already exists: {self.temporary}"
            )
        self.output = self.temporary.open("xb")
        self.output.write(MAGIC_V2)
        self.digest = hashlib.sha256()
        self.digest.update(MAGIC_V2)

    @staticmethod
    def _little_endian(values: array[float] | array[int]) -> bytes:
        payload = array(values.typecode, values)
        if sys.byteorder != "little":
            payload.byteswap()
        return payload.tobytes()

    def append(self, identifier: str, segment: SegmentBuffer) -> None:
        features = segment.features
        if not features or len(features) % FEATURE_DIM:
            raise ValueError("journal segment has an invalid feature count")
        count = len(features) // FEATURE_DIM
        if count > 1_000_000:
            raise ValueError("journal segment exceeds the reader contract")
        if not (
            count
            == len(segment.flow_ids)
            == len(segment.timestamps_ns)
            == len(segment.source_ordinals)
        ):
            raise ValueError("journal segment metadata is not aligned")
        if any(
            right < left
            for left, right in zip(segment.timestamps_ns, segment.timestamps_ns[1:])
        ):
            raise ValueError("journal timestamps are not monotonic")
        if any(
            right <= left
            for left, right in zip(segment.source_ordinals, segment.source_ordinals[1:])
        ):
            raise ValueError("journal source ordinals are not strictly increasing")
        encoded = b"".join(
            (
                self._little_endian(features),
                self._little_endian(segment.flow_ids),
                self._little_endian(segment.timestamps_ns),
                self._little_endian(segment.source_ordinals),
            )
        )
        header = b"".join(
            (
                struct.pack("<I", count),
                hashlib.sha256(identifier.encode()).digest(),
                hashlib.sha256(encoded).digest(),
            )
        )
        self.output.write(header)
        self.output.write(encoded)
        self.digest.update(header)
        self.digest.update(encoded)

    def finish(self) -> str:
        self.output.flush()
        os.fsync(self.output.fileno())
        self.output.close()
        os.replace(self.temporary, self.path)
        return self.digest.hexdigest()


def convert_archive(
    archive: Path,
    output: Path,
    *,
    segment_packets: int,
    max_packets: int,
    archive_sha256: str | None = None,
    source_uri: str | None = None,
    dataset: str = "CSE-CIC-IDS2018",
    member_pattern: str = "*",
) -> dict[str, object]:
    if segment_packets <= 30_000 or segment_packets > 1_000_000:
        raise ValueError("segment_packets must be between 30,001 and 1,000,000")
    if max_packets < segment_packets * 3:
        raise ValueError("max_packets must permit at least three complete segments")
    if dataset not in DATASET_SOURCES:
        raise ValueError(f"unsupported real packet corpus {dataset!r}")
    if source_uri is not None:
        source_uri = source_uri.strip()
        if not source_uri:
            raise ValueError("source_uri must not be empty")
        if dataset == "CSE-CIC-IDS2018" and not source_uri.startswith(
            DATASET_SOURCES[dataset]
        ):
            raise ValueError(
                "CSE-CIC-IDS2018 source_uri must be an object in the official "
                f"{DATASET_SOURCES[dataset]!r} bucket"
            )
    if not member_pattern:
        raise ValueError("member_pattern must not be empty")
    dataset_slug = dataset.lower().replace("-", "_")
    if archive_sha256 is None:
        archive_sha256 = _sha256(archive)
    elif len(archive_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in archive_sha256
    ):
        raise ValueError("archive_sha256 must be a lowercase SHA-256 digest")
    stats = ConversionStats()
    members: list[dict[str, object]] = []
    writer = JournalWriter(output)
    try:
        bundle: zipfile.ZipFile | None = None
        unrar: str | None = None
        if zipfile.is_zipfile(archive):
            archive_format = "zip"
            bundle = zipfile.ZipFile(archive)
            candidates = [
                member.filename
                for member in bundle.infolist()
                if not member.is_dir() and member.file_size >= 24
            ]
        elif archive.suffix.lower() == ".rar":
            archive_format = "rar"
            unrar = shutil.which("unrar")
            if unrar is None:
                raise RuntimeError("RAR conversion requires the unrar executable")
            listed = subprocess.run(
                [unrar, "lb", "-c-", str(archive)],
                check=True,
                capture_output=True,
                text=True,
            )
            candidates = [line.strip() for line in listed.stdout.splitlines()]
        else:
            raise ValueError("packet archive must be a valid ZIP or a RAR file")
        try:
            names = sorted(
                name
                for name in candidates
                if (
                    name.lower().replace("\\", "/").startswith("pcap/")
                    or Path(name).suffix.lower() in {".pcap", ".pcapng"}
                )
                and fnmatch.fnmatch(name.lower(), member_pattern.lower())
            )
            if not names:
                raise ValueError(
                    "archive contains no PCAP/PCAPNG members matching "
                    f"{member_pattern!r}"
                )
            for name in names:
                if stats.accepted_packets >= max_packets:
                    break
                member_packets = 0
                member_segments = 0
                previous_timestamp_ns: int | None = None
                clock = TimestampNormalizer()
                flows = FlowGenerationTracker()
                chunk = SegmentBuffer()
                truncated_tail = False
                truncated_tail_error = None
                rar_process: subprocess.Popen[bytes] | None = None
                if bundle is not None:
                    source = bundle.open(name)
                else:
                    assert unrar is not None
                    rar_process = subprocess.Popen(
                        [unrar, "p", "-inul", "-c-", str(archive), name],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    if rar_process.stdout is None:
                        rar_process.kill()
                        raise RuntimeError("unrar did not provide a packet stream")
                    source = rar_process.stdout
                with source:
                    try:
                        for source_ordinal, (timestamp, frame) in enumerate(
                            _reader(source)
                        ):
                            stats.source_records += 1
                            try:
                                packet = _parse_packet(frame)
                            except ValueError:
                                stats.malformed_packets += 1
                                continue
                            if packet is None:
                                stats.unsupported_packets += 1
                                continue
                            try:
                                normalized_timestamp_ns, rollback = clock.observe(
                                    float(timestamp)
                                )
                            except ValueError:
                                stats.malformed_packets += 1
                                continue
                            if rollback:
                                stats.clock_rollbacks += 1
                            delta_seconds = (
                                0.0
                                if previous_timestamp_ns is None
                                else (normalized_timestamp_ns - previous_timestamp_ns)
                                / 1_000_000_000.0
                            )
                            features = _features(packet, delta_seconds)
                            if not all(math.isfinite(value) for value in features):
                                stats.malformed_packets += 1
                                continue
                            flow_instance = flows.observe(
                                packet.flow_key,
                                normalized_timestamp_ns,
                                closes_flow=packet.closes_flow,
                            )
                            chunk.append(
                                features,
                                flow_instance,
                                normalized_timestamp_ns,
                                source_ordinal,
                            )
                            previous_timestamp_ns = normalized_timestamp_ns
                            member_packets += 1
                            stats.accepted_packets += 1
                            if chunk.count == segment_packets:
                                identifier = (
                                    f"{dataset_slug}:{archive_sha256}:{name}:"
                                    f"segment-{member_segments}"
                                )
                                writer.append(identifier, chunk)
                                chunk = SegmentBuffer()
                                member_segments += 1
                                stats.journal_segments += 1
                            if stats.accepted_packets >= max_packets:
                                break
                    except dpkt.dpkt.NeedData as error:
                        # Several official day archives contain a partial record
                        # header at a member tail. Preserve valid earlier packets,
                        # but make the truncation explicit in provenance.
                        truncated_tail = True
                        truncated_tail_error = str(error)
                        stats.truncated_capture_members += 1
                if rar_process is not None:
                    if stats.accepted_packets >= max_packets:
                        rar_process.terminate()
                        rar_process.wait(timeout=30)
                    else:
                        stderr = (
                            rar_process.stderr.read().decode(errors="replace")
                            if rar_process.stderr is not None
                            else ""
                        )
                        returncode = rar_process.wait(timeout=30)
                        if returncode != 0:
                            raise RuntimeError(
                                f"unrar failed for {name!r} with exit code "
                                f"{returncode}: {stderr.strip()}"
                            )
                if chunk.count > 30_000:
                    identifier = (
                        f"{dataset_slug}:{archive_sha256}:{name}:"
                        f"segment-{member_segments}"
                    )
                    writer.append(identifier, chunk)
                    member_segments += 1
                    stats.journal_segments += 1
                members.append(
                    {
                        "name": name,
                        "accepted_packets": member_packets,
                        "journal_segments": member_segments,
                        "truncated_tail": truncated_tail,
                        "truncated_tail_error": truncated_tail_error,
                    }
                )
        finally:
            if bundle is not None:
                bundle.close()
        if stats.journal_segments < 3:
            raise ValueError("conversion produced fewer than three trainable segments")
        journal_sha256 = writer.finish()
    except Exception:
        writer.output.close()
        writer.temporary.unlink(missing_ok=True)
        raise
    manifest = {
        "schema": "argus.real-pcap-journal/v3",
        "created_utc": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "official_source": DATASET_SOURCES[dataset],
        "synthetic_data": False,
        "task": "causal_next_packet_forecast_only",
        "not_intrusion_detection_accuracy": True,
        "sequence_contract": "argus.packet-sequence/v2",
        "capture_member_boundaries_preserved": True,
        "record_boundaries_preserved_during_training": True,
        "archive": {
            "name": archive.name,
            "source_uri": source_uri,
            "format": archive_format,
            "bytes": archive.stat().st_size,
            "sha256": archive_sha256,
            "member_pattern": member_pattern,
        },
        "journal": {
            "name": output.name,
            "bytes": output.stat().st_size,
            "sha256": journal_sha256,
            "binary_format": "ARGUSLR2",
            "payload_checksum": "sha256",
            "feature_schema": FEATURE_SCHEMA,
            "feature_dimension": FEATURE_DIM,
            "feature_contract": FEATURE_CONTRACT,
            "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
            "flow_identity": "directional_5tuple_plus_generation",
            "flow_identity_scope": "record_local_opaque_u64",
            "flow_idle_timeout_ns": FLOW_IDLE_TIMEOUT_NS,
            "timestamp_unit": "nanoseconds",
            "timestamp_normalization": "monotonic_offset_on_raw_clock_rollback",
            "source_ordinal_scope": "capture_member",
            "global_context_lane": True,
            "segment_packets": segment_packets,
            "max_packets": max_packets,
        },
        "members": members,
        "stats": asdict(stats),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-packets", type=int, default=250_000)
    parser.add_argument("--max-packets", type=int, default=3_000_000)
    parser.add_argument("--archive-sha256")
    parser.add_argument(
        "--source-uri",
        help="original official object/page URI recorded in converter provenance",
    )
    parser.add_argument("--member-pattern", default="*")
    parser.add_argument(
        "--dataset", choices=tuple(DATASET_SOURCES), default="CSE-CIC-IDS2018"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            convert_archive(
                args.archive,
                args.output,
                segment_packets=args.segment_packets,
                max_packets=args.max_packets,
                archive_sha256=args.archive_sha256,
                source_uri=args.source_uri,
                dataset=args.dataset,
                member_pattern=args.member_pattern,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

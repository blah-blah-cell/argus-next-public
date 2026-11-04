"""Fail-closed evaluation-label alignment for CIC identity-rich flow CSVs."""

from __future__ import annotations

import csv
import ipaddress
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from .cse_pcap_to_journal import FlowKey

CanonicalFlowKey = tuple[bytes, int, bytes, int, int]
HOUR_NS = 3_600_000_000_000


def _normalized_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.strip().lower())


HEADER_ALIASES = {
    "source_ip": {"srcip", "sourceip"},
    "destination_ip": {"dstip", "destinationip"},
    "source_port": {"srcport", "sourceport"},
    "destination_port": {"dstport", "destinationport"},
    "protocol": {"protocol", "proto"},
    "timestamp": {"timestamp", "flowstart", "starttime"},
    "duration": {"flowduration", "duration"},
    "label": {"label", "class", "attack"},
}


@dataclass(frozen=True)
class LabeledFlow:
    key: CanonicalFlowKey
    start_wall_ns: int
    end_wall_ns: int
    label: str
    source: str
    row_number: int

    @property
    def is_attack(self) -> bool:
        return self.label.strip().casefold() != "benign"


@dataclass(frozen=True)
class LabelMatch:
    status: str
    flow: LabeledFlow | None
    candidate_count: int


def canonical_flow_key(
    source_ip: bytes,
    destination_ip: bytes,
    protocol: int,
    source_port: int,
    destination_port: int,
) -> CanonicalFlowKey:
    if protocol not in (6, 17):
        raise ValueError("only TCP and UDP labels are supported")
    if not 0 <= source_port <= 65_535 or not 0 <= destination_port <= 65_535:
        raise ValueError("flow ports must be unsigned 16-bit integers")
    left = (bytes(source_ip), source_port)
    right = (bytes(destination_ip), destination_port)
    if right < left:
        left, right = right, left
    return left[0], left[1], right[0], right[1], protocol


def canonical_packet_key(key: FlowKey) -> CanonicalFlowKey:
    source, destination, protocol, source_port, destination_port = key
    return canonical_flow_key(
        source, destination, protocol, source_port, destination_port
    )


def _protocol(value: str) -> int:
    normalized = value.strip().casefold()
    mapping = {"6": 6, "tcp": 6, "17": 17, "udp": 17}
    if normalized not in mapping:
        raise ValueError(f"unsupported flow protocol {value!r}")
    return mapping[normalized]


TIMESTAMP_FORMATS = (
    "%d/%m/%Y %H:%M:%S.%f",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S.%f",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%d/%m/%Y %I:%M:%S.%f %p",
    "%d/%m/%Y %I:%M:%S %p",
    "%d/%m/%Y %I:%M %p",
    "%m/%d/%Y %I:%M:%S.%f %p",
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%Y %I:%M %p",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
)


def parse_wall_timestamp(value: str, expected_date: date) -> int:
    """Parse an unzoned CIC wall time only when its calendar date is known."""
    normalized = " ".join(value.strip().split())
    parsed: set[datetime] = set()
    for format_ in TIMESTAMP_FORMATS:
        try:
            candidate = datetime.strptime(normalized, format_)
        except ValueError:
            continue
        if candidate.date() == expected_date:
            parsed.add(candidate)
    if len(parsed) != 1:
        raise ValueError(
            f"timestamp {value!r} does not map uniquely to {expected_date.isoformat()}"
        )
    # UTC is used as a neutral encoding for an unzoned wall-clock coordinate.
    return round(parsed.pop().replace(tzinfo=UTC).timestamp() * 1_000_000_000)


def _columns(fieldnames: Sequence[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise ValueError("labeled-flow CSV has no header")
    normalized: dict[str, list[str]] = {}
    for original in fieldnames:
        normalized.setdefault(_normalized_header(original), []).append(original)
    result: dict[str, str] = {}
    for logical, aliases in HEADER_ALIASES.items():
        matches = [original for alias in aliases for original in normalized.get(alias, [])]
        if len(matches) != 1:
            raise ValueError(
                f"identity-rich labeled-flow CSV requires exactly one {logical} column"
            )
        result[logical] = matches[0]
    return result


def read_labeled_flows(path: Path, *, expected_date: date) -> list[LabeledFlow]:
    """Read identity/timing/label fields; completed-flow statistics stay unused."""
    flows: list[LabeledFlow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = _columns(reader.fieldnames)
        for row_number, row in enumerate(reader, start=2):
            try:
                source_ip = ipaddress.ip_address(row[columns["source_ip"]].strip()).packed
                destination_ip = ipaddress.ip_address(
                    row[columns["destination_ip"]].strip()
                ).packed
                source_port = int(row[columns["source_port"]])
                destination_port = int(row[columns["destination_port"]])
                protocol = _protocol(row[columns["protocol"]])
                start = parse_wall_timestamp(
                    row[columns["timestamp"]], expected_date
                )
                duration_us = float(row[columns["duration"]])
                label = row[columns["label"]].strip()
                if not label or not math.isfinite(duration_us) or duration_us < 0:
                    raise ValueError("label and duration must be valid")
                duration_ns = round(duration_us * 1_000)
                flows.append(
                    LabeledFlow(
                        key=canonical_flow_key(
                            source_ip,
                            destination_ip,
                            protocol,
                            source_port,
                            destination_port,
                        ),
                        start_wall_ns=start,
                        end_wall_ns=start + duration_ns,
                        label=label,
                        source=path.name,
                        row_number=row_number,
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid identity-rich label row {path.name}:{row_number}: {error}"
                ) from error
    if not flows:
        raise ValueError("labeled-flow CSV contains no data rows")
    return flows


class FlowLabelIndex:
    def __init__(self, flows: Iterable[LabeledFlow]) -> None:
        grouped: dict[CanonicalFlowKey, list[LabeledFlow]] = {}
        count = 0
        for flow in flows:
            grouped.setdefault(flow.key, []).append(flow)
            count += 1
        if count == 0:
            raise ValueError("label index requires at least one flow")
        self.flows = {
            key: tuple(sorted(values, key=lambda item: (item.start_wall_ns, item.end_wall_ns)))
            for key, values in grouped.items()
        }
        self.flow_count = count

    def matcher(
        self, *, clock_offset_ns: int, tolerance_ns: int = 1_000_000_000
    ) -> StreamingFlowLabelMatcher:
        return StreamingFlowLabelMatcher(
            self, clock_offset_ns=clock_offset_ns, tolerance_ns=tolerance_ns
        )


class StreamingFlowLabelMatcher:
    """Match globally chronological packet observations in amortized linear time."""

    def __init__(
        self,
        index: FlowLabelIndex,
        *,
        clock_offset_ns: int,
        tolerance_ns: int,
    ) -> None:
        if type(clock_offset_ns) is not int:
            raise ValueError("clock offset must be integer nanoseconds")
        if type(tolerance_ns) is not int or tolerance_ns < 0:
            raise ValueError("label tolerance must be non-negative integer nanoseconds")
        self.index = index
        self.clock_offset_ns = clock_offset_ns
        self.tolerance_ns = tolerance_ns
        self._cursor: dict[CanonicalFlowKey, int] = {}
        self._active: dict[CanonicalFlowKey, list[LabeledFlow]] = {}
        self._last_wall_ns: int | None = None

    def observe(self, packet_key: FlowKey, raw_timestamp_ns: int) -> LabelMatch:
        wall_ns = raw_timestamp_ns + self.clock_offset_ns
        if self._last_wall_ns is not None and wall_ns < self._last_wall_ns:
            raise ValueError("label matcher requires globally monotonic packet time")
        self._last_wall_ns = wall_ns
        key = canonical_packet_key(packet_key)
        flows = self.index.flows.get(key, ())
        cursor = self._cursor.get(key, 0)
        active = self._active.get(key, [])
        latest_start = wall_ns + self.tolerance_ns
        while cursor < len(flows) and flows[cursor].start_wall_ns <= latest_start:
            active.append(flows[cursor])
            cursor += 1
        earliest_end = wall_ns - self.tolerance_ns
        active = [flow for flow in active if flow.end_wall_ns >= earliest_end]
        self._cursor[key] = cursor
        self._active[key] = active
        candidates = [
            flow
            for flow in active
            if flow.start_wall_ns - self.tolerance_ns
            <= wall_ns
            <= flow.end_wall_ns + self.tolerance_ns
        ]
        if not candidates:
            return LabelMatch("unmatched", None, 0)
        if len(candidates) != 1:
            return LabelMatch("ambiguous", None, len(candidates))
        return LabelMatch("matched", candidates[0], 1)


def infer_clock_offset(
    index: FlowLabelIndex,
    packets: Sequence[tuple[FlowKey, int]],
    *,
    candidate_offsets_ns: Sequence[int] | None = None,
    tolerance_ns: int = 1_000_000_000,
    minimum_coverage: float = 0.50,
    minimum_margin: float = 0.05,
) -> dict[str, object]:
    """Infer timezone/clock shift using tuple coverage only, never attack scores."""
    if not packets:
        raise ValueError("clock inference requires packet observations")
    if candidate_offsets_ns is None:
        candidate_offsets_ns = tuple(
            half_hour * HOUR_NS // 2 for half_hour in range(-24, 29)
        )
    if len(set(candidate_offsets_ns)) != len(candidate_offsets_ns):
        raise ValueError("clock-offset candidates must be unique")
    scores: list[tuple[float, int, int, int]] = []
    for offset in candidate_offsets_ns:
        matcher = index.matcher(
            clock_offset_ns=int(offset), tolerance_ns=tolerance_ns
        )
        matched = 0
        ambiguous = 0
        for key, timestamp_ns in packets:
            result = matcher.observe(key, timestamp_ns)
            matched += result.status == "matched"
            ambiguous += result.status == "ambiguous"
        scores.append((matched / len(packets), int(offset), matched, ambiguous))
    scores.sort(key=lambda item: (-item[0], abs(item[1]), item[1]))
    best = scores[0]
    runner_up = scores[1] if len(scores) > 1 else (0.0, 0, 0, 0)
    if best[0] < minimum_coverage:
        raise ValueError(
            f"clock alignment coverage {best[0]:.3f} is below {minimum_coverage:.3f}"
        )
    if best[0] - runner_up[0] < minimum_margin:
        raise ValueError(
            "clock alignment is ambiguous: best and runner-up coverage are too close"
        )
    return {
        "clock_offset_ns": best[1],
        "coverage": best[0],
        "matched_packets": best[2],
        "ambiguous_packets": best[3],
        "sample_packets": len(packets),
        "runner_up_offset_ns": runner_up[1],
        "runner_up_coverage": runner_up[0],
        "selection_uses_attack_scores": False,
    }

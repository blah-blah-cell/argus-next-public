"""Causal packet-feature extraction from immutable PCAP members inside ZIP archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import dpkt
import numpy as np


FEATURE_SCHEMA = "packet-causal-v1"


def _features(
    timestamp: float, previous: float | None, frame: bytes
) -> tuple[np.ndarray, int] | None:
    try:
        ethernet = dpkt.ethernet.Ethernet(frame)
        ip = ethernet.data
        ipv4 = isinstance(ip, dpkt.ip.IP)
        ipv6 = isinstance(ip, dpkt.ip6.IP6)
        if not (ipv4 or ipv6):
            return None
        transport = ip.data
        tcp = isinstance(transport, dpkt.tcp.TCP)
        udp = isinstance(transport, dpkt.udp.UDP)
        if not (tcp or udp):
            return None
        flags = transport.flags if tcp else 0
        delta = 0.0 if previous is None else max(0.0, min(timestamp - previous, 60.0))
        feature = np.asarray(
            (
                np.log1p(len(frame)),
                np.log1p(delta),
                float(ipv4),
                float(ipv6),
                float(tcp),
                float(udp),
                float(bool(flags & dpkt.tcp.TH_SYN)),
                float(bool(flags & dpkt.tcp.TH_ACK)),
                float(bool(flags & dpkt.tcp.TH_FIN)),
                float(bool(flags & dpkt.tcp.TH_RST)),
                float(bool(flags & dpkt.tcp.TH_PUSH)),
                float(bool(flags & dpkt.tcp.TH_URG)),
            ),
            dtype=np.float32,
        )
        # This opaque directional key is retained only for runtime cache lookup.
        # It is never a learned feature or exported as endpoint identity.
        protocol = int(ip.p) if ipv4 else int(ip.nxt)
        flow_bytes = b"|".join(
            (
                ip.src,
                ip.dst,
                protocol.to_bytes(1, "big"),
                int(transport.sport).to_bytes(2, "big"),
                int(transport.dport).to_bytes(2, "big"),
            )
        )
        flow_id = int.from_bytes(
            hashlib.blake2b(flow_bytes, digest_size=8).digest(), "big"
        )
        return feature, flow_id
    except (dpkt.NeedData, dpkt.UnpackError, ValueError):
        return None


def extract_member(
    archive: Path, member: str, output: Path, packet_limit: int
) -> dict[str, object]:
    rows: list[np.ndarray] = []
    flow_ids: list[int] = []
    timestamps: list[float] = []
    parsed = skipped = 0
    previous: float | None = None
    with zipfile.ZipFile(archive) as bundle, bundle.open(member) as source:
        reader = dpkt.pcap.Reader(source)
        for timestamp, frame in reader:
            parsed += 1
            decoded = _features(float(timestamp), previous, frame)
            previous = float(timestamp)
            if decoded is None:
                skipped += 1
                continue
            vector, flow_id = decoded
            rows.append(vector)
            flow_ids.append(flow_id)
            timestamps.append(float(timestamp))
            if len(rows) >= packet_limit:
                break
    if len(rows) < 1025:
        raise ValueError("not enough decodable IP/TCP/UDP packets for self-supervision")
    output.parent.mkdir(parents=True, exist_ok=True)
    values = np.stack(rows)
    np.savez_compressed(
        output,
        features=values,
        flow_ids=np.asarray(flow_ids, dtype=np.uint64),
        timestamps=np.asarray(timestamps, dtype=np.float64),
    )
    metadata = {
        "source_archive": str(archive),
        "source_member": member,
        "feature_schema": FEATURE_SCHEMA,
        "parsed_frames": parsed,
        "accepted_packets": len(rows),
        "skipped_frames": skipped,
        "packet_limit": packet_limit,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packet-limit", type=int, default=500_000)
    args = parser.parse_args()
    print(
        json.dumps(
            extract_member(args.archive, args.member, args.output, args.packet_limit),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

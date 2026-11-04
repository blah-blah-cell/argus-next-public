"""Strict external capture input for full-frame Ethernet forecasting.

Frozen historical converters retain their behavior. This reader rejects partial
frames because their captured size is not the original frame size, and rejects
unsupported link layers rather than guessing Ethernet from payload bytes.
"""
from __future__ import annotations

import struct
from typing import BinaryIO, Iterator

MAX_BLOCK_BYTES = 64 * 1024 * 1024
PCAP_MAGICS = {
    b"\xd4\xc3\xb2\xa1": ("<", 1_000_000),
    b"\xa1\xb2\xc3\xd4": (">", 1_000_000),
    b"\x4d\x3c\xb2\xa1": ("<", 1_000_000_000),
    b"\xa1\xb2\x3c\x4d": (">", 1_000_000_000),
}
SECTION = b"\x0a\x0d\x0d\x0a"


def _read(source: BinaryIO, size: int) -> bytes:
    value = source.read(size)
    if len(value) != size:
        raise ValueError("incomplete capture header, block or packet payload")
    return value


def _lengths(captured: int, original: int, snaplen: int) -> None:
    if captured > MAX_BLOCK_BYTES or original < captured:
        raise ValueError("invalid captured/original packet lengths")
    if snaplen and captured > snaplen:
        raise ValueError("captured packet exceeds declared snapshot length")
    if captured != original:
        raise ValueError(
            f"snapshot-truncated frame: captured={captured}, original={original}; "
            "full-frame forecasting requires complete frames; select an audited full capture "
            "or explicitly freeze a separate length-semantics protocol")


def _ethernet(linktype: int) -> None:
    if linktype != 1:
        raise ValueError(f"unsupported capture link type {linktype}; Ethernet (1) required")


def _options(data: bytes, endian: str) -> Iterator[tuple[int, bytes]]:
    offset = 0
    while offset < len(data):
        if len(data) - offset < 4:
            raise ValueError("incomplete PCAPNG option header")
        kind, length = struct.unpack_from(endian + "HH", data, offset)
        offset += 4
        padded = (length + 3) & ~3
        if offset + padded > len(data):
            raise ValueError("incomplete PCAPNG option value")
        if kind == 0:
            if length or any(data[offset:]):
                raise ValueError("invalid PCAPNG end-of-options")
            return
        yield kind, data[offset:offset + length]
        offset += padded


def full_ethernet_packets(source: BinaryIO) -> Iterator[tuple[float, bytes]]:
    """Yield only complete, timestamped Ethernet records; fail on corrupt tails."""
    magic = _read(source, 4)
    if magic in PCAP_MAGICS:
        endian, scale = PCAP_MAGICS[magic]
        major, minor, _, _, snaplen, network = struct.unpack(endian + "HHiiII", _read(source, 20))
        if (major, minor) != (2, 4) or not snaplen:
            raise ValueError("unsupported classic-PCAP version or snapshot length")
        # Reject unimplemented upper-bit FCS metadata as well as other link types.
        _ethernet(network)
        while True:
            header = source.read(16)
            if not header:
                return
            if len(header) != 16:
                raise ValueError("incomplete classic-PCAP packet header")
            seconds, fraction, captured, original = struct.unpack(endian + "IIII", header)
            if fraction >= scale:
                raise ValueError("invalid classic-PCAP timestamp fraction")
            _lengths(captured, original, snaplen)
            yield seconds + fraction / scale, _read(source, captured)
    elif magic == SECTION:
        yield from _pcapng(source, magic)
    else:
        raise ValueError("unsupported capture magic; expected PCAP or PCAPNG")


def _pcapng(source: BinaryIO, first_magic: bytes) -> Iterator[tuple[float, bytes]]:
    interfaces: list[tuple[int, float, int]] = []
    endian = None
    magic = first_magic
    while magic:
        if len(magic) != 4:
            raise ValueError("incomplete PCAPNG block header")
        length_bytes = _read(source, 4)
        if magic == SECTION:
            order = _read(source, 4)
            if order == b"\x4d\x3c\x2b\x1a":
                endian = "<"
            elif order == b"\x1a\x2b\x3c\x4d":
                endian = ">"
            else:
                raise ValueError("invalid PCAPNG section byte order")
            size = struct.unpack(endian + "I", length_bytes)[0]
            if size < 28 or size % 4 or size > MAX_BLOCK_BYTES:
                raise ValueError("invalid PCAPNG section length")
            rest = _read(source, size - 12)
            if struct.unpack(endian + "I", rest[-4:])[0] != size:
                raise ValueError("PCAPNG section trailing length mismatch")
            if struct.unpack(endian + "HH", rest[:4]) != (1, 0):
                raise ValueError("unsupported PCAPNG section version")
            list(_options(rest[12:-4], endian))
            interfaces = []
        else:
            if endian is None:
                raise ValueError("PCAPNG interface/packet precedes section")
            size = struct.unpack(endian + "I", length_bytes)[0]
            if size < 12 or size % 4 or size > MAX_BLOCK_BYTES:
                raise ValueError("invalid PCAPNG block length")
            rest = _read(source, size - 8)
            if struct.unpack(endian + "I", rest[-4:])[0] != size:
                raise ValueError("PCAPNG block trailing length mismatch")
            body = rest[:-4]
            kind = struct.unpack(endian + "I", magic)[0]
            if kind == 1:
                if len(body) < 8:
                    raise ValueError("short PCAPNG interface description")
                linktype, reserved, snaplen = struct.unpack(endian + "HHI", body[:8])
                _ethernet(linktype)
                if reserved:
                    raise ValueError("nonzero PCAPNG interface reserved field")
                resolution, offset = 1e-6, 0
                seen = set()
                for key, value in _options(body[8:], endian):
                    if key in (9, 14):
                        if key in seen:
                            raise ValueError("duplicate PCAPNG timestamp option")
                        seen.add(key)
                    if key == 9:
                        if len(value) != 1:
                            raise ValueError("invalid PCAPNG timestamp resolution")
                        unit = value[0]
                        resolution = (2.0 if unit & 128 else 10.0) ** -(unit & 127)
                    elif key == 14:
                        if len(value) != 8:
                            raise ValueError("invalid PCAPNG timestamp offset")
                        offset = struct.unpack(endian + "q", value)[0]
                interfaces.append((snaplen, resolution, offset))
            elif kind == 6:
                if len(body) < 20:
                    raise ValueError("short PCAPNG enhanced packet")
                index, high, low, captured, original = struct.unpack(endian + "IIIII", body[:20])
                if index >= len(interfaces):
                    raise ValueError("PCAPNG packet references absent interface")
                snaplen, resolution, offset = interfaces[index]
                _lengths(captured, original, snaplen)
                padded = (captured + 3) & ~3
                if len(body) < 20 + padded:
                    raise ValueError("incomplete PCAPNG packet payload")
                list(_options(body[20 + padded:], endian))
                yield ((high << 32) | low) * resolution + offset, body[20:20 + captured]
            elif kind in (2, 3):
                raise ValueError("unsupported PCAPNG legacy/simple packet block; timestamped enhanced packets required")
        magic = source.read(4)

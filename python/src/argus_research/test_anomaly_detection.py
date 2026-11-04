"""Synthetic Diagnostic Packet Simulation & Anomaly Detection Test Harness.

Generates a controlled, self-contained test PCAP containing both benign traffic
baselines and synthetic diagnostic attack patterns (TCP Port Sweep, UDP Burst,
TCP Flag Anomalies, Out-of-Order Sequences), feeds it through the Argus NIDS
pipeline and the 11 temporal neural models, and evaluates the detection telemetry.
"""

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
NIDS_EXE = (WORKSPACE / "target" / "debug" / "argus-nids.exe").resolve()
MODELS_DIR = (WORKSPACE.parent / "argus-dev" / "models").resolve()


def build_ipv4_packet(
    src_ip: str,
    dst_ip: str,
    proto: int,
    payload: bytes,
) -> bytes:
    """Constructs an Ethernet II + IPv4 packet with standard headers."""
    src_bytes = bytes(map(int, src_ip.split(".")))
    dst_bytes = bytes(map(int, dst_ip.split(".")))

    # Ethernet Header (14 bytes)
    eth = (
        bytes([0x00, 0x1A, 0x2B, 0x3C, 0x4D, 0x5E])
        + bytes([0x00, 0x5E, 0x4D, 0x3C, 0x2B, 0x1A])
        + struct.pack(">H", 0x0800)
    )

    # IPv4 Header (20 bytes)
    total_len = 20 + len(payload)
    ip_header = struct.pack(
        ">BBHHHBBH4s4s",
        0x45,  # Version (4) + IHL (5)
        0x00,  # DSCP / ECN
        total_len,  # Total Length
        0x1234,  # Identification
        0x4000,  # Flags (DF) + Frag Offset (0)
        64,  # TTL
        proto,  # Protocol (6=TCP, 17=UDP)
        0x0000,  # Header Checksum (placeholder)
        src_bytes,  # Source IP
        dst_bytes,  # Destination IP
    )
    return eth + ip_header + payload


def build_tcp_payload(
    src_port: int,
    dst_port: int,
    seq: int,
    ack: int,
    flags: int,
    window: int = 65535,
    data: bytes = b"",
) -> bytes:
    """Constructs a 20-byte TCP header + optional payload data."""
    tcp_header = struct.pack(
        ">HHIIBBHHH",
        src_port,
        dst_port,
        seq,
        ack,
        0x50,  # Data Offset (5 * 4 = 20 bytes)
        flags,  # TCP Flags (SYN=0x02, ACK=0x10, PSH=0x08, FIN=0x01, etc.)
        window,  # Window Size
        0x0000,  # Checksum (placeholder)
        0x0000,  # Urgent Pointer
    )
    return tcp_header + data


def build_udp_payload(
    src_port: int,
    dst_port: int,
    data: bytes = b"",
) -> bytes:
    """Constructs an 8-byte UDP header + payload."""
    udp_header = struct.pack(
        ">HHHH",
        src_port,
        dst_port,
        8 + len(data),
        0x0000,
    )
    return udp_header + data


def write_pcap(path: Path, records: list[tuple[int, int, bytes]]) -> None:
    """Writes standard classic PCAP file format (Link Type 1 = Ethernet)."""
    with open(path, "wb") as f:
        # PCAP Global Header (24 bytes)
        f.write(
            struct.pack(
                "<IHHIIII",
                0xA1B2C3D4,  # Magic (microseconds)
                2,
                4,  # Major, Minor
                0,
                0,  # Timezone, Accuracy
                65535,  # Snaplen
                1,  # LinkType: Ethernet
            )
        )
        for sec, usec, frame in records:
            frame_len = len(frame)
            # PCAP Record Header (16 bytes)
            f.write(
                struct.pack(
                    "<IIII",
                    sec,
                    usec,
                    frame_len,  # Included length
                    frame_len,  # Original wire length
                )
            )
            f.write(frame)


def generate_synthetic_attack_pcap(output_pcap: Path) -> dict[str, int]:
    """Generates a mixture of benign baseline flows and 3 diagnostic attack vectors."""
    records: list[tuple[int, int, bytes]] = []
    base_time = 1723800000  # Unix timestamp
    current_time_us = 0
    stats = {
        "benign_packets": 0,
        "syn_scan_packets": 0,
        "udp_flood_packets": 0,
        "tcp_anomaly_packets": 0,
    }

    # 1. Benign Baseline Flow (Normal HTTPS exchange between 192.168.1.10 and 93.184.216.34:443)
    client_ip = "192.168.1.10"
    server_ip = "93.184.216.34"
    c_port = 52140
    s_port = 443

    # 3-Way Handshake
    # Client SYN
    records.append(
        (
            base_time,
            current_time_us,
            build_ipv4_packet(
                client_ip,
                server_ip,
                6,
                build_tcp_payload(c_port, s_port, 1000, 0, 0x02),
            ),
        )
    )
    current_time_us += 15000  # +15ms RTT
    # Server SYN-ACK
    records.append(
        (
            base_time,
            current_time_us,
            build_ipv4_packet(
                server_ip,
                client_ip,
                6,
                build_tcp_payload(s_port, c_port, 5000, 1001, 0x12),
            ),
        )
    )
    current_time_us += 15000
    # Client ACK
    records.append(
        (
            base_time,
            current_time_us,
            build_ipv4_packet(
                client_ip,
                server_ip,
                6,
                build_tcp_payload(c_port, s_port, 1001, 5001, 0x10),
            ),
        )
    )
    current_time_us += 5000

    # Client TLS ClientHello (512 bytes payload)
    tls_payload = b"\x16\x03\x01\x02\x00" + b"A" * 507
    records.append(
        (
            base_time,
            current_time_us,
            build_ipv4_packet(
                client_ip,
                server_ip,
                6,
                build_tcp_payload(c_port, s_port, 1001, 5001, 0x18, data=tls_payload),
            ),
        )
    )
    current_time_us += 20000
    # Server ACK + TLS ServerHello
    records.append(
        (
            base_time,
            current_time_us,
            build_ipv4_packet(
                server_ip,
                client_ip,
                6,
                build_tcp_payload(s_port, c_port, 5001, 1513, 0x18, data=b"B" * 1200),
            ),
        )
    )
    stats["benign_packets"] = 5

    # 2. Attack Vector 1: Rapid TCP Port Sweep (Attacker scanning ports 20..100 on target)
    attacker_ip = "10.0.0.99"
    target_ip = "192.168.1.1"
    for target_port in range(21, 51):  # 30 targeted ports
        current_time_us += 200  # Very rapid 200 microsecond intervals
        syn_payload = build_tcp_payload(
            src_port=44444,
            dst_port=target_port,
            seq=target_port * 100,
            ack=0,
            flags=0x02,
        )
        records.append(
            (
                base_time,
                current_time_us,
                build_ipv4_packet(attacker_ip, target_ip, 6, syn_payload),
            )
        )
        stats["syn_scan_packets"] += 1

    # 3. Attack Vector 2: High-rate UDP Flood (Burst of UDP packets to random ports)
    flood_attacker = "198.51.100.77"
    for i in range(50):
        current_time_us += 100  # 100us burst
        udp_data = b"X" * 1024  # Large identical payload
        udp_payload = build_udp_payload(
            src_port=53000 + i, dst_port=8080, data=udp_data
        )
        records.append(
            (
                base_time,
                current_time_us,
                build_ipv4_packet(flood_attacker, target_ip, 17, udp_payload),
            )
        )
        stats["udp_flood_packets"] += 1

    # 4. Attack Vector 3: TCP Flag Evasion & Out-of-Order Sequence Anomalies
    evasion_attacker = "203.0.113.55"
    evasion_port = 60000
    # Abnormal Flag combinations (e.g. SYN+FIN = 0x03, NULL flag = 0x00, Xmas Scan = 0x29)
    current_time_us += 50000
    records.append(
        (
            base_time,
            current_time_us,
            build_ipv4_packet(
                evasion_attacker,
                target_ip,
                6,
                build_tcp_payload(evasion_port, 80, seq=100, ack=0, flags=0x03),
            ),
        )
    )
    current_time_us += 1000
    records.append(
        (
            base_time,
            current_time_us,
            build_ipv4_packet(
                evasion_attacker,
                target_ip,
                6,
                build_tcp_payload(evasion_port, 80, seq=5000, ack=0, flags=0x29),
            ),
        )
    )
    current_time_us += 1000
    # Out of order sequence
    records.append(
        (
            base_time,
            current_time_us,
            build_ipv4_packet(
                evasion_attacker,
                target_ip,
                6,
                build_tcp_payload(
                    evasion_port, 80, seq=200, ack=0, flags=0x18, data=b"PAYLOAD"
                ),
            ),
        )
    )
    stats["tcp_anomaly_packets"] = 3

    write_pcap(output_pcap, records)
    return stats


def run_pipeline_test(pcap_path: Path) -> None:
    """Executes argus-nids flow parsing and causal feature export."""
    csv_path = pcap_path.with_suffix(".csv").resolve()
    if csv_path.exists():
        csv_path.unlink()

    print(f"\n[1/3] Running Argus NIDS Pipeline on {pcap_path.name}...")
    nids_cmd = [str(NIDS_EXE), "--json", "--pcap", str(pcap_path)]
    result = subprocess.run(nids_cmd, capture_output=True, text=True, check=True)
    print("Pipeline Flow Analysis Result:\n" + result.stdout)

    print(
        f"[2/3] Exporting 12-Dimensional Causal Feature Vectors to {csv_path.name}..."
    )
    export_cmd = [str(NIDS_EXE), "--export-features", str(pcap_path), str(csv_path)]
    subprocess.run(export_cmd, capture_output=True, text=True, check=True)
    print(f"Features successfully exported ({os.path.getsize(csv_path)} bytes).")

    print("\n[3/3] Ingesting into 11-Model Neural Ensemble & MoE Router...")
    from argus_research.experimental_cnn import TemporalEnsemble

    ensemble = TemporalEnsemble(MODELS_DIR)
    telemetry = ensemble.infer(
        features_csv=csv_path,
        history_scope="diagnostic-attack-simulation",
        predict=True,
    )

    print("\n" + "=" * 70)
    print("ARGUS NIDS DETECTION & TEMPORAL MODEL EVALUATION REPORT")
    print("=" * 70)
    print(f"• Total Packet Events:    {telemetry['new_packets']}")
    print(f"• Memory Scope:           {telemetry['history_scope']}")
    router = telemetry.get("router", {})
    print(
        f"• Selected MoE Expert:    {router.get('selected_expert', router.get('state', 'none'))}"
    )
    if "confidence" in router:
        print(f"• Router Confidence:      {router['confidence']:.4f}")
    if "used_fallback" in router:
        print(f"• Router Used Fallback:   {router['used_fallback']}")
    print("\nActive Model Checkpoint Telemetry (Forecast Errors):")
    print("-" * 70)
    for model_name, model_info in telemetry["models"].items():
        if model_info.get("state") == "forecast_only":
            err = model_info.get("forecast_error", 0.0)
            latency = model_info.get("execution_ms", 0.0)
            pred_len = model_info.get("predicted_log_packet_length", 0.0)
            obs_len = model_info.get("observed_log_packet_length", 0.0)
            print(
                f"  [{model_name:<12}] Error: {err:.5f} | Latency: {latency:.2f}ms | Pred: {pred_len:.2f} vs Obs: {obs_len:.2f}"
            )
        else:
            print(f"  [{model_name:<12}] State: {model_info.get('state')}")
    print("=" * 70)

    # Clean up generated test artifacts
    if pcap_path.exists():
        pcap_path.unlink()
    if csv_path.exists():
        csv_path.unlink()


def main() -> None:
    test_dir = (WORKSPACE.parent / "argus-dev" / "captures" / "experimental").resolve()
    test_dir.mkdir(parents=True, exist_ok=True)
    pcap_path = (test_dir / "attack_simulation_test.pcap").resolve()

    print("=" * 70)
    print("GENERATING SYNTHETIC DIAGNOSTIC ATTACK TRAFFIC")
    print("=" * 70)
    stats = generate_synthetic_attack_pcap(pcap_path)
    for k, v in stats.items():
        print(f"  + {k.replace('_', ' ').title():<25}: {v} packets")
    print(f"  Total generated test packets: {sum(stats.values())}")

    run_pipeline_test(pcap_path)


if __name__ == "__main__":
    main()

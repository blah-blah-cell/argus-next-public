"""Published CIC attack intervals used only for evaluation-label auditing.

The intervals below are a compact transcription of the official UNB dataset
pages. They are not model features. A non-benign flow label is accepted only
when it agrees with exactly one published interval on the expected capture
date; benign traffic remains benign even while an attack scenario is active.
"""

from __future__ import annotations

import re
import ipaddress
from dataclasses import dataclass
from datetime import UTC, date, datetime, time

from .cse_pcap_to_journal import FlowKey


@dataclass(frozen=True)
class AttackEpisode:
    episode_id: str
    day: date
    start_wall_ns: int
    end_wall_ns: int
    label_aliases: frozenset[str]
    required_ports: frozenset[int]


@dataclass(frozen=True)
class EpisodeMatch:
    status: str
    episode: AttackEpisode | None
    candidate_count: int


def normalize_attack_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _wall_ns(day: date, value: str) -> int:
    parsed = datetime.combine(day, time.fromisoformat(value), tzinfo=UTC)
    return round(parsed.timestamp() * 1_000_000_000)


def _episode(
    day: str,
    start: str,
    end: str,
    name: str,
    *aliases: str,
    ports: tuple[int, ...] = (),
) -> AttackEpisode:
    parsed_day = date.fromisoformat(day)
    return AttackEpisode(
        episode_id=f"{day}:{start}-{end}:{normalize_attack_label(name)}",
        day=parsed_day,
        start_wall_ns=_wall_ns(parsed_day, start),
        # Published schedules are minute-resolution. Include the complete end
        # minute rather than silently treating, for example, 10:20 as 10:20:00.
        end_wall_ns=_wall_ns(parsed_day, end) + 59_999_999_999,
        label_aliases=frozenset(normalize_attack_label(item) for item in (name, *aliases)),
        required_ports=frozenset(ports),
    )


# Source: https://www.unb.ca/cic/datasets/ids-2017.html
# Source: https://www.unb.ca/cic/datasets/ids-2018.html
ATTACK_EPISODES = (
    _episode("2017-07-04", "09:20", "10:20", "FTP-Patator", "FTP-BruteForce"),
    _episode("2017-07-04", "14:00", "15:00", "SSH-Patator", "SSH-Bruteforce"),
    _episode("2017-07-05", "09:47", "10:10", "DoS slowloris", "Slowloris"),
    _episode("2017-07-05", "10:14", "10:35", "DoS Slowhttptest", "SlowHTTPTest"),
    _episode("2017-07-05", "10:43", "11:00", "DoS Hulk", "Hulk"),
    _episode("2017-07-05", "11:10", "11:23", "DoS GoldenEye", "GoldenEye"),
    _episode("2017-07-05", "15:12", "15:32", "Heartbleed", ports=(444,)),
    _episode("2017-07-06", "09:20", "10:00", "Web Attack Brute Force", "Web Brute Force"),
    _episode("2017-07-06", "10:15", "10:35", "Web Attack XSS", "XSS"),
    _episode("2017-07-06", "10:40", "10:42", "Web Attack Sql Injection", "SQL Injection"),
    _episode("2017-07-06", "14:19", "14:21", "Infiltration"),
    _episode("2017-07-06", "14:33", "14:35", "Infiltration"),
    _episode("2017-07-06", "14:53", "15:00", "Infiltration"),
    _episode("2017-07-06", "15:04", "15:45", "Infiltration"),
    _episode("2017-07-07", "10:02", "11:02", "Bot", "Botnet ARES"),
    *(
        _episode("2017-07-07", start, end, "PortScan", "Port Scan")
        for start, end in (
            ("13:55", "13:57"),
            ("13:58", "14:00"),
            ("14:01", "14:04"),
            ("14:05", "14:07"),
            ("14:08", "14:10"),
            ("14:11", "14:13"),
            ("14:14", "14:16"),
            ("14:17", "14:19"),
            ("14:20", "14:21"),
            ("14:22", "14:24"),
            ("14:33", "14:33"),
            ("14:35", "14:35"),
            ("14:51", "14:53"),
            ("14:54", "14:56"),
            ("14:57", "14:59"),
            ("15:00", "15:02"),
            ("15:03", "15:05"),
            ("15:06", "15:07"),
            ("15:08", "15:10"),
            ("15:11", "15:12"),
            ("15:13", "15:15"),
            ("15:16", "15:18"),
            ("15:19", "15:21"),
            ("15:22", "15:24"),
            ("15:25", "15:25"),
            ("15:26", "15:27"),
            ("15:28", "15:29"),
        )
    ),
    _episode("2017-07-07", "15:56", "16:16", "DDoS", "DDoS LOIT", "DDoS LOIC"),
    _episode("2018-02-14", "10:32", "12:09", "FTP-BruteForce", "FTP-Patator"),
    _episode("2018-02-14", "14:01", "15:31", "SSH-Bruteforce", "SSH-Patator"),
    _episode("2018-02-15", "09:26", "10:09", "DoS-GoldenEye", "GoldenEye"),
    _episode("2018-02-15", "10:59", "11:40", "DoS-Slowloris", "Slowloris"),
    _episode("2018-02-16", "10:12", "11:08", "DoS-SlowHTTPTest", "SlowHTTPTest"),
    _episode("2018-02-16", "13:45", "14:19", "DoS-Hulk", "Hulk"),
    _episode("2018-02-20", "10:12", "11:17", "DDoS attacks-LOIC-HTTP", "DDoS-LOIC-HTTP"),
    _episode("2018-02-20", "13:13", "13:32", "DDoS-LOIC-UDP", "DDOS attack-LOIC-UDP"),
    _episode("2018-02-21", "10:09", "10:43", "DDoS-LOIC-UDP", "DDOS attack-LOIC-UDP"),
    _episode("2018-02-21", "14:05", "15:05", "DDoS-HOIC", "DDOS attack-HOIC"),
    _episode("2018-02-22", "10:17", "11:24", "Brute Force -Web", "Web Attack Brute Force"),
    _episode("2018-02-22", "13:50", "14:29", "Brute Force -XSS", "Web Attack XSS"),
    _episode("2018-02-22", "16:15", "16:29", "SQL Injection", "Web Attack Sql Injection"),
    _episode("2018-02-23", "10:03", "11:03", "Brute Force -Web", "Web Attack Brute Force"),
    _episode("2018-02-23", "13:00", "14:10", "Brute Force -XSS", "Web Attack XSS"),
    _episode("2018-02-23", "15:05", "15:18", "SQL Injection", "Web Attack Sql Injection"),
    _episode("2018-02-28", "10:50", "12:05", "Infiltration"),
    _episode("2018-02-28", "13:42", "14:40", "Infiltration"),
    _episode("2018-03-01", "09:57", "10:55", "Infiltration"),
    _episode("2018-03-01", "14:00", "15:37", "Infiltration"),
    _episode("2018-03-02", "10:11", "11:34", "Bot", "Botnet"),
    _episode("2018-03-02", "14:24", "15:55", "Bot", "Botnet"),
)


def _ips(*values: str) -> frozenset[bytes]:
    return frozenset(ipaddress.ip_address(value).packed for value in values)


# At least one endpoint in an attack-labeled packet must appear in the
# corresponding official day's attacker/victim/NAT path. This complements the
# exact bidirectional identity-rich flow join; IPs still never enter a model.
DAY_ATTACK_ENDPOINTS = {
    date(2017, 7, 4): _ips(
        "205.174.165.73", "205.174.165.80", "172.16.0.1", "192.168.10.50", "205.174.165.68"
    ),
    date(2017, 7, 5): _ips(
        "205.174.165.73", "205.174.165.80", "172.16.0.1", "172.16.0.11",
        "192.168.10.50", "192.168.10.51", "205.174.165.68", "205.174.165.66"
    ),
    date(2017, 7, 6): _ips(
        "205.174.165.73", "205.174.165.80", "172.16.0.1", "192.168.10.50",
        "205.174.165.68", "192.168.10.8", "192.168.10.25"
    ),
    date(2017, 7, 7): _ips(
        "205.174.165.69", "205.174.165.70", "205.174.165.71", "205.174.165.73",
        "205.174.165.80", "172.16.0.1", "192.168.10.50", "205.174.165.68",
        "192.168.10.5", "192.168.10.8", "192.168.10.9", "192.168.10.14",
        "192.168.10.15"
    ),
    date(2018, 2, 14): _ips(
        "172.31.70.4", "18.221.219.4", "172.31.70.6", "13.58.98.64",
        "172.31.69.25", "18.217.21.148"
    ),
    date(2018, 2, 15): _ips(
        "172.31.70.46", "18.219.211.138", "172.31.70.8", "18.217.165.70",
        "172.31.69.25", "18.217.21.148"
    ),
    date(2018, 2, 16): _ips(
        "172.31.70.23", "13.59.126.31", "172.31.70.16", "18.219.193.20",
        "172.31.69.25", "18.217.21.148"
    ),
    date(2018, 2, 20): _ips(
        "18.218.115.60", "18.219.9.1", "18.219.32.43", "18.218.55.126",
        "52.14.136.135", "18.219.5.43", "18.216.200.189", "18.218.229.235",
        "18.218.11.51", "18.216.24.42", "18.217.21.148", "172.31.69.25"
    ),
    date(2018, 2, 21): _ips(
        "18.218.115.60", "18.219.9.1", "18.219.32.43", "18.218.55.126",
        "52.14.136.135", "18.219.5.43", "18.216.200.189", "18.218.229.235",
        "18.218.11.51", "18.216.24.42", "18.218.83.150", "172.31.69.28"
    ),
    date(2018, 2, 22): _ips(
        "18.218.115.60", "18.218.83.150", "172.31.69.28"
    ),
    date(2018, 2, 23): _ips(
        "18.218.115.60", "18.218.83.150", "172.31.69.28"
    ),
    date(2018, 2, 28): _ips(
        "13.58.225.34", "18.221.148.137", "172.31.69.24"
    ),
    date(2018, 3, 1): _ips(
        "13.58.225.34", "18.216.254.154", "172.31.69.13"
    ),
    date(2018, 3, 2): _ips(
        "18.219.211.138", "18.217.218.111", "172.31.69.23", "18.222.10.237",
        "172.31.69.17", "18.222.86.193", "172.31.69.14", "18.222.62.221",
        "172.31.69.12", "13.59.9.106", "172.31.69.10", "18.222.102.2",
        "172.31.69.8", "18.219.212.0", "172.31.69.6", "18.216.105.13",
        "172.31.69.26", "18.219.163.126", "172.31.69.29", "18.216.164.12",
        "172.31.69.30"
    ),
}


def match_attack_episode(
    label: str,
    wall_timestamp_ns: int,
    expected_date: date,
    *,
    packet_key: FlowKey | None = None,
) -> EpisodeMatch:
    """Require a non-benign flow label to match one published interval."""
    normalized = normalize_attack_label(label)
    if not normalized:
        raise ValueError("flow label must not be empty")
    if normalized == "benign":
        return EpisodeMatch("benign", None, 0)
    candidates = [
        episode
        for episode in ATTACK_EPISODES
        if episode.day == expected_date
        and episode.start_wall_ns <= wall_timestamp_ns <= episode.end_wall_ns
        and normalized in episode.label_aliases
    ]
    if not candidates:
        return EpisodeMatch("schedule_disagreement", None, 0)
    if packet_key is not None:
        endpoints = DAY_ATTACK_ENDPOINTS.get(expected_date)
        if endpoints is None or not ({packet_key[0], packet_key[1]} & endpoints):
            return EpisodeMatch("schedule_endpoint_disagreement", None, 0)
        candidates = [
            episode
            for episode in candidates
            if not episode.required_ports
            or bool({packet_key[3], packet_key[4]} & episode.required_ports)
        ]
        if not candidates:
            return EpisodeMatch("schedule_port_disagreement", None, 0)
    if len(candidates) != 1:
        return EpisodeMatch("schedule_ambiguous", None, len(candidates))
    return EpisodeMatch("attack", candidates[0], 1)

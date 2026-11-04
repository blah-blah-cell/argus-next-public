"""Exact ordered feature contract for publication-eligible ARGUSLR2 journals."""

from __future__ import annotations

import hashlib
import json
from typing import Any

FEATURE_SCHEMA = "packet-causal-flow-v2"
DATASET_SOURCES = {
    "CSE-CIC-IDS2018": "s3://cse-cic-ids2018/",
    "CIC-IDS2017": "https://www.unb.ca/cic/datasets/ids-2017.html",
}
FEATURE_CONTRACT: dict[str, Any] = {
    "schema": FEATURE_SCHEMA,
    "dimension": 12,
    "fields": [
        {"index": 0, "name": "frame_length", "unit": "log1p(bytes)"},
        {
            "index": 1,
            "name": "per_flow_interarrival",
            "unit": "log1p(seconds)",
            "clip_seconds": [0.0, 60.0],
            "first_packet": 0.0,
        },
        {"index": 2, "name": "is_ipv4", "unit": "boolean"},
        {"index": 3, "name": "is_ipv6", "unit": "boolean"},
        {"index": 4, "name": "is_tcp", "unit": "boolean"},
        {"index": 5, "name": "is_udp", "unit": "boolean"},
        {"index": 6, "name": "tcp_syn", "unit": "boolean"},
        {"index": 7, "name": "tcp_ack", "unit": "boolean"},
        {"index": 8, "name": "tcp_fin", "unit": "boolean"},
        {"index": 9, "name": "tcp_rst", "unit": "boolean"},
        {"index": 10, "name": "tcp_push", "unit": "boolean"},
        {"index": 11, "name": "tcp_urg", "unit": "boolean"},
    ],
    "availability": "current_and_causally_prior_packets_only",
    "endpoint_identifiers_in_features": False,
    "payload_bytes_in_features": False,
}
FEATURE_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(FEATURE_CONTRACT, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

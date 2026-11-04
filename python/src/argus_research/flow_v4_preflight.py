"""Read-only source and coordinate preflight for the frozen AHEM-v5 study."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .flow_v2_protocol import write_manifest
from .flow_v2_train import _canonical_sha256, _source_provenance
from .flow_v3_protocol import ProtocolConfig, compile_source_disjoint
from .flow_v3_smoke import _long_gap_metadata
from .flow_v3_train import (
    PRIMARY_TEST_SOURCE,
    PRIMARY_TRAIN_SOURCES,
    PRIMARY_VALIDATION_SOURCE,
    V4_MODEL_NAMES,
    V4_PREREGISTRATION_SHA256,
    model_spec,
)

PREFLIGHT_SCHEMA = "argus.ahem-v5-preflight/v1"
FROZEN_MODELS = V4_MODEL_NAMES


def source_identity_bindings(
    journal: Path, expected_sources: set[str]
) -> dict[str, Any]:
    """Bind each logical source name to converter-recorded archive/member names."""

    def normalized(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    sources = {}
    for source in sorted(expected_sources):
        path = journal / f"{source}.arguslr"
        sidecar_path = path.with_suffix(path.suffix + ".manifest.json")
        problems = []
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
            sidecar = {}
            problems.append(f"missing or invalid sidecar: {type(error).__name__}")
        if not path.is_file():
            problems.append("journal file is absent")
        if sidecar.get("journal", {}).get("name") != path.name:
            problems.append("sidecar journal name does not match the source filename")
        dataset = sidecar.get("dataset")
        source_uri = sidecar.get("archive", {}).get("source_uri")
        identity_text = " ".join(
            [
                str(sidecar.get("archive", {}).get("name", "")),
                str(source_uri or ""),
            ]
            + [str(member.get("name", "")) for member in sidecar.get("members", [])]
        )
        if normalized(source) not in normalized(identity_text):
            problems.append("capture-day identity is absent from archive/member names")
        if dataset == "CSE-CIC-IDS2018":
            official_root = "s3://cse-cic-ids2018/"
            if not isinstance(source_uri, str) or not source_uri.startswith(
                official_root
            ):
                problems.append("official CSE-CIC-IDS2018 object URI is absent")
            elif normalized(source) not in normalized(source_uri):
                problems.append(
                    "capture-day identity is absent from the official object URI"
                )
        sources[source] = {
            "journal": path.name,
            "sidecar": sidecar_path.name,
            "archive_name": sidecar.get("archive", {}).get("name"),
            "source_uri": source_uri,
            "member_names": [
                member.get("name") for member in sidecar.get("members", [])
            ],
            "problems": problems,
            "bound": not problems,
        }
    return {
        "complete": bool(sources) and all(value["bound"] for value in sources.values()),
        "sources": sources,
    }


def _gap_counts(protocol: Any, partition: str, context: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for coordinate in protocol.coordinates[partition]:
        record = protocol.partitions[partition][coordinate.record_index]
        metadata = _long_gap_metadata(record.flow_ids, coordinate.anchor_index, context)
        counts[str(metadata["long_gap_bucket"])] += 1
    return dict(sorted(counts.items()))


def run(
    journal: Path,
    output: Path,
    *,
    allow_dirty_source: bool = False,
    context: int = 10_000,
    train_samples: int = 2_048,
    validation_samples: int = 512,
    test_samples: int = 512,
    protocol_seed: int = 20260830,
) -> dict[str, Any]:
    source = _source_provenance()
    source_clean = source.get("available") is True and source.get("git_dirty") is False
    if not allow_dirty_source and not source_clean:
        raise RuntimeError("v4 confirmatory preflight requires a clean Git revision")
    config = ProtocolConfig(
        context_packets=context,
        train_samples=train_samples,
        validation_samples=validation_samples,
        test_samples=test_samples,
        seed=protocol_seed,
        require_memory_match=True,
    )
    protocol = compile_source_disjoint(
        journal,
        PRIMARY_VALIDATION_SOURCE,
        PRIMARY_TEST_SOURCE,
        config,
    )
    protocol.manifest["preregistration_sha256"] = V4_PREREGISTRATION_SHA256
    expected_split = {
        "method": "journal_source_file_disjoint",
        "validation_source": PRIMARY_VALIDATION_SOURCE,
        "test_source": PRIMARY_TEST_SOURCE,
        "train_sources": sorted(PRIMARY_TRAIN_SOURCES),
    }
    split_ok = protocol.manifest.get("split") == expected_split
    journal_manifest = protocol.manifest.get("journal", {})
    provenance_ok = journal_manifest.get(
        "provenance_complete"
    ) is True and journal_manifest.get("datasets") == ["CSE-CIC-IDS2018"]
    source_bindings = source_identity_bindings(
        journal,
        set(expected_split["train_sources"])
        | {PRIMARY_VALIDATION_SOURCE, PRIMARY_TEST_SOURCE},
    )
    frozen_dimensions = (
        context == 10_000
        and train_samples == 2_048
        and validation_samples == 512
        and test_samples == 512
        and protocol_seed == 20260830
    )
    models = {}
    for name in FROZEN_MODELS:
        model = model_spec(name).constructor()
        models[name] = {
            "family": model_spec(name).family,
            "parameters": sum(value.numel() for value in model.parameters()),
        }
    gap_counts = {
        partition: _gap_counts(protocol, partition, context)
        for partition in ("train", "validation", "test")
    }
    populated_primary_buckets = {
        bucket: count
        for bucket, count in gap_counts["test"].items()
        if bucket in {"129_to_1023", "1024_to_4095", "4096_to_9999"} and count >= 50
    }
    confirmatory_ready = bool(
        source_clean
        and split_ok
        and provenance_ok
        and source_bindings["complete"]
        and frozen_dimensions
        and populated_primary_buckets
    )
    result = {
        "schema": PREFLIGHT_SCHEMA,
        "study_role": "diagnostic" if allow_dirty_source else "confirmatory_preflight",
        "preregistration_sha256": V4_PREREGISTRATION_SHA256,
        "confirmatory_ready": confirmatory_ready,
        "source_provenance": source,
        "checks": {
            "clean_source": source_clean,
            "frozen_source_split": split_ok,
            "cse_cic_ids2018_provenance": provenance_ok,
            "source_day_identity_bindings": source_bindings["complete"],
            "frozen_sampling_dimensions": frozen_dimensions,
            "at_least_one_populated_primary_long_gap_bucket": bool(
                populated_primary_buckets
            ),
        },
        "protocol": protocol.manifest,
        "source_identity_bindings": source_bindings,
        "protocol_sha256": _canonical_sha256(protocol.manifest),
        "coordinate_sha256": protocol.manifest["coordinate_sha256"],
        "long_gap_bucket_counts": gap_counts,
        "populated_primary_long_gap_buckets": populated_primary_buckets,
        "models": models,
    }
    write_manifest(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument("--context", type=int, default=10_000)
    parser.add_argument("--train-samples", type=int, default=2_048)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--protocol-seed", type=int, default=20260830)
    args = parser.parse_args()
    report = run(
        args.journal,
        args.output,
        allow_dirty_source=args.allow_dirty_source,
        context=args.context,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        protocol_seed=args.protocol_seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

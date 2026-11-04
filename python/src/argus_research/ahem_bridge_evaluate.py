"""Fit and evaluate the frozen AHEM forecast-to-detection bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .ahem_bridge_events import (
    EVENT_SCHEMA, MANIFEST_SCHEMA, EXPECTED_CHECKPOINT_SHA256,
)
from .ahem_detection_bridge import (
    bridge_metrics,
    load_frozen_plan,
    paired_hierarchical_pr_auc_bootstrap,
    select_threshold_for_false_alert_budget,
)
from .ahem_detection_probe import (
    VARIANTS,
    BridgeEvents,
    FittedProbe,
    ensemble_probabilities,
    fit_probe_ensemble,
    variant_matrix,
)

RESULT_SCHEMA = "argus.ahem-v61-detection-bridge-result/v1"
REFRACTORY_NS = 60_000_000_000


def validate_input_manifests(
    groups: dict[str, list[Path]], plan: dict[str, Any], plan_hash: str
) -> None:
    """Check the frozen source/model contract before opening any event arrays."""
    seen_archives: set[str] = set()
    seen_events: set[str] = set()
    expected_checkpoints = {
        f"{seed}/{name}": digest
        for seed, models in EXPECTED_CHECKPOINT_SHA256.items()
        for name, digest in models.items()
    }
    for role, paths in groups.items():
        source_spec = plan["data"][role]
        sources = source_spec.get("sources", [source_spec.get("source")])
        expected_dates = {
            datetime.strptime(source.split("-", 1)[1], "%d-%m-%Y").date().isoformat()
            for source in sources
        }
        dates: list[str] = []
        for path in paths:
            manifest = json.loads(
                path.with_suffix(path.suffix + ".manifest.json").read_text(encoding="utf-8")
            )
            if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("scientific_eligible") is not True:
                raise ValueError(f"{path}: ineligible event manifest")
            if manifest.get("frozen_plan_sha256") != plan_hash:
                raise ValueError(f"{path}: frozen plan SHA-256 mismatch")
            checkpoints = manifest.get("checkpoints", {})
            if set(checkpoints) != set(expected_checkpoints) or any(
                checkpoints[key].get("sha256") != digest
                for key, digest in expected_checkpoints.items()
            ):
                raise ValueError(f"{path}: frozen checkpoint identities mismatch")
            dates.append(manifest.get("expected_date"))
            for field, digest, seen in (
                ("archive", manifest.get("archive", {}).get("sha256"), seen_archives),
                ("event", manifest.get("event_npz", {}).get("sha256"), seen_events),
            ):
                if not isinstance(digest, str) or len(digest) != 64 or any(
                    char not in "0123456789abcdef" for char in digest
                ):
                    raise ValueError(f"{path}: invalid {field} SHA-256")
                if digest in seen:
                    raise ValueError(f"{path}: duplicate {field} source across inputs")
                seen.add(digest)
            labels = manifest.get("labels", [])
            if not labels or any(
                not isinstance(item.get("sha256"), str)
                or len(item["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in item["sha256"])
                for item in labels
            ):
                raise ValueError(f"{path}: missing or invalid label SHA-256")
        if len(dates) != len(expected_dates) or set(dates) != expected_dates:
            raise ValueError(f"{role}: requires exactly the frozen capture dates")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EventArtifact:
    events: BridgeEvents
    attack_families: np.ndarray
    episode_ids: np.ndarray
    capture_ids: np.ndarray
    flow_ids: np.ndarray
    timestamps_ns: np.ndarray
    source_ordinals: np.ndarray
    monitored_seconds: float
    provenance: dict[str, Any]


def load_event_artifact(path: Path) -> EventArtifact:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{manifest_path} has an incompatible schema")
    if manifest.get("scientific_eligible") is not True:
        raise ValueError(f"{path} was produced by a non-scientific preflight")
    expected_hash = manifest.get("event_npz", {}).get("sha256")
    if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
        raise ValueError(f"{path} does not match its manifest SHA-256")
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "schema",
            "base",
            "ahem_residuals",
            "lstm_residuals",
            "labels",
            "attack_families",
            "episode_ids",
            "capture_ids",
            "flow_ids",
            "timestamps_ns",
            "source_ordinals",
        }
        if set(payload.files) != required:
            raise ValueError(f"{path} has missing or unexpected event arrays")
        if str(payload["schema"].item()) != EVENT_SCHEMA:
            raise ValueError(f"{path} has an incompatible event schema")
        arrays = {name: payload[name].copy() for name in required - {"schema"}}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("event artifacts cannot contain pickle-backed arrays")
    events = BridgeEvents(
        arrays["base"],
        arrays["ahem_residuals"],
        arrays["lstm_residuals"],
        arrays["labels"],
    )
    events.validate(require_both_classes=False)
    count = events.event_count
    vector_names = (
        "attack_families",
        "episode_ids",
        "capture_ids",
        "flow_ids",
        "timestamps_ns",
        "source_ordinals",
    )
    if any(arrays[name].shape != (count,) for name in vector_names):
        raise ValueError("event metadata arrays are not aligned")
    if not np.issubdtype(arrays["timestamps_ns"].dtype, np.integer):
        raise ValueError("event timestamps must be integer nanoseconds")
    if set(arrays["capture_ids"].tolist()) != {manifest.get("expected_date")}:
        raise ValueError("event capture IDs do not match the manifest capture date")
    monitored_seconds = manifest.get("complete_monitored_capture_seconds")
    if (
        not isinstance(monitored_seconds, (int, float))
        or not np.isfinite(monitored_seconds)
        or monitored_seconds <= 0
    ):
        raise ValueError("manifest requires a positive complete capture duration")
    return EventArtifact(
        events=events,
        attack_families=arrays["attack_families"],
        episode_ids=arrays["episode_ids"],
        capture_ids=arrays["capture_ids"],
        flow_ids=arrays["flow_ids"],
        timestamps_ns=arrays["timestamps_ns"].astype(np.int64),
        source_ordinals=arrays["source_ordinals"].astype(np.int64),
        monitored_seconds=float(monitored_seconds),
        provenance={
            "path": path.name,
            "sha256": expected_hash,
            "manifest_path": manifest_path.name,
            "manifest_sha256": _sha256(manifest_path),
            "archive": manifest.get("archive"),
            "labels": manifest.get("labels"),
            "expected_date": manifest.get("expected_date"),
            "frozen_plan_sha256": manifest.get("frozen_plan_sha256"),
            "checkpoints": manifest.get("checkpoints"),
        },
    )


def combine_artifacts(artifacts: list[EventArtifact]) -> EventArtifact:
    if not artifacts:
        raise ValueError("at least one event artifact is required")
    return EventArtifact(
        events=BridgeEvents(
            np.concatenate([item.events.base for item in artifacts]),
            np.concatenate([item.events.ahem_residuals for item in artifacts]),
            np.concatenate([item.events.lstm_residuals for item in artifacts]),
            np.concatenate([item.events.labels for item in artifacts]),
        ),
        attack_families=np.concatenate([item.attack_families for item in artifacts]),
        episode_ids=np.concatenate([item.episode_ids for item in artifacts]),
        capture_ids=np.concatenate([item.capture_ids for item in artifacts]),
        flow_ids=np.concatenate([item.flow_ids for item in artifacts]),
        timestamps_ns=np.concatenate([item.timestamps_ns for item in artifacts]),
        source_ordinals=np.concatenate([item.source_ordinals for item in artifacts]),
        monitored_seconds=sum(item.monitored_seconds for item in artifacts),
        provenance={"inputs": [item.provenance for item in artifacts]},
    )


def _probe_identity(probe: FittedProbe) -> dict[str, Any]:
    return {
        "variant": probe.variant,
        "seed_index": probe.seed_index,
        "coefficient": probe.coefficient.tolist(),
        "intercept": probe.intercept,
        "temperature": probe.temperature,
        "preprocessor": {
            name: getattr(probe.preprocessor, name).tolist()
            for name in ("median", "low", "high", "mean", "std")
        },
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"result already exists: {path}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as destination:
        destination.write(encoded)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    access_path = args.output.with_suffix(args.output.suffix + ".access.json")
    if any(path.exists() for path in (
        args.output, args.output.with_suffix(args.output.suffix + ".partial"),
        access_path, access_path.with_suffix(access_path.suffix + ".partial"),
    )):
        raise FileExistsError("evaluation output or access record already exists; preserve the original run")
    plan = load_frozen_plan(args.plan)
    groups = {
        "detector_fit": args.fit, "probability_calibration": args.calibration,
        "operating_point_calibration": args.operating, "external_test": args.test,
    }
    validate_input_manifests(
        groups, plan, _sha256(args.plan),
    )
    # This is evaluator access, not a replacement for raw-test first-access ledgers.
    # Keep it even after a crash so a replacement run cannot silently erase access.
    _atomic_json(access_path, {
        "schema": "argus.ahem-v61-bridge-evaluation-access/v1",
        "stage": "before_event_array_access",
        "frozen_plan_sha256": _sha256(args.plan),
        "inputs": {
            role: [
                {"path": str(path), "manifest_sha256": _sha256(
                    path.with_suffix(path.suffix + ".manifest.json")
                )} for path in paths
            ] for role, paths in groups.items()
        },
    })
    fit_inputs = [load_event_artifact(path) for path in args.fit]
    calibration_inputs = [load_event_artifact(path) for path in args.calibration]
    operating_inputs = [load_event_artifact(path) for path in args.operating]
    test_inputs = [load_event_artifact(path) for path in args.test]
    fit = combine_artifacts(fit_inputs)
    calibration = combine_artifacts(calibration_inputs)
    operating = combine_artifacts(operating_inputs)
    test = combine_artifacts(test_inputs)
    fit.events.validate()
    calibration.events.validate()
    test.events.validate()
    operating.events.validate(require_both_classes=False)
    if np.any(operating.events.labels != 0):
        raise ValueError("operating-point calibration must be benign-only")
    if len(set(test.capture_ids.tolist())) < 4:
        raise ValueError("external test must contain all four frozen capture days")

    variant_results: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        probes = fit_probe_ensemble(fit.events, calibration.events, variant)
        operating_probabilities = ensemble_probabilities(probes, operating.events)
        threshold = select_threshold_for_false_alert_budget(
            operating_probabilities,
            operating.capture_ids.tolist(),
            operating.flow_ids.tolist(),
            operating.timestamps_ns,
            monitored_seconds=operating.monitored_seconds,
            maximum_false_alerts_per_hour=1.0,
            refractory_ns=REFRACTORY_NS,
        )
        test_probabilities = ensemble_probabilities(probes, test.events)
        probabilities[variant] = test_probabilities
        episode_ids = [value if value else None for value in test.episode_ids.tolist()]
        variant_results[variant] = {
            "threshold_source": "CIC-IDS2017 Monday benign only",
            "operating_threshold": threshold,
            "metrics": bridge_metrics(
                test_probabilities,
                test.events.labels,
                test.capture_ids.tolist(),
                test.flow_ids.tolist(),
                test.timestamps_ns,
                episode_ids,
                threshold=threshold,
                monitored_seconds=test.monitored_seconds,
                refractory_ns=REFRACTORY_NS,
            ),
            "per_seed_test_pr_auc": [
                bridge_metrics(
                    probe.probabilities(variant_matrix(test.events, variant, seed_index)),
                    test.events.labels,
                    test.capture_ids.tolist(),
                    test.flow_ids.tolist(),
                    test.timestamps_ns,
                    episode_ids,
                    threshold=threshold,
                    monitored_seconds=test.monitored_seconds,
                    refractory_ns=REFRACTORY_NS,
                )["attack_pr_auc"]
                for seed_index, probe in enumerate(probes)
            ],
            "probes": [_probe_identity(probe) for probe in probes],
        }

    bootstrap = paired_hierarchical_pr_auc_bootstrap(
        probabilities["base"],
        probabilities["base_plus_ahem"],
        probabilities["base_plus_lstm"],
        test.events.labels,
        test.capture_ids.tolist(),
        test.timestamps_ns,
        resamples=int(plan["primary_endpoints"]["bootstrap"]["resamples"]),
        seed=int(plan["primary_endpoints"]["bootstrap"]["seed"]),
    )
    ahem_metrics = variant_results["base_plus_ahem"]["metrics"]
    gates = {
        "both_pr_auc_superiority_bounds_positive": bootstrap["both_supported"],
        "external_test_false_alerts_per_hour_at_most_one": (
            ahem_metrics["false_alerts_per_hour"] <= 1.0
        ),
    }
    result = {
        "schema": RESULT_SCHEMA,
        "decision": "supported" if all(gates.values()) else "not_supported",
        "reportable_regardless_of_verdict": True,
        "plan_sha256": _sha256(args.plan),
        "inputs": {
            "fit": [item.provenance for item in fit_inputs],
            "probability_calibration": [
                item.provenance for item in calibration_inputs
            ],
            "operating_point": [item.provenance for item in operating_inputs],
            "external_test": [item.provenance for item in test_inputs],
        },
        "event_counts": {
            "fit": fit.events.event_count,
            "probability_calibration": calibration.events.event_count,
            "operating_point": operating.events.event_count,
            "external_test": test.events.event_count,
        },
        "variants": variant_results,
        "primary_bootstrap": bootstrap,
        "gates": gates,
        "limitations": [
            "CIC-IDS2017 and CSE-CIC-IDS2018 are same-lab, related-generation corpora.",
            "A pass establishes a forecast-residual detection bridge, not universal LSTM superiority.",
            "The frozen v6.1 learned-retrieval top-1 gate remains failed and is not repaired here.",
        ],
    }
    _atomic_json(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fit", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration", type=Path, nargs="+", required=True)
    parser.add_argument("--operating", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()

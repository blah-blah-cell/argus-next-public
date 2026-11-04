"""Fail-closed admission decisions for Argus research artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

POLICY = {
    "version": "argus.model-admission/v1",
    "security": {
        "min_attack_average_precision": 0.95,
        "min_recall": 0.80,
        "max_false_positive_rate": 0.01,
        "max_ece": 0.05,
        "max_brier": 0.10,
    },
    "forecast_router": {
        "min_test_samples": 128,
        "min_gain_over_best_fixed": 0.0,
        "max_regret_fraction_of_best_fixed": 0.25,
    },
}

EXPERT_FILES = {
    "ahem_fixed": "ahem_fixed_real_pcap_30k.pt",
    "cnn_local": "cnn_local_real_pcap_30k.pt",
    "lstm": "lstm_real_pcap_30k.pt",
    "tcn_long": "tcn_long_real_pcap_30k.pt",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check(name: str, passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
    }


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(value)


def _disjoint_nonempty(groups: list[list[str]]) -> bool:
    if any(not group for group in groups):
        return False
    sets = [set(group) for group in groups]
    return all(
        sets[left].isdisjoint(sets[right])
        for left in range(len(sets))
        for right in range(left + 1, len(sets))
    )


def _load_manifest_pair(
    json_path: Path, checkpoint_path: Path
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        embedded = payload.get("manifest") if isinstance(payload, dict) else None
        return manifest, embedded, None
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        return manifest, None, str(error)


def evaluate_security_detector(
    manifest: dict[str, Any],
    embedded: dict[str, Any] | None,
    checkpoint_path: Path,
    policy: dict[str, Any] = POLICY,
) -> dict[str, Any]:
    security = policy["security"]
    train = manifest.get("train_files", [])
    calibration = manifest.get("calibration_files", [])
    test = manifest.get("test_files", [])
    all_sources = train + calibration + test
    hashes = manifest.get("source_sha256", {})
    preprocessor = manifest.get("preprocessor", {})
    names = preprocessor.get("feature_names", [])
    arrays = [
        preprocessor.get(key, [])
        for key in ("median", "clip_low", "clip_high", "mean", "std")
    ]
    metrics = manifest.get("test_metrics", {})
    metric_requirements = (
        ("attack_average_precision", security["min_attack_average_precision"], "min"),
        ("recall", security["min_recall"], "min"),
        ("false_positive_rate", security["max_false_positive_rate"], "max"),
        ("ece_15_bin", security["max_ece"], "max"),
        ("brier", security["max_brier"], "max"),
    )
    checks = [
        _check(
            "checkpoint_exists",
            checkpoint_path.is_file(),
            checkpoint_path.name,
            "existing file",
        ),
        _check(
            "manifest_bound_to_checkpoint",
            embedded == manifest,
            embedded is not None,
            True,
        ),
        _check(
            "artifact_type",
            manifest.get("artifact_type") == "calibrated_retrospective_flow_detector",
            manifest.get("artifact_type"),
            "calibrated_retrospective_flow_detector",
        ),
        _check(
            "live_feature_compatibility",
            manifest.get("admitted_for_live_packet_detection") is True,
            manifest.get("admitted_for_live_packet_detection"),
            True,
        ),
        _check(
            "chronological_split",
            manifest.get("split_policy") == "whole_day_disjoint",
            manifest.get("split_policy"),
            "whole_day_disjoint",
        ),
        _check(
            "split_files_disjoint",
            _disjoint_nonempty([train, calibration, test]),
            {"train": train, "calibration": calibration, "test": test},
            "non-empty pairwise-disjoint files",
        ),
        _check(
            "source_hashes_complete",
            set(hashes) == set(all_sources)
            and all(
                isinstance(value, str) and len(value) == 64 for value in hashes.values()
            ),
            sorted(hashes),
            sorted(all_sources),
        ),
        _check(
            "preprocessor_contract",
            bool(names)
            and all(len(values) == len(names) for values in arrays)
            and preprocessor.get("fit_scope")
            == "training_days_deterministic_stride_sample_only",
            {
                "features": len(names),
                "array_lengths": [len(values) for values in arrays],
                "fit_scope": preprocessor.get("fit_scope"),
            },
            "ordered features and equal-length train-only preprocessing arrays",
        ),
        _check(
            "temperature",
            _finite(manifest.get("temperature")) and manifest["temperature"] > 0,
            manifest.get("temperature"),
            "> 0",
        ),
        _check(
            "threshold",
            _finite(manifest.get("threshold")) and 0 <= manifest["threshold"] <= 1,
            manifest.get("threshold"),
            "0..1",
        ),
    ]
    for name, limit, direction in metric_requirements:
        observed = metrics.get(name)
        passed = _finite(observed) and (
            observed >= limit if direction == "min" else observed <= limit
        )
        checks.append(_check(f"test_{name}", passed, observed, f"{direction} {limit}"))
    return {
        "id": "flow-detector-calibrated",
        "scope": "live_packet_security",
        "admitted": all(check["passed"] for check in checks),
        "artifact_sha256": _sha256(checkpoint_path)
        if checkpoint_path.is_file()
        else None,
        "checks": checks,
    }


def evaluate_forecast_router(
    manifest: dict[str, Any],
    embedded: dict[str, Any] | None,
    checkpoint_path: Path,
    model_directory: Path,
    policy: dict[str, Any] = POLICY,
) -> dict[str, Any]:
    router_policy = policy["forecast_router"]
    split = manifest.get("split", {})
    partitions = [split.get(name, []) for name in ("train", "calibration", "test")]
    expected_hashes = manifest.get("expert_sha256", {})
    actual_hashes = {
        name: _sha256(model_directory / file_name)
        for name, file_name in EXPERT_FILES.items()
        if (model_directory / file_name).is_file()
    }
    metrics = manifest.get("metrics", {}).get("test", {})
    samples = metrics.get("samples")
    gain = metrics.get("router_gain_over_best_fixed")
    regret = metrics.get("mean_route_regret")
    best_fixed = metrics.get("best_fixed_expert_loss")
    regret_limit = (
        best_fixed * router_policy["max_regret_fraction_of_best_fixed"]
        if _finite(best_fixed)
        else None
    )
    checks = [
        _check(
            "checkpoint_exists",
            checkpoint_path.is_file(),
            checkpoint_path.name,
            "existing file",
        ),
        _check(
            "manifest_bound_to_checkpoint",
            embedded == manifest,
            embedded is not None,
            True,
        ),
        _check(
            "real_data",
            manifest.get("real_data") is True,
            manifest.get("real_data"),
            True,
        ),
        _check(
            "not_security_detector",
            manifest.get("security_detector") is False,
            manifest.get("security_detector"),
            False,
        ),
        _check(
            "feature_schema",
            manifest.get("feature_schema") == "packet-causal-v1",
            manifest.get("feature_schema"),
            "packet-causal-v1",
        ),
        _check(
            "causal_inputs",
            manifest.get("expert_outputs_are_router_inputs") is False,
            manifest.get("expert_outputs_are_router_inputs"),
            False,
        ),
        _check(
            "chronological_split",
            _disjoint_nonempty(partitions),
            split,
            "non-empty pairwise-disjoint chronological files",
        ),
        _check(
            "expert_hashes",
            expected_hashes == actual_hashes,
            actual_hashes,
            expected_hashes,
        ),
        _check(
            "test_samples",
            _finite(samples) and samples >= router_policy["min_test_samples"],
            samples,
            f">= {router_policy['min_test_samples']}",
        ),
        _check(
            "test_gain",
            _finite(gain) and gain > router_policy["min_gain_over_best_fixed"],
            gain,
            f"> {router_policy['min_gain_over_best_fixed']}",
        ),
        _check(
            "test_route_regret",
            _finite(regret) and _finite(regret_limit) and regret <= regret_limit,
            regret,
            f"<= {regret_limit}",
        ),
    ]
    return {
        "id": "temporal-router",
        "scope": "forecast_compute_routing_only",
        "admitted": all(check["passed"] for check in checks),
        "security_detector": False,
        "artifact_sha256": _sha256(checkpoint_path)
        if checkpoint_path.is_file()
        else None,
        "checks": checks,
    }


def audit(model_directory: Path, output: Path | None = None) -> dict[str, Any]:
    detector_json = model_directory / "flow_detector_calibrated_cse_cic_ids2018.json"
    detector_pt = detector_json.with_suffix(".pt")
    router_json = model_directory / "temporal_router_real_live.json"
    router_pt = router_json.with_suffix(".pt")
    if not detector_json.is_file() or not router_json.is_file():
        raise FileNotFoundError("required detector and router manifests are missing")
    detector, detector_embedded, detector_error = _load_manifest_pair(
        detector_json, detector_pt
    )
    router, router_embedded, router_error = _load_manifest_pair(router_json, router_pt)
    detector_decision = evaluate_security_detector(
        detector, detector_embedded, detector_pt
    )
    router_decision = evaluate_forecast_router(
        router, router_embedded, router_pt, model_directory
    )
    report = {
        "report_type": "argus_model_admission",
        "report_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "policy": POLICY,
        "model_directory": str(model_directory.resolve()),
        "load_errors": {
            "security_detector": detector_error,
            "forecast_router": router_error,
        },
        "decisions": {
            "security_detector": detector_decision,
            "forecast_router": router_decision,
        },
        "runtime": {
            "security_verdict_enabled": detector_decision["admitted"],
            "forecast_router_enabled": router_decision["admitted"],
        },
    }
    destination = output or model_directory / "argus_admission_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.models, args.output), indent=2))


if __name__ == "__main__":
    main()

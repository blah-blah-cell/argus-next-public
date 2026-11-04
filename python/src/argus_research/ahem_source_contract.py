"""Fail-closed declarations for future source-disjoint experiments.

Validation checks declared identities, not truth of a dataset or statistical
independence. Byte hashes and duplicate groups require a separate preparation
audit. No download, file access or final-data opening happens in this module.
"""
import hashlib
import json


def validate_sources(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema") != "argus.source-contract/v1":
        raise ValueError("unsupported source contract")
    entries = manifest.get("sources")
    if not isinstance(entries, list) or not entries:
        raise ValueError("nonempty source declarations required")
    roles = {"train", "development", "final"}
    seen_ids, ownership = set(), {}
    counts = {role: 0 for role in roles}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("role") not in roles:
            raise ValueError("invalid source role")
        role = entry["role"]
        for key in ("source_id", "parent_capture_id", "sha256", "url", "license"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError(f"missing source provenance: {key}")
        if entry["source_id"] in seen_ids:
            raise ValueError("duplicate source identity")
        seen_ids.add(entry["source_id"])
        digest = entry["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid source SHA-256")
        if type(entry.get("previously_accessed")) is not bool:
            raise ValueError("prior-access status cannot be unknown")
        if role == "final" and entry["previously_accessed"]:
            raise ValueError("previously accessed data cannot be declared untouched final")
        groups = entry.get("duplicate_groups")
        if not isinstance(groups, list) or any(not isinstance(g, str) or not g.strip() for g in groups):
            raise ValueError("explicit duplicate-group declaration required")
        # Converted shards from the same parent or byte-identical data cannot
        # masquerade as independent sources merely by receiving new filenames.
        identities = [("parent", entry["parent_capture_id"]), ("bytes", digest)]
        identities += [("duplicate", g) for g in groups]
        for identity in identities:
            if identity in ownership and ownership[identity] != role:
                raise ValueError(f"cross-role source leakage: {identity[0]}")
            ownership[identity] = role
        counts[role] += 1
    if any(count == 0 for count in counts.values()):
        raise ValueError("train/development/final roles must all be present")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"counts": counts, "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
            "declared_intersections_empty": True, "physical_hashes_verified": False}


def authorize_role(manifest, role, *, frozen_checkpoint_sha256=None):
    """Return declarations only; final access requires a frozen checkpoint hash.

    This is not a one-access ledger. The future evaluator must independently
    record access once and enforce its ledger before opening final files.
    """
    validate_sources(manifest)
    if role not in {"train", "development", "final"}:
        raise ValueError("unknown source role")
    if role == "final":
        h = frozen_checkpoint_sha256
        if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            raise ValueError("freeze checkpoint before authorizing final data")
    return [dict(entry) for entry in manifest["sources"] if entry["role"] == role]

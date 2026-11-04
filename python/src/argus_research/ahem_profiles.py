"""Named AHEM capacity tiers and auditable smaller-to-larger warm starts."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from .ahem_v3 import AhemV32NextPacket


@dataclass(frozen=True)
class AhemProfile:
    name: str
    width: int
    key_dim: int
    chunk_size: int
    local_packets: int
    group_size: int
    coarse_top_k: int
    fine_top_k: int
    flat_retrieval_until_chunks: int = 512

    def build(self, feature_dim: int = 12) -> AhemV32NextPacket:
        return AhemV32NextPacket(
            feature_dim,
            width=self.width,
            key_dim=self.key_dim,
            chunk_size=self.chunk_size,
            local_packets=self.local_packets,
            group_size=self.group_size,
            coarse_top_k=self.coarse_top_k,
            fine_top_k=self.fine_top_k,
            flat_retrieval_until_chunks=self.flat_retrieval_until_chunks,
        )

    def manifest(self) -> dict[str, int | str]:
        return asdict(self)


PROFILES = {
    "flash": AhemProfile(
        name="flash",
        width=60,
        key_dim=32,
        chunk_size=64,
        local_packets=128,
        group_size=16,
        coarse_top_k=4,
        fine_top_k=8,
    ),
    "pro": AhemProfile(
        name="pro",
        width=256,
        key_dim=96,
        chunk_size=64,
        local_packets=128,
        group_size=16,
        coarse_top_k=8,
        fine_top_k=16,
    ),
    "max": AhemProfile(
        name="max",
        width=512,
        key_dim=192,
        chunk_size=64,
        local_packets=128,
        group_size=16,
        coarse_top_k=12,
        fine_top_k=24,
    ),
}


def get_profile(name: str) -> AhemProfile:
    try:
        return PROFILES[name.lower()]
    except KeyError as error:
        choices = ", ".join(PROFILES)
        raise ValueError(f"unknown AHEM profile {name!r}; choose {choices}") from error


@dataclass(frozen=True)
class TransferReport:
    source_sha256: str
    copied_tensors: int
    copied_elements: int
    target_elements: int

    @property
    def target_fraction_initialized(self) -> float:
        return self.copied_elements / max(self.target_elements, 1)

    def manifest(self) -> dict[str, int | float | str]:
        return {
            **asdict(self),
            "target_fraction_initialized": self.target_fraction_initialized,
        }


def warm_start_from_smaller(
    model: AhemV32NextPacket, checkpoint: Path
) -> TransferReport:
    """Copy a smaller model into the leading subspace of a larger profile.

    Exact-shape tensors are copied directly. A tensor may be prefix-copied only
    when it has the same rank and every source dimension fits the destination.
    The rest of the larger model retains its normal initialization. The source
    artifact is loaded with ``weights_only=True`` and is never unpickled.
    """

    raw = checkpoint.read_bytes()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("warm-start checkpoint has no state_dict")
    source = payload["state_dict"]
    target = model.state_dict()
    copied_tensors = 0
    copied_elements = 0
    for name, source_value in source.items():
        target_value = target.get(name)
        if not isinstance(source_value, Tensor) or target_value is None:
            continue
        if source_value.ndim != target_value.ndim or any(
            source_size > target_size
            for source_size, target_size in zip(
                source_value.shape, target_value.shape, strict=True
            )
        ):
            continue
        converted = source_value.to(dtype=target_value.dtype)
        if source_value.ndim == 0:
            target_value.copy_(converted)
        else:
            slices = tuple(slice(0, size) for size in source_value.shape)
            target_value[slices].copy_(converted)
        copied_tensors += 1
        copied_elements += source_value.numel()
    if copied_tensors == 0:
        raise ValueError("checkpoint has no shape-compatible AHEM tensors")
    model.load_state_dict(target, strict=True)
    return TransferReport(
        source_sha256=hashlib.sha256(raw).hexdigest(),
        copied_tensors=copied_tensors,
        copied_elements=copied_elements,
        target_elements=sum(value.numel() for value in target.values()),
    )

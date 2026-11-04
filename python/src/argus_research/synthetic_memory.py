"""Deterministic, labelled long-range evidence streams for Gate 1 only.

This is deliberately *not* an intrusion dataset.  Each final query packet is
ambiguous until paired with a preceding marker at a controlled distance.  The
benchmark lets us falsify the bounded-memory claim before data-label alignment
can confound it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SyntheticMemoryConfig:
    sequence_length: int = 1024
    feature_dim: int = 12
    key_vocab: int = 8
    evidence_distance: int = 768
    marker_strength: float = 12.0
    noise_std: float = 0.15

    def validate(self) -> None:
        if self.feature_dim < self.key_vocab + 4:
            raise ValueError("feature_dim needs key slots plus marker/query fields")
        if not 1 < self.evidence_distance < self.sequence_length - 1:
            raise ValueError("evidence_distance must be inside the sequence")
        if self.key_vocab < 2 or self.sequence_length < 8:
            raise ValueError("invalid synthetic stream dimensions")


def make_streams(
    count: int, config: SyntheticMemoryConfig, seed: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Create causal streams and binary labels, with no label-derived features."""
    config.validate()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    streams = (
        torch.randn(
            count, config.sequence_length, config.feature_dim, generator=generator
        )
        * config.noise_std
    )
    keys = torch.randint(config.key_vocab, (count,), generator=generator)
    labels = torch.randint(2, (count,), generator=generator)
    marker_index = config.sequence_length - 1 - config.evidence_distance
    row = torch.arange(count)

    # Key is observable at the marker and query; polarity exists only at marker.
    streams[row, marker_index, keys] = config.marker_strength
    streams[row, -1, keys] = config.marker_strength
    streams[:, marker_index, config.key_vocab] = labels.float() * 2.0 - 1.0
    streams[:, marker_index, config.key_vocab + 1] = 1.0  # marker indicator
    streams[:, -1, config.key_vocab + 2] = 1.0  # query indicator
    streams[:, -1, config.key_vocab + 3] = (
        float(config.evidence_distance) / config.sequence_length
    )
    return streams.to(device), labels.to(device)


def make_mixed_distance_streams(
    count: int,
    config: SyntheticMemoryConfig,
    distances: tuple[int, ...],
    seed: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Balanced distance regimes, useful only for router falsification tests."""
    if not distances or count % len(distances):
        raise ValueError("count must divide evenly across nonempty distance regimes")
    per_regime = count // len(distances)
    groups = []
    labels = []
    for index, distance in enumerate(distances):
        group_config = SyntheticMemoryConfig(
            sequence_length=config.sequence_length,
            feature_dim=config.feature_dim,
            key_vocab=config.key_vocab,
            evidence_distance=distance,
            marker_strength=config.marker_strength,
            noise_std=config.noise_std,
        )
        stream, label = make_streams(
            per_regime, group_config, seed + index * 10_000, device
        )
        groups.append(stream)
        labels.append(label)
    order = torch.randperm(count, device=device)
    return torch.cat(groups, dim=0)[order], torch.cat(labels, dim=0)[order]

"""Research-only bounded online retention and frozen causal embedding helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Entry:
    source: int
    successor: int
    predecessor: int
    footprint: int
    key: np.ndarray


class Retention:
    """At most sixteen retained entries; no archive or historical refill API.

    Random is bounded random replacement, not an unbiased history reservoir.
    The key-only diversity reserve never inspects outcomes or query targets.
    """

    def __init__(self, arm: str, *, seed: int = 20261002, horizon: int = 10_000):
        if arm not in ("fifo", "diversity", "random") or horizon < 1:
            raise ValueError("explicit retention arm and positive horizon required")
        self.arm = arm
        self.horizon = horizon
        self.recent: list[Entry] = []
        self.reserve: list[Entry] = []
        self.rng = random.Random(seed)
        self.clock = -1
        self.last_source = -1
        self.max_retained = 0
        self.max_transient = 0
        self.evictions = 0

    def update(self, ordinal: int, incoming: Entry | None) -> list[Entry]:
        if type(ordinal) is not int or ordinal <= self.clock:
            raise ValueError("strictly increasing flow-update ordinals required")
        self.clock = ordinal
        cutoff = max(0, ordinal - self.horizon + 1)

        # The FIFO control retains phase-expired entries until base expiry,
        # then masks them at read time, matching the original implementation.
        def legal(entry: Entry) -> bool:
            return entry.predecessor >= cutoff and (
                self.arm == "fifo" or entry.footprint >= cutoff
            )

        self.recent = [e for e in self.recent if legal(e)]
        self.reserve = [e for e in self.reserve if legal(e)]
        if incoming is not None:
            if (
                not 0 <= incoming.footprint <= incoming.source < incoming.successor
                or incoming.successor != ordinal
                or not 0 <= incoming.predecessor < incoming.source
                or incoming.source <= self.last_source
                or incoming.key.ndim != 1
                or not len(incoming.key)
                or not np.isfinite(incoming.key).all()
            ):
                raise ValueError("new causal completed transition required")
            self.last_source = incoming.source
            if legal(incoming):
                self.recent.insert(0, incoming)
        self.max_transient = max(
            self.max_transient, len(self.recent) + len(self.reserve)
        )
        if self.arm == "fifo":
            if len(self.recent) > 16:
                self.evictions += 1
            self.recent = self.recent[:16]
        else:
            overflow, self.recent = self.recent[8:], self.recent[:8]
            self.reserve.extend(overflow)
            if len(self.reserve) > 8:
                if len(self.reserve) != 9:
                    raise ValueError("more than one incoming entry at a time")
                self.reserve.sort(key=lambda e: e.source)
                if self.arm == "random":
                    drop = self.rng.randrange(9)
                else:
                    keys = np.stack([e.key for e in self.reserve]).astype(np.float64)
                    norms = np.linalg.norm(keys, axis=1, keepdims=True)
                    keys = keys / np.maximum(norms, 1e-12)
                    distance = np.clip(1 - keys @ keys.T, 0, 2)
                    np.fill_diagonal(distance, np.inf)
                    objective = []
                    for i in range(9):
                        keep = [j for j in range(9) if j != i]
                        objective.append(float(distance[np.ix_(keep, keep)].min()))
                    # Sorted oldest-first; np.argmax picks oldest on exact ties.
                    drop = int(np.argmax(objective))
                self.reserve.pop(drop)
                self.evictions += 1
        entries = sorted(self.recent + self.reserve, key=lambda e: -e.source)
        if len(entries) > 16 or len({e.source for e in entries}) != len(entries):
            raise ValueError("retention capacity/identity invariant broken")
        self.max_retained = max(self.max_retained, len(entries))
        return [e for e in entries if e.footprint >= cutoff]

    def telemetry(self) -> dict[str, int]:
        return {
            "max_retained": self.max_retained,
            "max_transient": self.max_transient,
            "evictions": self.evictions,
        }


@torch.no_grad()
def encode_record(
    reader: Any, record: Any, *, batch: int = 256
) -> dict[str, torch.Tensor]:
    """Offline batching of source-only representations; not online memory proof.

    Actual retention operates on Entry objects only. This archive is diagnostic
    acceleration and is never available as a history-refill interface.
    """
    x, previous = record.features, record.previous_same_flow
    n = len(x)
    if n == 0 or batch < 1 or x.shape[1] != reader.packet_dim:
        raise ValueError("nonempty conditioned record and positive batch required")
    if next(reader.parameters()).device.type != "cpu":
        raise ValueError("this fixed diagnostic encodes on CPU only")
    local = x.new_zeros(n, reader.width)
    statistics = x.new_zeros(n, 2 * reader.packet_dim)
    # Evaluate true prefixes, never padded through a biased convolution.
    for end in range(min(n, reader.local_packets - 1)):
        window = x[: end + 1][None]
        local[end] = reader.local(window)[0]
        statistics[end] = torch.cat((window.mean(1), window.std(1, unbiased=False)), 1)[
            0
        ]
    if n >= reader.local_packets:
        windows = x.unfold(0, reader.local_packets, 1).transpose(1, 2)
        for start in range(0, len(windows), batch):
            group = windows[start : start + batch].contiguous()
            section = slice(
                start + reader.local_packets - 1,
                start + reader.local_packets - 1 + len(group),
            )
            local[section] = reader.local(group)
            statistics[section] = torch.cat(
                (group.mean(1), group.std(1, unbiased=False)), 1
            )
    ordinals = torch.arange(n)
    cutoff = (ordinals - 9999).clamp_min(0)
    prefix = x.new_zeros(n, 4, reader.packet_dim + 2)
    ids = ordinals.clone()
    active = torch.ones(n, dtype=torch.bool)
    footprint = ordinals.clone()
    for slot in range(4):
        prefix[:, slot, : reader.packet_dim] = x[ids.clamp_min(0)] * active[:, None]
        prefix[:, slot, reader.packet_dim] = active.to(x.dtype)
        footprint = torch.where(active, ids, footprint)
        older = previous[ids.clamp_min(0)]
        next_active = active & (older >= cutoff)
        if slot < 3:
            gap = torch.log1p((ids - older).clamp_min(0).to(x.dtype)) / math.log1p(
                10000
            )
            prefix[:, slot, reader.packet_dim + 1] = gap.clamp(max=1) * next_active
        ids, active = older, next_active
    phase = torch.cat((prefix.flatten(1), local, statistics), 1)
    has_previous = previous >= cutoff
    prior = x[previous.clamp_min(0)] * has_previous[:, None]
    gap = torch.log1p((ordinals - previous).clamp_min(0).to(x.dtype)) / math.log1p(
        10000
    )
    gap = gap.clamp(max=1) * has_previous
    keys = x.new_zeros(n, reader.key_dim)
    for start in range(0, n, batch):
        section = slice(start, start + batch)
        base = reader.project_state(prior[section], x[section], gap[section])
        keys[section] = F.normalize(base + reader.phase_encoder(phase[section]), dim=-1)
    return {
        "keys": keys,
        "phase": phase,
        "footprint": torch.minimum(
            footprint, (ordinals - reader.local_packets + 1).clamp_min(0)
        ),
    }


def source_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float | int]:
    prediction, target = (
        np.asarray(prediction, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
    )
    if (
        prediction.shape != target.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 2
        or not len(target)
    ):
        raise ValueError("nonempty aligned two-field predictions required")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("nonfinite metric inputs")
    error = np.abs(prediction - target)
    huber = np.where(error < 1, error * error / 2, error - 0.5)
    physical = np.abs(np.expm1(prediction) - np.expm1(target)).mean(0)
    bounds = np.log1p([65535, 60])
    return {
        "queries": len(target),
        "weighted_smooth_l1": float(((huber[:, 0] + 4 * huber[:, 1]) / 5).mean()),
        "byte_mae": float(physical[0]),
        "delay_mae_ms": float(physical[1] * 1000),
        "invalid_outputs": int(
            ((prediction < -1e-6) | (prediction > bounds + 1e-6)).any(1).sum()
        ),
    }


def candidate_oracle(
    outcomes: np.ndarray, current: np.ndarray, target: np.ndarray
) -> float:
    choices = np.concatenate(
        (np.asarray(outcomes).reshape(-1, 2), np.asarray(current)[None])
    )
    error = np.abs(choices.astype(np.float64) - np.asarray(target, dtype=np.float64))
    fields = np.where(error < 1, error * error / 2, error - 0.5)
    return float(((fields[:, 0] + 4 * fields[:, 1]) / 5).min())

"""Research-only causal policies. No app mutation, raw capture, or verdict learning.

The replay uses an offline archive of prefix-only frozen representations to avoid
repeated CNN work. Policies cannot refill from it: only physical arrival admits an
Entry. This is NOT an end-to-end streaming memory/latency implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any

import numpy as np
import torch

from .ahem_nlm_loop_20261019 import Entry, Retention, encode_record


BOUNDS = np.log1p([65535.0, 60.0])


class Surprise:
    """Score an earlier forecast against a now-observed successor, then update."""

    def __init__(self, multiple: float = 2.0) -> None:
        self.scale = np.array([0.1, 0.01], dtype=np.float64)
        self.multiple = multiple

    def observe(self, forecast: np.ndarray, observed: np.ndarray) -> bool:
        residual = np.abs(np.asarray(forecast) - np.asarray(observed))
        if residual.shape != (2,) or not np.isfinite(residual).all():
            raise ValueError("finite two-field observed residual required")
        surprising = bool((residual > self.multiple * self.scale).any())
        self.scale = 0.95 * self.scale + 0.05 * residual
        self.scale = np.maximum(self.scale, [1e-6, 1e-8])
        return surprising


class SelectiveBank:
    """Sixteen entries, no archive/refill. Pure gate or recent8/reserve8.

    Hybrid arms always write recent history. Surprise therefore does not mean
    skipping every write in these arms. Entries marked useful can move to the
    reserve only when displaced from recent history, avoiding duplicated slots.
    """

    def __init__(self, arm: str, *, seed: int, horizon: int = 10000) -> None:
        if arm not in {"surprise_only", "surprise_recent", "random_recent"}:
            raise ValueError("unknown selective retention arm")
        if horizon < 1:
            raise ValueError("positive horizon required")
        self.arm, self.horizon = arm, horizon
        self.recent: list[tuple[Entry, bool]] = []
        self.reserve: list[Entry] = []
        self.rng = random.Random(seed)
        self.clock = self.last_source = -1
        self.recent_writes = self.reserve_writes = self.rejected = 0
        self.max_retained = 0

    def update(
        self, ordinal: int, incoming: Entry | None, surprising: bool
    ) -> list[Entry]:
        if type(ordinal) is not int or ordinal <= self.clock:
            raise ValueError("strictly increasing update ordinals required")
        if incoming is not None:
            if (
                not 0 <= incoming.footprint <= incoming.source < incoming.successor
                or incoming.successor != ordinal
                or not 0 <= incoming.predecessor < incoming.source
                or incoming.source <= self.last_source
                or incoming.key.ndim != 1
                or not np.isfinite(incoming.key).all()
            ):
                raise ValueError("causally completed finite entry required")
            self.last_source = incoming.source
        self.clock = ordinal
        oldest = max(0, ordinal - self.horizon + 1)
        self.recent = [(e, flag) for e, flag in self.recent if e.footprint >= oldest]
        self.reserve = [e for e in self.reserve if e.footprint >= oldest]
        if incoming is not None and incoming.footprint >= oldest:
            mark = (
                self.rng.random() < 0.25 if self.arm == "random_recent" else surprising
            )
            if self.arm == "surprise_only":
                if mark:
                    self.recent.insert(0, (incoming, mark))
                    self.recent = self.recent[:16]
                    self.recent_writes += 1
                else:
                    self.rejected += 1
            else:
                self.recent.insert(0, (incoming, mark))
                self.recent_writes += 1
                if len(self.recent) > 8:
                    displaced, keep = self.recent.pop()
                    if keep:
                        # Compare observed state keys only, not future outcomes.
                        duplicate = next(
                            (
                                i
                                for i, e in enumerate(self.reserve)
                                if cosine(e.key, displaced.key) >= 0.995
                            ),
                            None,
                        )
                        if duplicate is not None:
                            self.reserve.pop(duplicate)
                        self.reserve.insert(0, displaced)
                        self.reserve = self.reserve[:8]
                        self.reserve_writes += 1
                    else:
                        self.rejected += 1
        entries = sorted(
            [e for e, _ in self.recent] + self.reserve, key=lambda e: -e.source
        )
        self.max_retained = max(self.max_retained, len(entries))
        if len(entries) > 16 or len({e.source for e in entries}) != len(entries):
            raise AssertionError("memory capacity/uniqueness violated")
        return entries


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(first @ second) / max(denominator, 1e-12)


class ReadSchedule:
    """Per-flow scheduling: observes every packet, including on skipped work."""

    def __init__(self, adaptive: bool) -> None:
        self.adaptive = adaptive
        self.clock = self.last_read = -1
        self.since = 0
        self.previous: np.ndarray | None = None

    def step(self, ordinal: int, packet: np.ndarray) -> bool:
        packet = np.asarray(packet)
        if (
            ordinal <= self.clock
            or packet.shape != (12,)
            or not np.isfinite(packet).all()
        ):
            raise ValueError("increasing ordinals and finite packet required")
        self.clock = ordinal
        critical = bool((packet[[6, 8, 9]] > 0.5).any())  # SYN / FIN / RST
        changed = False
        if self.previous is not None and self.adaptive:
            difference = np.abs(packet[:2] - self.previous[:2])
            changed = bool(
                difference[0] > 0.1
                or difference[1] > 0.01
                or not np.array_equal(packet[2:], self.previous[2:])
            )
        do_work = (
            self.last_read < 0
            or self.since >= 3
            or critical
            or changed
            or ordinal - self.last_read >= 128
        )
        self.previous = packet.copy()
        self.since += 1
        if do_work:
            self.last_read, self.since = ordinal, 0
        return do_work


@torch.no_grad()
def prepare_record(reader: Any, record: Any) -> dict[str, torch.Tensor]:
    encoded = encode_record(reader, record, batch=32)
    start = reader.phase_packets * (reader.packet_dim + 2)
    local = encoded["phase"][:, start : start + reader.width]
    encoded["local_prediction"] = reader._replace_time(
        reader.local_predict(local), reader.local_time(local)
    )
    return encoded


def policy_slots(record: Any, encoded: dict, arm: str, seed: int) -> dict:
    """Only current packet and previous pointers drive policy; no next pointers."""
    n = len(record.features)
    src = torch.full((n, 16), -1, dtype=torch.long)
    nxt = torch.full_like(src, -1)
    valid = torch.zeros(n, 16, dtype=torch.bool)
    work = torch.ones(n, dtype=torch.bool)
    x = record.features.numpy()
    keys = encoded["keys"].numpy()
    local = encoded["local_prediction"].numpy()
    footprints = encoded["footprint"].tolist()
    previous = record.previous_same_flow.tolist()
    lanes: dict[int, Any] = {}
    surprises: dict[int, Surprise] = {}
    schedules: dict[int, ReadSchedule] = {}
    observed = candidates = admitted = max_retained = 0
    for ordinal, flow in enumerate(record.flow_ids.tolist()):
        observed += 1
        if arm in {"stride", "selective"}:
            schedule = schedules.setdefault(flow, ReadSchedule(arm == "selective"))
            work[ordinal] = schedule.step(ordinal, x[ordinal])
        lane = lanes.get(flow)
        if lane is None:
            lane = (
                SelectiveBank(arm, seed=seed + flow % 1000003)
                if arm in {"surprise_only", "surprise_recent", "random_recent"}
                else Retention("fifo")
            )
            lanes[flow] = lane
            surprises[flow] = Surprise()
        source = previous[ordinal]
        before = previous[source] if source >= 0 else -1
        surprising = False
        if source >= 0:
            surprising = surprises[flow].observe(local[source], x[ordinal, :2])
        incoming = (
            Entry(source, ordinal, before, footprints[source], keys[source])
            if before >= max(0, ordinal - 9999)
            else None
        )
        candidates += incoming is not None
        if not bool(work[ordinal]):
            incoming = None  # Skip memory writes, not packet observation.
        if isinstance(lane, SelectiveBank):
            entries = lane.update(ordinal, incoming, surprising)
        else:
            entries = lane.update(ordinal, incoming)
        admitted += bool(
            incoming is not None and any(e.source == source for e in entries)
        )
        max_retained = max(max_retained, len(entries))
        if entries:
            count = len(entries)
            src[ordinal, :count] = torch.tensor([e.source for e in entries])
            nxt[ordinal, :count] = torch.tensor([e.successor for e in entries])
            valid[ordinal, :count] = True
    return {
        "source": src,
        "successor": nxt,
        "valid": valid,
        "work": work,
        "telemetry": {
            "observed_packets": observed,
            "candidate_transitions": candidates,
            "admitted_transitions": admitted,
            "memory_work_packets": int(work.sum()),
            "max_entries_per_flow": max_retained,
            "flow_states": len(lanes),
            "offline_archive_not_streaming_ram_proof": True,
        },
    }


@torch.no_grad()
def read_packets(
    reader: Any,
    readout: Any,
    scale: torch.Tensor,
    record: Any,
    encoded: dict,
    slots: dict,
    indices: torch.Tensor,
) -> dict:
    """Only rows requesting work execute reader scoring and head; others persist."""
    current = record.features[indices, :2]
    prediction = current.clone()
    weights = current.new_zeros(len(indices), 16)
    active = slots["work"][indices] & slots["valid"][indices].any(1)
    chosen = indices[active]
    if len(chosen):
        src = slots["source"][chosen]
        nxt = slots["successor"][chosen]
        mask = slots["valid"][chosen]
        values = record.features[nxt.clamp_min(0), :2] * mask[..., None]
        age = reader._normalized_gap(
            src, chosen[:, None], torch.full((len(chosen),), 10000.0)
        )
        scores = reader._scores(
            encoded["keys"][chosen], encoded["keys"][src.clamp_min(0)], age
        )
        w, _, _, p = readout(
            scores.masked_fill(~mask, -1e4),
            mask,
            values,
            record.features[chosen, :2],
            encoded["phase"][chosen],
        )
        corrected = p + (1 - scale) * (record.features[chosen, :2] - p)
        prediction[active] = torch.where((chosen < 127)[:, None], corrected, p)
        weights[active] = w
    return {"prediction": prediction, "weights": weights, "executed": active}


def retrieval_diagnostic(record: Any, slots: dict, predictions: dict) -> dict:
    """Offline outcome oracle, never provided to a policy or forecaster."""
    anchors = torch.nonzero(record.next_same_flow >= 0).flatten()
    valid = slots["valid"][anchors]
    executed = predictions["executed"][anchors]
    target = record.features[record.next_same_flow[anchors], :2]
    candidates = record.features[slots["successor"][anchors].clamp_min(0), :2]
    error = (candidates - target[:, None]).abs().double()
    h = torch.where(error < 1, error.square() / 2, error - 0.5)
    loss = (h[..., 0] + 4 * h[..., 1]) / 5
    loss = loss.masked_fill(~valid, torch.inf)
    available = valid.any(1)
    oracle = loss.min(1).values
    chosen = predictions["weights"][anchors].argmax(1)
    hit = (loss.gather(1, chosen[:, None])[:, 0] <= oracle + 1e-8) & executed
    return {
        "queries": len(anchors),
        "available": int(available.sum()),
        "executed": int(executed.sum()),
        "tie_aware_hits": int(hit.sum()),
        "oracle_loss_sum_available": float(oracle[available].sum()),
        "definition": "oracle and hits on own retained bank; skipped reads count as no hit, not removed",
        "not_comparable_as_fixed_candidate_accuracy": True,
    }


def residual_features(
    current: np.ndarray, parent: np.ndarray, available: bool
) -> np.ndarray:
    return np.array(
        [
            1.0,
            current[0] / BOUNDS[0],
            current[1] / BOUNDS[1],
            (parent[0] - current[0]) / BOUNDS[0],
            (parent[1] - current[1]) / BOUNDS[1],
            float(available),
        ],
        dtype=np.float64,
    )


def corrected(
    weights: np.ndarray, features: np.ndarray, parent: np.ndarray, eligible: bool
) -> np.ndarray:
    if not eligible:
        return parent.copy()
    margin = np.minimum(np.minimum(parent, BOUNDS - parent), 1.0)
    return parent + 0.1 * np.tanh(weights @ features) * margin


def residual_gradient(
    weights: np.ndarray, features: np.ndarray, parent: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Exact raw SGD derivative; verified against autograd, no AdamW guarantee."""
    margin = np.minimum(np.minimum(parent, BOUNDS - parent), 1.0)
    tangent = np.tanh(weights @ features)
    p = parent + 0.1 * tangent * margin
    derivative = np.clip(p - target, -1, 1) * np.array([0.2, 0.8])
    physical = (math.expm1(float(p[0])) - math.expm1(float(target[0]))) / 1500
    derivative[0] += 0.1 * np.clip(physical, -1, 1) * math.exp(float(p[0])) / 1500
    return (derivative * 0.1 * (1 - tangent * tangent) * margin)[:, None] * features[
        None
    ]


def error_totals(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    error = np.abs(prediction - target)
    h = np.where(error < 1, error * error / 2, error - 0.5)
    physical = np.abs(np.expm1(prediction) - np.expm1(target))
    return np.array([(h[0] + 4 * h[1]) / 5, physical[0], physical[1]])


@dataclass
class Pending:
    ordinal: int
    features: np.ndarray
    parent: np.ndarray
    eligible: bool
    fast: np.ndarray
    active: np.ndarray
    challenger: np.ndarray | None
    version: int


class OnlineResidual:
    """Fast shadow SGD plus frozen-candidate, subsequent-forecast validation.

    Each flow has at most one unresolved forecast. Validation cannot use a
    prediction made before candidate freeze. The retained backbone never changes.
    """

    def __init__(self, block: int = 64, *, consolidate: bool = True) -> None:
        if block < 1:
            raise ValueError("positive consolidation block required")
        self.block = block
        self.consolidate = consolidate
        self.fast = np.zeros((2, 6))
        self.active = self.fast.copy()
        self.challenger: np.ndarray | None = None
        self.version = 0
        self.clock = self.last_observed = -1
        self.pending: dict[int, Pending] = {}
        self.completed = self.since_freeze = self.validation_count = 0
        self.steps = self.promotions = self.rollbacks = 0
        self.frozen_at = -1
        self.totals = np.zeros((3, 3))  # challenger, active, original anchor
        self.decisions: list[dict] = []

    def arrive(self, ordinal: int, flow: int, observed: np.ndarray) -> None:
        if (
            ordinal <= self.clock
            or np.asarray(observed).shape != (2,)
            or not np.isfinite(observed).all()
        ):
            raise ValueError("increasing arrivals and finite observed fields required")
        self.clock = ordinal
        # Horizon expiry is enforced on unresolved predictions, not just weights.
        self.pending = {
            f: p for f, p in self.pending.items() if p.ordinal >= ordinal - 9999
        }
        pending = self.pending.pop(flow, None)
        if pending is None:
            return
        if pending.ordinal >= ordinal:
            raise AssertionError("future-target update")
        self.completed += 1
        self.last_observed = ordinal
        if (
            self.challenger is not None
            and pending.version == self.version
            and pending.challenger is not None
            and pending.eligible
        ):
            if pending.ordinal < self.frozen_at:
                raise AssertionError("validation forecast predates freeze")
            self.totals += np.stack(
                [
                    error_totals(p, observed)
                    for p in (pending.challenger, pending.active, pending.parent)
                ]
            )
            self.validation_count += 1
        if pending.eligible:
            gradient = residual_gradient(
                self.fast, pending.features, pending.parent, observed
            )
            if not np.isfinite(gradient).all():
                raise ValueError("nonfinite online gradient")
            norm = float(np.linalg.norm(gradient))
            self.fast -= 0.05 * gradient / max(1.0, norm)
            self.fast /= max(1.0, float(np.linalg.norm(self.fast)))
            self.steps += 1
            self.since_freeze += 1
        if self.challenger is not None and self.validation_count >= self.block:
            wins = bool(
                self.totals[0, 0] < self.totals[1, 0]
                and (self.totals[0] <= self.totals[1] + 1e-12).all()
                and (self.totals[0] <= 1.05 * self.totals[2] + 1e-12).all()
            )
            self.decisions.append(
                {
                    "ordinal": ordinal,
                    "frozen_at": self.frozen_at,
                    "count": self.validation_count,
                    "promoted": wins,
                    "totals": self.totals.tolist(),
                }
            )
            if wins:
                self.active = self.challenger.copy()
                self.promotions += 1
            else:
                self.rollbacks += 1
            self.fast = self.active.copy()
            self.challenger = None
            self.since_freeze = 0
        if (
            self.consolidate
            and self.challenger is None
            and self.since_freeze >= self.block
        ):
            self.challenger = self.fast.copy()
            self.version += 1
            self.frozen_at = ordinal
            self.validation_count = 0
            self.totals[:] = 0
            self.since_freeze = 0

    def issue(
        self,
        ordinal: int,
        flow: int,
        current: np.ndarray,
        parent: np.ndarray,
        available: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if ordinal != self.clock or flow in self.pending:
            raise ValueError("arrive then issue exactly once per flow arrival")
        current, parent = np.asarray(current), np.asarray(parent)
        if (
            current.shape != (2,)
            or parent.shape != (2,)
            or not np.isfinite(parent).all()
            or not np.isfinite(current).all()
            or (parent < 0).any()
            or (parent > BOUNDS + 1e-6).any()
        ):
            raise ValueError("finite bounded parent required")
        features = residual_features(current, parent, available)
        eligible = available and ordinal >= 127
        fast = corrected(self.fast, features, parent, eligible)
        active = corrected(self.active, features, parent, eligible)
        candidate = (
            corrected(self.challenger, features, parent, eligible)
            if self.challenger is not None
            else None
        )
        self.pending[flow] = Pending(
            ordinal,
            features,
            parent.copy(),
            eligible,
            fast.copy(),
            active.copy(),
            candidate,
            self.version,
        )
        return fast, active

"""Bounded, single-owner Pro stream reference for CPU or CUDA tensors.

The caller supplies stable directional flow generations and already encoded
packet-causal-flow-v2 features. Capture boundaries/loss require explicit reset.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Hashable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .ahem_pro import AhemProNextPacket
from .ahem_v6 import AhemV6Output


@dataclass
class _Transition:
    predecessor_ordinal: int
    source_ordinal: int
    successor_ordinal: int
    key: Tensor
    value: Tensor
    successor: Tensor


@dataclass
class _Lane:
    events: deque[tuple[int, Tensor]] = field(default_factory=lambda: deque(maxlen=2))
    transitions: deque[_Transition] = field(default_factory=deque)


class ProStream:
    """One ordered capture stream; independent instances can run concurrently.

    Keep weights immutable for the lifetime of this object. To replace weights
    construct a new stream. This reference has no automatic GPU routing or
    within-stream concurrency; ``read_projected`` supports batched snapshots.
    """

    def __init__(
        self, model: AhemProNextPacket, capture_id: str, *, flow_capacity: int = 1024
    ):
        model.resource_budget(flow_capacity)  # validate before allocating state
        if model.training:
            raise ValueError("Pro stream requires eval mode and immutable weights")
        self.model = model
        self.flow_capacity = flow_capacity
        self.reset(capture_id)

    def reset(self, capture_id: str) -> None:
        if not isinstance(capture_id, str) or not capture_id:
            raise ValueError("a nonempty capture identity is required")
        self.capture_id = capture_id
        self.next_ordinal = 0
        self.evictions = 0
        self.expired_transitions = 0
        self.local_state: tuple[Tensor, ...] | None = None
        self.lanes: OrderedDict[Hashable, _Lane] = OrderedDict()

    @torch.inference_mode()
    def step(
        self,
        packet: Tensor,
        flow_generation: Hashable,
        ordinal: int,
        *,
        capture_id: str,
    ) -> AhemV6Output | None:
        if self.model.training:
            raise ValueError("training cannot share a projected inference cache")
        if capture_id != self.capture_id:
            raise ValueError("capture changed; explicitly reset Pro state")
        if type(ordinal) is not int or ordinal != self.next_ordinal:
            raise ValueError(
                "Pro expects contiguous global ordinals; reset on loss or rollback"
            )
        hash(flow_generation)
        parameter = next(self.model.parameters())
        if parameter.dtype not in (torch.float32, torch.float64):
            raise ValueError("Pro streaming reference requires FP32/FP64 weights")
        if (
            packet.shape != (12,)
            or packet.dtype != parameter.dtype
            or packet.device != parameter.device
        ):
            raise ValueError("packet must have 12 fields and share model dtype/device")
        if not torch.isfinite(packet).all():
            raise ValueError("packet features must be finite")
        # Clone prevents a view retaining the entire caller's capture tensor.
        packet = packet.detach().clone()
        local_encoded, self.local_state = self.model.local.forward_step(
            packet[None], self.local_state
        )
        lane = self.lanes.pop(flow_generation, _Lane())
        self.lanes[flow_generation] = lane
        if len(self.lanes) > self.flow_capacity:
            self.lanes.popitem(last=False)
            self.evictions += 1
        oldest = ordinal - self.model.config.horizon_packets + 1
        while lane.events and lane.events[0][0] < oldest:
            lane.events.popleft()
        while lane.transitions and lane.transitions[0].predecessor_ordinal < oldest:
            lane.transitions.popleft()
            self.expired_transitions += 1

        def position(value: int) -> Tensor:
            return torch.tensor([value], device=packet.device, dtype=torch.long)

        predecessor = lane.events[-1][1] if lane.events else torch.zeros_like(packet)
        query_gap = (
            self.model.gap(position(lane.events[-1][0]), position(ordinal))
            if lane.events
            else packet.new_zeros(1)
        )
        if len(lane.events) == 2:
            p_ordinal, p = lane.events[0]
            s_ordinal, source = lane.events[1]
            lane.transitions.append(
                _Transition(
                    p_ordinal,
                    s_ordinal,
                    ordinal,
                    self.model.project_state(
                        p[None],
                        source[None],
                        self.model.gap(position(p_ordinal), position(s_ordinal)),
                    )[0].clone(),
                    self.model.project_value(source[None], packet[None], query_gap)[
                        0
                    ].clone(),
                    packet[:2].clone(),
                )
            )
            if len(lane.transitions) > self.model.max_transitions:
                lane.transitions.popleft()
        lane.events.append((ordinal, packet))
        self.next_ordinal += 1
        if self.next_ordinal < self.model.local_packets:
            return None

        entries = list(reversed(lane.transitions))
        if entries:
            keys = torch.stack([entry.key for entry in entries])[None]
            values = torch.stack([entry.value for entry in entries])[None]
            successors = torch.stack([entry.successor for entry in entries])[None]
            sources = torch.tensor(
                [[entry.source_ordinal for entry in entries]],
                device=packet.device,
                dtype=torch.long,
            )
            valid = torch.ones(
                (1, len(entries)), device=packet.device, dtype=torch.bool
            )
        else:
            keys = packet.new_zeros(1, 1, self.model.key_dim)
            values = packet.new_zeros(1, 1, self.model.width)
            successors = packet.new_zeros(1, 1, 2)
            sources = torch.full((1, 1), -1, device=packet.device, dtype=torch.long)
            valid = torch.zeros((1, 1), device=packet.device, dtype=torch.bool)
        return self.model.read_projected(
            local_encoded,
            predecessor[None],
            packet[None],
            query_gap,
            keys,
            values,
            successors,
            sources,
            valid,
            position(ordinal),
        )

    def telemetry(self) -> dict[str, object]:
        return {
            "capture_id": self.capture_id,
            "contract_sha256": self.model.config.contract_sha256(),
            "processed_packets": self.next_ordinal,
            "retained_flows": len(self.lanes),
            "flow_capacity": self.flow_capacity,
            "flow_evictions": self.evictions,
            "expired_transitions": self.expired_transitions,
            "stored_transitions": sum(
                len(lane.transitions) for lane in self.lanes.values()
            ),
            "stored_raw_events": sum(len(lane.events) for lane in self.lanes.values()),
            "resource_budget": self.model.resource_budget(self.flow_capacity),
            "security_detector": False,
        }

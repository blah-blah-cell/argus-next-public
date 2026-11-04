"""Exact bounded streaming reference for frozen AHEM v6.1 inference."""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Hashable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .ahem_v6 import (
    AhemV6Output,
    AhemV6StateTransitionNextPacket,
    FlowLaneLstmNextPacket,
)


@dataclass
class _Transition:
    predecessor_ordinal: int
    source_ordinal: int
    key: Tensor
    value: Tensor
    successor: Tensor


@dataclass
class _Lane:
    events: deque[tuple[int, Tensor]] = field(default_factory=lambda: deque(maxlen=2))
    transitions: deque[_Transition] = field(default_factory=deque)


class AhemV6Stream:
    """Process each packet once while matching a fixed-horizon v6.1 window.

    Capture loss, clock rollback, or capture identity changes require an
    explicit reset. The caller supplies a stable flow-generation identifier;
    it routes state but never enters a learned feature.
    """

    def __init__(
        self,
        model: AhemV6StateTransitionNextPacket,
        capture_id: str,
        *,
        horizon_packets: int = 10_000,
        emit_warmup: bool = False,
    ) -> None:
        if model.training:
            raise ValueError("AHEM v6 stream requires immutable eval-mode weights")
        if horizon_packets < model.local_packets + 2:
            raise ValueError("stream horizon is too short for the frozen model")
        self.model = model
        self.horizon_packets = horizon_packets
        self.emit_warmup = emit_warmup
        self.reset(capture_id)

    def reset(self, capture_id: str) -> None:
        if not isinstance(capture_id, str) or not capture_id:
            raise ValueError("a non-empty capture identity is required")
        self.capture_id = capture_id
        self.next_ordinal = 0
        self.local_state: tuple[Tensor, ...] | None = None
        self.lanes: OrderedDict[Hashable, _Lane] = OrderedDict()
        self.expired_flows = 0
        self.expired_transitions = 0

    def _gap(self, earlier: int, later: int, packet: Tensor) -> Tensor:
        before = torch.tensor([earlier], device=packet.device, dtype=torch.long)
        after = torch.tensor([later], device=packet.device, dtype=torch.long)
        span = packet.new_full((1,), float(self.horizon_packets))
        return self.model._normalized_gap(before, after, span)

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
            raise ValueError("training cannot share an AHEM inference cache")
        if capture_id != self.capture_id:
            raise ValueError("capture changed; explicitly reset AHEM state")
        if type(ordinal) is not int or ordinal != self.next_ordinal:
            raise ValueError("AHEM stream requires contiguous global ordinals")
        hash(flow_generation)
        parameter = next(self.model.parameters())
        if parameter.dtype not in (torch.float32, torch.float64):
            raise ValueError("AHEM v6 streaming reference supports FP32/FP64")
        if (
            packet.shape != (self.model.packet_dim,)
            or packet.dtype != parameter.dtype
            or packet.device != parameter.device
            or not torch.isfinite(packet).all()
        ):
            raise ValueError(
                "packet must be finite and share the frozen model shape, dtype, and device"
            )
        packet = packet.detach().clone()
        local_encoded, self.local_state = self.model.local.forward_step(
            packet[None], self.local_state
        )
        oldest = ordinal - self.horizon_packets + 1
        while self.lanes:
            _, oldest_lane = next(iter(self.lanes.items()))
            if oldest_lane.events[-1][0] >= oldest:
                break
            self.lanes.popitem(last=False)
            self.expired_flows += 1

        lane = self.lanes.pop(flow_generation, _Lane())
        self.lanes[flow_generation] = lane
        while lane.events and lane.events[0][0] < oldest:
            lane.events.popleft()
        while lane.transitions and lane.transitions[0].predecessor_ordinal < oldest:
            lane.transitions.popleft()
            self.expired_transitions += 1

        predecessor = lane.events[-1][1] if lane.events else torch.zeros_like(packet)
        query_gap = (
            self._gap(lane.events[-1][0], ordinal, packet)
            if lane.events
            else packet.new_zeros(1)
        )
        if len(lane.events) == 2:
            predecessor_ordinal, transition_predecessor = lane.events[0]
            source_ordinal, source = lane.events[1]
            state_gap = self._gap(predecessor_ordinal, source_ordinal, packet)
            lane.transitions.append(
                _Transition(
                    predecessor_ordinal=predecessor_ordinal,
                    source_ordinal=source_ordinal,
                    key=self.model.project_state(
                        transition_predecessor[None], source[None], state_gap
                    )[0].clone(),
                    value=self.model.project_value(
                        source[None], packet[None], query_gap
                    )[0].clone(),
                    successor=packet[: self.model.output_dim].clone(),
                )
            )
            if len(lane.transitions) > self.model.max_transitions:
                lane.transitions.popleft()
        lane.events.append((ordinal, packet))
        self.next_ordinal += 1
        if self.next_ordinal < self.horizon_packets and not self.emit_warmup:
            return None

        entries = list(reversed(lane.transitions))
        if entries:
            keys = torch.stack([entry.key for entry in entries])[None]
            values = torch.stack([entry.value for entry in entries])[None]
            successors = torch.stack([entry.successor for entry in entries])[None]
            source_ordinals = torch.tensor(
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
            successors = packet.new_zeros(1, 1, self.model.output_dim)
            source_ordinals = torch.full(
                (1, 1), -1, device=packet.device, dtype=torch.long
            )
            valid = torch.zeros((1, 1), device=packet.device, dtype=torch.bool)
        span = packet.new_full((1,), float(self.horizon_packets))
        query = self.model.project_state(predecessor[None], packet[None], query_gap)
        return self._read_packet(
            packet,
            local_encoded,
            query,
            keys,
            values,
            successors,
            source_ordinals,
            valid,
            torch.tensor([ordinal], device=packet.device, dtype=torch.long),
            span,
        )

    def _read_packet(self, packet: Tensor, *inputs) -> AhemV6Output:
        """Extension point for research readers needing current observed fields."""
        return self.model._read(*inputs)

    def telemetry(self) -> dict[str, int | str]:
        return {
            "capture_id": self.capture_id,
            "processed_packets": self.next_ordinal,
            "horizon_packets": self.horizon_packets,
            "retained_flows": len(self.lanes),
            "stored_transitions": sum(
                len(lane.transitions) for lane in self.lanes.values()
            ),
            "stored_raw_events": sum(len(lane.events) for lane in self.lanes.values()),
            "expired_flows": self.expired_flows,
            "expired_transitions": self.expired_transitions,
        }


class FlowLaneLstmStream:
    """Ordinary recurrent comparator state, isolated by flow generation."""

    def __init__(self, model: FlowLaneLstmNextPacket, capture_id: str) -> None:
        if model.training:
            raise ValueError("flow LSTM stream requires immutable eval-mode weights")
        self.model = model
        self.reset(capture_id)

    def reset(self, capture_id: str) -> None:
        if not isinstance(capture_id, str) or not capture_id:
            raise ValueError("a non-empty capture identity is required")
        self.capture_id = capture_id
        self.next_ordinal = 0
        self.states: dict[Hashable, tuple[Tensor, Tensor]] = {}

    @torch.inference_mode()
    def step(
        self,
        packet: Tensor,
        flow_generation: Hashable,
        ordinal: int,
        *,
        capture_id: str,
    ) -> Tensor:
        if self.model.training:
            raise ValueError("training cannot share a flow LSTM inference state")
        if capture_id != self.capture_id:
            raise ValueError("capture changed; explicitly reset LSTM state")
        if type(ordinal) is not int or ordinal != self.next_ordinal:
            raise ValueError("flow LSTM stream requires contiguous global ordinals")
        hash(flow_generation)
        parameter = next(self.model.parameters())
        if (
            packet.shape != (self.model.packet_dim,)
            or packet.dtype != parameter.dtype
            or packet.device != parameter.device
            or not torch.isfinite(packet).all()
        ):
            raise ValueError(
                "packet must be finite and share the LSTM shape, dtype, and device"
            )
        prediction, state = self.model.forward_step(
            packet.detach().clone()[None], self.states.get(flow_generation)
        )
        self.states[flow_generation] = tuple(value.clone() for value in state)
        self.next_ordinal += 1
        return prediction[0]

    def telemetry(self) -> dict[str, int | str]:
        return {
            "capture_id": self.capture_id,
            "processed_packets": self.next_ordinal,
            "retained_flows": len(self.states),
        }

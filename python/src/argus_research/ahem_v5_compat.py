"""Versioned inference repairs for frozen v5 weights, not a new trained model."""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .ahem_v4 import AhemV5SourceAlignedStreamNextPacket


class CompatibleV5(AhemV5SourceAlignedStreamNextPacket):
    """Preserve the trained 16-slot gate distribution for compact cache reads."""

    inference_contract = "argus.v5-fixed-layout/v1"

    def __init__(self) -> None:
        # This compatibility contract is for the frozen default checkpoint only.
        super().__init__()

    def _read_projected(
        self,
        local_state,
        anchor,
        keys,
        values,
        successor_dynamic,
        source_ordinals,
        valid,
        anchor_ordinals,
        context_spans,
    ):
        count = keys.size(1)
        if not 1 <= count <= self.max_transitions:
            raise ValueError("cache slot count is outside the frozen model capacity")
        entries = [keys, values, successor_dynamic, source_ordinals, valid]
        if count < self.max_transitions:
            entries = [
                torch.cat(
                    (
                        value,
                        value.new_zeros(
                            value.size(0),
                            self.max_transitions - count,
                            *value.shape[2:],
                        ),
                    ),
                    dim=1,
                )
                for value in entries
            ]
        output = super()._read_projected(
            local_state, anchor, *entries, anchor_ordinals, context_spans
        )
        # Preserve compact diagnostic shape; only the internal gate sees padding.
        return replace(
            output,
            selected_chunks=output.selected_chunks[:, :count],
            selected_weights=output.selected_weights[:, :count],
        )


@dataclass(frozen=True)
class ProjectionContract:
    capture_id: str
    feature_version: str
    context_span: int
    weights_sha256: str


class V5ProjectionCache:
    """CPU inference cache with private fixed weights and fixed normalization.

    Predictions use a private inference copy; mutation/hot-training of that
    copy is unsupported. Recreate the cache on any model,
    feature, capture, or normalization change. No raw projected writes accepted.
    """

    def __init__(
        self,
        model: AhemV5SourceAlignedStreamNextPacket,
        *,
        capture_id: str,
        feature_version: str,
        context_span: int,
        flow_capacity: int = 256,
    ) -> None:
        if not capture_id or not feature_version:
            raise ValueError("capture and feature identities are required")
        if type(context_span) is not int or context_span < 128:
            raise ValueError("fixed context span must be at least 128")
        if type(flow_capacity) is not int or flow_capacity < 1:
            raise ValueError("flow capacity must be positive")
        self._model = CompatibleV5().to(dtype=next(model.parameters()).dtype).eval()
        self._model.load_state_dict(model.state_dict(), strict=True)
        self._model.requires_grad_(False)
        digest = hashlib.sha256()
        for name, value in self._model.state_dict().items():
            if not torch.isfinite(value).all():
                raise ValueError("model contains non-finite weights")
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(value.contiguous().numpy().tobytes())
        self._contract = ProjectionContract(
            capture_id, feature_version, context_span, digest.hexdigest()
        )
        self.flow_capacity = flow_capacity
        self._lanes: OrderedDict[str, list[tuple[int, int, Tensor, Tensor, Tensor]]] = (
            OrderedDict()
        )
        self._observed_ordinal = -1

    @property
    def contract(self) -> ProjectionContract:
        return self._contract

    def _check(self, contract: ProjectionContract, ordinal: int) -> None:
        if contract != self.contract:
            raise ValueError("capture/feature/normalization/weights contract mismatch")
        if type(ordinal) is not int or ordinal < self._observed_ordinal or ordinal < 0:
            raise ValueError("packet ordinal rollback; start a new capture cache")

    @torch.inference_mode()
    def append(
        self,
        flow: str,
        source: Tensor,
        successor: Tensor,
        *,
        source_ordinal: int,
        successor_ordinal: int,
        observed_ordinal: int,
        contract: ProjectionContract,
    ) -> None:
        self._check(contract, observed_ordinal)
        if not isinstance(flow, str) or not flow:
            raise ValueError("an opaque flow-generation identity is required")
        if (
            type(source_ordinal) is not int
            or type(successor_ordinal) is not int
            or not 0 <= source_ordinal < successor_ordinal <= observed_ordinal
            or source_ordinal < observed_ordinal - self.contract.context_span + 1
        ):
            raise ValueError("transition is future, expired, or has invalid ordinals")
        for packet in (source, successor):
            if packet.shape != (12,) or not torch.isfinite(packet).all():
                raise ValueError("transition packet must contain 12 finite features")
        lane = self._lanes.get(flow, [])
        if lane and source_ordinal != lane[-1][1]:
            raise ValueError("transition must continue the same observed flow lane")
        dtype = next(self._model.parameters()).dtype
        source = source.detach().to(device="cpu", dtype=dtype)
        successor = successor.detach().to(device="cpu", dtype=dtype)
        gap = source.new_tensor(
            [
                math.log1p(successor_ordinal - source_ordinal)
                / math.log1p(self.contract.context_span)
            ]
        )
        key, value = self._model.project_transition(source[None], successor[None], gap)
        if not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise ValueError("projection produced non-finite values")
        # All validation/projection precedes mutation: a rejected write is atomic.
        self._lanes[flow] = (
            lane
            + [
                (
                    source_ordinal,
                    successor_ordinal,
                    key[0].clone(),
                    value[0].clone(),
                    successor[:2].clone(),
                )
            ]
        )[-self._model.max_transitions :]
        self._lanes.move_to_end(flow)
        while len(self._lanes) > self.flow_capacity:
            self._lanes.popitem(last=False)
        self._observed_ordinal = observed_ordinal

    def read(
        self,
        flow: str,
        *,
        anchor_ordinal: int,
        contract: ProjectionContract,
    ) -> dict[str, Tensor]:
        self._check(contract, anchor_ordinal)
        lane = [
            entry
            for entry in reversed(self._lanes.get(flow, []))
            if anchor_ordinal - self.contract.context_span + 1 <= entry[0]
            and entry[1] <= anchor_ordinal
        ]
        dtype = next(self._model.parameters()).dtype
        count = max(len(lane), 1)
        result = {
            "keys": torch.zeros(1, count, 32, dtype=dtype),
            "values": torch.zeros(1, count, 60, dtype=dtype),
            "successor_dynamic": torch.zeros(1, count, 2, dtype=dtype),
            "source_ordinals": torch.zeros(1, count, dtype=torch.int64),
            "valid": torch.zeros(1, count, dtype=torch.bool),
            "anchor_ordinals": torch.tensor([anchor_ordinal]),
            "context_spans": torch.tensor([self.contract.context_span], dtype=dtype),
        }
        for index, entry in enumerate(lane):
            result["source_ordinals"][0, index] = entry[0]
            result["keys"][0, index] = entry[2]
            result["values"][0, index] = entry[3]
            result["successor_dynamic"][0, index] = entry[4]
            result["valid"][0, index] = True
        if flow in self._lanes:
            self._lanes.move_to_end(flow)
        self._observed_ordinal = anchor_ordinal
        return result

    @torch.inference_mode()
    def predict(
        self,
        local_packets: Tensor,
        *,
        flow: str,
        anchor_ordinal: int,
        contract: ProjectionContract,
    ):
        """Read using the same fixed weights/transform as the cached projections.

        The supplied 128 events must be the causal global prefix ending at the
        declared anchor. This bounded reference call is not an end-to-end timer.
        """
        if (
            local_packets.shape != (1, 128, 12)
            or not torch.isfinite(local_packets).all()
        ):
            raise ValueError("prediction requires 128 finite causal packet features")
        if type(anchor_ordinal) is not int or anchor_ordinal < 127:
            raise ValueError(
                "prediction requires 128 observed events; cache is warming up"
            )
        packets = local_packets.to(
            device="cpu", dtype=next(self._model.parameters()).dtype
        )
        projected = self.read(flow, anchor_ordinal=anchor_ordinal, contract=contract)
        return self._model.forward_from_projected_cache(
            packets, packets[:, -1], **projected
        )

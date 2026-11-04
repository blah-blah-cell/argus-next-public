"""Experimental repairs to the frozen hybrid; a distinct checkpoint contract.

Flow-local convolution and a memory correction centered on the local forecast.
The null action is the local forecast. Utility labels score the final proposals.
This is a bounded-window reference, not a streaming deployment implementation.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .ahem_hybrid import AhemHybrid, HybridConfig
from .ahem_v4 import StreamableLocalCausalBranch
from .ahem_v6 import transition_slots
from .flow_v6_objective import dynamic_loss


def flow_local_state(local: StreamableLocalCausalBranch, packets: Tensor) -> Tensor:
    """Last 31 query-flow events; mask padding at every biased layer.

    The dilated stack's receptive field is 31 events. Masking avoids treating
    missing history as zero-valued events passed through learned biases.
    """
    positions = torch.arange(packets.shape[1], device=packets.device)[None]
    selected = (
        positions.expand(packets.shape[0], -1)
        .masked_fill(packets[..., -1] == 0, -1)
        .topk(min(31, packets.shape[1]), dim=1)
        .values.flip(1)
    )
    valid = (selected >= 0)[..., None]
    lane = packets[..., :12].gather(
        1, selected.clamp_min(0)[..., None].expand(-1, -1, 12)
    )
    lane = lane.masked_fill(~valid, 0)
    state = local.input(lane).masked_fill(~valid, 0)
    for dilation, filter_, mix in zip(
        local.dilations, local.filters, local.mix, strict=True
    ):
        causal = filter_(F.pad(state.transpose(1, 2), (2 * dilation, 0))).transpose(
            1, 2
        )
        state = (state + F.gelu(mix(causal))).masked_fill(~valid, 0)
    return state[:, -1]


@dataclass
class RepairOutput:
    prediction: Tensor
    local_prediction: Tensor
    candidates: Tensor
    weights: Tensor
    valid: Tensor


class AhemHybridRepair(AhemHybrid):
    """Same key encoder; replace the persistence mixture with local corrections.

    No old checkpoint is silently accepted as this architecture. Readers learned,
    recency and none provide fixed ablations. `flow_local=False` isolates the local
    lane change. Existing trained weights require explicit experimental transfer.
    """

    schema = "argus.ahem-hybrid-repair/v1"

    def __init__(self, config: HybridConfig | None = None, *, flow_local: bool = True):
        super().__init__(config)
        self.flow_local = flow_local
        # There is no secondary mixing gate: the null attention action is local.
        del self.gate_head

    def forward_diagnostics(self, packets: Tensor) -> RepairOutput:
        c = self.config
        if (
            packets.ndim != 3
            or packets.shape[0] < 1
            or packets.shape[-1] != 13
            or packets.shape[1] < c.local_packets
        ):
            raise ValueError("repair expects [batch,context,13] with local warmup")
        if not torch.isfinite(packets).all() or (packets[..., :2] < 0).any():
            raise ValueError("repair requires finite features and nonnegative dynamics")
        if not ((packets[..., -1] == 0) | (packets[..., -1] == 1)).all():
            raise ValueError("flow relation must be binary")
        packets = packets[:, -c.horizon :]
        previous, current, slots = transition_slots(packets, max_transitions=c.capacity)
        persistence = current[:, :2]
        h = (
            flow_local_state(self.local, packets)
            if self.flow_local
            else self.local(packets[:, -c.local_packets :, :12])
        )
        local = (persistence + c.radius * self.local_delta(h).tanh()).clamp_min(0)
        if c.reader == "none":
            # Exact bypass also avoids evaluating a reader with zero influence.
            valid = torch.ones(
                (*local.shape[:1], 1), dtype=torch.bool, device=local.device
            )
            return RepairOutput(
                local, local, local[:, None], valid.to(local.dtype), valid
            )

        def clean(value):
            return value.masked_fill(~slots.valid[..., None], 0)

        def gap(a, b):
            return (
                torch.log1p((b - a).clamp_min(0).to(packets.dtype))
                / math.log1p(c.horizon)
            ).clamp_max(1)

        positions = torch.arange(packets.shape[1], device=packets.device)[None]
        prior = (
            positions[:, :-1]
            .expand(packets.shape[0], -1)
            .masked_fill(packets[:, :-1, -1] == 0, -1)
            .amax(-1)
        )
        anchor = torch.full_like(prior, packets.shape[1] - 1)
        query = self.state_key(
            previous, current, torch.where(prior >= 0, gap(prior, anchor), 0)
        )
        keys = self.state_key(
            clean(slots.predecessor),
            clean(slots.source),
            gap(slots.predecessor_ordinals, slots.source_ordinals),
        )
        logits = self.scores(query, keys, gap(slots.source_ordinals, anchor[:, None]))
        valid = torch.cat((torch.ones_like(slots.valid[:, :1]), slots.valid), 1)
        weights = (
            torch.cat((self.null_head(query), logits), 1)
            .masked_fill(~valid, -torch.inf)
            .softmax(-1)
        )
        raw = (
            (1 - c.transport) * clean(slots.successor)[..., :2]
            + c.transport
            * (
                persistence[:, None]
                + clean(slots.successor)[..., :2]
                - clean(slots.source)[..., :2]
            )
        ).clamp_min(0)
        proposed = (
            local[:, None]
            + c.radius
            * self.memory_gain.tanh()
            * ((raw - local[:, None]) / c.radius).tanh()
        )
        lower = (persistence - c.radius).clamp_min(0)[:, None]
        upper = (persistence + c.radius)[:, None]
        proposed = torch.maximum(lower, torch.minimum(proposed, upper))
        candidates = torch.cat((local[:, None], proposed), 1)
        # Residual form makes gain=0 exactly local, including floating arithmetic.
        prediction = local + (weights[..., None] * (candidates - local[:, None])).sum(1)
        return RepairOutput(prediction, local, candidates, weights, valid)


def repair_loss(output: RepairOutput, target: Tensor):
    """Expected regret of the *actual* forecast proposals, null included.

    Stop-gradient utility labels cannot be manipulated through the auxiliary loss.
    The main forecast loss still trains proposal values and the reader jointly.
    """
    if (
        target.shape != output.prediction.shape
        or not torch.isfinite(target).all()
        or (target < 0).any()
    ):
        raise ValueError("target must be finite nonnegative [batch,2]")
    costs = dynamic_loss(
        output.candidates.detach(),
        target[:, None].expand_as(output.candidates),
        reduction="none",
    )
    best = costs.masked_fill(~output.valid, torch.inf).amin(-1, keepdim=True)
    regret = (costs - best).masked_fill(~output.valid, 0).detach()
    retrieval = (output.weights * regret).sum(-1).mean()
    forecast = dynamic_loss(output.prediction, target)
    local = dynamic_loss(output.local_prediction, target)
    return forecast + 0.1 * local + 0.1 * retrieval, {
        "forecast": forecast.detach(),
        "local": local.detach(),
        "retrieval": retrieval.detach(),
    }

"""Experimental persistence-anchored transition memory; no trained claim.

The bounded window implementation is a research reference, not a stateful
deployment kernel. Targets enter only hybrid_loss, never the forward path.
"""

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ahem_v4 import StreamableLocalCausalBranch
from .ahem_v6 import transition_slots
from .flow_v6_objective import dynamic_loss
from .flow_v2_features import FEATURE_CONTRACT_SHA256


@dataclass(frozen=True)
class HybridConfig:
    width: int = 60
    key_dim: int = 32
    capacity: int = 16
    horizon: int = 10_000
    local_packets: int = 128
    transport: float = 0.5
    radius: float = 1.0
    reader: str = "learned"

    def __post_init__(self):
        if any(
            type(x) is not int or x <= 0
            for x in (
                self.width,
                self.key_dim,
                self.capacity,
                self.horizon,
                self.local_packets,
            )
        ):
            raise ValueError("hybrid dimensions must be positive integers")
        if self.capacity < 2 or not 31 <= self.local_packets <= self.horizon:
            raise ValueError("hybrid requires capacity >=2 and 31 <= local <= horizon")
        if not math.isfinite(self.transport) or not 0 <= self.transport <= 1:
            raise ValueError("transport must be in [0,1]")
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be finite and positive")
        if self.reader not in {"learned", "recency", "none"}:
            raise ValueError("unknown hybrid reader")

    def contract(self):
        return {
            "schema": "argus.ahem-hybrid/v1",
            "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
            **asdict(self),
            "initialization": "exact_packet_persistence",
            "task": "next_same_flow_dynamic_forecast",
            "security_detector": False,
        }


@dataclass
class HybridOutput:
    prediction: Tensor
    local_prediction: Tensor
    memory_prediction: Tensor
    weights: Tensor
    candidates: Tensor
    valid: Tensor
    gate: Tensor
    source_ordinals: Tensor


class AhemHybrid(nn.Module):
    """v6 two-event keys, bounded Pro age prior, persistence residual heads.

    Slot zero is an explicit persistence option. The other slots contain
    completed same-flow transitions, newest first. Learned key projections
    never receive the successor of the transition being keyed.
    """

    def __init__(self, config: HybridConfig | None = None):
        super().__init__()
        self.config = config or HybridConfig()
        c = self.config
        self.local = StreamableLocalCausalBranch(12, c.width)
        self.signature = nn.Sequential(
            nn.Linear(25, c.width),
            nn.LayerNorm(c.width),
            nn.GELU(),
            nn.Linear(c.width, c.key_dim),
        )
        self.local_delta = nn.Linear(c.width, 2)
        self.memory_gain = nn.Parameter(torch.zeros(2))
        self.gate_head = nn.Linear(c.width + 4, 2)
        self.null_head = nn.Linear(c.key_dim, 1)
        self.semantic_scale = nn.Parameter(torch.tensor(math.atanh(0.25)))
        self.age_strength = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.local_delta.weight)
        nn.init.zeros_(self.local_delta.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.0)
        nn.init.zeros_(self.null_head.weight)
        nn.init.zeros_(self.null_head.bias)

    def state_key(self, previous: Tensor, current: Tensor, gap: Tensor) -> Tensor:
        return F.normalize(
            self.signature(
                torch.cat((current, current - previous, gap[..., None]), -1)
            ),
            dim=-1,
        )

    def scores(self, query: Tensor, keys: Tensor, gap: Tensor) -> Tensor:
        c = self.config
        rank = torch.arange(keys.shape[1], device=keys.device, dtype=keys.dtype)
        prior = (
            -8
            * self.age_strength.sigmoid()
            * torch.log1p(rank + gap)
            / math.log1p(c.capacity)
        )
        if c.reader == "recency":
            return prior
        return prior + 8 * self.semantic_scale.tanh() * torch.einsum(
            "bd,bkd->bk", query, keys
        )

    def forward_diagnostics(self, packets: Tensor) -> HybridOutput:
        c = self.config
        if packets.ndim != 3 or packets.shape[0] < 1 or packets.shape[-1] != 13:
            raise ValueError("hybrid expects [batch,context,13]")
        if packets.shape[1] < c.local_packets or not torch.isfinite(packets).all():
            raise ValueError("hybrid requires finite features and local warmup")
        if not ((packets[..., -1] == 0) | (packets[..., -1] == 1)).all():
            raise ValueError("flow relation must be binary")
        if (packets[..., :2] < 0).any():
            raise ValueError("log dynamic features must be nonnegative")
        packets = packets[:, -c.horizon :]
        previous, current, slots = transition_slots(packets, max_transitions=c.capacity)

        # Missing slots contain gathered placeholders: remove them before any
        # neural operation so they cannot affect attention or its gradients.
        def clean(value):
            return value.masked_fill(~slots.valid[..., None], 0)

        def gap(a, b):
            return (
                torch.log1p((b - a).clamp_min(0).to(packets.dtype))
                / math.log1p(c.horizon)
            ).clamp_max(1)

        keys = self.state_key(
            clean(slots.predecessor),
            clean(slots.source),
            gap(slots.predecessor_ordinals, slots.source_ordinals),
        )
        # Locate the previous event independently: query gap remains valid for
        # a two-event flow with no complete three-event cache transition.
        positions = torch.arange(packets.shape[1], device=packets.device)[None]
        prior_position = (
            positions[:, :-1]
            .expand(packets.shape[0], -1)
            .masked_fill(packets[:, :-1, -1] == 0, -1)
            .amax(-1)
        )
        anchor = torch.full_like(prior_position, packets.shape[1] - 1)
        query_gap = torch.where(prior_position >= 0, gap(prior_position, anchor), 0)
        query = self.state_key(previous, current, query_gap)
        logits = self.scores(query, keys, gap(slots.source_ordinals, anchor[:, None]))
        valid = torch.cat((torch.ones_like(slots.valid[:, :1]), slots.valid), 1)
        if c.reader == "none":
            valid[:, 1:] = False
        null_logit = self.null_head(query)
        weights = (
            torch.cat((null_logit, logits), 1)
            .masked_fill(~valid, -torch.inf)
            .softmax(-1)
        )
        persistence = current[:, :2]
        # transport=0 copies observed successors; transport=1 transports their
        # changes onto today's state. Both are hypotheses in log-feature space.
        source = clean(slots.source)[..., :2]
        successor = clean(slots.successor)[..., :2]
        proposals = (
            (1 - c.transport) * successor
            + c.transport * (persistence[:, None] + successor - source)
        ).clamp_min(0)
        candidates = torch.cat((persistence[:, None], proposals), 1)
        read = torch.einsum("bk,bkd->bd", weights, candidates)
        h = self.local(packets[:, -c.local_packets :, :12])
        local = (persistence + c.radius * self.local_delta(h).tanh()).clamp_min(0)
        memory = (
            persistence
            + c.radius
            * self.memory_gain.tanh()
            * ((read - persistence) / c.radius).tanh()
        ).clamp_min(0)
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(
            -1, keepdim=True
        ) / math.log(c.capacity + 1)
        occupancy = slots.valid.float().mean(-1, keepdim=True).to(h.dtype)
        gate = self.gate_head(
            torch.cat((h, read - persistence, entropy, occupancy), -1)
        ).sigmoid()
        gate = gate * valid[:, 1:].any(-1, keepdim=True)
        prediction = local + gate * (memory - local)
        return HybridOutput(
            prediction,
            local,
            memory,
            weights,
            candidates,
            valid,
            gate,
            slots.source_ordinals,
        )

    def forward(self, packets: Tensor) -> Tensor:
        return self.forward_diagnostics(packets).prediction


def hybrid_loss(
    output: HybridOutput, target: Tensor
) -> tuple[Tensor, dict[str, Tensor]]:
    """Expected candidate regret; no forced oracle label on tied candidates.

    The persistence slot competes in the same units as all memory candidates.
    Detaching candidate utility prevents manipulating the supervised labels.
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
    oracle = costs.masked_fill(~output.valid, torch.inf).amin(-1, keepdim=True)
    regret = (costs - oracle).masked_fill(~output.valid, 0).detach()
    retrieval = (output.weights * regret).sum(-1).mean()
    forecast = dynamic_loss(output.prediction, target)
    local = dynamic_loss(output.local_prediction, target)
    available = output.valid[:, 1:].any(-1)
    memory_errors = dynamic_loss(output.memory_prediction, target, reduction="none")
    memory = (
        memory_errors[available].mean() if available.any() else memory_errors.sum() * 0
    )
    # Same forecast objective teaches the gate; remove Pro's separate gate
    # classification target, which can conflict with the primary forecast.
    total = forecast + 0.1 * local + 0.1 * memory + 0.1 * retrieval
    return total, {
        "forecast": forecast.detach(),
        "local": local.detach(),
        "memory": memory.detach(),
        "retrieval_regret": retrieval.detach(),
    }

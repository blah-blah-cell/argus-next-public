"""Versioned full-gradient causal AHEM candidate; no target-dependent forward path.

Unlike frozen projection adapters, recomputes all learned representations from
observed packets. Larger research model, not a replacement for the admitted app.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .ahem_v6 import AhemV6StateTransitionNextPacket
from .flow_v6_objective import (
    dynamic_loss,
    frozen_utility_distribution,
    retrieval_cross_entropy,
)


class EndToEndAhem(nn.Module):
    def __init__(
        self,
        width=192,
        key_dim=64,
        phase_width=128,
        local_packets=128,
        slots=16,
        horizon=10000,
    ):
        super().__init__()
        self.config = dict(
            width=width,
            key_dim=key_dim,
            phase_width=phase_width,
            local_packets=local_packets,
            slots=slots,
            horizon=horizon,
        )
        if min(width, key_dim, phase_width, slots) < 1 or horizon < local_packets + 2:
            raise ValueError("Invalid end-to-end configuration")
        self.backbone = AhemV6StateTransitionNextPacket(
            width=width,
            key_dim=key_dim,
            local_packets=local_packets,
            max_transitions=slots,
        )
        self.phase_dim = 4 * 14 + width + 24
        self.phase_encoder = nn.Sequential(
            nn.Linear(self.phase_dim, phase_width),
            nn.GELU(),
            nn.Linear(phase_width, key_dim),
        )
        self.forecast = nn.Sequential(
            nn.Linear(2 * width + self.phase_dim + 11, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 2),
        )
        # Starts as persistence on mature-memory rows, not an identity claim to
        # the retained model. Parent is a separate baseline and protected fallback.
        nn.init.zeros_(self.forecast[-1].weight)
        nn.init.zeros_(self.forecast[-1].bias)
        self.register_buffer(
            "bounds", torch.tensor([math.log1p(65535), math.log1p(60)])
        )
        self.requires_grad_(True)

    def gap(self, earlier, later):
        return (
            torch.log1p((later - earlier).clamp_min(0).float())
            / math.log1p(self.config["horizon"])
        ).clamp(max=1)

    def raw_layout(self, previous, anchors):
        """Ordinal-only admission. Historical successor must already be observed."""
        horizon, slots = self.config["horizon"], self.config["slots"]
        oldest = (anchors - horizon + 1).clamp_min(0)
        chain = [anchors]
        for _ in range(slots + 1):
            before = previous[chain[-1].clamp_min(0)]
            before = torch.where((chain[-1] >= 0) & (before >= oldest), before, -1)
            chain.append(before)
        chain = torch.stack(chain, 1)
        successor, source, predecessor = (
            chain[:, :slots],
            chain[:, 1 : slots + 1],
            chain[:, 2 : slots + 2],
        )
        valid = predecessor >= 0
        if (
            (source[valid] >= anchors[:, None].expand_as(source)[valid])
            | (successor[valid] > anchors[:, None].expand_as(source)[valid])
        ).any():
            raise ValueError("Unobserved transition admitted")
        return chain, valid

    def encode_nodes(self, x, previous, nodes):
        """Encode each source at its own arrival, never at successor arrival."""
        count, local_packets = len(nodes), self.config["local_packets"]
        lengths = (nodes + 1).clamp(max=local_packets)
        local = x.new_zeros(count, self.config["width"])
        stats = x.new_zeros(count, 24)
        for length in lengths.unique().tolist():
            rows = (lengths == length).nonzero().flatten()
            positions = nodes[rows, None] + torch.arange(1 - length, 1, device=x.device)
            windows = x[positions]
            local[rows] = self.backbone.local(windows)
            stats[rows] = torch.cat(
                (windows.mean(1), windows.std(1, unbiased=False)), 1
            )
        phase_indices = [nodes]
        for _ in range(3):
            before = previous[phase_indices[-1].clamp_min(0)]
            phase_indices.append(
                torch.where(
                    (phase_indices[-1] >= 0)
                    & (before >= (nodes - self.config["horizon"] + 1).clamp_min(0)),
                    before,
                    -1,
                )
            )
        indices = torch.stack(phase_indices, 1)
        available = indices >= 0
        packets = x[indices.clamp_min(0)] * available[:, :, None]
        gaps = x.new_zeros(count, 4)
        gaps[:, :3] = self.gap(indices[:, 1:], indices[:, :-1]) * available[:, 1:]
        packed = torch.cat(
            (packets, available[:, :, None].float(), gaps[:, :, None]), 2
        )
        phase = torch.cat((packed.flatten(1), local, stats), 1)
        earliest = torch.minimum(
            nodes - lengths + 1,
            indices.masked_fill(~available, torch.iinfo(torch.long).max).min(1).values,
        )
        return local, phase, earliest

    def forward(self, x, previous, anchors, fallback):
        """x/previous are causal feature history, fallback is frozen parent output.

        No labels, future successors of the query, or attack classes are inputs.
        """
        if anchors.ndim != 1 or fallback.shape != (len(anchors), 2):
            raise ValueError("Aligned anchors/fallback required")
        chain, valid = self.raw_layout(previous, anchors)
        slots = self.config["slots"]
        sources = chain[:, 1 : slots + 1]
        combined = torch.cat((anchors[:, None], sources.clamp_min(0)), 1)
        nodes, inverse = torch.unique(combined, return_inverse=True)
        local_all, phase_all, earliest = self.encode_nodes(x, previous, nodes)
        mapping = inverse.reshape_as(combined)
        local = local_all[mapping[:, 0]]
        query_phase = phase_all[mapping[:, 0]]
        key_phase = phase_all[mapping[:, 1:]]
        oldest = (anchors - self.config["horizon"] + 1).clamp_min(0)
        valid = valid & (earliest[mapping[:, 1:]] >= oldest[:, None])
        pred = x[chain[:, 2 : slots + 2].clamp_min(0)]
        src = x[sources.clamp_min(0)]
        successor = x[chain[:, :slots].clamp_min(0)]
        before = x[chain[:, 1].clamp_min(0)] * (chain[:, 1] >= 0)[:, None]
        current = x[anchors]
        query = self.backbone.project_state(
            before, current, self.gap(chain[:, 1], anchors) * (chain[:, 1] >= 0)
        )
        keys = self.backbone.project_state(
            pred.flatten(0, 1),
            src.flatten(0, 1),
            self.gap(chain[:, 2 : slots + 2], sources).flatten(),
        ).reshape(len(anchors), slots, -1)
        values = self.backbone.project_value(
            src.flatten(0, 1),
            successor.flatten(0, 1),
            self.gap(sources, chain[:, :slots]).flatten(),
        ).reshape(len(anchors), slots, -1)
        q = F.normalize(query + self.phase_encoder(query_phase), dim=-1)
        k = F.normalize(keys + self.phase_encoder(key_phase), dim=-1)
        age = self.gap(sources, anchors[:, None])
        logits = self.backbone._scores(q, k, age).masked_fill(~valid, -1e4)
        weights = logits.softmax(-1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        memory = torch.einsum("bk,bkd->bd", weights, values)
        raw = torch.einsum("bk,bko->bo", weights, successor[:, :, :2])
        mixed = torch.cat((local, memory), 1)
        local_prediction = self.backbone._replace_time(
            self.backbone.local_predict(local), self.backbone.local_time(local)
        )
        memory_prediction = self.backbone._replace_time(
            0.2 * local_prediction
            + 0.8 * raw
            + 0.2 * self.backbone.memory_delta(mixed),
            self.backbone.memory_time(mixed),
        )
        evidence = valid.float().mean(-1, keepdim=True)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(
            -1, keepdim=True
        ) / math.log(max(slots, 2))
        gate = self.backbone.utility_gate(
            torch.cat((mixed, evidence, entropy), 1)
        ).sigmoid()
        features = torch.cat(
            (
                mixed,
                query_phase,
                current[:, :2],
                raw,
                evidence,
                entropy,
                local_prediction,
                memory_prediction,
                gate,
            ),
            1,
        )
        shift = self.forecast(features)
        persistence_logit = torch.logit(
            (current[:, :2] / self.bounds).clamp(1e-6, 1 - 1e-6)
        )
        proposal = self.bounds * (persistence_logit + shift).sigmoid()
        protected = (~valid.any(-1)) | (anchors < self.config["local_packets"] - 1)
        prediction = torch.where(protected[:, None], fallback, proposal)
        return {
            "prediction": prediction,
            "weights": weights,
            "logits": logits,
            "valid": valid,
            "successors": successor[:, :, :2],
            "protected": protected,
            "source_ordinals": sources,
        }


def budgeted_loss(model, output, target, reference, preservation=0.1):
    forecast = dynamic_loss(output["prediction"], target) + preservation * dynamic_loss(
        output["prediction"], reference
    )
    distribution, _ = frozen_utility_distribution(
        output["successors"].detach(), target, output["valid"]
    )
    retrieval = retrieval_cross_entropy(output["logits"], distribution, output["valid"])
    params = tuple(model.parameters())
    gf = torch.autograd.grad(forecast, params, retain_graph=True, allow_unused=True)
    gr = torch.autograd.grad(retrieval, params, retain_graph=True, allow_unused=True)

    def norm(values):
        return (
            torch.stack(
                [g.detach().double().square().sum() for g in values if g is not None]
            )
            .sum()
            .sqrt()
        )

    fn, rn = norm(gf), norm(gr)
    if not torch.isfinite(fn + rn):
        raise ValueError("Nonfinite gradient geometry")
    alpha = torch.minimum(fn.new_tensor(0.02), 0.25 * fn / (rn + 1e-12)).detach()
    # Raw-gradient budget only; not a guarantee on AdamW's preconditioned step.
    return forecast + alpha.float() * retrieval, {
        "forecast": float(forecast.detach()),
        "retrieval": float(retrieval.detach()),
        "aux_weight": float(alpha),
        "forecast_gradient_norm": float(fn),
        "retrieval_gradient_norm": float(rn),
    }

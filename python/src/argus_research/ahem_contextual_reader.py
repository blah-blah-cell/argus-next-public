"""Untrained research candidate: a 161-parameter contextual retrieval correction."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .ahem_v6 import AhemV6StateTransitionNextPacket


class AhemContextualReader(AhemV6StateTransitionNextPacket):
    """Rank known successor outcomes relative to the causal local forecast.

    All v6.1 weights are copied and frozen. A zero-initialized score correction
    starts at identical forecasts and attention. The future query target is
    absent from the forward interface and is permitted only in offline loss.
    This class is not an admitted Argus model or a published result.
    """

    def __init__(self, backbone: AhemV6StateTransitionNextPacket, *, reader_width: int = 16):
        if type(backbone) is not AhemV6StateTransitionNextPacket:
            raise ValueError("contextual reader requires the exact v6.1 backbone class")
        if type(reader_width) is not int or reader_width <= 0:
            raise ValueError("reader width must be a positive integer")
        super().__init__(packet_dim=backbone.packet_dim, output_dim=backbone.output_dim,
                         width=backbone.width, key_dim=backbone.key_dim,
                         local_packets=backbone.local_packets, max_transitions=backbone.max_transitions)
        if backbone.output_dim != 2:
            raise ValueError("contextual reader requires size and delay outputs")
        self.to(device=next(backbone.parameters()).device, dtype=next(backbone.parameters()).dtype)
        self.load_state_dict(backbone.state_dict(), strict=True)
        self.requires_grad_(False)
        self.reader_head = nn.Sequential(nn.Linear(8, reader_width), nn.GELU(), nn.Linear(reader_width, 1))
        self.reader_head.to(device=next(backbone.parameters()).device, dtype=next(backbone.parameters()).dtype)
        self.to(device=next(backbone.parameters()).device, dtype=next(backbone.parameters()).dtype)
        nn.init.zeros_(self.reader_head[-1].weight)
        nn.init.zeros_(self.reader_head[-1].bias)
        self.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        for name, child in self.named_children():
            if name != "reader_head":
                child.eval()
        return self

    def _read(self, local_state: Tensor, query: Tensor, keys: Tensor, values: Tensor,
              successor_dynamic: Tensor, source_ordinals: Tensor, valid: Tensor,
              anchor_ordinals: Tensor, context_spans: Tensor, *, logits_override=None):
        if logits_override is not None:
            raise ValueError("external score overrides cannot bypass the contextual reader")
        local_prediction = self._replace_time(self.local_predict(local_state), self.local_time(local_state))
        age = self._normalized_gap(source_ordinals, anchor_ordinals[:, None], context_spans)
        similarity = torch.einsum("bd,bkd->bk", query, keys)
        scale = successor_dynamic.new_tensor((math.log1p(65_535), math.log1p(60)))
        historical = successor_dynamic / scale
        difference = (successor_dynamic - local_prediction[:, None]) / scale
        features = torch.cat((similarity[..., None], age[..., None], historical,
                              difference, difference.abs()), dim=-1)
        correction = self.reader_head(features).squeeze(-1)
        logits = self._scores(query, keys, age) + correction
        return super()._read(local_state, query, keys, values, successor_dynamic,
                             source_ordinals, valid, anchor_ordinals, context_spans,
                             logits_override=logits)

    def research_contract(self):
        return {"schema": "argus.contextual-retrieval-candidate/v1",
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "total_parameters": sum(p.numel() for p in self.parameters()),
                "retention": "unchanged newest completed transitions",
                "capacity": self.max_transitions, "baseline_initialization": "zero_score_correction",
                "online_future_target_access": False, "trained_result": False}

"""Untrained, separately versioned query-conditioned reader experiment."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .ahem_contextual_reader import AhemContextualReader
from .ahem_v6 import AhemV6StateTransitionNextPacket


class AhemQueryConditionedReader(AhemContextualReader):
    """Use low-rank query/key and local/value interactions, not cosine alone.

    Retention, the backbone and forecasting gate remain frozen. No current
    future packet is accepted by the forward interface. This is an untrained
    research candidate, not an admitted app model or an improvement claim.
    """

    def __init__(self, backbone, *, interaction_rank=8, reader_width=32):
        if type(interaction_rank) is not int or interaction_rank <= 0:
            raise ValueError('interaction rank must be a positive integer')
        super().__init__(backbone, reader_width=reader_width)
        self.interaction_rank = interaction_rank
        self.query_projection = nn.Linear(self.key_dim, interaction_rank, bias=False)
        self.key_projection = nn.Linear(self.key_dim, interaction_rank, bias=False)
        self.local_projection = nn.Linear(self.width, interaction_rank, bias=False)
        self.value_projection = nn.Linear(self.width, interaction_rank, bias=False)
        self.reader_head = nn.Sequential(nn.Linear(8 + 2 * interaction_rank, reader_width),
                                         nn.GELU(), nn.Linear(reader_width, 1))
        self.to(device=next(backbone.parameters()).device, dtype=next(backbone.parameters()).dtype)
        nn.init.zeros_(self.reader_head[-1].weight)
        nn.init.zeros_(self.reader_head[-1].bias)
        self.eval()

    def _read(self, local_state: Tensor, query: Tensor, keys: Tensor, values: Tensor,
              successor_dynamic: Tensor, source_ordinals: Tensor, valid: Tensor,
              anchor_ordinals: Tensor, context_spans: Tensor, *, logits_override=None):
        if logits_override is not None:
            raise ValueError('external score overrides cannot bypass the reader')
        local_prediction = self._replace_time(self.local_predict(local_state), self.local_time(local_state))
        age = self._normalized_gap(source_ordinals, anchor_ordinals[:, None], context_spans)
        similarity = torch.einsum('bd,bkd->bk', query, keys)
        scale = successor_dynamic.new_tensor((math.log1p(65535), math.log1p(60)))
        difference = (successor_dynamic - local_prediction[:, None]) / scale
        query_key = self.query_projection(query)[:, None] * self.key_projection(keys)
        context_value = self.local_projection(local_state)[:, None] * self.value_projection(values)
        features = torch.cat((similarity[..., None], age[..., None], successor_dynamic / scale,
                              difference, difference.abs(), query_key, context_value), dim=-1)
        logits = self._scores(query, keys, age) + self.reader_head(features).squeeze(-1)
        return AhemV6StateTransitionNextPacket._read(self, local_state, query, keys, values,
                                                     successor_dynamic, source_ordinals, valid,
                                                     anchor_ordinals, context_spans,
                                                     logits_override=logits)

    def research_contract(self):
        contract = super().research_contract()
        return contract | {'schema': 'argus.query-conditioned-reader-candidate/v2',
                           'interaction_rank': self.interaction_rank,
                           'new_features': ['query_key_product', 'local_value_product'],
                           'trained_result': False, 'runtime_admission': False}

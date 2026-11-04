"""Bounded causal transition reranking; frozen phase reader and forecast head."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .ahem_phase_reader import AhemPhaseReader
from .ahem_v6 import AhemV6StateTransitionNextPacket
from .ahem_observed_outcome_read import observed_outcome_read


class ExactSqrt(torch.autograd.Function):
    """Exact sqrt forward; choose finite zero derivative at the zero boundary."""
    @staticmethod
    def forward(ctx, value):
        root = value.sqrt()
        ctx.save_for_backward(root)
        return root

    @staticmethod
    def backward(ctx, grad):
        root, = ctx.saved_tensors
        positive = root > 0
        denominator = torch.where(positive, 2 * root, torch.ones_like(root))
        return torch.where(positive, grad / denominator, torch.zeros_like(grad))


def stable_gate_inputs(out, current, capacity):
    """Same values as gate_inputs; only zero-variance backward is stabilized."""
    raw = observed_outcome_read(out.retrieval_weights, out.successor_dynamic, out.valid, current)
    variance = (out.retrieval_weights[..., None] * (out.successor_dynamic - raw[:, None]).square()).sum(1)
    entropy = -(out.retrieval_weights * out.retrieval_weights.clamp_min(1e-8).log()).sum(-1, keepdim=True)
    evidence = out.valid.sum(-1, keepdim=True).to(current.dtype) / capacity
    features = torch.cat((current, raw, raw - current, ExactSqrt.apply(variance.clamp_min(0)), entropy, evidence), -1)
    return features, raw, out.valid.any(-1, keepdim=True)


class TransitionReranker(AhemPhaseReader):
    def __init__(self, reader, *, width=16, correction_cap=.25):
        if type(reader) is not AhemPhaseReader or reader.top_k is not None or width != 16 or correction_cap != .25:
            raise ValueError('exact full-soft parent and declared small reranker required')
        base = AhemV6StateTransitionNextPacket(packet_dim=reader.packet_dim, output_dim=reader.output_dim,
            width=reader.width, key_dim=reader.key_dim, local_packets=reader.local_packets, max_transitions=reader.max_transitions)
        parameter = next(reader.parameters())
        base.to(device=parameter.device, dtype=parameter.dtype)
        base.load_state_dict({n: t for n, t in reader.state_dict().items() if not n.startswith('phase_encoder.')}, strict=True)
        super().__init__(base, phase_width=reader.phase_encoder[0].out_features)
        self.load_state_dict(reader.state_dict(), strict=True)
        self.requires_grad_(False)
        self.correction_cap = correction_cap
        self.rerank_head = nn.Sequential(nn.Linear(18, width), nn.GELU(), nn.Linear(width, 1)).to(device=parameter.device, dtype=parameter.dtype)
        nn.init.zeros_(self.rerank_head[-1].weight)
        nn.init.zeros_(self.rerank_head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'rerank_head'):
            self.rerank_head.train(mode)
        self.phase_encoder.eval()
        return self

    def _read(self, local_state, query, keys, values, successor_dynamic, source_ordinals,
              valid, anchor_ordinals, context_spans, *, query_phase, key_phase, phase_valid):
        if ((source_ordinals >= anchor_ordinals[:, None]) & phase_valid).any() or ((source_ordinals < 0) & phase_valid).any():
            raise ValueError('unobserved source cannot enter causal ranking')
        original = super()._read(local_state, query, keys, values, successor_dynamic, source_ordinals,
            valid, anchor_ordinals, context_spans, query_phase=query_phase, key_phase=key_phase, phase_valid=phase_valid)
        scale = successor_dynamic.new_tensor((math.log1p(65535), math.log1p(60)))
        current = query_phase[:, :2]
        source = key_phase[..., :2]
        previous = query_phase[:, 14:16]
        has_previous = query_phase[:, 26:27]
        trend = (current - previous) * has_previous
        anchor_difference = (current[:, None] - source) / scale
        historical_change = (successor_dynamic - source) / scale
        trend = trend[:, None].expand_as(successor_dynamic) / scale
        change_difference = historical_change - trend
        outcome_difference = (successor_dynamic - current[:, None]) / scale
        age = self._normalized_gap(source_ordinals, anchor_ordinals[:, None], context_spans)
        features = torch.cat((original.retrieval_logits[..., None] / 8, age[..., None],
            anchor_difference, anchor_difference.abs(), historical_change, trend,
            change_difference, change_difference.abs(), outcome_difference, outcome_difference.abs()), -1)
        if features.shape[-1] != 18 or not torch.isfinite(features).all():
            raise ValueError('finite causal transition features required')
        correction = self.correction_cap * torch.tanh(self.rerank_head(features.clamp(-4, 4)).squeeze(-1))
        scores = (original.retrieval_logits + correction).masked_fill(~phase_valid, -1e4)
        q = F.normalize(query + self.phase_encoder(query_phase), dim=-1)
        k = F.normalize(keys + self.phase_encoder(key_phase), dim=-1)
        return AhemV6StateTransitionNextPacket._read(self, local_state, q, k, values, successor_dynamic,
            source_ordinals, phase_valid, anchor_ordinals, context_spans, logits_override=scores)

    def research_contract(self):
        return super().research_contract() | {'schema': 'argus.transition-reranker/v1',
            'additional_parameters': 321, 'correction_cap': self.correction_cap,
            'future_target_inputs': False, 'source_outcome': 'already observed completed successor',
            'frozen_phase_encoder': True, 'forecast_head_training': False}

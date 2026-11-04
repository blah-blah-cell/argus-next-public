"""Separate development repair: top-1-aligned loss and observed-anchor reader."""
from __future__ import annotations

import math

import torch
from torch import nn

from .ahem_contextual_reader import AhemContextualReader
from .ahem_v6 import AhemV6StateTransitionNextPacket
from .ahem_v6_stream import AhemV6Stream
from .flow_v6_objective import dynamic_loss


def best_oracle_margin(logits, successors, target, valid, *, tie_tolerance=1e-8, margin=.1):
    """Push the strongest oracle above the strongest non-oracle.

    Any oracle-equivalent historical successor is acceptable. Close mistakes
    still receive supervision; no 0.01 utility cutoff is used. Future outcomes
    label the offline loss only, never the reader inputs. This nonconvex latent
    positive loss is top-1-aligned, not a guarantee of generalization.
    """
    if any(not math.isfinite(v) or v < 0 for v in (tie_tolerance, margin)):
        raise ValueError('margin and tolerance must be finite and nonnegative')
    if (logits.ndim != 2 or not logits.shape[0] or not logits.shape[1]
            or valid.shape != logits.shape or valid.dtype != torch.bool
            or successors.shape != (*logits.shape, 2) or target.shape != (len(logits), 2)):
        raise ValueError('top-1 inputs must be nonempty and aligned')
    if any(not torch.isfinite(x).all() for x in (logits, successors, target)):
        raise ValueError('top-1 inputs must be finite')
    with torch.no_grad():
        utility = dynamic_loss(successors, target[:, None].expand_as(successors), reduction='none')
        minimum = utility.masked_fill(~valid, torch.inf).amin(-1)
        minimum = torch.where(valid.any(-1), minimum, torch.zeros_like(minimum))
        positives = valid & (utility - minimum[:, None] <= tie_tolerance)
        negatives = valid & ~positives
        informative = (valid.sum(-1) >= 2) & negatives.any(-1)
    # Restrict before masked reductions so empty/all-tied rows never generate
    # infinite intermediates or undefined masked gradients.
    if not informative.any():
        return logits.sum() * 0, {'informative_queries': 0, 'unsupervised_queries': len(logits)}
    scores = logits[informative]
    best_positive = scores.masked_fill(~positives[informative], -torch.inf).amax(-1)
    strongest_negative = scores.masked_fill(~negatives[informative], -torch.inf).amax(-1)
    loss = torch.nn.functional.softplus(strongest_negative - best_positive + margin).mean()
    return loss, {'informative_queries': int(informative.sum()),
                  'unsupervised_queries': int((~informative).sum())}


class AhemAnchorReader(AhemContextualReader):
    """Condition scores on the current observed size/IAT, not the future target.

    Frozen backbone, 16 transitions and horizon semantics stay unchanged.
    No EWMA summary of pre-horizon traffic is introduced. This research-only
    reader requires explicit causal anchor quantities; absent inputs fail closed.
    """
    def __init__(self, backbone, *, reader_width=16):
        super().__init__(backbone, reader_width=reader_width)
        self.reader_head = nn.Sequential(nn.Linear(12, reader_width), nn.GELU(), nn.Linear(reader_width, 1))
        self.to(device=next(backbone.parameters()).device, dtype=next(backbone.parameters()).dtype)
        nn.init.zeros_(self.reader_head[-1].weight)
        nn.init.zeros_(self.reader_head[-1].bias)
        self.eval()

    def _read(self, local_state, query, keys, values, successor_dynamic, source_ordinals,
              valid, anchor_ordinals, context_spans, *, causal_anchor=None, logits_override=None):
        if logits_override is not None:
            raise ValueError('external overrides cannot bypass anchor reader')
        if (causal_anchor is None or causal_anchor.shape != (len(query), 2)
                or causal_anchor.device != query.device or causal_anchor.dtype != query.dtype
                or not torch.isfinite(causal_anchor).all()):
            raise ValueError('explicit finite observed anchor size/IAT required')
        forecast = self._replace_time(self.local_predict(local_state), self.local_time(local_state))
        age = self._normalized_gap(source_ordinals, anchor_ordinals[:, None], context_spans)
        similarity = torch.einsum('bd,bkd->bk', query, keys)
        scale = successor_dynamic.new_tensor((math.log1p(65535), math.log1p(60)))
        difference = (successor_dynamic - forecast[:, None]) / scale
        observed_difference = (successor_dynamic - causal_anchor[:, None]) / scale
        features = torch.cat((similarity[..., None], age[..., None], successor_dynamic / scale,
                              difference, difference.abs(), observed_difference,
                              observed_difference.abs()), dim=-1)
        logits = self._scores(query, keys, age) + self.reader_head(features).squeeze(-1)
        return AhemV6StateTransitionNextPacket._read(self, local_state, query, keys, values,
                                                     successor_dynamic, source_ordinals, valid,
                                                     anchor_ordinals, context_spans, logits_override=logits)

    def research_contract(self):
        return super().research_contract() | {'schema': 'argus.observed-anchor-reader/v3',
            'causal_anchor': 'current observed log size and per-flow IAT',
            'extra_history': False, 'runtime_admission': False}


class AhemAnchorBlendReader(AhemAnchorReader):
    """Two train-only convex-blend coefficients, not extra historical context.

    Persistence is a serious control on repetitive traffic. Rather than hide
    that weakness, test whether combining it with the frozen forecast helps.
    Zero coefficients preserve initial baseline predictions exactly.
    """
    def __init__(self, backbone):
        super().__init__(backbone)
        parameter = next(backbone.parameters())
        self.persistence_blend = nn.Parameter(torch.zeros(2, device=parameter.device, dtype=parameter.dtype))

    def _read(self, *inputs, causal_anchor=None, **kwargs):
        from dataclasses import replace
        result = super()._read(*inputs, causal_anchor=causal_anchor, **kwargs)
        coefficient = self.persistence_blend.clamp(0, 1)
        prediction = result.prediction + coefficient * (causal_anchor - result.prediction)
        return replace(result, prediction=prediction)

    @torch.no_grad()
    def project_parameters(self):
        self.persistence_blend.clamp_(0, 1)

    def research_contract(self):
        return super().research_contract() | {'schema': 'argus.anchor-convex-blend/v1',
            'forecast_blend': 'two train-only global coefficients; not context-adaptive',
            'baseline_initialization': 'zero_reader_correction_and_zero_persistence_blend'}


class AhemConvexForecaster(AhemV6StateTransitionNextPacket):
    """Original memory selector untouched; only two convex forecast coefficients."""
    def __init__(self, backbone):
        if type(backbone) is not AhemV6StateTransitionNextPacket or backbone.output_dim != 2:
            raise ValueError('exact two-output frozen backbone required')
        super().__init__(packet_dim=backbone.packet_dim, output_dim=backbone.output_dim,
                         width=backbone.width, key_dim=backbone.key_dim,
                         local_packets=backbone.local_packets, max_transitions=backbone.max_transitions)
        parameter = next(backbone.parameters())
        self.to(device=parameter.device, dtype=parameter.dtype)
        self.load_state_dict(backbone.state_dict(), strict=True)
        self.requires_grad_(False)
        self.persistence_blend = nn.Parameter(torch.zeros(2, device=parameter.device, dtype=parameter.dtype))
        self.eval()

    def _read(self, *inputs, causal_anchor=None, **kwargs):
        from dataclasses import replace
        query = inputs[1]
        if (causal_anchor is None or causal_anchor.shape != (len(query), 2)
                or causal_anchor.device != query.device or causal_anchor.dtype != query.dtype
                or not torch.isfinite(causal_anchor).all()):
            raise ValueError('explicit finite observed anchor size/IAT required')
        result = super()._read(*inputs, **kwargs)
        prediction = result.prediction + self.persistence_blend.clamp(0, 1) * (causal_anchor - result.prediction)
        return replace(result, prediction=prediction)

    def research_contract(self):
        return {'schema': 'argus.original-reader-convex-forecast/v1',
                'trainable_parameters': 2, 'original_retrieval_unchanged_for_all_inputs': True,
                'extra_history': False, 'runtime_admission': False,
                'causal_anchor': 'current observed log size and per-flow IAT'}


class AhemAnchorStream(AhemV6Stream):
    """Research streaming adapter: current packet fields are already observed."""
    def __init__(self, model, capture_id, **kwargs):
        if type(model) not in (AhemAnchorReader, AhemAnchorBlendReader, AhemConvexForecaster):
            raise ValueError('anchor stream requires the explicit observed-anchor reader')
        super().__init__(model, capture_id, **kwargs)

    def _read_packet(self, packet, *inputs):
        return self.model._read(*inputs, causal_anchor=packet[None, :2])


def top1_selection(metrics, reference, *, forecast_ceiling=1.03, regret_ceiling=1.0):
    """Accuracy first; ties use regret, with explicit forecast/utility budgets."""
    accuracy = metrics['retrieval']['multi_candidate_top1_tie_aware']
    regret = metrics['retrieval']['selected_oracle_regret']
    reference_regret = reference['retrieval']['selected_oracle_regret']
    values = (accuracy, regret, reference_regret, metrics['forecast']['weighted_smooth_l1'],
              reference['forecast']['weighted_smooth_l1'], forecast_ceiling, regret_ceiling)
    if any(v is not None and (not math.isfinite(v) or v < 0) for v in values):
        raise ValueError('selection metrics and budgets must be finite and nonnegative')
    if accuracy is not None and accuracy > 1:
        raise ValueError('accuracy must be a fraction')
    if (accuracy is None or regret is None or reference_regret is None
            or metrics['forecast']['weighted_smooth_l1'] > reference['forecast']['weighted_smooth_l1'] * forecast_ceiling
            or regret > reference_regret * regret_ceiling):
        return None
    return float(accuracy), -float(regret)

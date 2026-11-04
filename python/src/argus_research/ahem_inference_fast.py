"""Inference-only shared readout; retain boundary checks, elide redundant work.

The softmax constructs nonnegative, exactly masked weights. We therefore validate
input finiteness and derived normalization once, rather than treat the internally
constructed weights as arbitrary caller-supplied weights. No training API.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .ahem_read_sharpness_constant import ConstantBetaReadout


class FastReadout(nn.Module):
    def __init__(self, head, rho, *, enabled=True):
        super().__init__()
        frozen = ConstantBetaReadout(copy.deepcopy(head), rho, enabled=enabled)
        self.head = frozen.head
        self.register_buffer('beta', frozen.beta.clone())
        self.enabled = enabled
        self.requires_grad_(False).eval()

    def forward(self, logits, valid, successors, current, context):
        n, k = logits.shape if logits.ndim == 2 else (0, 0)
        if (not n or not k or valid.shape != logits.shape or valid.dtype != torch.bool
                or successors.shape != (n, k, 2) or current.shape != (n, 2)
                or context.shape != (n, len(self.head.context_mean))
                or not logits.is_floating_point()):
            raise ValueError('unaligned finite readout inputs required')
        parameter = next(self.head.parameters())
        if logits.device != parameter.device or logits.dtype != parameter.dtype or valid.device != logits.device:
            raise ValueError('inputs must share model dtype/device')
        if any(t.dtype != logits.dtype or t.device != logits.device for t in (successors,current,context)):
            raise ValueError('inputs must share dtype/device')
        finite_inputs = torch.cat((logits.reshape(-1),successors.reshape(-1),current.reshape(-1),context.reshape(-1)))
        if not torch.isfinite(finite_inputs).all():
            raise ValueError('nonfinite causal inputs')
        scores = self.beta * logits if self.enabled else logits
        weights = scores.masked_fill(~valid,-1e4).softmax(-1) * valid.to(logits.dtype)
        weights = weights / weights.sum(-1,keepdim=True).clamp_min(1e-8)
        available = valid.any(-1,keepdim=True)
        # Important edge case: a finite but extremely negative valid logit can
        # underflow behind the invalid -1e4 sentinel. Do not accept zero mass.
        if not torch.allclose(weights.sum(-1),available[:,0].to(logits.dtype),rtol=1e-5,atol=1e-6):
            raise ValueError('retrieval mass must be one or zero without memory')
        raw = torch.where(available,(weights[...,None]*successors).sum(1),current)
        variance = (weights[...,None]*(successors-raw[:,None]).square()).sum(1)
        entropy = -(weights*weights.clamp_min(1e-8).log()).sum(-1,keepdim=True)
        evidence = valid.sum(-1,keepdim=True).to(current.dtype)/16
        features = torch.cat((current,raw,raw-current,variance.clamp_min(0).sqrt(),entropy,evidence),-1)
        if not torch.isfinite(features).all():
            raise ValueError('nonfinite derived read features')
        gate_logits = self.head.network(features)
        if self.head.context_enabled:
            z = ((context-self.head.context_mean)/self.head.context_scale).clamp(-10,10)
            gate_logits = gate_logits+self.head.context_delta(z)
        coefficient = gate_logits.sigmoid()*available.to(gate_logits.dtype)
        prediction = current+coefficient*(raw-current)
        return weights,raw,features,prediction


class FastConsolidatedAhem(nn.Module):
    """Reuse the frozen bundle; don't compute discarded legacy forecast heads."""
    def __init__(self, model, rho, *, enabled=True):
        super().__init__()
        self.reader = model.reader
        self.readout = FastReadout(model.readout.head,rho,enabled=enabled)
        self.requires_grad_(False).eval()

    def forward(self, local_state, query, keys, values, successors, source_ordinals,
                valid, anchor_ordinals, context_spans, query_phase, key_phase,
                phase_valid, current):
        # Historical raw phase-footprint eligibility remains the projector's
        # responsibility. Fail closed on mask expansion or future source slots.
        n,k = valid.shape
        if (valid.dtype != torch.bool or phase_valid.dtype != torch.bool or phase_valid.shape != valid.shape
                or query.shape != (n,self.reader.key_dim) or keys.shape != (n,k,self.reader.key_dim)
                or query_phase.shape != (n,self.reader.phase_dim) or key_phase.shape != (n,k,self.reader.phase_dim)
                or source_ordinals.shape != (n,k) or anchor_ordinals.shape != (n,) or context_spans.shape != (n,)):
            raise ValueError('unaligned projected phase inputs')
        if ((phase_valid & ~valid).any() or ((source_ordinals>=anchor_ordinals[:,None])&phase_valid).any()
                or ((source_ordinals<0)&phase_valid).any() or (context_spans<=0).any()):
            raise ValueError('illegal causal phase mask')
        if not torch.isfinite(torch.cat((query.reshape(-1),keys.reshape(-1),query_phase.reshape(-1),key_phase.reshape(-1),context_spans))).all():
            raise ValueError('nonfinite projected phase')
        q = F.normalize(query+self.reader.phase_encoder(query_phase),dim=-1)
        key = F.normalize(keys+self.reader.phase_encoder(key_phase),dim=-1)
        age = self.reader._normalized_gap(source_ordinals,anchor_ordinals[:,None],context_spans)
        scores = self.reader._scores(q,key,age).masked_fill(~phase_valid,-1e4)
        weights,raw,features,prediction = self.readout(scores,phase_valid,successors,current,query_phase)
        return prediction,weights,raw,features

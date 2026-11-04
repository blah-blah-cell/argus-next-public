"""Frozen-beta inference only; same arithmetic, no per-query rho machinery."""
from types import SimpleNamespace
import math
import torch
from torch import nn
from .ahem_read_sharpness import causal_weights
from .ahem_transition_reranker import stable_gate_inputs


class ConstantBetaReadout(nn.Module):
    def __init__(self, head, rho, *, enabled=True):
        super().__init__()
        parameter = next(head.parameters())
        rho = torch.as_tensor(rho, device=parameter.device, dtype=parameter.dtype).detach().clone()
        if rho.numel() != 1 or not torch.isfinite(rho) or not 0 <= float(rho) <= math.log(2.5) + 1e-7:
            raise ValueError('finite frozen fitted rho in declared range required')
        self.head = head
        self.head.requires_grad_(False).eval()
        self.enabled = enabled
        # Device-specific float32 exp matches reference exactly on this device.
        self.register_buffer('beta', rho.exp())

    def weights(self, logits, valid):
        # The parent must not perform an unnecessary beta=1 multiplication.
        return causal_weights(self.beta * logits if self.enabled else logits, valid)

    def forward(self, logits, valid, successors, current, context):
        weights = self.weights(logits, valid)
        result = SimpleNamespace(retrieval_weights=weights, successor_dynamic=successors, valid=valid)
        features, raw, available = stable_gate_inputs(result, current, 16)
        prediction = self.head(features, context, current, raw, available)
        return weights, raw, features, prediction

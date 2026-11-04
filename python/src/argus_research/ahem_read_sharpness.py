"""One-scalar positive temperature intervention on frozen causal reader logits."""
import math
import torch
from torch import nn


def causal_weights(logits, valid):
    if logits.shape != valid.shape or valid.dtype != torch.bool or not torch.isfinite(logits).all():
        raise ValueError('finite aligned causal logits required')
    weights = logits.masked_fill(~valid, -1e4).softmax(-1) * valid.to(logits.dtype)
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)


class ReadSharpness(nn.Module):
    def __init__(self, *, dtype=torch.float32, device='cpu'):
        super().__init__()
        self.rho = nn.Parameter(torch.zeros((), dtype=dtype, device=device))

    @property
    def beta(self):
        return self.rho.exp()

    @torch.no_grad()
    def project(self):
        self.rho.clamp_(0, math.log(2.5))

    def forward(self, logits, valid):
        if not torch.isfinite(self.rho) or not 0 <= float(self.rho.detach()) <= math.log(2.5) + 1e-7:
            raise ValueError('projected finite rho required')
        return causal_weights(self.beta * logits, valid)

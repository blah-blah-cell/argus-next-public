"""Small research-only gate between an observed prior and retrieved outcomes."""
import torch
from torch import nn

from .ahem_observed_outcome_read import observed_outcome_read


def gate_inputs(out, current, *, max_transitions=16, variance_floor=0.):
    if type(max_transitions) is not int or max_transitions < out.valid.size(1) or max_transitions <= 0:
        raise ValueError('invalid fixed cache capacity')
    raw = observed_outcome_read(out.retrieval_weights, out.successor_dynamic, out.valid, current)
    variance = (out.retrieval_weights[..., None] * (out.successor_dynamic - raw[:, None]).square()).sum(1)
    entropy = -(out.retrieval_weights * out.retrieval_weights.clamp_min(1e-8).log()).sum(-1, keepdim=True)
    # Compact streaming omits unused slots. It must retain the same fixed
    # denominator as full-capacity training, not treat two slots as a full bank.
    evidence = out.valid.sum(-1, keepdim=True).to(current.dtype) / max_transitions
    if not 0 <= variance_floor < float('inf'):
        raise ValueError('variance floor must be finite and nonnegative')
    # Joint reader training differentiates through variance. sqrt(0) has an
    # undefined derivative; the explicit joint protocol uses a tiny floor.
    features = torch.cat((current, raw, raw - current, variance.clamp_min(variance_floor).sqrt(), entropy, evidence), -1)
    return features, raw, out.valid.any(-1, keepdim=True)


class ObservedPriorGate(nn.Module):
    """210 parameters; independent size/delay trust, causal convex prediction."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(10, 16), nn.GELU(), nn.Linear(16, 2))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, -4.)

    def forward(self, features, current, raw, available):
        if (features.shape != (len(current), 10) or current.shape != raw.shape
                or current.shape[-1] != 2 or available.shape != (len(current), 1)
                or available.dtype != torch.bool):
            raise ValueError('unaligned observed-prior gate inputs')
        gate = self.network(features).sigmoid() * available.to(current.dtype)
        return current + gate * (raw - current)

"""Research-only causal mixture of persistence, soft mean and weighted median.

No new memory entries or future targets. This is a readout candidate, not a
promoted retrieval architecture or a runtime deployment.
"""
import torch
from torch import nn
from .ahem_outcome_gate import ObservedPriorGate


class RobustObservedReadout(nn.Module):
    def __init__(self, *, median_enabled=True):
        super().__init__()
        if type(median_enabled) is not bool:
            raise ValueError('explicit median arm required')
        self.median_enabled = median_enabled
        self.network = nn.Sequential(nn.Linear(12,16), nn.GELU(), nn.Linear(16,6))

    @torch.no_grad()
    def initialize_from_gate(self, gate):
        if type(gate) is not ObservedPriorGate:
            raise ValueError('exact observed-prior gate required')
        first, last = self.network[0], self.network[2]
        first.weight.zero_()
        first.weight[:, :10].copy_(gate.network[0].weight)
        first.bias.copy_(gate.network[0].bias)
        last.weight.zero_()
        last.bias.zero_()
        # Per field, logits are [persistence=0, mean=old gate logit, median=-8].
        # Masking median yields exactly the original sigmoid mixture (up to
        # floating point). Enabled median has a small, explicitly nonzero prior.
        for field in range(2):
            last.weight[3*field+1].copy_(gate.network[2].weight[field])
            last.bias[3*field+1].copy_(gate.network[2].bias[field])
            last.bias[3*field+2] = -8.

    def forward(self, features, current, mean, median, available):
        if (features.shape != (len(current),10) or current.shape != mean.shape
            or median.shape != current.shape or current.shape != (len(current),2)
            or available.shape != (len(current),1) or available.dtype != torch.bool):
            raise ValueError('unaligned causal mixture inputs')
        if any(not torch.isfinite(t).all() for t in (features,current,mean,median)):
            raise ValueError('finite observations required')
        logits = self.network(torch.cat((features,median-current),dim=1)).reshape(-1,2,3)
        if not self.median_enabled:
            logits = logits.clone()
            logits[:,:,2] = -torch.inf
        weights = logits.softmax(-1)
        observations = torch.stack((current,mean,median),dim=-1)
        prediction = (weights*observations).sum(-1)
        return torch.where(available,prediction,current)

    def research_contract(self):
        return {'schema': 'argus.robust-observed-readout/v1', 'parameters': 310,
            'median_enabled': self.median_enabled, 'future_targets_as_inputs': False,
            'memory_policy_changed': False, 'runtime_admission': False,
            'claim': 'untrained readout candidate; no established accuracy gain'}

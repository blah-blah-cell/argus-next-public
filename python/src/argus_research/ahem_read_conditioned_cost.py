"""Experimental read-conditioned residual; preserves its trained parent at init."""
import copy
import torch
from torch import nn
from .ahem_cost_context_readout import CostContextReadout


class ReadConditionedCost(nn.Module):
    def __init__(self, parent, read_scale, *, interaction_enabled=True):
        super().__init__()
        if type(parent) is not CostContextReadout or not parent.context_enabled or type(interaction_enabled) is not bool:
            raise ValueError('exact trained context parent required')
        if read_scale.shape != (2,) or not torch.isfinite(read_scale).all() or (read_scale < 1e-6).any():
            raise ValueError('positive finite training-only read scales required')
        self.parent = copy.deepcopy(parent)
        self.interaction_enabled = interaction_enabled
        self.register_buffer('read_scale', read_scale.detach().clone())
        self.interaction = nn.Linear(len(parent.context_mean), 2, bias=False,
            device=parent.context_mean.device, dtype=parent.context_mean.dtype)
        nn.init.zeros_(self.interaction.weight)
        self.interaction.requires_grad_(interaction_enabled)

    def trust(self, features, context, available):
        # Reuse the parent's fail-closed checks without reconstructing logits
        # from probabilities, which could disturb saturated parent predictions.
        self.parent.trust(features, context, available)
        z = ((context - self.parent.context_mean) / self.parent.context_scale).clamp(-10, 10)
        logits = self.parent.network(features) + self.parent.context_delta(z)
        if self.interaction_enabled:
            read_direction = torch.tanh(features[:, 4:6] / self.read_scale)
            logits = logits + self.interaction(z) * read_direction
        return logits.sigmoid() * available.to(logits.dtype)

    def forward(self, features, context, current, raw, available):
        if current.shape != (len(features), 2) or raw.shape != current.shape or not torch.isfinite(current).all() or not torch.isfinite(raw).all():
            raise ValueError('aligned finite causal predictions required')
        if features.shape != (len(current), 10) or not torch.allclose(features[:, 4:6], raw - current, atol=1e-6, rtol=1e-5):
            raise ValueError('read-disagreement contract mismatch')
        return current + self.trust(features, context, available) * (raw - current)

    def research_contract(self):
        return {'schema': 'argus.read-conditioned-cost/v1', 'interaction_enabled': self.interaction_enabled,
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'stored_parameters': sum(p.numel() for p in self.parameters()),
            'initial_prediction': 'identical to trained context parent', 'future_targets_as_inputs': False,
            'memory_policy_changed': False, 'runtime_admission': False}

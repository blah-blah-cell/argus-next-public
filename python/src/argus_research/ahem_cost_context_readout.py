"""Experimental continuous cost-trained gate; no teacher labels at inference."""
import copy
import torch
from torch import nn
from .ahem_outcome_gate import ObservedPriorGate


class CostContextReadout(nn.Module):
    def __init__(self, parent, mean, scale, *, context_enabled=True):
        super().__init__()
        if type(parent) is not ObservedPriorGate or type(context_enabled) is not bool:
            raise ValueError('exact parent gate and explicit comparison arm required')
        if mean.ndim!=1 or scale.shape!=mean.shape or not len(mean) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale<1e-6).any():
            raise ValueError('finite training-only context statistics required')
        self.context_enabled=context_enabled
        self.network=copy.deepcopy(parent.network)
        self.register_buffer('context_mean',mean.detach().clone())
        self.register_buffer('context_scale',scale.detach().clone())
        self.context_delta=nn.Linear(len(mean),2,device=mean.device,dtype=mean.dtype)
        nn.init.zeros_(self.context_delta.weight)
        nn.init.zeros_(self.context_delta.bias)
        self.context_delta.requires_grad_(context_enabled)

    def trust(self, features, context, available):
        if features.shape!=(len(context),10) or context.shape!=(len(features),len(self.context_mean)) or available.shape!=(len(features),1) or available.dtype!=torch.bool:
            raise ValueError('unaligned causal inputs')
        if not torch.isfinite(features).all() or not torch.isfinite(context).all():
            raise ValueError('nonfinite causal features')
        logits=self.network(features)
        if self.context_enabled:
            # Predeclared numerical safeguard, not a validation-fit threshold.
            z=((context-self.context_mean)/self.context_scale).clamp(-10,10)
            logits=logits+self.context_delta(z)
        return logits.sigmoid()*available.to(logits.dtype)

    def forward(self, features, context, current, raw, available):
        if current.shape!=(len(features),2) or raw.shape!=current.shape or not torch.isfinite(current).all() or not torch.isfinite(raw).all():
            raise ValueError('aligned finite observed predictions required')
        coefficient=self.trust(features,context,available)
        return current+coefficient*(raw-current)

    def research_contract(self):
        return {'schema':'argus.cost-context-readout/v1','context_enabled':self.context_enabled,
            'context_dim':len(self.context_mean),'trainable_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad),
            'stored_parameters':sum(p.numel() for p in self.parameters()),'future_targets_as_inputs':False,
            'initial_prediction':'identical to frozen parent','action':'continuous convex per-field blend; no fitted operating threshold',
            'abstention':'exact persistence when memory unavailable; otherwise continuous shrinkage, not exact hard abstention',
            'memory_policy_changed':False,'runtime_admission':False}

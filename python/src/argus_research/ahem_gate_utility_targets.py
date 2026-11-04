"""Offline coefficient labels; targets must never enter inference features."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def utility_targets(current, raw, target, available, *, minimum_span=1e-4):
    if current.shape != raw.shape or target.shape != current.shape or current.ndim != 2 or current.shape[1]!=2 or available.shape!=(len(current),1) or available.dtype!=torch.bool:
        raise ValueError('aligned fields and boolean memory availability required')
    if minimum_span<=0 or any(not torch.isfinite(t).all() for t in (current,raw,target)):
        raise ValueError('finite fields and positive span threshold required')
    delta=raw-current
    active=(delta.abs()>minimum_span)&available
    safe=torch.where(active,delta,torch.ones_like(delta))
    coefficient=torch.where(active,((target-current)/safe).clamp(0,1),torch.zeros_like(delta))
    oracle=current+coefficient*delta
    base=F.smooth_l1_loss(current,target,reduction='none')
    endpoint=F.smooth_l1_loss(raw,target,reduction='none')
    best=F.smooth_l1_loss(oracle,target,reduction='none')
    # Both useful history and a harmful endpoint carry supervision. This is
    # the worst endpoint's regret to the nearest reachable target, not a class
    # count weight that would reward always predicting persistence.
    regret=(torch.maximum(base,endpoint)-best).clamp_min(0)*active
    return coefficient.detach(),regret.detach(),active.detach()


def utility_loss(logits, coefficient, regret):
    if logits.shape != coefficient.shape or regret.shape!=logits.shape or logits.ndim!=2 or logits.shape[1]!=2:
        raise ValueError('aligned two-field supervision required')
    if any(not torch.isfinite(t).all() for t in (logits,coefficient,regret)) or (regret<0).any() or (coefficient<0).any() or (coefficient>1).any():
        raise ValueError('finite logits, unit coefficients and nonnegative regret required')
    values=F.binary_cross_entropy_with_logits(logits,coefficient,reduction='none')
    mass=regret.sum(0)
    field=(values*regret).sum(0)/mass.clamp_min(1e-12)
    # Zero-regret field is exactly zero, including its gradient.
    field=torch.where(mass>0,field,torch.zeros_like(field))
    return (field[0]+4*field[1])/5

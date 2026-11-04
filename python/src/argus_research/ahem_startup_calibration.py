"""Constrained startup-only field shrinkage; no neural parameter changes.

For fixed parent p, persistence c and observed training target y, Huber of
p + (1-s)(c-p) - y is convex in s in [0,1]. Each source's sublevel set at
its identity loss is an interval containing 1. Their intersection therefore
contains 1. Fit the equal-source mean within that interval, per field.
This is a training-set geometry guarantee, not a generalization guarantee.
"""
import torch
from torch import nn


def calibrate_prediction(parent, current, ready, scale):
    if (parent.shape != current.shape or parent.ndim != 2 or parent.shape[1] != 2
            or ready.shape != (len(parent),) or ready.dtype != torch.bool
            or scale.shape != (2,) or not torch.isfinite(scale).all()
            or (scale < 0).any() or (scale > 1).any()):
        raise ValueError("aligned bounded startup calibration required")
    correction = parent + (1 - scale) * (current - parent)
    # Identity is exact at s=1; no-memory parent==current is also unchanged.
    return torch.where(ready[:, None], parent, correction)


def fit_startup_scale(sources, iterations=64):
    if iterations != 64 or not sources:
        raise ValueError("fixed 64-step solver and nonempty training sources required")
    data = []
    for parent, current, target in sources:
        if (parent.shape != current.shape or target.shape != parent.shape or
            parent.ndim != 2 or parent.shape[1] != 2 or not len(parent)):
            raise ValueError("nonempty aligned training startup fields required")
        if not all(torch.isfinite(t).all() for t in (parent,current,target)):
            raise ValueError("nonfinite training startup fields")
        data.append(tuple(t.detach().double().cpu() for t in (parent,current,target)))
    def loss(d, field, s):
        p,c,y = (t[:,field] for t in d)
        error = (p + (1-s)*(c-p)-y).abs()
        return float(torch.where(error < 1, .5*error.square(), error-.5).mean())
    def derivative(d, field, s):
        p,c,y = (t[:,field] for t in d)
        return float(((p+(1-s)*(c-p)-y).clamp(-1,1)*(p-c)).mean())
    scale, diagnostics = [], []
    for field in (0,1):
        bounds = []
        for d in data:
            baseline = loss(d,field,1.)
            if loss(d,field,0.) <= baseline:
                bound = 0.
            elif derivative(d,field,1.) <= 0:
                bound = 1.
            else:
                lo,hi = 0.,1.
                for _ in range(iterations):
                    mid = (lo+hi)/2
                    if loss(d,field,mid) > baseline:
                        lo = mid
                    else:
                        hi = mid
                bound = hi
            bounds.append(bound)
        lower = max(bounds)
        def pooled_derivative(s):
            return sum(derivative(d,field,s) for d in data)/len(data)
        if lower == 1. or pooled_derivative(1.) <= 0:
            value = 1.
        elif pooled_derivative(lower) >= 0:
            value = lower
        else:
            lo,hi = lower,1.
            for _ in range(iterations):
                mid = (lo+hi)/2
                if pooled_derivative(mid) < 0:
                    lo = mid
                else:
                    hi = mid
            value = (lo+hi)/2
        # Round towards the feasible identity endpoint if FP32 crosses lower.
        rounded = torch.tensor(value,dtype=torch.float32)
        if float(rounded) < lower:
            rounded = torch.nextafter(rounded,torch.tensor(1.,dtype=torch.float32))
        scale.append(rounded)
        diagnostics.append({"feasible_lower":lower,"source_lower_bounds":bounds,
                            "coefficient":float(rounded),
                            "derivatives_at_identity":[derivative(d,field,1.) for d in data],
                            "training_parent_huber":[loss(d,field,1.) for d in data],
                            "training_calibrated_huber":[loss(d,field,float(rounded)) for d in data]})
    return torch.stack(scale), diagnostics


class StartupCalibration(nn.Module):
    def __init__(self, model, scale):
        super().__init__()
        self.model = model
        self.window_packets = model.reader.local_packets
        scale = torch.as_tensor(scale).detach().clone()
        if scale.shape != (2,) or not torch.isfinite(scale).all() or (scale<0).any() or (scale>1).any():
            raise ValueError("bounded field scale required")
        parameter = next(model.parameters())
        self.register_buffer("scale",scale.to(device=parameter.device,dtype=parameter.dtype))
        self.requires_grad_(False).eval()

    def forward(self, local_state, query, keys, values, successors, source_ordinals,
                valid, anchor_ordinals, context_spans, query_phase, key_phase,
                phase_valid, current):
        if anchor_ordinals.ndim != 1 or anchor_ordinals.dtype not in (torch.int32,torch.int64) or (anchor_ordinals<0).any():
            raise ValueError("nonnegative integer segment ordinals required")
        prediction,weights,raw,features = self.model(local_state,query,keys,values,
             successors,source_ordinals,valid,anchor_ordinals,context_spans,
             query_phase,key_phase,phase_valid,current)
        ready = anchor_ordinals >= self.window_packets-1
        return calibrate_prediction(prediction,current,ready,self.scale),weights,raw,features

"""Common projected engine with aggregated causal checks and inline calibration.

Parent and candidate share this engine; original fast parent remains a separate
timing control. Math/operation order of neural scoring and field correction is
unchanged. No training API or reduction in boundary checks.
"""
import torch
from torch import nn
from torch.nn import functional as F


class InlineStartupAhem(nn.Module):
    def __init__(self,fast_model,scale,*,enabled=True):
        super().__init__()
        self.reader=fast_model.reader;self.readout=fast_model.readout
        self.window_packets=self.reader.local_packets;self.enabled=enabled
        parameter=next(self.reader.parameters())
        with torch.inference_mode(False):
            scale=torch.as_tensor(scale).detach().to(device=parameter.device,dtype=parameter.dtype).clone()
            if scale.shape!=(2,) or not torch.isfinite(scale).all() or (scale<0).any() or (scale>1).any():
                raise ValueError('bounded startup calibration scale required')
            self.register_buffer('scale',scale)
            self.register_buffer('complement',(1-scale).clone())
        self._fixed=((self.scale._version,self.scale.data_ptr()),(self.complement._version,self.complement.data_ptr()))
        self.requires_grad_(False).eval()

    def forward(self,local_state,query,keys,values,successors,source_ordinals,
                valid,anchor_ordinals,context_spans,query_phase,key_phase,
                phase_valid,current):
        if self.enabled and ((self.scale._version,self.scale.data_ptr()),(self.complement._version,self.complement.data_ptr()))!=self._fixed:
            raise ValueError('fixed calibration buffers changed')
        if valid.ndim!=2:raise ValueError('unaligned projected phase inputs')
        n,k=valid.shape
        if (not n or valid.dtype!=torch.bool or phase_valid.dtype!=torch.bool or phase_valid.shape!=valid.shape
                or query.shape!=(n,self.reader.key_dim) or keys.shape!=(n,k,self.reader.key_dim)
                or query_phase.shape!=(n,self.reader.phase_dim) or key_phase.shape!=(n,k,self.reader.phase_dim)
                or source_ordinals.shape!=(n,k) or anchor_ordinals.shape!=(n,) or context_spans.shape!=(n,)
                or anchor_ordinals.dtype not in (torch.int32,torch.int64)):
            raise ValueError('unaligned integer-ordinal projected phase inputs')
        first=int(anchor_ordinals.item()) if n==1 else int(anchor_ordinals.min().item())
        if first<0:raise ValueError('negative segment ordinal')
        illegal=(phase_valid & ~valid) | (((source_ordinals>=anchor_ordinals[:,None]) | (source_ordinals<0)) & phase_valid)
        if torch.cat((illegal.reshape(-1),context_spans<=0)).any():
            raise ValueError('illegal causal phase mask')
        if not torch.isfinite(torch.cat((query.reshape(-1),keys.reshape(-1),query_phase.reshape(-1),key_phase.reshape(-1),context_spans))).all():
            raise ValueError('nonfinite projected phase')
        q=F.normalize(query+self.reader.phase_encoder(query_phase),dim=-1)
        key=F.normalize(keys+self.reader.phase_encoder(key_phase),dim=-1)
        age=self.reader._normalized_gap(source_ordinals,anchor_ordinals[:,None],context_spans)
        scores=self.reader._scores(q,key,age).masked_fill(~phase_valid,-1e4)
        weights,raw,features,prediction=self.readout(scores,phase_valid,successors,current,query_phase)
        if self.enabled and first<self.window_packets-1:
            if n==1:
                # FastReadout always creates a fresh prediction (current + ...),
                # never aliases input/current/raw/features. Preserve subtraction,
                # multiplication, addition order while reusing owned temporaries.
                correction=current-prediction
                correction.mul_(self.complement)
                prediction.add_(correction)
            else:
                corrected=prediction+self.complement*(current-prediction)
                prediction=torch.where((anchor_ordinals>=self.window_packets-1)[:,None],prediction,corrected)
        return prediction,weights,raw,features

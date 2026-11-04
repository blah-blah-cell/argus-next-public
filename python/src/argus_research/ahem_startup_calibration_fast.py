"""Inference-only fixed calibration; validate immutable scale once, not per row."""
import torch
from .ahem_startup_calibration import StartupCalibration


class FastStartupCalibration(StartupCalibration):
    def __init__(self, model, scale):
        with torch.inference_mode(False):
            super().__init__(model,scale)
        self._scale_version = self.scale._version
        self._scale_pointer = self.scale.data_ptr()

    def forward(self, local_state, query, keys, values, successors, source_ordinals,
                valid, anchor_ordinals, context_spans, query_phase, key_phase,
                phase_valid, current):
        if self.scale._version != self._scale_version or self.scale.data_ptr() != self._scale_pointer:
            raise ValueError('fixed startup scale changed; reconstruct validated adapter')
        if anchor_ordinals.ndim != 1 or anchor_ordinals.dtype not in (torch.int32,torch.int64) or (anchor_ordinals<0).any():
            raise ValueError('nonnegative integer segment ordinals required')
        # The underlying fast reader checks shape/device, input finiteness,
        # causal phase validity and derived normalization. Scale is fixed above.
        prediction,weights,raw,features = self.model(local_state,query,keys,values,
             successors,source_ordinals,valid,anchor_ordinals,context_spans,
             query_phase,key_phase,phase_valid,current)
        ready = anchor_ordinals >= self.window_packets-1
        corrected = prediction+(1-self.scale)*(current-prediction)
        return torch.where(ready[:,None],prediction,corrected),weights,raw,features

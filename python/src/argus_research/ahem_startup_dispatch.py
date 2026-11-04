"""Exact calibration with a mature-batch fast path; no tensor work after parent.

One integer-min reduction both rejects negative segment ordinals and detects a
fully mature batch. Mature batches need no calibration arithmetic or allocation.
Mixed/startup batches retain the exact frozen FP32 expression and operation order.
"""
import torch
from .ahem_startup_calibration_fast import FastStartupCalibration


class DispatchedStartupCalibration(FastStartupCalibration):
    def forward(self, local_state, query, keys, values, successors, source_ordinals,
                valid, anchor_ordinals, context_spans, query_phase, key_phase,
                phase_valid, current):
        if self.scale._version != self._scale_version or self.scale.data_ptr() != self._scale_pointer:
            raise ValueError('fixed startup scale changed; reconstruct validated adapter')
        if (anchor_ordinals.ndim != 1 or not anchor_ordinals.numel()
                or anchor_ordinals.dtype not in (torch.int32,torch.int64)):
            raise ValueError('nonnegative integer segment ordinals required')
        first = int(anchor_ordinals.min().item())
        if first < 0:
            raise ValueError('nonnegative integer segment ordinals required')
        output = self.model(local_state,query,keys,values,successors,source_ordinals,
                            valid,anchor_ordinals,context_spans,query_phase,key_phase,
                            phase_valid,current)
        if first >= self.window_packets-1:
            return output
        prediction,weights,raw,features = output
        ready = anchor_ordinals >= self.window_packets-1
        corrected = prediction+(1-self.scale)*(current-prediction)
        return torch.where(ready[:,None],prediction,corrected),weights,raw,features

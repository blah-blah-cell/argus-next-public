"""Exact singleton fast path and precomputed frozen complement; same calibration."""
import torch
from .ahem_startup_dispatch import DispatchedStartupCalibration


class FusedStartupCalibration(DispatchedStartupCalibration):
    def __init__(self,model,scale):
        super().__init__(model,scale)
        with torch.inference_mode(False):
            self.register_buffer('complement',(1-self.scale).detach().clone())
        self._complement_version=self.complement._version
        self._complement_pointer=self.complement.data_ptr()

    def forward(self, local_state, query, keys, values, successors, source_ordinals,
                valid, anchor_ordinals, context_spans, query_phase, key_phase,
                phase_valid, current):
        if (self.scale._version!=self._scale_version or self.scale.data_ptr()!=self._scale_pointer
                or self.complement._version!=self._complement_version
                or self.complement.data_ptr()!=self._complement_pointer):
            raise ValueError('fixed calibration buffers changed; reconstruct adapter')
        if (anchor_ordinals.ndim!=1 or not anchor_ordinals.numel()
                or anchor_ordinals.dtype not in (torch.int32,torch.int64)):
            raise ValueError('nonnegative integer segment ordinals required')
        singleton=anchor_ordinals.numel()==1
        first=int(anchor_ordinals.item()) if singleton else int(anchor_ordinals.min().item())
        if first<0:raise ValueError('nonnegative integer segment ordinals required')
        output=self.model(local_state,query,keys,values,successors,source_ordinals,
                          valid,anchor_ordinals,context_spans,query_phase,key_phase,
                          phase_valid,current)
        if first>=self.window_packets-1:return output
        prediction,weights,raw,features=output
        corrected=prediction+self.complement*(current-prediction)
        # One startup query cannot be a mixed batch, so no mask or where needed.
        if singleton:return corrected,weights,raw,features
        ready=anchor_ordinals>=self.window_packets-1
        return torch.where(ready[:,None],prediction,corrected),weights,raw,features

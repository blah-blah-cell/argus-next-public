"""Exploratory startup guard using the existing complete local-window contract.

This is a new policy, NOT parity-equivalent optimization of the frozen candidate.
Caller supplies zero-based contiguous global ordinals within the current capture
segment. Reset/discontinuity must start a new segment, not reuse old ordinals.
"""
import torch
from torch import nn


class StartupGuard(nn.Module):
    def __init__(self, model):
        super().__init__()
        if not hasattr(model, "reader") or type(model.reader.local_packets) is not int or model.reader.local_packets < 1:
            raise ValueError("model must declare a positive local window")
        self.model = model
        self.window_packets = model.reader.local_packets
        self.requires_grad_(False).eval()

    def ready(self, anchor_ordinals):
        if (anchor_ordinals.ndim != 1 or anchor_ordinals.dtype not in (torch.int32, torch.int64)
                or (anchor_ordinals < 0).any()):
            raise ValueError("nonnegative integer segment-global ordinals required")
        return anchor_ordinals >= self.window_packets - 1

    def forward(self, local_state, query, keys, values, successors, source_ordinals,
                valid, anchor_ordinals, context_spans, query_phase, key_phase,
                phase_valid, current):
        ready = self.ready(anchor_ordinals)
        if phase_valid.shape != valid.shape or phase_valid.dtype != torch.bool or phase_valid.device != ready.device:
            raise ValueError("aligned causal phase mask required")
        # Validate original mask expansion even when warm-up would hide it.
        if (phase_valid & ~valid).any():
            raise ValueError("illegal original phase mask")
        if (((source_ordinals >= anchor_ordinals[:, None]) | (source_ordinals < 0)) & phase_valid).any():
            raise ValueError("illegal original source ordinal")
        safe_mask = phase_valid & ready[:, None]
        return self.model(local_state, query, keys, values, successors,
                          source_ordinals, valid, anchor_ordinals, context_spans,
                          query_phase, key_phase, safe_mask, current)

    def research_contract(self):
        return {"schema": "argus.startup-guard/v1", "window_packets": self.window_packets,
                "ordinal_scope": "zero-based contiguous segment-global; not same-flow count",
                "startup_prediction": "current observed size and same-flow delay (persistence)",
                "mature_prediction": "unchanged frozen candidate",
                "learned_parameters_added": 0, "runtime_admission": False}

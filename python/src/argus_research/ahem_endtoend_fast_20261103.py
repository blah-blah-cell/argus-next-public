"""Inference-only equivalent of the recovered end-to-end research model.

The four causal width-3 layers have dilations 1,2,4,8: receptive field 31.
Statistics retain the complete configured history; only the final local CNN
state uses the minimal suffix. Never alters checkpoint weights or training.
"""
import torch
from .ahem_endtoend_20261026 import EndToEndAhem


class FastEndToEndAhem(EndToEndAhem):
    def encode_nodes(self, x, previous, nodes):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('fast path is evaluation-only under inference/no_grad')
        count, local_packets = len(nodes), self.config['local_packets']
        lengths = (nodes + 1).clamp(max=local_packets)
        local = x.new_zeros(count, self.config['width'])
        stats = x.new_zeros(count, 24)
        receptive_field = 1 + sum(2 * d for d in (1, 2, 4, 8))
        for length in lengths.unique().tolist():
            rows = (lengths == length).nonzero().flatten()
            positions = nodes[rows, None] + torch.arange(1-length, 1, device=x.device)
            windows = x[positions]
            local[rows] = self.backbone.local(windows[:, -receptive_field:])
            stats[rows] = torch.cat((windows.mean(1), windows.std(1, unbiased=False)), 1)
        phase_indices = [nodes]
        for _ in range(3):
            before = previous[phase_indices[-1].clamp_min(0)]
            phase_indices.append(torch.where((phase_indices[-1] >= 0) & (before >= (nodes-self.config['horizon']+1).clamp_min(0)), before, -1))
        indices = torch.stack(phase_indices, 1)
        available = indices >= 0
        packets = x[indices.clamp_min(0)] * available[:, :, None]
        gaps = x.new_zeros(count, 4)
        gaps[:, :3] = self.gap(indices[:, 1:], indices[:, :-1]) * available[:, 1:]
        packed = torch.cat((packets, available[:, :, None].float(), gaps[:, :, None]), 2)
        phase = torch.cat((packed.flatten(1), local, stats), 1)
        earliest = torch.minimum(nodes-lengths+1, indices.masked_fill(~available, torch.iinfo(torch.long).max).min(1).values)
        return local, phase, earliest

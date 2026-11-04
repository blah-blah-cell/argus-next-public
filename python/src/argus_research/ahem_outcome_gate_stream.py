"""Research correctness wrapper; not an admitted/optimized application path."""
import torch

from .ahem_phase_reader import AhemPhaseStream
from .ahem_outcome_gate import gate_inputs


class ObservedPriorStream:
    def __init__(self, reader, gate, capture_id, *, horizon_packets=10_000, emit_warmup=True):
        if gate.training:
            raise ValueError('streaming gate requires frozen eval mode')
        self.stream = AhemPhaseStream(reader, capture_id, horizon_packets=horizon_packets, emit_warmup=emit_warmup)
        self.gate = gate

    def reset(self, capture_id):
        self.stream.reset(capture_id)

    @torch.inference_mode()
    def step(self, packet, flow_generation, ordinal, *, capture_id):
        if self.gate.training:
            raise ValueError('training gate cannot share streaming state')
        out = self.stream.step(packet, flow_generation, ordinal, capture_id=capture_id)
        if out is None:
            return None
        current = packet[None, :2]
        features, raw, available = gate_inputs(out, current, max_transitions=self.stream.model.max_transitions)
        return self.gate(features, current, raw, available)

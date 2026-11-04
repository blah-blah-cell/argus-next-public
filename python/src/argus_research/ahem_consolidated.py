"""One research-only composition of the retained causal reader and readout.

Accepts projected causal state, not targets. Raw-packet feature extraction and
historical cache ownership remain the audited projection/streaming boundary.
This is not the deployed v6.1 ONNX model and does not grant runtime admission.
"""
import torch
from torch import nn
from .ahem_cost_context_readout import CostContextReadout
from .ahem_outcome_gate import ObservedPriorGate
from .ahem_phase_reader import AhemPhaseReader
from .ahem_read_sharpness_constant import ConstantBetaReadout
from .ahem_v6 import AhemV6StateTransitionNextPacket


class ConsolidatedAhem(nn.Module):
    def __init__(self, reader, head, rho):
        super().__init__()
        if type(reader) is not AhemPhaseReader or reader.top_k is not None:
            raise ValueError('exact retained soft phase reader required')
        self.reader = reader
        self.readout = ConstantBetaReadout(head, rho)
        self.requires_grad_(False).eval()

    def forward(self, local_state, query, keys, values, successor_dynamic,
                source_ordinals, valid, anchor_ordinals, context_spans,
                query_phase, key_phase, phase_valid, current):
        output = self.reader._read(local_state, query, keys, values,
            successor_dynamic, source_ordinals, valid, anchor_ordinals,
            context_spans, query_phase=query_phase, key_phase=key_phase,
            phase_valid=phase_valid)
        weights, raw, features, prediction = self.readout(
            output.retrieval_logits, output.valid, output.successor_dynamic,
            current, query_phase)
        return prediction, weights, raw, features

    @classmethod
    def from_components(cls, reader_state, head_state, rho, device='cpu'):
        reader = AhemPhaseReader(AhemV6StateTransitionNextPacket(), phase_width=32)
        reader.load_state_dict(reader_state, strict=True)
        head = CostContextReadout(ObservedPriorGate(), head_state['context_mean'],
                                 head_state['context_scale'])
        head.load_state_dict(head_state, strict=True)
        return cls(reader.to(device), head.to(device), torch.as_tensor(rho).to(device))

    @classmethod
    def from_bundle(cls, path, device='cpu'):
        saved = torch.load(path, weights_only=True, map_location='cpu')
        if (saved['schema'] != 'argus.ahem-consolidated-research/v1'
                or saved['runtime_admission'] is not False):
            raise ValueError('unsupported bundle or runtime claim')
        state = saved['state_dict']
        reader = {k.removeprefix('reader.'): v for k, v in state.items() if k.startswith('reader.')}
        head = {k.removeprefix('readout.head.'): v for k, v in state.items() if k.startswith('readout.head.')}
        model = cls.from_components(reader, head, saved['rho'], device)
        # Recompute beta on the execution device from the frozen rho; a saved
        # CUDA exp buffer differs from CPU exp by one ULP on this candidate.
        state = {k: v.to(device) for k, v in state.items()}
        state['readout.beta'] = model.readout.beta.clone()
        model.load_state_dict(state, strict=True)
        if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
            raise ValueError('non-finite consolidated weights')
        return model

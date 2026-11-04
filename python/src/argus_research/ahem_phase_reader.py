"""Research-only causal phase encoder; frozen v6.1 value/prediction branches."""
from collections import deque
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .ahem_reader_experiment import INPUTS, project_queries
from .ahem_v6 import AhemV6StateTransitionNextPacket
from .ahem_v6_stream import AhemV6Stream

PHASE_INPUTS = ('query_phase', 'key_phase', 'phase_valid')


class AhemPhaseReader(AhemV6StateTransitionNextPacket):
    def __init__(self, backbone, *, phase_packets=4, phase_width=32, top_k=None):
        if type(backbone) is not AhemV6StateTransitionNextPacket:
            raise ValueError('exact frozen v6.1 backbone required')
        if phase_packets != 4 or type(phase_width) is not int or phase_width <= 0:
            raise ValueError('phase pilot requires four packets and positive width')
        if top_k not in (None, 2):
            raise ValueError('only full soft reading or predeclared top-2 supported')
        super().__init__(packet_dim=backbone.packet_dim, output_dim=backbone.output_dim,
                         width=backbone.width, key_dim=backbone.key_dim,
                         local_packets=backbone.local_packets, max_transitions=backbone.max_transitions)
        parameter = next(backbone.parameters())
        self.to(device=parameter.device, dtype=parameter.dtype)
        self.load_state_dict(backbone.state_dict(), strict=True)
        self.requires_grad_(False)
        self.phase_packets, self.top_k = phase_packets, top_k
        self.phase_dim = phase_packets * (self.packet_dim + 2) + self.width + 2 * self.packet_dim
        self.phase_encoder = nn.Sequential(nn.Linear(self.phase_dim, phase_width), nn.GELU(),
                                           nn.Linear(phase_width, self.key_dim))
        self.phase_encoder.to(device=parameter.device, dtype=parameter.dtype)
        nn.init.zeros_(self.phase_encoder[-1].weight)
        nn.init.zeros_(self.phase_encoder[-1].bias)
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        for name, module in self.named_children():
            if name != 'phase_encoder':
                module.eval()
        return self

    def _read(self, local_state, query, keys, values, successor_dynamic, source_ordinals,
              valid, anchor_ordinals, context_spans, *, query_phase, key_phase, phase_valid):
        if (query_phase.shape != (len(query), self.phase_dim)
                or key_phase.shape != (*keys.shape[:2], self.phase_dim)
                or phase_valid.shape != valid.shape or phase_valid.dtype != torch.bool
                or (phase_valid & ~valid).any()):
            raise ValueError('unaligned or illegal causal phase inputs')
        for tensor in (query_phase, key_phase):
            if tensor.device != query.device or tensor.dtype != query.dtype or not torch.isfinite(tensor).all():
                raise ValueError('phase inputs must be finite and share reader dtype/device')
        q = F.normalize(query + self.phase_encoder(query_phase), dim=-1)
        k = F.normalize(keys + self.phase_encoder(key_phase), dim=-1)
        age = self._normalized_gap(source_ordinals, anchor_ordinals[:, None], context_spans)
        scores = self._scores(q, k, age)
        ranking_scores = scores.masked_fill(~phase_valid, -1e4)
        mask = phase_valid
        if self.top_k is not None:
            ranked = scores.masked_fill(~mask, -torch.inf).argsort(dim=-1, descending=True, stable=True)
            keep = torch.zeros_like(mask).scatter_(1, ranked[:, :min(self.top_k, mask.size(1))], True)
            # Preserve candidate availability for the gate/evaluation. Sparsity
            # changes the read, not the denominator or oracle candidate set.
            scores = scores.masked_fill(~keep, -1e4)
        output = super()._read(local_state, q, k, values, successor_dynamic, source_ordinals,
                              mask, anchor_ordinals, context_spans, logits_override=scores)
        # Train ranking over all eligible entries, including entries outside
        # the current top-2. Otherwise truncation would block their gradients.
        return replace(output, retrieval_logits=ranking_scores)

    def research_contract(self):
        return {'schema': 'argus.phase-reader/v1', 'phase_packets': self.phase_packets,
                'phase_dim': self.phase_dim, 'top_k': self.top_k,
                'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'total_parameters': sum(p.numel() for p in self.parameters()),
                'historical_context': 'frozen at source arrival, excludes source successor',
                'global_context_statistics': ['causal_mean', 'causal_population_std'],
                'frozen_local_receptive_field': 31,
                'phase_horizon': 'all raw prefix and local context within current global horizon',
                'eligibility': 'FIFO slots additionally masked if historical phase footprint expired',
                'runtime_admission': False, 'publication_claim': False}


def phase_vector(model, packets, ordinals, local_state, horizon, global_statistics):
    """Newest-first observed prefix only; padding distinguished by a validity bit."""
    if not packets or len(packets) > model.phase_packets or len(packets) != len(ordinals):
        raise ValueError('invalid observed phase prefix')
    if global_statistics.shape != (2 * model.packet_dim,) or not torch.isfinite(global_statistics).all():
        raise ValueError('causal global mean/std required')
    packed = local_state.new_zeros(model.phase_packets, model.packet_dim + 2)
    for i, (packet, ordinal) in enumerate(zip(packets, ordinals, strict=True)):
        packed[i, :model.packet_dim] = packet.to(local_state.device)
        packed[i, model.packet_dim] = 1
        if i + 1 < len(ordinals):
            gap = model._normalized_gap(torch.tensor([ordinals[i + 1]], device=local_state.device),
                                        torch.tensor([ordinal], device=local_state.device),
                                        local_state.new_tensor([float(horizon)]))[0]
            packed[i, model.packet_dim + 1] = gap
    return torch.cat((packed.flatten(), local_state, global_statistics))


@torch.no_grad()
def project_phase_queries(model, record, anchors, *, horizon_packets=10_000, phase_cache=None):
    """Build historical context at its source, never from a later outcome."""
    base = project_queries(model, record, anchors, horizon_packets=horizon_packets)
    x, previous = record.features, record.previous_same_flow
    device = next(model.parameters()).device
    # Optional record-scoped reuse avoids encoding the same source once per
    # queried successor. Cache stores observed context, never learned keys/targets.
    if phase_cache is not None:
        binding = (id(model), x.data_ptr(), x._version, previous.data_ptr(), previous._version, horizon_packets)
        if phase_cache.get('binding') != binding:
            phase_cache.clear()
            phase_cache.update(binding=binding, phases={}, earliest={})
        phases, earliest = phase_cache['phases'], phase_cache['earliest']
    else:
        phases, earliest = {}, {}
    # Cache local representations for distinct source/query ordinals in this batch.
    indices = set(anchors)
    indices.update(int(i) for i in base['source_ordinals'][base['valid']].tolist())
    by_length = {}
    for index in sorted(indices):
        if index in phases:
            continue
        length = min(index + 1, model.local_packets)
        by_length.setdefault(length, []).append(index)
    for length, group in by_length.items():
        windows = torch.stack([x[i - length + 1:i + 1] for i in group]).to(device)
        local = model.local(windows)
        statistics = torch.cat((windows.mean(1), windows.std(1, unbiased=False)), dim=-1)
        for row, index in enumerate(group):
            oldest = max(0, index - horizon_packets + 1)
            chain = [index]
            while len(chain) < model.phase_packets:
                before = int(previous[chain[-1]])
                if before < oldest:
                    break
                chain.append(before)
            phases[index] = phase_vector(model, [x[i] for i in chain], chain, local[row], horizon_packets, statistics[row])
            earliest[index] = min(chain[-1], index - length + 1)
    query_phase = torch.stack([phases[a] for a in anchors])
    key_phase = x.new_zeros(len(anchors), model.max_transitions, model.phase_dim, device=device)
    mask = base['valid'].clone()
    for row, anchor in enumerate(anchors):
        for slot in range(model.max_transitions):
            if base['valid'][row, slot]:
                source = int(base['source_ordinals'][row, slot])
                key_phase[row, slot] = phases[source]
                mask[row, slot] &= earliest[source] >= max(0, anchor - horizon_packets + 1)
    return base | {'query_phase': query_phase, 'key_phase': key_phase, 'phase_valid': mask}


def phase_infer(model, tensors, indices, device):
    ordinary = [tensors[name][indices].to(device) for name in INPUTS]
    if isinstance(model, AhemPhaseReader):
        return model._read(*ordinary, **{name: tensors[name][indices].to(device) for name in PHASE_INPUTS})
    # Information-footprint-matched baseline, not original full-mask result.
    ordinary[6] = tensors['phase_valid'][indices].to(device)
    return model._read(*ordinary)


class AhemPhaseStream(AhemV6Stream):
    """Correctness reference, not an optimized kernel or admitted app path."""
    def __init__(self, model, capture_id, *, horizon_packets=10_000, emit_warmup=False):
        if not isinstance(model, AhemPhaseReader):
            raise ValueError('phase stream requires phase reader')
        self.requested_warmup = emit_warmup
        super().__init__(model, capture_id, horizon_packets=horizon_packets, emit_warmup=True)

    def reset(self, capture_id):
        super().reset(capture_id)
        self.phase_lanes = {}
        self.phase_global_packets = deque(maxlen=self.model.local_packets)

    def step(self, packet, flow_generation, ordinal, *, capture_id):
        # Parent validates packet, ordinal, capture and immutable mode before callback.
        self.active_phase_flow = flow_generation
        result = super().step(packet, flow_generation, ordinal, capture_id=capture_id)
        for flow in list(self.phase_lanes):
            if flow not in self.lanes:
                del self.phase_lanes[flow]
        if self.next_ordinal < self.horizon_packets and not self.requested_warmup:
            return None
        return result

    def _read_packet(self, packet, *inputs):
        local, query, keys, values, successors, ordinals, valid, anchor, span = inputs
        now, flow = int(anchor[0]), self.active_phase_flow
        state = self.phase_lanes.setdefault(flow, {'prefix': deque(maxlen=4), 'pending': None, 'bank': {}})
        oldest = max(0, now - self.horizon_packets + 1)
        while state['prefix'] and state['prefix'][0][0] < oldest:
            state['prefix'].popleft()
        pending = state['pending']
        retained = {int(i) for i in ordinals[valid].tolist()}
        if pending is not None and pending[0] in retained:
            state['bank'][pending[0]] = pending[1:]
        state['bank'] = {i: value for i, value in state['bank'].items() if i in retained}
        state['prefix'].append((now, packet.detach().clone()))
        prefix = list(reversed(state['prefix']))
        self.phase_global_packets.append(packet.detach().clone())
        window = torch.stack(list(self.phase_global_packets))
        statistics = torch.cat((window.mean(0), window.std(0, unbiased=False)))
        context = phase_vector(self.model, [p for _, p in prefix], [i for i, _ in prefix], local[0], self.horizon_packets, statistics)
        first = min(prefix[-1][0], max(0, now - self.model.local_packets + 1))
        state['pending'] = (now, context.detach().clone(), first)
        phase_keys = packet.new_zeros(1, keys.size(1), self.model.phase_dim)
        phase_valid = valid.clone()
        for slot in range(keys.size(1)):
            if valid[0, slot]:
                source = int(ordinals[0, slot])
                if source not in state['bank']:
                    raise ValueError('missing causal source phase')
                value, start = state['bank'][source]
                phase_keys[0, slot] = value
                phase_valid[0, slot] &= start >= oldest
        return self.model._read(*inputs, query_phase=context[None], key_phase=phase_keys,
                                phase_valid=phase_valid)

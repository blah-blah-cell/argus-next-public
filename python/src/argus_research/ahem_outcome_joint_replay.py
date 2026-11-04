"""Real-packet arithmetic verification of a selected joint checkpoint."""
import argparse
import json
from pathlib import Path

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_outcome_gate import ObservedPriorGate, gate_inputs
from .ahem_outcome_joint import forward
from .ahem_phase_reader import AhemPhaseReader, AhemPhaseStream, project_phase_queries
from .ahem_population_audit import sha256, strict_records
from .ahem_reader_experiment import atomic_json
from .flow_v3_protocol import condition_record
from .flow_v6_final_evaluate import _load_frozen_checkpoint


def run(args):
    if args.output.exists():
        raise FileExistsError('immutable replay evidence exists')
    result = json.loads((args.trial / 'RESULT.json').read_text(encoding='utf-8'))
    path = args.trial / 'best.pt'
    if not result['selected_epoch'] or sha256(path) != result['best_sha256']:
        raise ValueError('no verified selected checkpoint')
    state = torch.load(path, weights_only=True, map_location='cpu')
    base, _, _ = _load_frozen_checkpoint(args.checkpoints, state['seed'], AHEM_NAME,
                                       EXPECTED_CHECKPOINT_SHA256[state['seed']][AHEM_NAME])
    reader = AhemPhaseReader(base, phase_width=32).eval()
    reader.load_state_dict(state['reader_state'], strict=True)
    gate = ObservedPriorGate().eval()
    gate.load_state_dict(state['gate_state'], strict=True)
    expected_hash = state['binding']['source_sha256'][args.journal.name]
    if sha256(args.journal) != expected_hash:
        raise ValueError('real replay source digest mismatch')
    record = next(condition_record(r) for r in strict_records(args.journal) if len(r.features) >= 512)
    anchors = [0, 1, 2, 31, 127, 128, 255, 256, 511]
    tensors = project_phase_queries(reader, record, anchors, horizon_packets=256)
    tensors['persistence'] = record.features[anchors, :2]
    floor = state['binding']['plan']['variance_floor']
    stream = AhemPhaseStream(reader, 'real-joint-replay', horizon_packets=256, emit_warmup=True)
    maximum = 0.
    with torch.no_grad():
        expected, _ = forward(reader, gate, tensors, slice(None), 'cpu', floor)
        # Previous links establish identities without consulting any next link.
        identities = []
        for index, packet in enumerate(record.features[:512]):
            prior = int(record.previous_same_flow[index])
            flow = identities[prior] if prior >= 0 else index
            identities.append(flow)
            out = stream.step(packet, flow, index, capture_id='real-joint-replay')
            features, raw, available = gate_inputs(out, packet[None, :2], max_transitions=reader.max_transitions,
                                                  variance_floor=floor)
            prediction = gate(features, packet[None, :2], raw, available)
            if index in anchors:
                row = anchors.index(index)
                difference = float((prediction[0] - expected[row]).abs().max())
                maximum = max(maximum, difference)
                torch.testing.assert_close(prediction[0], expected[row], rtol=1e-5, atol=1e-6)
            assert all(len(lane['bank']) <= 16 and len(lane['prefix']) <= 4 for lane in stream.phase_lanes.values())
        stream.reset('reset-joint-replay')
        assert not stream.phase_lanes and not stream.phase_global_packets
        out = stream.step(record.features[0], 0, 0, capture_id='reset-joint-replay')
        features, raw, available = gate_inputs(out, record.features[None, 0, :2],
            max_transitions=reader.max_transitions, variance_floor=floor)
        cold = gate(features, record.features[None, 0, :2], raw, available)
        torch.testing.assert_close(cold[0], record.features[0, :2], rtol=0, atol=0)
    atomic_json({'checkpoint_sha256': sha256(path), 'source_sha256': sha256(args.journal),
                 'actual_packets': 512, 'anchors': anchors, 'max_log_coordinate_difference': maximum,
                 'shortened_horizon_for_expiry_check': 256, 'variance_floor': floor,
                 'targets_used': False, 'cold_fallback_exact': True, 'reset_verified': True,
                 'role': 'real_packet_arithmetic_not_quality_or_latency', 'runtime_admission': False}, args.output)


def main():
    parser = argparse.ArgumentParser()
    for name in ('trial', 'checkpoints', 'journal', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    torch.set_num_threads(4)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

"""Real-packet correctness replay; no downloads, labels, training or latency claim."""
import argparse
import json
from pathlib import Path

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_outcome_gate import ObservedPriorGate, gate_inputs
from .ahem_outcome_gate_stream import ObservedPriorStream
from .ahem_phase_reader import AhemPhaseReader, phase_infer, project_phase_queries
from .ahem_population_audit import sha256, strict_records
from .ahem_reader_experiment import atomic_json
from .flow_v3_protocol import condition_record
from .flow_v6_final_evaluate import _load_frozen_checkpoint


def run(args):
    if args.output.exists():
        raise FileExistsError('immutable replay check exists')
    torch.set_num_threads(4)
    seed = 20260906
    binding = json.loads((args.gate_run / 'FROZEN.json').read_text(encoding='utf-8'))['selections'][str(seed)]
    gate_path = args.gate_run / str(seed) / 'best.pt'
    prior = json.loads((args.study / 'FROZEN.json').read_text(encoding='utf-8'))['selections'][f'{seed}/phase-soft']
    reader_path = args.study / 'trials' / f'{seed}-phase-soft' / 'best.pt'
    if binding['fallback'] or sha256(gate_path) != binding['best_sha256'] or sha256(reader_path) != prior['checkpoint_sha256']:
        raise ValueError('actual frozen reader/gate required')
    base, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
    reader = AhemPhaseReader(base, phase_width=32).eval()
    reader.load_state_dict(torch.load(reader_path, weights_only=True, map_location='cpu')['model_state'], strict=True)
    gate = ObservedPriorGate().eval()
    gate.load_state_dict(torch.load(gate_path, weights_only=True, map_location='cpu')['model_state'], strict=True)
    record = condition_record(next(r for r in strict_records(args.journal) if len(r.features) >= 512))
    anchors, horizon = [0, 1, 2, 31, 127, 128, 255, 256, 511], 256
    inputs = project_phase_queries(reader, record, anchors, horizon_packets=horizon)
    with torch.inference_mode():
        out = phase_infer(reader, inputs, slice(None), 'cpu')
        features, raw, available = gate_inputs(out, record.features[anchors, :2], max_transitions=16)
        # The compact-stream fix is bit-exact to the emitted training feature
        # on full-capacity tensors, so selected checkpoint semantics stay fixed.
        torch.testing.assert_close(features[:, -1], out.valid.float().mean(-1), rtol=0, atol=0)
        expected = gate(features, record.features[anchors, :2], raw, available)
    stream = ObservedPriorStream(reader, gate, 'real-replay-check', horizon_packets=horizon)
    maximum_error = 0.
    for index in range(512):
        actual = stream.step(record.features[index], int(record.flow_ids[index]), index, capture_id='real-replay-check')
        if index in anchors:
            difference = float((actual[0] - expected[anchors.index(index)]).abs().max())
            maximum_error = max(maximum_error, difference)
            torch.testing.assert_close(actual[0], expected[anchors.index(index)], rtol=1e-5, atol=1e-6)
        assert all(len(lane['bank']) <= 16 and len(lane['prefix']) <= 4 for lane in stream.stream.phase_lanes.values())
    stream.reset('fresh')
    assert not stream.stream.phase_lanes and not stream.stream.phase_global_packets and not stream.stream.lanes
    prediction = stream.step(record.features[0], int(record.flow_ids[0]), 0, capture_id='fresh')
    torch.testing.assert_close(prediction, record.features[:1, :2], rtol=0, atol=0)
    atomic_json({'role': 'real_packet_arithmetic_correctness_only', 'packets': 512, 'anchors': anchors,
                 'test_horizon_packets': horizon, 'test_horizon_not_a_replacement_for_10000_contract': True,
                 'max_forecast_difference': maximum_error, 'reset_verified': True,
                 'compact_denominator_verified': True, 'full_capacity_feature_bit_exact': True,
                 'journal_sha256': sha256(args.journal), 'reader_sha256': sha256(reader_path),
                 'gate_sha256': sha256(gate_path), 'targets_used': False,
                 'runtime_admission': False, 'latency_claim': False}, args.output)
    print(json.dumps({'stage': 'verified', 'max_difference': maximum_error}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('journal', 'study', 'gate-run', 'checkpoints', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

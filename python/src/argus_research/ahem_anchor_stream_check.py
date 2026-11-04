"""Replay an existing real capture through a restored research reader on CPU."""
import argparse
import json
from pathlib import Path
import time

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_population_audit import sha256, strict_records
from .ahem_reader_experiment import INPUTS, atomic_json, project_queries
from .ahem_top1_reader import AhemAnchorReader, AhemAnchorStream, AhemConvexForecaster
from .flow_v3_protocol import condition_record
from .flow_v6_final_evaluate import _load_frozen_checkpoint


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'checkpoints', 'journal', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('immutable stream check exists')
    if not args.journal.name.startswith('2026-') or not args.journal.name.endswith('.v2.arguslr'):
        raise ValueError('only local engineering journal accepted')
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    original_reader = state.get('schema') == 'argus.convex-calibrated-reader/v1' and state['binding'].get('original_reader_preserved') is True
    if not original_reader and state['binding'].get('variant') != 'anchor-top1':
        raise ValueError('only declared anchor reader supported')
    seed = state['binding']['seed']
    base, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
    reader = (AhemConvexForecaster(base) if original_reader else AhemAnchorReader(base)).eval()
    reader.load_state_dict(state['model_state'], strict=True)
    for name, value in base.state_dict().items():
        if not torch.equal(reader.state_dict()[name], value):
            raise ValueError('frozen backbone mismatch')
    records = [condition_record(r) for r in strict_records(args.journal)]
    record = max(records, key=lambda r: len(r.features))
    if len(record.features) <= 10000:
        raise ValueError('real replay must cross the 10000-packet horizon')
    anchors = sorted(set([0, 1, 64, 128, 512, 4096, 9999, 10000, len(record.features) - 1]))
    inputs = project_queries(base, record, anchors)
    with torch.no_grad():
        expected = reader._read(*(inputs[name] for name in INPUTS), causal_anchor=record.features[anchors, :2])
    stream = AhemAnchorStream(reader, 'engineering-stream', emit_warmup=True)
    prediction_error, weight_error = 0., 0.
    start = time.perf_counter()
    for index, packet in enumerate(record.features):
        output = stream.step(packet, int(record.flow_ids[index]), index, capture_id='engineering-stream')
        if index in anchors:
            row = anchors.index(index)
            torch.testing.assert_close(output.prediction[0], expected.prediction[row], rtol=2e-5, atol=2e-6)
            prediction_error = max(prediction_error, float((output.prediction[0] - expected.prediction[row]).abs().max()))
            n = int(output.valid.sum())
            if n:
                torch.testing.assert_close(output.retrieval_weights[0, :n], expected.retrieval_weights[row, :n], rtol=2e-5, atol=2e-6)
                weight_error = max(weight_error, float((output.retrieval_weights[0, :n] - expected.retrieval_weights[row, :n]).abs().max()))
            if int(inputs['valid'][row].sum()) != n:
                raise ValueError('stream and projection disagree on admitted transitions')
    elapsed = time.perf_counter() - start
    telemetry = stream.telemetry()
    if telemetry['stored_transitions'] > telemetry['retained_flows'] * reader.max_transitions:
        raise ValueError('unbounded transition state')
    stream.reset('next-capture')
    output = stream.step(record.features[0], int(record.flow_ids[0]), 0, capture_id='next-capture')
    torch.testing.assert_close(output.prediction[0], expected.prediction[0], rtol=2e-5, atol=2e-6)
    if output.valid.any():
        raise ValueError('previous capture memory leaked into reset')
    result = {'role': 'real_local_stream_engineering_only', 'journal_sha256': sha256(args.journal),
              'checkpoint_sha256': sha256(args.checkpoint), 'checked_ordinals': anchors,
              'packets': len(record.features), 'stream_projection_parity': True,
              'max_prediction_absolute_error': prediction_error, 'max_weight_absolute_error': weight_error,
              'reset_parity': True, 'telemetry': telemetry, 'elapsed_seconds_including_checks': elapsed,
              'publication_latency_claim': False, 'original_locked_test_opened': False}
    atomic_json(result, args.output)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

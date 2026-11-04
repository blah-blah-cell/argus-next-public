"""Real existing local-capture engineering check; not the official experiment."""
import argparse
import json
from pathlib import Path
import time

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_phase_experiment import prepare_cache, fit_phase
from .ahem_phase_reader import AhemPhaseReader, AhemPhaseStream, phase_infer, project_phase_queries
from .ahem_population_audit import sha256, strict_records
from .ahem_reader_experiment import atomic_json
from .flow_v3_protocol import condition_record
from .flow_v6_final_evaluate import _load_frozen_checkpoint


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'checkpoints', 'journals', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('immutable phase engineering preflight exists')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    seed = 20260906
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    small = plan | {'epochs': 2, 'patience': 2, 'batch': 64, 'projection_batch': 16}
    sources = {'train': args.journals / '2026-09-12T16.v2.arguslr',
               'validation': args.journals / '2026-09-15T08.v2.arguslr'}
    binding = {'seed': seed, 'role': 'local_mechanical_real_capture_preflight_not_publication',
               'original_plan_sha256': sha256(args.plan), 'settings': small,
               'source_sha256': {role: sha256(path) for role, path in sources.items()},
               'synthetic_training': False, 'new_dataset_download': False}
    atomic_json(binding, args.output / 'PREFLIGHT.json')
    base, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
    base = base.to(args.device).eval()
    caches = {role: prepare_cache(base, path, role, seed, small, sha256(args.plan),
                                  args.output / (role + '.pt'), official=False,
                                  samples=256 if role == 'train' else 128)
              for role, path in sources.items()}
    results = {}
    for variant, top_k in plan['variants'].items():
        torch.manual_seed(seed)
        candidate = AhemPhaseReader(base, phase_width=plan['phase_width'], top_k=top_k)
        results[variant] = fit_phase(base, candidate, caches, small, args.output / variant,
                                     binding | {'variant': variant}, args.device)
    # Exercise restored nonzero trained weights against an entire real segment.
    state = torch.load(args.output / 'phase-soft/latest.pt', weights_only=True, map_location='cpu')
    base = base.cpu().eval()
    model = AhemPhaseReader(base, phase_width=plan['phase_width'])
    model.load_state_dict(state['model_state'], strict=True)
    if not all(torch.isfinite(v).all() for v in state['model_state'].values()):
        raise ValueError('nonfinite restored phase weights')
    record = max((condition_record(r) for r in strict_records(sources['validation'])), key=lambda r: len(r.features))
    anchors = sorted({0, 1, 128, len(record.features) - 1, min(9999, len(record.features) - 1), min(10000, len(record.features) - 1)})
    torch.set_num_threads(1)
    projected = project_phase_queries(model, record, anchors)
    with torch.no_grad():
        expected = phase_infer(model, projected, slice(None), 'cpu')
    stream = AhemPhaseStream(model, 'real-local-preflight', emit_warmup=True)
    # Recurrence links define opaque flow clusters; no endpoint identities enter model.
    identities = []
    for i, before in enumerate(record.previous_same_flow.tolist()):
        identities.append(i if before < 0 else identities[before])
    start, prediction_error, weight_error = time.perf_counter(), 0., 0.
    for i, packet in enumerate(record.features):
        out = stream.step(packet, identities[i], i, capture_id='real-local-preflight')
        if i in anchors:
            row = anchors.index(i)
            p = float((out.prediction[0] - expected.prediction[row]).abs().max())
            w = float((out.retrieval_weights[0] - expected.retrieval_weights[row, :out.retrieval_weights.size(1)]).abs().max())
            prediction_error, weight_error = max(prediction_error, p), max(weight_error, w)
            torch.testing.assert_close(out.prediction[0], expected.prediction[row], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(out.retrieval_weights[0], expected.retrieval_weights[row, :out.retrieval_weights.size(1)], rtol=1e-5, atol=1e-6)
    elapsed = time.perf_counter() - start
    stream.reset('fresh')
    if stream.phase_lanes:
        raise ValueError('phase state survived capture reset')
    result = {'trials': results, 'stream_packets': len(record.features), 'checked_anchors': anchors,
              'prediction_max_error': prediction_error, 'weight_max_error': weight_error,
              'correctness_replay_seconds_not_kernel_latency': elapsed,
              'restored_checkpoint_sha256': sha256(args.output / 'phase-soft/latest.pt'),
              'runtime_admission': False, 'publication_claim': False}
    atomic_json(result, args.output / 'RESULT.json')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

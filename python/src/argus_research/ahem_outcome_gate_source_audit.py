"""Read completed temporary projections to break the fixed assessment by date.

No training, selections, or primary-run files are changed. Stop after 15 minutes.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_forecast_audit import forecast_quality
from .ahem_outcome_gate import ObservedPriorGate
from .ahem_outcome_gate_study import project_gate, evaluate
from .ahem_phase_reader import AhemPhaseReader
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json
from .flow_v6_final_evaluate import _load_frozen_checkpoint


def run(args):
    if args.output.exists():
        raise FileExistsError('immutable source audit exists')
    run = json.loads((args.run / 'RUN.json').read_text(encoding='utf-8'))
    prior = json.loads((args.study / 'FROZEN.json').read_text(encoding='utf-8'))
    plan = run['plan']
    pending = set(plan['seeds'])
    report = {}
    deadline = time.monotonic() + 900
    while pending and time.monotonic() < deadline:
        frozen_path = args.run / 'FROZEN.json'
        if not frozen_path.exists():
            time.sleep(.5)
            continue
        frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
        for seed in sorted(pending):
            paths = [args.run / 'temporary-projections' / f'{seed}-assessment-{i}.pt'
                     for i in range(len(plan['sources']['assessment']))]
            if not all(path.exists() and path.with_suffix('.manifest.json').exists() for path in paths):
                continue
            try:
                caches = []
                for index, path in enumerate(paths):
                    manifest = json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
                    if (sha256(path) != manifest['cache_sha256'] or manifest['seed'] != seed
                            or manifest['source'] != plan['sources']['assessment'][index]
                            or manifest['plan_sha256'] != run['plan_sha256']):
                        raise ValueError('source audit cache binding mismatch')
                    caches.append(torch.load(path, weights_only=True, map_location='cpu'))
            except FileNotFoundError:
                continue  # Primary cleanup is authoritative; never interrupt it.
            base, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
            reader_path = args.study / 'trials' / f'{seed}-phase-soft' / 'best.pt'
            if sha256(reader_path) != prior['selections'][f'{seed}/phase-soft']['checkpoint_sha256']:
                raise ValueError('source audit reader changed')
            reader = AhemPhaseReader(base, phase_width=plan['phase_width'])
            reader.load_state_dict(torch.load(reader_path, weights_only=True, map_location='cpu')['model_state'], strict=True)
            reader.eval()
            selection = frozen['selections'][str(seed)]
            gate = None
            if not selection['fallback']:
                path = args.run / str(seed) / 'best.pt'
                if sha256(path) != selection['best_sha256']:
                    raise ValueError('source audit gate changed')
                gate = ObservedPriorGate().eval()
                gate.load_state_dict(torch.load(path, weights_only=True, map_location='cpu')['model_state'], strict=True)
            rows = {}
            for source, cache in zip(plan['sources']['assessment'], caches, strict=True):
                data = project_gate(reader, cache['tensors'], plan['batch'], 'cpu')
                persistence = forecast_quality(data['current'].numpy(), data['target'].numpy())
                rows[source] = {'persistence': persistence,
                                'selected_gate': evaluate(gate, data, 'cpu') if gate is not None else persistence | {'baseline_fallback': True}}
            report[str(seed)] = rows
            pending.remove(seed)
            atomic_json({'role': 'same_fixed_assessment_source_breakdown', 'device': 'cpu',
                         'frozen_sha256': sha256(frozen_path), 'report': report,
                         'pending_seeds': sorted(pending), 'training': False}, args.output)
            print(json.dumps({'stage': 'source_audit_complete', 'seed': seed}), flush=True)
        if pending and (args.run / 'CLEANUP.json').exists():
            raise RuntimeError('primary cleaned projections before remaining source audit; no results fabricated')
        if pending:
            time.sleep(.5)
    if pending:
        raise TimeoutError('source audit wait limit reached')


def main():
    parser = argparse.ArgumentParser()
    for name in ('run', 'study', 'checkpoints', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    torch.set_num_threads(4)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

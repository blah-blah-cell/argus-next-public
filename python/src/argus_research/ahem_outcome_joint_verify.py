"""Verify selected development artifacts without reopening traffic assessments."""
import argparse
import json
from pathlib import Path

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_outcome_gate import ObservedPriorGate
from .ahem_outcome_joint import eligible
from .ahem_phase_reader import AhemPhaseReader
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json
from .flow_v6_final_evaluate import _load_frozen_checkpoint


def run(args):
    if args.output.exists():
        raise FileExistsError('immutable verification summary exists')
    frozen = json.loads((args.study / 'FROZEN.json').read_text(encoding='utf-8'))
    binding = json.loads((args.study / 'RUN.json').read_text(encoding='utf-8'))
    if frozen['assessment_opened'] or binding['assessment_opened']:
        raise ValueError('unexpected assessment access')
    for name, digest in binding['source_sha256'].items():
        if sha256(args.journals / name) != digest:
            raise ValueError('source digest changed')
    for name, digest in binding['code_sha256'].items():
        if sha256(Path(__file__).parent / name) != digest:
            raise ValueError('training source changed after binding')
    cleanup = json.loads((args.study / 'CLEANUP.json').read_text(encoding='utf-8'))
    if len(cleanup['removed']) != 15 or list((args.study / 'temporary-projections').glob('*.pt')):
        raise ValueError('expected temporary projection cleanup missing')
    rows = {}
    for key, selection in frozen['results'].items():
        seed, arm = key.split('/')
        trial = args.study / f'{seed}-{arm}'
        provenance = json.loads((trial / 'RUN.json').read_text(encoding='utf-8'))
        if not selection['selected_epoch']:
            rows[key] = {'baseline_retained': True, 'initial_validation': provenance['initial_validation']}
            continue
        path = trial / 'best.pt'
        if sha256(path) != selection['best_sha256']:
            raise ValueError('selected artifact digest changed')
        state = torch.load(path, weights_only=True, map_location='cpu')
        if state['seed'] != int(seed) or state['arm'] != arm or state['epoch'] != selection['selected_epoch']:
            raise ValueError('selected artifact metadata mismatch')
        base, _, _ = _load_frozen_checkpoint(args.checkpoints, int(seed), AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[int(seed)][AHEM_NAME])
        reader = AhemPhaseReader(base, phase_width=32)
        reader.load_state_dict(state['reader_state'], strict=True)
        gate = ObservedPriorGate()
        gate.load_state_dict(state['gate_state'], strict=True)
        if any(not torch.isfinite(v).all() for k in ('reader_state', 'gate_state') for v in state[k].values()):
            raise ValueError('nonfinite selected weights')
        if any(not torch.equal(state['reader_state'][name], value) for name, value in base.state_dict().items()):
            raise ValueError('original backbone changed')
        controls = provenance['controls']
        reference = min((controls['persistence'], controls['ewma'], controls['fixed_blend']['validation']),
                        key=lambda m: m['weighted_smooth_l1'])
        if not eligible(state['validation'], reference, provenance['initial_validation'], float('inf')):
            raise ValueError('selected checkpoint does not clear recorded guard')
        m = state['validation']['forecast']
        if abs(m['weighted_smooth_l1'] - (m['size_smooth_l1'] + 4 * m['delay_smooth_l1']) / 5) > 1e-12:
            raise ValueError('weighted metric arithmetic mismatch')
        rows[key] = {'selected_epoch': state['epoch'], 'checkpoint_sha256': sha256(path),
                     'trainable_parameters': provenance['trainable_parameters'],
                     'retained_total_parameters': sum(p.numel() for p in reader.parameters()) + sum(p.numel() for p in gate.parameters()),
                     'initial_validation': provenance['initial_validation'], 'selected_validation': state['validation'],
                     'strongest_validation_control': reference, 'baseline_retained': False}
    atomic_json({'rows': rows, 'assessment_opened': False, 'role': 'training_validation_only_development',
                 'selected_checkpoints_verified': sum(not r['baseline_retained'] for r in rows.values()),
                 'temporary_projections_deleted': 15, 'sources_unchanged': True,
                 'training_code_unchanged': True, 'runtime_admission': False, 'publication_claim': False}, args.output)


def main():
    parser = argparse.ArgumentParser()
    for name in ('study', 'journals', 'checkpoints', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    torch.set_num_threads(4)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

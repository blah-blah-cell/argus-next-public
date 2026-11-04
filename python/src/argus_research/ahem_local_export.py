"""Export verified local research artifacts; never admit them into the app."""
import argparse
import json
from pathlib import Path
import shutil

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json
from .ahem_top1_reader import AhemAnchorReader, AhemConvexForecaster
from .flow_v6_final_evaluate import _load_frozen_checkpoint


def run(args):
    if args.output.exists():
        raise FileExistsError('immutable local export exists')
    trial = json.loads((args.study / 'RESULT.json').read_text())
    calibration = json.loads((args.calibration / 'RESULT.json').read_text())
    entries = []
    for seed in (20260906, 20260907, 20260908):
        base, _, _ = _load_frozen_checkpoint(args.checkpoints, seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        for variant, source, constructor, expected in (
            ('original-reader-convex', args.calibration / str(seed) / 'calibrated.pt', AhemConvexForecaster,
             calibration['records'][str(seed)]['checkpoint_sha256']),
            ('experimental-anchor-reader', args.study / 'runs' / f'{seed}-anchor-top1' / 'best.pt', AhemAnchorReader,
             trial['frozen'][f'{seed}/anchor-top1']['checkpoint_sha256']),
        ):
            if sha256(source) != expected:
                raise ValueError('final artifact hash mismatch')
            state = torch.load(source, weights_only=True, map_location='cpu')
            model = constructor(base).eval()
            model.load_state_dict(state['model_state'], strict=True)
            if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
                raise ValueError('nonfinite artifact')
            for name, value in base.state_dict().items():
                if not torch.equal(model.state_dict()[name], value):
                    raise ValueError('backbone modified')
            if variant == 'original-reader-convex' and state['binding'].get('original_reader_preserved') is not True:
                raise ValueError('forecaster provenance does not preserve original reader')
            entries.append((source, f'{seed}/{variant}.pt', expected, model.research_contract()))
    args.output.mkdir(parents=True)
    files = {}
    contracts = {}
    for source, name, expected, contract in entries:
        destination = args.output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.pt.partial')
        shutil.copyfile(source, temporary)
        if sha256(temporary) != expected:
            raise ValueError('copied artifact hash mismatch')
        temporary.replace(destination)
        files[name], contracts[name] = expected, contract
    for label, directory, names in (
        ('trial', args.study, ('STUDY.json', 'FROZEN.json', 'ACCESS.json', 'RESULT.json')),
        ('calibration', args.calibration, ('RESULT.json',)),
    ):
        for name in names:
            destination = args.output / label / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix('.json.partial')
            shutil.copyfile(directory / name, temporary)
            if sha256(temporary) != sha256(directory / name):
                raise ValueError('copied provenance hash mismatch')
            temporary.replace(destination)
            files[f'{label}/{name}'] = sha256(destination)
    atomic_json({'schema': 'argus.local-research-export/v1', 'files': files, 'contracts': contracts,
                 'publication_claim': False, 'runtime_admission': False,
                 'default_app_models_changed': False, 'contains_raw_packet_captures': False,
                 'strict_load_and_finite_weights_verified': True, 'original_backbones_unchanged': True,
                 'role': 'private_local_research_staging_only'}, args.output / 'MANIFEST.json')
    print(json.dumps({'output': str(args.output), 'verified_model_files': len(entries),
                      'runtime_admission': False, 'manifest_sha256': sha256(args.output / 'MANIFEST.json')}))


def main():
    parser = argparse.ArgumentParser()
    for name in ('study', 'calibration', 'checkpoints', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

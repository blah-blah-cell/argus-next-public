"""Official-source preparation and the fixed nine-trial outcome development run."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import torch

from .ahem_bridge_events import AHEM_NAME, EXPECTED_CHECKPOINT_SHA256
from .ahem_outcome_gate import ObservedPriorGate
from .ahem_outcome_joint import ARMS, fit
from .ahem_phase_experiment import prepare_cache, load_cache, validate_plan
from .ahem_phase_reader import AhemPhaseReader
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json
from .cse_pcap_to_journal import convert_archive
from .flow_v6_final_evaluate import _load_frozen_checkpoint


def completed_archive(root, day):
    raw = root / 'raw'
    pid_file = raw / (day + '.download.pid')
    while not pid_file.exists():
        time.sleep(2)
    pid = int(pid_file.read_text().strip())
    while True:
        process = Path('/proc') / str(pid) / 'cmdline'
        if not process.exists() or b'aws' not in process.read_bytes():
            break
        time.sleep(5)
    temporary = raw / (day + '.zip.partial')
    metadata = json.loads((raw / (day + '.head.json')).read_text())
    if not temporary.exists() or temporary.stat().st_size != metadata['ContentLength'] or not zipfile.is_zipfile(temporary):
        raise ValueError(f'official download incomplete or invalid: {day}; inspect download log')
    digest = sha256(temporary)
    final = raw / (day + '.zip')
    if final.exists():
        raise FileExistsError('refusing to replace existing archive')
    temporary.replace(final)
    provenance = {'uri': f's3://cse-cic-ids2018/Original Network Traffic and Log data/{day}/pcap.zip',
                  'sha256': digest, 'bytes': final.stat().st_size, 's3_metadata': metadata,
                  'synthetic_data': False}
    atomic_json(provenance, final.with_suffix('.source.json'))
    return final, provenance


def run(args):
    if sys.platform == 'win32' or not torch.cuda.is_available():
        raise RuntimeError('Linux GPU VM only; no local acquisition')
    plan = json.loads(args.plan.read_text())
    sources = json.loads(args.sources.read_text())
    validate_plan(sources)
    if (plan['schema'] != 'argus.outcome-vm-development/v1' or plan['arms'] != list(ARMS)
            or plan['seeds'] != sources['seeds'] or plan['assessment_access'] is not False
            or plan['source_plan_sha256'] != sha256(args.sources)):
        raise ValueError('VM protocol/source binding mismatch')
    args.root.mkdir(parents=True, exist_ok=True)
    marker = args.root / 'ACTIVE.json'
    with marker.open('x') as stream:
        json.dump({'pid': os.getpid(), 'plan_sha256': sha256(args.plan)}, stream)
    binding = {'plan': plan, 'plan_sha256': sha256(args.plan), 'sources_sha256': sha256(args.sources),
               'source_sha256': sources['source_sha256'], 'assessment_opened': False,
               'torch_version': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
               'role': 'official_previously_used_development_sources_not_confirmatory',
               'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    atomic_json(binding, args.root / 'RUN.json')
    def project(role, journal):
        for seed in sources['seeds']:
            base, _, _ = _load_frozen_checkpoint(args.package / 'checkpoints', seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
            base = base.to('cuda').eval()
            path = args.root / 'cache' / str(seed) / (role + '.pt')
            prepare_cache(base, journal, role, seed, sources, sha256(args.sources), path)
            del base
    def convert(role, filename, gpu):
        day = filename.removesuffix('.arguslr')
        archive, provenance = completed_archive(args.root, day)
        journal = args.root / 'journals' / filename
        journal.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps({'stage': 'convert', 'role': role, 'archive_bytes': archive.stat().st_size}), flush=True)
        convert_archive(archive, journal, segment_packets=sources['segment_packets'],
                        max_packets=sources['max_packets_per_source'], archive_sha256=provenance['sha256'], source_uri=provenance['uri'])
        if sha256(journal) != sources['source_sha256'][role]:
            raise ValueError('converted journal differs from frozen original prefix')
        return gpu.submit(project, role, journal)
    with ThreadPoolExecutor(max_workers=1) as gpu, ThreadPoolExecutor(max_workers=1) as cpu:
        tasks = [cpu.submit(convert, role, filename, gpu) for role, filename in sources['sources'].items()]
        for task in tasks:
            task.result().result()
    atomic_json({'cache_sha256': {f'{s}/{r}': sha256(args.root / 'cache' / str(s) / (r + '.pt'))
                  for s in sources['seeds'] for r in ('train', 'validation')}, 'assessment_opened': False}, args.root / 'PREPARED.json')
    selections = {}
    for seed in sources['seeds']:
        data = {role: load_cache(args.root / 'cache' / str(seed) / (role + '.pt'), role, seed, sha256(args.sources), sources)['tensors']
                for role in ('train', 'validation')}
        base, _, _ = _load_frozen_checkpoint(args.package / 'checkpoints', seed, AHEM_NAME, EXPECTED_CHECKPOINT_SHA256[seed][AHEM_NAME])
        base = base.to('cuda').eval()
        parent_reader = args.package / 'parents' / str(seed) / 'reader.pt'
        parent_gate = args.package / 'parents' / str(seed) / 'gate.pt'
        parents = json.loads((args.package / 'PARENTS.json').read_text())[str(seed)]
        if sha256(parent_reader) != parents['reader'] or sha256(parent_gate) != parents['gate']:
            raise ValueError('parent digest mismatch')
        reader = AhemPhaseReader(base, phase_width=sources['phase_width']).to('cuda')
        reader.load_state_dict(torch.load(parent_reader, weights_only=True, map_location='cuda')['model_state'], strict=True)
        gate = ObservedPriorGate().to('cuda')
        gate.load_state_dict(torch.load(parent_gate, weights_only=True, map_location='cuda')['model_state'], strict=True)
        for arm in ARMS:
            selections[f'{seed}/{arm}'] = fit(copy.deepcopy(reader), copy.deepcopy(gate), data, plan, arm, seed,
                args.root / 'results' / f'{seed}-{arm}', binding | {'seed': seed, 'parent_sha256': parents}, 'cuda')
        del data, base, reader, gate
        torch.cuda.empty_cache()
    atomic_json({'results': selections, 'assessment_opened': False, 'runtime_admission': False}, args.root / 'FROZEN.json')
    marker.unlink()
    print(json.dumps({'stage': 'complete', 'results': selections}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'package', 'plan', 'sources'):
        parser.add_argument('--' + name, type=Path, required=True)
    torch.set_num_threads(4)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

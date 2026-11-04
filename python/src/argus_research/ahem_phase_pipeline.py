"""Linux-only acquisition/preparation with one GPU worker and bounded training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from .ahem_phase_experiment import validate_plan, load_cache
from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json
from .ahem_reader_pipeline import acquire
from .cse_pcap_to_journal import convert_archive


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'plan', 'checkpoints'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--journals-root', type=Path)
    args = parser.parse_args()
    if sys.platform == 'win32' or not torch.cuda.is_available():
        raise RuntimeError('phase pipeline requires Linux GPU VM; no local dataset acquisition')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    validate_plan(plan)
    digest = sha256(args.plan)
    args.root.mkdir(parents=True, exist_ok=True)
    # Exclusive run marker is intentionally retained on failure. Inspect its PID
    # before any manual recovery; never delete it to restart a healthy process.
    marker = args.root / 'ACTIVE.json'
    with marker.open('x', encoding='utf-8') as handle:
        json.dump({'pid': os.getpid(), 'plan_sha256': digest}, handle)
    def launch(arguments):
        subprocess.run([sys.executable, '-m', *arguments], check=True)
    def project(role, journal):
        for seed in plan['seeds']:
            path = args.root / 'cache' / str(seed) / (role + '.pt')
            if path.exists():
                load_cache(path, role, seed, digest, plan)
            else:
                launch(['argus_research.ahem_phase_experiment', 'prepare', '--plan', str(args.plan),
                        '--checkpoints', str(args.checkpoints), '--journal', str(journal), '--role', role,
                        '--seed', str(seed), '--output', str(path), '--device', 'cuda'])
    def convert(role, day, archive, provenance, gpu_pool):
        journal = args.root / 'journals' / (day + '.arguslr')
        journal.parent.mkdir(parents=True, exist_ok=True)
        convert_archive(archive, journal, segment_packets=plan['segment_packets'],
                        max_packets=plan['max_packets_per_source'], archive_sha256=provenance['sha256'],
                        source_uri=provenance['uri'])
        return gpu_pool.submit(project, role, journal)
    pending = []
    with ThreadPoolExecutor(max_workers=1) as gpu, ThreadPoolExecutor(max_workers=1) as cpu:
        for role, filename in plan['sources'].items():
            supplied = args.journals_root / filename if args.journals_root else None
            existing = supplied if supplied and supplied.exists() else args.root / 'journals' / filename
            if existing.exists():
                pending.append(cpu.submit(gpu.submit, project, role, existing))
            else:
                day = filename.removesuffix('.arguslr')
                archive, provenance = acquire(args.root, day)
                pending.append(cpu.submit(convert, role, day, archive, provenance, gpu))
        for task in pending:
            task.result().result()
    if sha256(args.plan) != digest:
        raise ValueError('plan changed during preparation')
    caches = {f'{seed}/{role}': sha256(args.root / 'cache' / str(seed) / (role + '.pt'))
              for seed in plan['seeds'] for role in ('train', 'validation')}
    atomic_json({'plan_sha256': digest, 'cache_sha256': caches, 'locked_test_opened': False}, args.root / 'PREPARED.json')
    completed = {}
    for seed in plan['seeds']:
        for variant in plan['variants']:
            result = args.root / 'results' / f'{seed}-{variant}'
            # Immutable trials are never automatically resumed/replaced.
            launch(['argus_research.ahem_phase_experiment', 'train', '--plan', str(args.plan),
                    '--checkpoints', str(args.checkpoints), '--train', str(args.root / 'cache' / str(seed) / 'train.pt'),
                    '--validation', str(args.root / 'cache' / str(seed) / 'validation.pt'),
                    '--seed', str(seed), '--variant', variant, '--output', str(result), '--device', 'cuda'])
            completed[f'{seed}/{variant}'] = {'result_sha256': sha256(result / 'RESULT.json'),
                                             'latest_sha256': sha256(result / 'latest.pt'),
                                             'best_sha256': sha256(result / 'best.pt') if (result / 'best.pt').exists() else None}
    atomic_json({'plan_sha256': digest, 'trials': completed, 'locked_test_opened': False,
                 'independent_assessment_started': False}, args.root / 'FROZEN.json')
    marker.unlink()


if __name__ == '__main__':
    main()

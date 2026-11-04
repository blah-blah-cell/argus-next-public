"""VM-only real-data pilot: overlap download, CPU conversion and GPU projection."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .ahem_population_audit import sha256
from .ahem_reader_experiment import atomic_json
from .cse_pcap_to_journal import convert_archive


def run_command(arguments):
    subprocess.run(arguments, check=True)


def acquire(root, day):
    """Anonymous official S3 acquisition on the VM only, atomic at completion."""
    if sys.platform == 'win32':
        raise RuntimeError('dataset acquisition is VM-only; do not download to the laptop')
    key = f'Original Network Traffic and Log data/{day}/pcap.zip'
    aws = ['aws', '--no-sign-request', '--region', 'ca-central-1']
    metadata = json.loads(subprocess.check_output(aws + ['s3api', 'head-object', '--bucket', 'cse-cic-ids2018', '--key', key], text=True))
    destination = root / 'raw' / f'{day}.zip'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        record = json.loads(destination.with_suffix('.source.json').read_text())
        if record['sha256'] != sha256(destination) or destination.stat().st_size != metadata['ContentLength']:
            raise ValueError('existing archive hash/size mismatch')
        return destination, record
    required = metadata['ContentLength'] + 12 * 1024**3
    if shutil.disk_usage(root).free < required:
        raise RuntimeError(f'insufficient disk headroom for {day}: require {required} free bytes')
    temporary = destination.with_suffix('.zip.partial')
    run_command(aws + ['s3', 'cp', '--only-show-errors', f's3://cse-cic-ids2018/{key}', str(temporary)])
    if temporary.stat().st_size != metadata['ContentLength']:
        raise ValueError('downloaded object length mismatch')
    record = {'uri': f's3://cse-cic-ids2018/{key}', 'sha256': sha256(temporary),
              'bytes': metadata['ContentLength'], 's3_etag': metadata.get('ETag'),
              's3_last_modified': metadata.get('LastModified'), 'synthetic_data': False}
    temporary.replace(destination)
    atomic_json(record, destination.with_suffix('.source.json'))
    return destination, record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    args = parser.parse_args()
    import torch
    if sys.platform == 'win32' or not torch.cuda.is_available():
        raise RuntimeError('pipeline requires the Linux GPU VM')
    args.root.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    if plan['schema'] != 'argus.contextual-reader-development/v1' or plan['locked_test_access']:
        raise ValueError('wrong pilot plan')
    run_plan_hash = sha256(args.plan)
    jobs = []

    def convert_and_queue(role, day, archive, provenance):
        journal = args.root / 'journals' / f'{day}.arguslr'
        if journal.exists():
            manifest = json.loads(journal.with_suffix('.arguslr.manifest.json').read_text())
            if manifest['journal']['sha256'] != sha256(journal) or manifest['archive']['sha256'] != provenance['sha256']:
                raise ValueError('existing conversion provenance mismatch')
        else:
            journal.parent.mkdir(parents=True, exist_ok=True)
            convert_archive(archive, journal, segment_packets=plan['segment_packets'],
                            max_packets=plan['max_packets_per_source'], archive_sha256=provenance['sha256'],
                            source_uri=provenance['uri'])
        return projection_pool.submit(project_source, role, journal)

    def project_source(role, journal):
        for seed in plan['seeds']:
            output = args.root / 'cache' / str(seed) / f'{role}.pt'
            if output.exists():
                from .ahem_reader_experiment import read_cache
                read_cache(output, role, seed)
                continue
            run_command([sys.executable, '-m', 'argus_research.ahem_reader_experiment', 'prepare',
                         '--journal', str(journal), '--role', role, '--samples', str(plan['samples'][role]),
                         '--seed', str(seed), '--checkpoints', str(args.checkpoints), '--output', str(output),
                         '--device', 'cuda', '--threads', '4', '--require-official-manifest'])

    # Exactly one conversion worker and one GPU worker. No competing final training.
    with ThreadPoolExecutor(max_workers=1) as projection_pool, ThreadPoolExecutor(max_workers=1) as conversion_pool:
        for role, filename in plan['sources'].items():
            day = filename.removesuffix('.arguslr')
            if day not in {'Wednesday-14-02-2018', 'Thursday-01-03-2018', 'Friday-02-03-2018'}:
                raise ValueError('source is outside this bounded development pilot')
            journal = args.root / 'journals' / filename
            if journal.exists():
                manifest = json.loads(journal.with_suffix('.arguslr.manifest.json').read_text())
                if (manifest['journal']['sha256'] != sha256(journal)
                        or manifest['journal']['segment_packets'] != plan['segment_packets']
                        or manifest['journal']['max_packets'] != plan['max_packets_per_source']):
                    raise ValueError('existing journal differs from planned conversion')
                jobs.append(conversion_pool.submit(projection_pool.submit, project_source, role, journal))
                continue
            print(json.dumps({'stage': 'download', 'source': day}), flush=True)
            archive, provenance = acquire(args.root, day)
            jobs.append(conversion_pool.submit(convert_and_queue, role, day, archive, provenance))
        projection_jobs = [job.result() for job in jobs]
        for job in projection_jobs:
            job.result()
    if sha256(args.plan) != run_plan_hash:
        raise ValueError('plan changed during preparation')
    atomic_json({'plan_sha256': run_plan_hash, 'locked_test_opened': False,
                 'sources': plan['sources'], 'samples': plan['samples']}, args.root / 'PREPARED.json')
    for seed in plan['seeds']:
        run_command([sys.executable, '-m', 'argus_research.ahem_reader_experiment', 'train',
                     '--plan', str(args.plan), '--checkpoints', str(args.checkpoints), '--seed', str(seed),
                     '--train', str(args.root / 'cache' / str(seed) / 'train.pt'),
                     '--validation', str(args.root / 'cache' / str(seed) / 'validation.pt'),
                     '--screen', str(args.root / 'cache' / str(seed) / 'screen.pt'),
                     '--output', str(args.root / 'results' / str(seed)), '--device', 'cuda', '--threads', '4'])


if __name__ == '__main__':
    main()

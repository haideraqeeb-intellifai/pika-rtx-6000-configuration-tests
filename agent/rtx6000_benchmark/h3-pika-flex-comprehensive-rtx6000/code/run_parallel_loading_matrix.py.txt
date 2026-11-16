"""Compare independent-chunk model-copy loading strategies on one idle GPU.

The runner stops on the first failed or unsafe configuration. Each supervisor
calibrates its model placement before admitting concurrent copies and validates
the assembled MP4. Dry-run is the default.
"""

import argparse
import json
import subprocess
import sys

from .parallel_chunk_processes import RUNS


CASES = [
    ('resident_1_default', 'resident', 1, 'default', 'dynamic', 10, 'hard'),
    ('eager_1_default', 'eager', 1, 'default', 'dynamic', 10, 'hard'),
    ('resident_2_default', 'resident', 2, 'default', 'dynamic', 10, 'hard'),
    ('eager_2_default', 'eager', 2, 'default', 'dynamic', 10, 'hard'),
    ('resident_2_cpu_text', 'resident', 2, 'cpu', 'dynamic', 10, 'hard'),
    ('eager_2_cpu_text', 'eager', 2, 'cpu', 'dynamic', 10, 'hard'),
    ('resident_3_cpu_text', 'resident', 3, 'cpu', 'dynamic', 10, 'hard'),
    ('resident_2_highvram', 'resident', 2, 'default', 'highvram', 10, 'hard'),
    ('resident_2_anchor_every2', 'resident', 2, 'default', 'dynamic', 10, 'every-2'),
    ('resident_2_seven_chunks', 'resident', 2, 'default', 'dynamic', 7, 'hard'),
    ('resident_2_fourteen_chunks', 'resident', 2, 'default', 'dynamic', 14, 'hard'),
]


def command(case):
    suffix, loader, copies, clip_device, residency, chunks, anchor = case
    module = ('agent.rtx6000_benchmark.parallel_chunk_resident' if loader == 'resident'
              else 'agent.rtx6000_benchmark.parallel_chunk_processes')
    return [sys.executable, '-m', module, '--name', f'original_image_parallel_{suffix}',
            '--copies', str(copies), '--chunks', str(chunks), '--chunk-plan', 'aligned',
            '--clip-device', clip_device, '--residency', residency,
            '--anchor-mode', anchor]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Execute on the GPU; otherwise print the plan')
    args = parser.parse_args()
    for number, case in enumerate(CASES, 1):
        cmd = command(case)
        name = cmd[cmd.index('--name') + 1]
        result_path = RUNS / name / 'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get('status') == 'success':
                print(f'{number}/{len(CASES)} {name}: existing valid result', flush=True)
                continue
            raise RuntimeError(f'{name} has a failed run; inspect it before continuing')
        print(f'{number}/{len(CASES)} {" ".join(cmd)}', flush=True)
        if args.run:
            with (RUNS / f'{name}.log').open('w') as log:
                subprocess.run([*cmd, '--run'], check=True, stdout=log,
                               stderr=subprocess.STDOUT)


if __name__ == '__main__':
    main()

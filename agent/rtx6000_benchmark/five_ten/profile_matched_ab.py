"""Separate Nsight Systems captures for the matched A/B; never used as timing results."""
import json
import os
import subprocess
import sys
from pathlib import Path

from .analyze_profiles import analyze

P = Path(__file__).resolve().parent


if __name__ == '__main__':
    profiles = P / 'profiles'
    profiles.mkdir(exist_ok=True)
    result = {}
    for label, mode, vaes, control in [('matched_simple', 'simple', 1, 'matched_simple_a2'),
                                       ('matched_parallel', 'sequential', 2, 'matched_parallel_b2')]:
        prefix = profiles / label
        command = ['nsys', 'profile', '--trace=cuda-sw,nvtx,osrt', '--sample=none',
                   '--cpuctxsw=none', '--gpu-metrics-devices=0', '--gpu-metrics-frequency=1000',
                   '--wait=all', '--output', str(prefix), sys.executable,
                   '-m', 'agent.rtx6000_benchmark.five_ten.experiment', '--mode', mode,
                   '--name', 'profile_' + label, '--transformers', '1', '--vaes', str(vaes),
                   '--repeats', '1']
        (profiles / f'{label}_command.json').write_text(json.dumps({
            'command': command, 'note': 'Full five-chunk warmup plus one profiled replay. Runtime excluded from A/B latency.'}, indent=2))
        with (profiles / f'{label}.log').open('w') as log:
            subprocess.run(command, env={**os.environ, 'H3_BATCH_NSYS': '1',
                           'H3_NSYS_GPU_METRICS': '1'}, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=2400)
        subprocess.run(['nsys', 'export', '--type=sqlite', '--output', str(prefix) + '.sqlite',
                        str(prefix) + '.nsys-rep'], check=True, timeout=300)
        result[label] = analyze(Path(str(prefix) + '.sqlite'), control_root=control)
        (P / 'matched_ab_profiles.json').write_text(json.dumps(result, indent=2) + '\n')
        print(label, result[label]['kernel_overlap'], flush=True)

"""Reproduce the winning 20-chunk layout with production fast inference."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .run_chunk_copy_matrix import run
from .parallel_chunk_processes import RUNS, file_hash


def main():
    root = Path(__file__).resolve().parent
    provenance = root / 'fast_comparison_source'
    provenance.mkdir(exist_ok=True)
    for name in ('stage_pipeline.py', 'parallel_chunk_resident.py', 'run_fast_comparison.py',
                 'mps_session.py', 'gpu_preencode.py', 'preencoded_parallel.py'):
        shutil.copy2(root / name, provenance / name)
    for name in ('runtime.py', 'fast_inference.py'):
        shutil.copy2(root.parents[1] / 'ml_modules/minimax_h3_turbo' / name, provenance / name)
    (provenance / 'runtime.patch').write_text(subprocess.check_output(
        ['git', 'diff', '--', 'ml_modules/minimax_h3_turbo/runtime.py'], text=True))
    (provenance / 'provenance.json').write_text(json.dumps({
        'repo_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'sage3_revision': 'd1a57a546c3d395b1ffcbeecc66d81db76f3b4b5',
        'cutlass_revision': '0b55a2f691d69981583568fd9eb69687b1f0de8a',
        'packages': {p: importlib.metadata.version(p) for p in
                     ('torch', 'sageattention', 'sageattn3', 'comfy-kitchen', 'comfy-aimdo')},
        'python': sys.executable, 'overlay': os.environ.get('PYTHONPATH'),
        'source_hashes': {p.name: file_hash(p) for p in provenance.iterdir() if p.is_file()},
        'protocol': 'Same 49.408s podcast, 20 independent hard-anchor chunks, 1T+2V MPS, two steps. '
                    'Fresh processes per video, resident loaders across chunks. GPU preparation included. '
                    'Calibration excluded. Repeated controls and fast runs alternate. '
                    'VAE-only fast validation is stage aware; request-local hook nodes refresh per chunk.'
    }, indent=2))
    (RUNS / 'owned_coordinator_pids.json').write_text(json.dumps([os.getpid()]))
    stop = RUNS / 'followup_monitor.stop'
    stop.unlink(missing_ok=True)
    with (RUNS / 'fast_ownership_monitor.log').open('w') as log:
        monitor = subprocess.Popen([sys.executable, '-m', 'agent.rtx6000_benchmark.watch_gpu_ownership'],
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            run('followup_fast_cal20', ['--chunks', '20', '--inference-mode', 'fast', '--calibrate-only'])
            common = ['--chunks', '20', '--schedule', 'pipeline', '--transformers', '1', '--vaes', '2', '--mps']
            for repeat in (1, 2):
                for mode in ('default', 'fast'):
                    calibration = 'followup_fast_cal20' if mode == 'fast' else 'followup_cal_20'
                    run(f'followup_fast_comparison_{mode}_{repeat}', common + [
                        '--inference-mode', mode, '--reuse-calibration', calibration])
            print('Fast-mode comparison complete', flush=True)
        finally:
            stop.touch()
            monitor.wait(timeout=10)


if __name__ == '__main__':
    main()

"""Run the calibrated VAE stage matrix sequentially on the dedicated GPU."""
import json
import subprocess
import sys
from .parallel_chunk_processes import RUNS
CASES = [(s,1,v) for s in ('staged','pipeline') for v in (1,2,4)] + [('pipeline',2,1),('pipeline',2,2)]
for schedule, transformers, vaes in CASES:
    name = f'vae_{schedule}_{transformers}t_{vaes}v'
    print(f'Starting {name}', flush=True)
    with (RUNS / f'{name}.log').open('w') as log:
        subprocess.run([sys.executable,'-m','agent.rtx6000_benchmark.stage_pipeline',
            '--name',name,'--schedule',schedule,'--transformers',str(transformers),'--vaes',str(vaes),
            '--reuse-calibration','vae_stages_calibration','--run'], stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((RUNS/name/'result.json').read_text())
    print(json.dumps({k:result.get(k) for k in ('name','status','timed_total_seconds','gpu_trace_summary')}),flush=True)

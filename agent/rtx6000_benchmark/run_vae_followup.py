"""Wait for ordinary matrix completion, then run MPS and batch diagnostics."""
import json
import subprocess
import sys
import time
from .parallel_chunk_processes import RUNS


def run(name,command):
    print('Starting '+name,flush=True)
    with (RUNS/f'{name}.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    path=RUNS/name/'result.json'
    if path.exists():
        r=json.loads(path.read_text())
        print(json.dumps({k:r.get(k) for k in ('name','status','timed_total_seconds','decoder_phase_seconds','gpu_trace_summary')}),flush=True)
    else: print('Completed '+name,flush=True)


def main():
    names=[f'vae_{s}_{t}t_{v}v' for s,t,v in [(s,1,v) for s in ('staged','pipeline') for v in (1,2,4)]+[('pipeline',2,1),('pipeline',2,2)]]
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline:
        records=[json.loads((RUNS/n/'result.json').read_text()) for n in names if (RUNS/n/'result.json').exists()]
        if any(r['status']=='failure' for r in records): raise RuntimeError('Ordinary matrix failed; inspect before follow-up')
        if len(records)==len(names) and all(r['status']=='success' for r in records): break
        time.sleep(10)
    else: raise TimeoutError('Ordinary matrix did not finish in two hours')
    py=sys.executable
    if not (RUNS/'vae_mps_smoke/result.json').exists():
        run('vae_mps_smoke',[py,'-m','agent.rtx6000_benchmark.check_mps'])
    elif json.loads((RUNS/'vae_mps_smoke/result.json').read_text())['status']!='success':
        raise RuntimeError('MPS smoke check did not pass')
    if not (RUNS/'vae_profiler_baseline/result.json').exists():
        run('vae_profiler_baseline',['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
            '--wait=all','--output',str(RUNS/'vae_profiler_baseline_trace'),py,'-m',
            'agent.rtx6000_benchmark.profile_vae_pool','--name','vae_profiler_baseline','--baseline-only'])
    run('vae_profile_mps_four_retry',['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
        '--wait=all','--output',str(RUNS/'vae_profile_mps_four_retry_trace'),py,'-m',
        'agent.rtx6000_benchmark.profile_vae_pool','--name','vae_profile_mps_four_retry','--mps','--profiler-baseline','vae_profiler_baseline'])
    for schedule,transformers,vaes in [('staged',1,4),('pipeline',1,1),('pipeline',2,1)]:
        name=f'vae_mps_{schedule}_{transformers}t_{vaes}v'
        run(name,[py,'-m','agent.rtx6000_benchmark.stage_pipeline','--name',name,'--schedule',schedule,
            '--transformers',str(transformers),'--vaes',str(vaes),'--mps','--reuse-calibration','vae_stages_calibration','--run'])
    for size in (1,2,4):
        name=f'vae_batch_{size}'
        run(name,[py,'-m','agent.rtx6000_benchmark.batch_decode','--name',name,
            '--source-run','vae_staged_1t_1v','--batch-size',str(size)])

if __name__=='__main__':main()

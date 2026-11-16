"""Finish with one MPS resource split and repeated full-video controls."""
import json
import subprocess
import sys
import time
from .parallel_chunk_processes import RUNS


def main():
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline:
        path=RUNS/'vae_batch_4/result.json'
        if path.exists():
            result=json.loads(path.read_text())
            if result['status']=='success': break
            if result['status']=='failure': raise RuntimeError('Batch diagnostic failed; inspect before final checks')
        time.sleep(10)
    else: raise TimeoutError('Follow-up tests did not complete')
    py=sys.executable
    def run(name,extra):
        print('Starting '+name,flush=True)
        with (RUNS/f'{name}.log').open('w') as log:
            subprocess.run([py,'-m','agent.rtx6000_benchmark.stage_pipeline','--name',name,
                '--reuse-calibration','vae_stages_calibration','--run',*extra],check=True,stdout=log,stderr=subprocess.STDOUT)
        r=json.loads((RUNS/name/'result.json').read_text())
        print(json.dumps({k:r.get(k) for k in ('name','status','timed_total_seconds','gpu_trace_summary')}),flush=True)
    run('vae_mps_pipeline_1t_1v_prime',['--schedule','pipeline','--transformers','1','--vaes','1','--mps','--prime-longest'])
    run('vae_mps_pipeline_1t_1v_65_35',['--schedule','pipeline','--transformers','1','--vaes','1',
        '--mps','--denoise-percent','65','--decode-percent','35'])
    run('vae_pipeline_1t_1v_repeat2',['--schedule','pipeline','--transformers','1','--vaes','1'])
    candidates=[]
    for p in RUNS.glob('vae_*/result.json'):
        r=json.loads(p.read_text())
        if r.get('loader')=='separate stage processes' and r['status']=='success' and not p.parent.name.endswith('_repeat2'):
            candidates.append(r)
    best=min(candidates,key=lambda r:r['timed_total_seconds'])
    if best['name']!='vae_pipeline_1t_1v':
        extra=['--schedule',best['schedule'],'--transformers',str(best['transformers']),'--vaes',str(best['vaes'])]
        if best.get('prime_longest'): extra+=['--prime-longest']
        if best.get('mps'):
            extra+=['--mps','--denoise-percent',str(best['mps_percentages']['denoise']),
                    '--decode-percent',str(best['mps_percentages']['decode'])]
        run(best['name']+'_repeat2',extra)

if __name__=='__main__':main()

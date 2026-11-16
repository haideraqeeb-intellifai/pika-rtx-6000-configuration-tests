"""Confirm the best one/two-transformer layouts, then screen residency/export."""
import json
import subprocess
import sys
from .run_chunk_copy_matrix import run
from .parallel_chunk_processes import RUNS
from .parallel_chunk_resident import wait_for_empty_gpu

def flags(r):
    return ['--chunks',str(r['chunks']),'--schedule','pipeline','--transformers',str(r['transformers']),
            '--vaes',str(r['vaes'])]

def run_variant(name, options):
    try:
        return run(name, options)
    except subprocess.CalledProcessError:
        # Retain failed results; proceed only if the GPU fully returns to idle.
        wait_for_empty_gpu(max_wait_seconds=60)
        result=json.loads((RUNS/name/'result.json').read_text())
        print(json.dumps({'name':name,'status':result['status'],'error':result.get('error')}),flush=True)
        return result

def main():
    candidates=[]
    for p in RUNS.glob('followup_*/result.json'):
        r=json.loads(p.read_text())
        if r.get('loader')=='separate stage processes' and r['status']=='success' and r.get('residency','dynamic')=='dynamic' and not r.get('async_encoding') and not r.get('mps') and '_repeat' not in r['name'] and (r['name'].startswith('followup_c') or r['name']=='followup_baseline_10'):
            candidates.append(r)
    selected=[]
    for count in (1,2):
        choices=[r for r in candidates if r['transformers']==count]
        if not choices: continue
        best=min(choices,key=lambda r:r['time_to_assembled_valid_video_seconds'])
        cal='followup_baseline_10' if best['chunks']==10 else f'followup_cal_{best["chunks"]}'
        repeated=run(best['name']+'_repeat2',flags(best)+['--reuse-calibration',cal])
        selected.append((best,repeated))
    best,repeat=min(selected,key=lambda pair:sum(r['time_to_assembled_valid_video_seconds'] for r in pair)/2)
    # Test residency with new per-stage calibration; do not reuse dynamic peaks.
    resident=run_variant('followup_best_resident',flags(best)+['--residency','highvram'])
    cal='followup_baseline_10' if best['chunks']==10 else f'followup_cal_{best["chunks"]}'
    background=run_variant('followup_best_async',flags(best)+['--async-encoding','--reuse-calibration',cal])
    # Repeat an improved variant before recommending it. Avoid combining two
    # changes before their individual effects are measured.
    reference=sum(r['time_to_assembled_valid_video_seconds'] for r in (best,repeat))/2
    for variant,extra in [(resident,['--residency','highvram']),
                          (background,['--async-encoding'])]:
        if variant['status']=='success' and variant['time_to_assembled_valid_video_seconds'] < reference:
            run(variant['name']+'_repeat2',flags(best)+extra+['--reuse-calibration',variant['name']])
    # Last: ordinary 100/100 MPS, never the previously failed 65/35 split.
    # Keep this after all ordinary runs so a device fault cannot invalidate them.
    with (RUNS/'followup_mps_smoke.log').open('w') as log:
        subprocess.run([sys.executable,'-m','agent.rtx6000_benchmark.check_mps',
            '--name','followup_mps_smoke'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    pair=next(pair for pair in selected if pair[0]['transformers']==2)
    candidate=pair[0]
    cal='followup_baseline_10' if candidate['chunks']==10 else f'followup_cal_{candidate["chunks"]}'
    mps=run_variant('followup_best_two_mps',flags(candidate)+['--mps','--reuse-calibration',cal])
    if mps['status']=='success' and mps['time_to_assembled_valid_video_seconds'] < sum(r['time_to_assembled_valid_video_seconds'] for r in pair)/2:
        run('followup_best_two_mps_repeat2',flags(candidate)+['--mps','--reuse-calibration',cal])
    print('Confirmation complete',flush=True)

if __name__=='__main__':main()

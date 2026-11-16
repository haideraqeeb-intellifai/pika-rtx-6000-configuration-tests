"""Test whether extra VAE copies provide MPS gains without a second transformer."""
import json
from .parallel_chunk_processes import RUNS
from .run_chunk_copy_matrix import run

def options(vaes):
    return ['--schedule','pipeline','--chunks','20','--transformers','1','--vaes',str(vaes),
            '--mps','--reuse-calibration','followup_cal_20']

def main():
    run('followup_one_mps_repeat3',['--schedule','pipeline','--chunks','20','--transformers','1','--vaes','1','--mps','--reuse-calibration','followup_cal_20'])
    results=[run(f'followup_mps_1t_{v}v',options(v)) for v in (2,4)]
    successful=[r for r in results if r['status']=='success']
    if not successful:return
    best=min(successful,key=lambda r:r['time_to_assembled_valid_video_seconds'])
    controls=[json.loads((RUNS/('followup_best_two_mps'+s)/'result.json').read_text()) for s in ('','_repeat2')]
    control=sum(r['time_to_assembled_valid_video_seconds'] for r in controls)/len(controls)
    if best['time_to_assembled_valid_video_seconds'] <= control * 1.02:
        run(best['name']+'_repeat2',options(best['vaes']))
    print('MPS VAE sizing complete',flush=True)

if __name__=='__main__':main()

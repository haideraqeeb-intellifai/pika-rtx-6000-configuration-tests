"""Isolated stage profile and the remaining MPS copy allocation."""
import json,os,subprocess,sys
from .parallel_chunk_processes import RUNS
from .run_chunk_copy_matrix import run

def main():
    (RUNS/'owned_coordinator_pids.json').write_text(json.dumps([os.getpid()]))
    (RUNS/'followup_monitor.stop').unlink(missing_ok=True)
    with (RUNS/'followup_monitor_finish.log').open('w') as log:
      monitor=subprocess.Popen([sys.executable,'-m','agent.rtx6000_benchmark.watch_gpu_ownership'],stdout=log,stderr=subprocess.STDOUT)
      try:
        for vaes in (2,4):
            name=f'followup_mps_1t_{vaes}v'
            candidate=json.loads((RUNS/name/'result.json').read_text())
            if candidate['status']=='success' and candidate['time_to_assembled_valid_video_seconds']<=225.85:
                run(name+'_repeat2',['--schedule','pipeline','--chunks','20','--transformers','1','--vaes',str(vaes),'--mps','--reuse-calibration','followup_cal_20'])
        run('followup_isolated_20',['--schedule','staged','--chunks','20','--transformers','1','--vaes','1','--reuse-calibration','followup_cal_20'])
        flags=['--schedule','pipeline','--chunks','20','--transformers','2','--vaes','1','--mps','--reuse-calibration','followup_cal_20']
        trial=run('followup_mps_2t_1v',flags)
        if trial['status']=='success' and trial['time_to_assembled_valid_video_seconds']<=225.85:
            run('followup_mps_2t_1v_repeat2',flags)
        print('All resumed comparisons complete',flush=True)
      finally:
        (RUNS/'followup_monitor.stop').touch();monitor.wait(timeout=15)
if __name__=='__main__':main()

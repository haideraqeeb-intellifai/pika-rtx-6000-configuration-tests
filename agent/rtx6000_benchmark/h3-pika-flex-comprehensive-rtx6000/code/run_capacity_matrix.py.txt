"""Fill practical memory envelopes while measuring simultaneous stage pools."""
import json, os, subprocess, sys
from .parallel_chunk_processes import RUNS
from .run_chunk_copy_matrix import run

def main():
    (RUNS/'owned_coordinator_pids.json').write_text(json.dumps([os.getpid()]))
    (RUNS/'followup_monitor.stop').unlink(missing_ok=True)
    with (RUNS/'followup_capacity_monitor.log').open('w') as log:
        monitor=subprocess.Popen([sys.executable,'-m','agent.rtx6000_benchmark.watch_gpu_ownership'],stdout=log,stderr=subprocess.STDOUT)
        try:
            common=['--schedule','pipeline','--chunks','20','--mps','--reuse-calibration','followup_cal_20','--memory-budget-fraction','0.95']
            control=run('followup_capacity_1t_2v_control',[*common,'--transformers','1','--vaes','2'])
            trials=[]
            # User narrowed the sweep after 1T8V completed: retain practical
            # multi-transformer layouts and do not launch decoder-heavy extensions.
            for transformers,vaes in [(3,1),(2,4),(2,3)]:
                name=f'followup_capacity_{transformers}t_{vaes}v'
                flags=[*common,'--transformers',str(transformers),'--vaes',str(vaes)]
                if transformers==3:flags+=['--reuse-transformer-pool-memory','followup_mps_3t_2v_staged']
                result=run(name,flags)
                if result['status']=='success':trials.append((result['timed_total_seconds'],name,flags))
            if trials:
                seconds,name,flags=min(trials)
                if seconds <= control['timed_total_seconds']*1.01:
                    run(name+'_repeat2',flags)
            print('Capacity comparisons complete',flush=True)
        finally:
            (RUNS/'followup_monitor.stop').touch()
            monitor.wait(timeout=15)
if __name__=='__main__':main()

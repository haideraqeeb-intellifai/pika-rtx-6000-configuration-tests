"""Requested three-transformer test, with recorded memory-constrained fallback."""
import json, os, subprocess, sys
from .parallel_chunk_processes import RUNS
from .run_chunk_copy_matrix import run

def main():
    (RUNS/'owned_coordinator_pids.json').write_text(json.dumps([os.getpid()]))
    (RUNS/'followup_monitor.stop').unlink(missing_ok=True)
    with (RUNS/'followup_three_transformers_monitor.log').open('w') as log:
        monitor=subprocess.Popen([sys.executable,'-m','agent.rtx6000_benchmark.watch_gpu_ownership'],stdout=log,stderr=subprocess.STDOUT)
        try:
            flags=['--chunks','20','--transformers','3','--vaes','2','--mps','--reuse-calibration','followup_cal_20']
            result=run('followup_mps_3t_2v',['--schedule','pipeline',*flags])
            if result['status']=='memory_guard_rejected':
                run('followup_mps_3t_2v_staged',['--schedule','staged','--memory-budget-fraction','0.95',*flags])
        finally:
            (RUNS/'followup_monitor.stop').touch()
            monitor.wait(timeout=15)
if __name__=='__main__':main()

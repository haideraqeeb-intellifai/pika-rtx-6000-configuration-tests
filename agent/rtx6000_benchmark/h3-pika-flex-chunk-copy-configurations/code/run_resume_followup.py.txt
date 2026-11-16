"""Resume interrupted comparisons without replacing previous attempts."""
import json, os, subprocess, sys, time
from .parallel_chunk_processes import RUNS
from .run_chunk_copy_matrix import run
from .run_mps_vae_sizing import main as size_vaes

def main():
    (RUNS/'owned_coordinator_pids.json').write_text(json.dumps([os.getpid()]))
    (RUNS/'followup_monitor.stop').unlink(missing_ok=True)
    with (RUNS/'followup_monitor_resume.log').open('w') as log:
        monitor=subprocess.Popen([sys.executable,'-m','agent.rtx6000_benchmark.watch_gpu_ownership'],stdout=log,stderr=subprocess.STDOUT)
        try:
            size_vaes()
            flags=['--schedule','pipeline','--chunks','20','--transformers','1','--vaes','1',
                '--residency','highvram','--async-encoding','--reuse-calibration','followup_best_resident']
            for suffix in ('repeat3','repeat4'):
                run('followup_best_combined_'+suffix,flags)
            print('Resumed GPU comparisons complete',flush=True)
        finally:
            (RUNS/'followup_monitor.stop').touch()
            monitor.wait(timeout=15)
if __name__=='__main__':main()

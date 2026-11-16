"""Nsight Systems reproductions; instrumented runtimes are not benchmarks."""
import json, os, subprocess, sys, time
from pathlib import Path
from agent.rtx6000_benchmark.parallel_chunk_resident import wait_for_empty_gpu
from agent.rtx6000_benchmark.parallel_chunk_processes import RUNS

ROOT=Path(__file__).resolve().parent
LAYOUTS=[('1t2v_control',1,2,'pipeline'),('2t2v',2,2,'pipeline'),('3t1v',3,1,'pipeline'),('3t2v_staged',3,2,'staged')]
def main():
    (RUNS/'owned_coordinator_pids.json').write_text(json.dumps([os.getpid()]))
    stop=RUNS/'followup_monitor.stop';stop.unlink(missing_ok=True)
    with (ROOT/'ownership_monitor.log').open('w') as log:
        monitor=subprocess.Popen([sys.executable,'-m','agent.rtx6000_benchmark.watch_gpu_ownership'],stdout=log,stderr=subprocess.STDOUT)
        try:
            for label,t,v,schedule in LAYOUTS:
                prefix=ROOT/(label+'_v2')
                name='followup_nsys_v2_'+label
                if (ROOT/(label+'_v2_status.json')).exists():
                    raise RuntimeError(f'Existing capture: {label}')
                wait_for_empty_gpu(max_wait_seconds=600)
                args=['nsys','profile','--trace=cuda-sw,nvtx,osrt','--sample=none','--cpuctxsw=none',
                    '--gpu-metrics-devices=0','--gpu-metrics-frequency=1000','--wait=all',
                    '--output',str(prefix),sys.executable,'-m','agent.rtx6000_benchmark.stage_pipeline',
                    '--name',name,'--chunks','20','--schedule',schedule,'--transformers',str(t),'--vaes',str(v),
                    '--mps','--reuse-calibration','followup_cal_20','--memory-budget-fraction','0.95','--run']
                if t==3 and schedule=='pipeline':
                    args+=['--reuse-transformer-pool-memory','followup_mps_3t_2v_staged']
                started=time.time()
                record={'command':args,'started_wall_utc':started,'profile_only':True}
                (ROOT/(label+'_v2_command.json')).write_text(json.dumps(record,indent=2))
                print('START',label,flush=True)
                with (ROOT/(label+'_v2.log')).open('w') as f:
                    completed=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,
                        env={**os.environ,'H3_PROFILE_NVTX':'1','H3_NSYS_GPU_METRICS':'1'},timeout=1800)
                result_path=RUNS/name/'result.json'
                r=json.loads(result_path.read_text()) if result_path.exists() else {}
                servers=set()
                if result_path.exists():
                    for metadata in result_path.parent.rglob('mps_clients.txt'):
                        for line in metadata.read_text().splitlines():
                            p=line.split()
                            if len(p)>2 and p[2].isdigit():servers.add(int(p[2]))
                ended=time.time()
                checks=[json.loads(x) for x in (RUNS/'followup_ownership_continuous.jsonl').read_text().splitlines()]
                foreign=[s for s in checks if started<=s['time']<=ended and any(not p['owned'] and p['pid'] not in servers for p in s['gpu_processes'])]
                record.update(ended_wall_utc=ended,returncode=completed.returncode,result_status=r.get('status'),
                    instrumented_total_seconds=r.get('timed_total_seconds'),competing_samples=len(foreign),
                    output_sha256=r.get('output_sha256'))
                (ROOT/(label+'_v2_status.json')).write_text(json.dumps(record,indent=2))
                if foreign:(ROOT/(label+'_foreign.json')).write_text(json.dumps(foreign,indent=2))
                print('END',label,json.dumps(record),flush=True)
                if completed.returncode or r.get('status')!='success' or foreign:
                    raise RuntimeError('Capture failed or contended; stop new launches')
                subprocess.run(['nsys','export','--type=sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],check=True,timeout=180)
        finally:
            stop.touch();monitor.wait(timeout=15)
if __name__=='__main__':main()

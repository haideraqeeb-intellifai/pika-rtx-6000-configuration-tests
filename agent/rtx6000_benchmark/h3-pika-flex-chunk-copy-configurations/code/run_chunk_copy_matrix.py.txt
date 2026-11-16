"""Sequential GPU matrix; fail closed on any execution error; resumable records."""
import json
import os
import subprocess
import sys
import time
from .parallel_chunk_processes import RUNS
from .parallel_chunk_resident import wait_for_empty_gpu

LAYOUTS = [(1,1),(2,1),(1,2),(2,2),(1,4)]
def run(name, flags):
    ownership=RUNS/'owned_coordinator_pids.json'
    roots=json.loads(ownership.read_text()) if ownership.exists() else []
    if os.getpid() not in roots:ownership.write_text(json.dumps([*roots,os.getpid()]))
    path=RUNS/name/'result.json'
    if path.exists():
        result=json.loads(path.read_text())
        exclusions=RUNS/'timing_exclusions.json'
        if exclusions.exists() and name in json.loads(exclusions.read_text()):
            raise RuntimeError(f'Existing timing excluded; use a distinct retry name: {name}')
        if result['status'] not in ('success','calibrated','memory_guard_rejected'):
            raise RuntimeError(f'Existing incomplete/failed run: {name}')
        print(f'Existing {name}: {result["status"]}',flush=True)
        return result
    wait_for_empty_gpu(max_wait_seconds=600)
    command=[sys.executable,'-m','agent.rtx6000_benchmark.stage_pipeline','--name',name,*flags,'--run']
    print(f'Starting {name}',flush=True)
    with (RUNS/f'{name}.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=5400)
    result=json.loads(path.read_text())
    monitor=RUNS/'followup_ownership_continuous.jsonl'
    if monitor.exists() and result.get('empty_gpu_gate'):
        start=result['empty_gpu_gate'][0]['time'];end=time.time();servers=set()
        for metadata in path.parent.rglob('mps_clients.txt'):
            for line in metadata.read_text().splitlines():
                parts=line.split()
                if len(parts)>2 and parts[0].isdigit() and parts[2].isdigit():servers.add(int(parts[2]))
        samples=[json.loads(line) for line in monitor.read_text().splitlines()]
        foreign=[s for s in samples if start<=s['time']<=end and any(not p['owned'] and p['pid'] not in servers for p in s['gpu_processes'])]
        if foreign:
            exclusions=RUNS/'timing_exclusions.json'
            values=json.loads(exclusions.read_text()) if exclusions.exists() else {}
            values[name]={'reason':'Competing workload observed during timed run; coordinator stopped before launching another test.','evidence':'ownership_check.json'}
            exclusions.write_text(json.dumps(values,indent=2))
            (path.parent/'ownership_check.json').write_text(json.dumps(foreign,indent=2))
            raise RuntimeError(f'Contention detected in {name}; retain output, exclude timing, stop new tests')
    print(json.dumps({k:result.get(k) for k in ('name','status','timed_total_seconds','gpu_trace_summary','calibration_peak_gib')}),flush=True)
    return result

def main():
    # The baseline was already launched separately. Do not overlap it.
    baseline=RUNS/'followup_baseline_10'/'result.json'
    deadline=time.monotonic()+1800
    while not baseline.exists() or json.loads(baseline.read_text()).get('status')=='running':
        if time.monotonic()>deadline: raise TimeoutError('Baseline did not finish')
        time.sleep(5)
    if json.loads(baseline.read_text())['status']!='success':
        raise RuntimeError('Baseline failed')
    for chunks in (20,5,10):
        cal='followup_baseline_10' if chunks==10 else f'followup_cal_{chunks}'
        if chunks!=10:
            run(cal,['--chunks',str(chunks),'--calibrate-only'])
        for t,v in LAYOUTS:
            if chunks==10 and (t,v)==(1,1): continue
            run(f'followup_c{chunks}_{t}t_{v}v',['--chunks',str(chunks),'--schedule','pipeline',
                '--transformers',str(t),'--vaes',str(v),'--reuse-calibration',cal])
    print('Matrix complete',flush=True)

if __name__=='__main__':main()

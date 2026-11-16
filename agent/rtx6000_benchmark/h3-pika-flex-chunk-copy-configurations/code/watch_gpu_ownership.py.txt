"""Record GPU process ownership once per second; never terminate workloads."""
import json,time,subprocess
from pathlib import Path
import pynvml
RUNS=Path(__file__).resolve().parent/'runs'
pynvml.nvmlInit();gpu=pynvml.nvmlDeviceGetHandleByIndex(0)
try:
    while not (RUNS/'followup_monitor.stop').exists():
        roots=set(json.loads((RUNS/'owned_coordinator_pids.json').read_text()))
        parents={};names={}
        for line in subprocess.check_output(['ps','-eo','pid=,ppid=,comm='],text=True).splitlines():
            a,b,c=line.split(None,2);parents[int(a)]=int(b);names[int(a)]=c
        def owned(pid):
            seen=set()
            while pid>1 and pid not in seen:
                if pid in roots:return True
                seen.add(pid);pid=parents.get(pid,1)
            return False
        servers=set()
        for path in RUNS.glob('followup_*/*/mps_clients.txt'):
            for line in path.read_text().splitlines():
                parts=line.split()
                if len(parts)>2 and parts[0].isdigit() and parts[2].isdigit() and owned(int(parts[0])):
                    servers.add(int(parts[2]))
        processes=[]
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu):
            processes.append({'pid':p.pid,'ppid':parents.get(p.pid),'name':names.get(p.pid),
                'used_mib':round(p.usedGpuMemory/1024**2,2),'owned':owned(p.pid) or p.pid in servers})
        unknown=[p['pid'] for p in processes if not p['owned']]
        record={'time':time.time(),'gpu_processes':processes,'unowned_pids':unknown}
        with (RUNS/'followup_ownership_continuous.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        (RUNS/'followup_ownership_current.json').write_text(json.dumps(record))
        if unknown:print(json.dumps(record),flush=True)
        time.sleep(1)
finally:pynvml.nvmlShutdown()

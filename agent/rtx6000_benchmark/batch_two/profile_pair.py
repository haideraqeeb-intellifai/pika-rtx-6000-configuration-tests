"""Diagnostic captures only; excluded from performance means."""
import json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;D=P/'profiles';D.mkdir(exist_ok=True)
for mode in ['sequential','batch']:
 name='profile_'+mode+'_v2';prefix=D/mode
 cmd=['nsys','profile','--trace=cuda-sw,nvtx,osrt','--sample=none','--cpuctxsw=none','--wait=all','--output',str(prefix),sys.executable,'-m','agent.rtx6000_benchmark.batch_two.experiment','--mode',mode,'--name',name,'--repeats','1']
 (D/(mode+'_command.json')).write_text(json.dumps({'command':cmd,'purpose':'Diagnostic launch counts; not included in performance averages'},indent=2))
 with (D/(mode+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'H3_BATCH_NSYS':'1','H3_NSYS_GPU_METRICS':'1'},timeout=1200)
 print(mode,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
 subprocess.run(['nsys','export','--type=sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],check=True,timeout=180)

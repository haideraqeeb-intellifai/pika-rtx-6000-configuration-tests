"""Full five-by-ten-second Nsight diagnostics, excluded from performance means."""
import json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent
if __name__=='__main__':
 (P/'profiles').mkdir(exist_ok=True)
 for name,mode,t,v in [('sequential_1t2v','sequential',1,2),('concurrent_2t1v','concurrent',2,1),('concurrent_2t2v','concurrent',2,2)]:
  prefix=P/'profiles'/name
  cmd=['nsys','profile','--trace=cuda-sw,nvtx,osrt','--sample=none','--cpuctxsw=none','--gpu-metrics-devices=0','--gpu-metrics-frequency=1000','--wait=all','--output',str(prefix),sys.executable,'-m','agent.rtx6000_benchmark.five_ten.experiment','--mode',mode,'--name','profile_'+name,'--transformers',str(t),'--vaes',str(v),'--repeats','1']
  (P/'profiles'/f'{name}_command.json').write_text(json.dumps({'command':cmd,'note':'Full 5x10s warmup + one profile repeat; not a speed measurement.'},indent=2))
  with (P/'profiles'/f'{name}.log').open('w') as log:r=subprocess.run(cmd,env={**os.environ,'H3_BATCH_NSYS':'1','H3_NSYS_GPU_METRICS':'1'},stdout=log,stderr=subprocess.STDOUT,timeout=2400)
  print(name,r.returncode,flush=True)
  if r.returncode:raise SystemExit(r.returncode)
  subprocess.run(['nsys','export','--type=sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],check=True,timeout=300)

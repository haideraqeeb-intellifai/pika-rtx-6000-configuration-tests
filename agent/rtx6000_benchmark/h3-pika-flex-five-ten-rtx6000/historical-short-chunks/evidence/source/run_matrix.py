import json,subprocess,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent
for mode in ['sequential','batch','concurrent']:
 name=mode+'_v2';start=time.time()
 with (p/(name+'.log')).open('w') as log:
  r=subprocess.run([sys.executable,'-m','agent.rtx6000_benchmark.batch_two.experiment','--mode',mode,'--name',name,'--repeats','3'],stdout=log,stderr=subprocess.STDOUT,timeout=1800)
 print(json.dumps({'mode':mode,'returncode':r.returncode,'elapsed':time.time()-start}),flush=True)
 if r.returncode:raise SystemExit(r.returncode)

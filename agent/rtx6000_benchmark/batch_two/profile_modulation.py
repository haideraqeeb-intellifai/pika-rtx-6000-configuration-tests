import json,os,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent/'profiles';prefix=p/'batch_modulation'
cmd=['nsys','profile','--trace=cuda-sw,nvtx,osrt','--sample=none','--cpuctxsw=none','--wait=all','--output',str(prefix),sys.executable,'-m','agent.rtx6000_benchmark.batch_two.experiment','--mode','batch','--name','profile_batch_modulation_v1','--repeats','1']
(p/'batch_modulation_command.json').write_text(json.dumps({'command':cmd,'env_flags':{'H3_BATCH_MODULATION':'1','H3_BATCH_NSYS':'1','H3_NSYS_GPU_METRICS':'1'},'purpose':'Diagnostic; exclude from timing means'},indent=2))
with (p/'batch_modulation.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'H3_BATCH_MODULATION':'1','H3_BATCH_NSYS':'1','H3_NSYS_GPU_METRICS':'1'},check=True,timeout=1200)
subprocess.run(['nsys','export','--type=sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],check=True,timeout=180)

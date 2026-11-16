import json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent
if __name__=='__main__':
 deadline=time.monotonic()+2400
 while not (P/'runs/sequential_1t2v/result.json').exists():
  if (P/'runs/sequential_1t2v/failure.json').exists():raise RuntimeError('Baseline failed')
  if time.monotonic()>deadline:raise TimeoutError('Baseline waiting deadline')
  time.sleep(5)
 for name,mode,t,v in [('batch_1t2v','batch',1,2),('concurrent_2t1v','concurrent',2,1),('concurrent_2t2v','concurrent',2,2)]:
  # Fresh host gate is enforced by each experiment before workers are created.
  command=[sys.executable,'-m','agent.rtx6000_benchmark.five_ten.experiment','--name',name,'--mode',mode,'--transformers',str(t),'--vaes',str(v),'--repeats','3']
  (P/(name+'_command.json')).write_text(json.dumps(command,indent=2));start=time.time()
  with (P/(name+'.log')).open('w') as log:r=subprocess.run(command,env={**os.environ,'H3_BATCH_MODULATION':'1'},stdout=log,stderr=subprocess.STDOUT,timeout=3600)
  print(json.dumps({'name':name,'returncode':r.returncode,'elapsed':time.time()-start}),flush=True)
  if r.returncode:raise SystemExit(r.returncode)

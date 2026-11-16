"""Serial post-matrix profiling and CPU validation; no competing GPU work."""
import json,subprocess,sys,time,os
from pathlib import Path
P=Path(__file__).resolve().parent
if __name__=='__main__':
 deadline=time.monotonic()+15000
 while not (P/'runs/concurrent_2t2v/result.json').exists():
  failures=list((P/'runs').glob('*/failure.json'))
  if failures:raise RuntimeError('Matrix failure: '+str(failures))
  if time.monotonic()>deadline:raise TimeoutError('Matrix completion deadline')
  time.sleep(10)
 subprocess.run([sys.executable,'-m','agent.rtx6000_benchmark.five_ten.check_compact'],check=True,timeout=120)
 with (P/'batch_compact_1t2v.log').open('w') as log:
  subprocess.run([sys.executable,'-m','agent.rtx6000_benchmark.five_ten.experiment','--mode','batch','--name','batch_compact_1t2v','--transformers','1','--vaes','2','--repeats','3'],env={**os.environ,'H3_BATCH_MODULATION':'1','H3_BATCH_COMPACT':'1'},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
 print('Compact batch repeated',flush=True)
 for mod in ['summarize','profile','analyze_profiles','profile_ncu','analyze_ncu']:
  with (P/(mod+'_execution.log')).open('w') as log:r=subprocess.run([sys.executable,'-m','agent.rtx6000_benchmark.five_ten.'+mod],stdout=log,stderr=subprocess.STDOUT,timeout=9000)
  print(mod,r.returncode,flush=True)
  if r.returncode:raise SystemExit(r.returncode)
 subprocess.run([sys.executable,'-m','agent.rtx6000_benchmark.five_ten.normalize_timestamps'],check=True,timeout=300)
 for name in ['sequential_1t2v','batch_compact_1t2v','concurrent_2t1v','concurrent_2t2v','batch_1t2v']:
  subprocess.run([sys.executable,'-m','agent.rtx6000_benchmark.five_ten.validate',name],check=True,timeout=1800)
 subprocess.run([sys.executable,'-m','agent.rtx6000_benchmark.five_ten.quality'],check=True,timeout=1800)

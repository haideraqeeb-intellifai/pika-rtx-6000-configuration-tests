"""Resume only the corrected diagnostic capture and outstanding CPU checks."""
import subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent
for mod in ['profile_ncu','analyze_ncu']:
 with (P/(mod+'_retry_execution.log')).open('w') as log:
  subprocess.run([sys.executable,'-m','agent.rtx6000_benchmark.five_ten.'+mod],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=9000)
 print(mod,'complete',flush=True)
subprocess.run([sys.executable,'-m','agent.rtx6000_benchmark.five_ten.normalize_timestamps'],check=True,timeout=300)
for name in ['sequential_1t2v','batch_compact_1t2v','concurrent_2t1v','concurrent_2t2v','batch_1t2v']:
 subprocess.run([sys.executable,'-m','agent.rtx6000_benchmark.five_ten.validate',name],check=True,timeout=1800)
subprocess.run([sys.executable,'-m','agent.rtx6000_benchmark.five_ten.quality'],check=True,timeout=1800)

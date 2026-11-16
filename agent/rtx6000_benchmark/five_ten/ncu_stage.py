"""Isolated 10-second transformer kernel diagnostics, never throughput data."""
import os,queue,json
from pathlib import Path
from agent.rtx6000_benchmark.five_ten.experiment import worker,Monitor
from agent.rtx6000_benchmark.run_case import empty_gpu_gate
P=Path(__file__).resolve().parent
if __name__=='__main__':
 root=P/'runs/ncu_transformer';root.mkdir(parents=True)
 (root/'empty_gpu_gate.json').write_text(json.dumps(empty_gpu_gate(),indent=2))
 tasks=queue.SimpleQueue();results=queue.SimpleQueue()
 for rep in ['warmup',1]:
  for i in range(5):tasks.put((rep,[i]))
 tasks.put(None);os.environ['H3_BATCH_NSYS']='1'
 mon=Monitor({os.getpid(),os.getppid()});mon.thread.start()
 try:
  worker('t',0,'sequential',str(root),tasks,results);mon.check()
 finally:
  mon.stop.set();mon.thread.join();(root/'gpu_trace.json').write_text(json.dumps(mon.samples))
 messages=[]
 while not results.empty():messages.append(results.get())
 (root/'messages.json').write_text(json.dumps(messages,indent=2))
 errors=[m for m in messages if m['kind']=='error']
 if errors:raise RuntimeError(errors)
 assert len([m for m in messages if m['kind']=='result'])==10

"""Warm resident pair comparison; one common VAE, no denoise/decode overlap."""
import argparse,copy,json,multiprocessing as mp,os,queue,subprocess,sys,threading,time,traceback,uuid
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent;SOURCE=ROOT/'runs/followup_fast_comparison_fast_1'

def write(p,d):Path(p).write_text(json.dumps(d,indent=2,default=str))

def worker(role,lane,mode,directory,tasks,results):
 try:
  os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']='100'
  import torch
  from ml_modules.minimax_h3_turbo.infer import load_models,infer
  from agent.rtx6000_benchmark.run_case import MODEL_ROOT
  from agent.rtx6000_benchmark.gpu_preencode import register
  from agent.rtx6000_benchmark.latent_stage_nodes import NODE_CLASS_MAPPINGS
  from agent.rtx6000_benchmark import batch_two
  directory=Path(directory);models=load_models(MODEL_ROOT,directory/f'runtime_{role}{lane}');main=models['main'];register(models);main.nodes.NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS)
  for name in ('MiniMaxH3SageAttentionPatch','MiniMaxH3PikaSampler'):main.nodes.NODE_CLASS_MAPPINGS[name].IS_CHANGED=classmethod(lambda cls,**kwargs:float('nan'))
  if mode=='batch' and role=='t':
   from .batch_support import install,STATS
   install(models,directory/'batch_source')
  models['executor']=main.execution.PromptExecutor(models['progress'],cache_type=main.execution.CacheType.CLASSIC,cache_args={'lru':0,'ram':0,'ram_inactive':0})
  results.put({'kind':'ready','role':role,'lane':lane,'pid':os.getpid()})
  while (task:=tasks.get()) is not None:
   rep,indices=task;out=directory/f'repeat_{rep}';out.mkdir(exist_ok=True)
   g=json.loads((ROOT/'runs'/f'followup_fast_comparison_fast_1_{"denoise_" if role=="t" else ""}chunk_0/workflow.json').read_text())
   deadline=time.monotonic()+1200
   if role=='t':
    if mode=='batch':
     seeds=[9901+i for i in indices];g['c0_5']={'class_type':'H3BatchPrepared','inputs':{'paths':json.dumps([str(SOURCE/f'prepared/prepared_{i}.pt') for i in indices]),'seeds':json.dumps(seeds)}}
     g['c0_6']={'class_type':'H3BatchNoise','inputs':{'seeds':json.dumps(seeds)}}
     g['save_latent']={'class_type':'H3BatchSave','inputs':{'samples':['c0_10',0],'paths':json.dumps([str(out/f'latent_{i}.pt') for i in indices])}}
    else:
     i=indices[0];g['c0_5']['inputs']['path']=str(SOURCE/f'prepared/prepared_{i}.pt');g['c0_6']['inputs']['noise_seed']=9901+i;g['save_latent']['inputs']['path']=str(out/f'latent_{i}.pt')
   else:
    i=indices[0];g['load_latent']['inputs']['path']=str(out/f'latent_{i}.pt')
    frames=json.loads((SOURCE/'result.json').read_text())['chunk_frames'][i]
    g['buffer_0']['inputs'].update(directory=str(out/f'chunk_{i}'),deadline=deadline,frames=frames)
    g['save_chunks']['inputs'].update(output=str(out/f'chunk_{i}.mp4'),audio_master=str(SOURCE/f'audio_{i}.wav'),deadline=deadline)
   write(out/f'graph_{role}_{lane}_{indices[0]}.json',g)
   torch.cuda.synchronize();started=time.time();perf=time.perf_counter()
   profiling=os.environ.get('H3_BATCH_NSYS')=='1'
   if profiling:torch.cuda.nvtx.range_push(f'h3pair.{mode}.{role}.repeat_{rep}.items_'+','.join(map(str,indices)))
   try:result=infer(models,g,prompt_id=uuid.uuid4().hex,timeout_seconds=1200,inference_mode='fast')
   finally:
    torch.cuda.synchronize()
    if profiling:torch.cuda.nvtx.range_pop()
   torch.cuda.synchronize();duration=time.perf_counter()-perf
   response={'kind':'result','role':role,'lane':lane,'indices':indices,'repeat':rep,'started':started,'ended':time.time(),'seconds':duration,'telemetry':result['telemetry'],'torch_allocated_peak':torch.cuda.max_memory_allocated(),'torch_reserved_peak':torch.cuda.max_memory_reserved()}
   if role=='t' and mode=='batch':response['batch_stats']=copy.deepcopy(STATS)
   write(out/f'result_{role}_{lane}_{indices[0]}.json',response);results.put(response)
 except BaseException:
  results.put({'kind':'error','role':role,'lane':lane,'error':traceback.format_exc()});traceback.print_exc()

class Monitor:
 def __init__(self,allowed):self.allowed=allowed;self.samples=[];self.stop=threading.Event();self.thread=threading.Thread(target=self.loop,daemon=True)
 def loop(self):
  import pynvml
  from agent.rtx6000_benchmark.mps_session import control
  pynvml.nvmlInit();h=pynvml.nvmlDeviceGetHandleByIndex(0)
  try:
   while not self.stop.is_set():
    processes=pynvml.nvmlDeviceGetComputeRunningProcesses(h)+pynvml.nvmlDeviceGetGraphicsRunningProcesses(h);pids={p.pid for p in processes}
    unknown=pids-set(self.allowed)
    if unknown:
     parents={}
     for line in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines():
      pid,ppid=map(int,line.split());parents[pid]=ppid
     for pid in unknown:
      ancestor=pid;seen=set()
      while ancestor>1 and ancestor not in seen:
       if ancestor in self.allowed:self.allowed.add(pid);break
       seen.add(ancestor);ancestor=parents.get(ancestor,1)
     try:self.allowed.update(int(v) for v in control('get_server_list').split() if v.isdigit())
     except Exception:pass
    self.samples.append({'time':time.time(),'used_gib':pynvml.nvmlDeviceGetMemoryInfo(h).used/1024**3,'util_pct':pynvml.nvmlDeviceGetUtilizationRates(h).gpu,'pids':sorted(pids),'foreign_pids':sorted(pids-set(self.allowed))})
    self.stop.wait(.2)
  except BaseException:self.samples.append({'time':time.time(),'error':traceback.format_exc(),'foreign_pids':['telemetry_error']})
  finally:pynvml.nvmlShutdown()
 def check(self):
  if any(s.get('foreign_pids') for s in self.samples):raise RuntimeError('Competing GPU process or unavailable telemetry; timings excluded')

def run(mode,name,repeats,denoise_only=False):
 from agent.rtx6000_benchmark.run_case import empty_gpu_gate
 from agent.rtx6000_benchmark.mps_session import run_with_mps
 root=BASE/'runs'/name
 if root.exists():raise FileExistsError(root)
 root.mkdir(parents=True);gate=empty_gpu_gate();write(root/'empty_gpu_gate.json',gate)
 def phase():
  ctx=mp.get_context('spawn');results=ctx.Queue();workers=[];tq=[];vq=None
  for role,lane in [('t',i) for i in range(2 if mode=='concurrent' else 1)]+([] if denoise_only else [('v',0)]):
   q=ctx.Queue();p=ctx.Process(target=worker,args=(role,lane,mode,str(root),q,results));p.start();workers.append((p,q));tq.append(q) if role=='t' else None
   if role=='v':vq=q
  mon=Monitor({p.pid for p,q in workers}|{os.getpid()});mon.thread.start();records=[]
  def receive(n):
   got=[];deadline=time.monotonic()+1500
   while len(got)<n:
    mon.check()
    try:r=results.get(timeout=1)
    except queue.Empty:
     if time.monotonic()>deadline or any(p.exitcode not in (None,0) for p,q in workers):raise RuntimeError('Worker exited or timeout')
     continue
    if r['kind']=='error':raise RuntimeError(r['error'])
    got.append(r)
   return got
  try:
   ready=receive(len(workers));write(root/'workers.json',ready)
   for repeat in ['warmup']+list(range(1,repeats+1)):
    mon.check();start=time.time();perf=time.perf_counter()
    if mode=='batch':tq[0].put((repeat,[0,1]));denoised=receive(1)
    elif mode=='sequential':
     tq[0].put((repeat,[0]));denoised=receive(1);tq[0].put((repeat,[1]));denoised+=receive(1)
    else:
     tq[0].put((repeat,[0]));tq[1].put((repeat,[1]));denoised=receive(2)
    dt=time.perf_counter()-perf;decoded=[]
    if not denoise_only:
     for i in [0,1]:vq.put((repeat,[i]));decoded+=receive(1)
    total=time.perf_counter()-perf;end=time.time();samples=[s for s in mon.samples if start<=s['time']<=end]
    r={'repeat':repeat,'mode':mode,'time_to_both_saved_seconds':total,'denoise_pair_seconds':dt,'decode_and_save_seconds':total-dt,'started':start,'ended':end,'peak_vram_gib':max(s['used_gib'] for s in samples),'gpu_util_mean_pct':sum(s['util_pct'] for s in samples)/len(samples),'samples':len(samples),'foreign_samples':sum(bool(s.get('foreign_pids')) for s in samples),'denoise':denoised,'decode':decoded}
    records.append(r);write(root/'results.json',records);print(json.dumps({k:v for k,v in r.items() if k not in ('denoise','decode')}),flush=True)
   return {'status':'success','mode':mode,'repeats':repeats,'denoise_only':denoise_only,'records':records}
  finally:
   for p,q in workers:q.put(None)
   for p,q in workers:
    p.join(timeout=15)
    if p.is_alive():p.terminate();p.join(timeout=10)
   mon.stop.set();mon.thread.join(timeout=5);write(root/'gpu_trace.json',mon.samples)
 try:
  r=run_with_mps(phase,root/'mps');write(root/'result.json',r)
 except BaseException:
  write(root/'failure.json',{'error':traceback.format_exc()});raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['sequential','batch','concurrent'],required=True);p.add_argument('--name',required=True);p.add_argument('--repeats',type=int,default=3);p.add_argument('--denoise-only',action='store_true');a=p.parse_args();run(a.mode,a.name,a.repeats,a.denoise_only)

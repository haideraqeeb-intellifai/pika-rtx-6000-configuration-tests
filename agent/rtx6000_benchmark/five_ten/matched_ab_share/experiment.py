"""Fixed five-by-ten-second resident fast-mode benchmark. No chunk-count tuning."""
import argparse,copy,json,multiprocessing as mp,os,queue,subprocess,sys,threading,time,traceback,uuid
from pathlib import Path
from agent.rtx6000_benchmark.batch_two.experiment import Monitor,write
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
def worker(role,lane,mode,directory,tasks,results):
 try:
  if mode!='simple':os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']='100'
  import torch
  from ml_modules.minimax_h3_turbo.infer import load_models,infer
  from agent.rtx6000_benchmark.run_case import MODEL_ROOT
  from agent.rtx6000_benchmark.gpu_preencode import register
  from agent.rtx6000_benchmark.latent_stage_nodes import NODE_CLASS_MAPPINGS
  from agent.rtx6000_benchmark import batch_two
  directory=Path(directory);models=load_models(MODEL_ROOT,directory/f'runtime_{role}{lane}');main=models['main'];register(models);main.nodes.NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS)
  for name in ('MiniMaxH3SageAttentionPatch','MiniMaxH3PikaSampler'):main.nodes.NODE_CLASS_MAPPINGS[name].IS_CHANGED=classmethod(lambda cls,**kwargs:float('nan'))
  if mode=='batch' and role=='t':
   from agent.rtx6000_benchmark.batch_two.batch_support import install,STATS
   install(models,directory/'batch_source')
   if os.environ.get('H3_BATCH_COMPACT')=='1':
    from .compact_batch import install as install_compact
    install_compact(models)
  models['executor']=main.execution.PromptExecutor(models['progress'],cache_type=main.execution.CacheType.CLASSIC,cache_args={'lru':0,'ram':0,'ram_inactive':0})
  results.put({'kind':'ready','role':role,'lane':lane,'pid':os.getpid()})
  while (task:=tasks.get()) is not None:
   rep,indices=task;out=directory/f'repeat_{rep}';out.mkdir(exist_ok=True)
   g=json.loads((BASE/'templates'/('denoise.json' if role=='t' else 'decode.json')).read_text())
   deadline=time.monotonic()+1200
   if role=='t':
    if mode=='batch' and len(indices)==2:
     seeds=[9901+i for i in indices];g['c0_5']={'class_type':'H3BatchPrepared','inputs':{'paths':json.dumps([str(BASE/f'prepared/prepared_{i}.pt') for i in indices]),'seeds':json.dumps(seeds)}}
     g['c0_6']={'class_type':'H3BatchNoise','inputs':{'seeds':json.dumps(seeds)}}
     g['save_latent']={'class_type':'H3BatchSave','inputs':{'samples':['c0_10',0],'paths':json.dumps([str(out/f'latent_{i}.pt') for i in indices])}}
    else:
     i=indices[0];g['c0_5']['inputs']['path']=str(BASE/f'prepared/prepared_{i}.pt');g['c0_6']['inputs']['noise_seed']=9901+i;g['save_latent']['inputs']['path']=str(out/f'latent_{i}.pt')
   else:
    i=indices[0];g['load_latent']['inputs']['path']=str(out/f'latent_{i}.pt')
    frames=240
    g['buffer_0']['inputs'].update(directory=str(out/f'chunk_{i}'),deadline=deadline,frames=frames)
    g['save_chunks']['inputs'].update(output=str(out/f'chunk_{i}.mp4'),audio_master=str(BASE/f'inputs/audio_{i}.wav'),deadline=deadline)
   write(out/f'graph_{role}_{lane}_{indices[0]}.json',g)
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.time();perf=time.perf_counter()
   profiling=os.environ.get('H3_BATCH_NSYS')=='1'
   if profiling:torch.cuda.nvtx.range_push(f'h3five.{mode}.{role}.repeat_{rep}.items_'+','.join(map(str,indices)))
   try:result=infer(models,g,prompt_id=uuid.uuid4().hex,timeout_seconds=1200,inference_mode='fast')
   finally:
    torch.cuda.synchronize()
    if profiling:torch.cuda.nvtx.range_pop()
   torch.cuda.synchronize();duration=time.perf_counter()-perf
   response={'kind':'result','role':role,'lane':lane,'indices':indices,'repeat':rep,'started':started,'ended':time.time(),'seconds':duration,'pid':os.getpid(),'telemetry':result['telemetry'],'torch_allocated_peak':torch.cuda.max_memory_allocated(),'torch_reserved_peak':torch.cuda.max_memory_reserved()}
   if role=='t' and mode=='batch':response['batch_stats']=copy.deepcopy(STATS)
   write(out/f'result_{role}_{lane}_{indices[0]}.json',response);results.put(response)
 except BaseException:
  results.put({'kind':'error','role':role,'lane':lane,'error':traceback.format_exc()});traceback.print_exc()

def assemble(out):
 playlist=out/'concat.txt';playlist.write_text(''.join("file '"+str(out/f'chunk_{i}.mp4')+"'\n" for i in range(5)))
 original=out/'original_assembly.mp4'
 subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(playlist),'-i',str(BASE/'inputs/audio_master.wav'),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-t','50','-movflags','+faststart',str(original)],check=True,timeout=120)
 subprocess.run(['ffmpeg','-v','error','-y','-i',str(original),'-map','0:v:0','-map','0:a:0','-c','copy','-bsf:v','setts=pts=PTS-STARTPTS:dts=DTS-STARTPTS','-avoid_negative_ts','disabled','-movflags','+faststart',str(out/'video.mp4')],check=True,timeout=120)

def run(mode,name,transformers,vaes,repeats):
 from agent.rtx6000_benchmark.run_case import empty_gpu_gate
 from agent.rtx6000_benchmark.mps_session import run_with_mps
 root=BASE/'runs'/name
 if root.exists():raise FileExistsError(root)
 root.mkdir(parents=True);write(root/'empty_gpu_gate.json',empty_gpu_gate())
 assert mode!='batch' or transformers==1
 def phase():
  ctx=mp.get_context('spawn');results=ctx.Queue();workers=[];qs={};init=time.perf_counter()
  for role,n in [('t',transformers),('v',vaes)]:
   for lane in range(n):
    q=ctx.Queue();p=ctx.Process(target=worker,args=(role,lane,mode,str(root),q,results));p.start();workers.append((p,q));qs[(role,lane)]=q
  mon=Monitor({p.pid for p,q in workers}|{os.getpid()});mon.thread.start();records=[]
  def receive():
   deadline=time.monotonic()+2400
   while time.monotonic()<deadline:
    mon.check()
    try:r=results.get(timeout=1)
    except queue.Empty:
     if any(p.exitcode not in (None,0) for p,q in workers):raise RuntimeError('Worker exited')
     continue
    if r['kind']=='error':raise RuntimeError(r['error'])
    return r
   raise TimeoutError('Worker timeout')
  try:
   ready=[receive() for _ in workers];write(root/'workers.json',ready);startup=time.perf_counter()-init
   for rep in ['warmup']+list(range(1,repeats+1)):
    mon.check();start=time.time();perf=time.perf_counter();events=[]
    pending=[[0,1],[2,3],[4]] if mode=='batch' else [[i] for i in range(5)]
    waiting=[];idle_t=list(range(transformers));idle_v=list(range(vaes));done=0;denoise_end=None
    def dispatch():
     if mode=='simple':
      if waiting and idle_v:
       lane=idle_v.pop(0);qs[('v',lane)].put((rep,[waiting.pop(0)]))
      elif pending and idle_t and not waiting and len(idle_v)==vaes:
       lane=idle_t.pop(0);qs[('t',lane)].put((rep,pending.pop(0)))
      return
     while pending and idle_t:
      lane=idle_t.pop(0);qs[('t',lane)].put((rep,pending.pop(0)))
     while waiting and idle_v:
      lane=idle_v.pop(0);qs[('v',lane)].put((rep,[waiting.pop(0)]))
    dispatch()
    while done<5:
     r=receive();assert r['kind']=='result' and r['repeat']==rep;events.append(r)
     if r['role']=='t':
      idle_t.append(r['lane']);waiting.extend(r['indices'])
      if sum(len(e['indices']) for e in events if e['role']=='t')==5:denoise_end=time.perf_counter()-perf
     else:idle_v.append(r['lane']);done+=1
     dispatch()
    chunks_saved=time.perf_counter()-perf;assemble(root/f'repeat_{rep}');total=time.perf_counter()-perf;end=time.time()
    samples=[s for s in mon.samples if start<=s['time']<=end];mon.check()
    r={'repeat':rep,'mode':mode,'transformers':transformers,'vaes':vaes,'chunks':5,'frames_per_chunk':240,'fps':24,'seconds':total,'chunks_saved_seconds':chunks_saved,'last_denoise_finished_seconds':denoise_end,'assembly_seconds':total-chunks_saved,'started':start,'ended':end,'peak_vram_gib':max(s['used_gib'] for s in samples),'gpu_util_mean_pct':sum(s['util_pct'] for s in samples)/len(samples),'samples':len(samples),'foreign_samples':sum(bool(s.get('foreign_pids')) for s in samples),'events':events}
    records.append(r);write(root/'results.json',records);print(json.dumps({k:v for k,v in r.items() if k!='events'}),flush=True)
   return {'status':'success','mode':mode,'transformers':transformers,'vaes':vaes,'repeats':repeats,'startup_seconds':startup,'records':records,'protocol':'One full-video warmup, then repeated full 5x10s videos; resident models; GPU preparation separately measured; denoise/decode pipelined; final audio remux included; validation excluded.'}
  finally:
   for p,q in workers:q.put(None)
   for p,q in workers:
    p.join(timeout=15)
    if p.is_alive():p.terminate();p.join(timeout=10)
   mon.stop.set();mon.thread.join(timeout=5);write(root/'gpu_trace.json',mon.samples)
 try:
  r=phase() if mode=='simple' else run_with_mps(phase,root/'mps');write(root/'result.json',r)
 except BaseException:
  write(root/'failure.json',{'error':traceback.format_exc()});raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['simple','sequential','batch','concurrent'],required=True);p.add_argument('--name',required=True);p.add_argument('--transformers',type=int,default=1);p.add_argument('--vaes',type=int,default=2);p.add_argument('--repeats',type=int,default=3);a=p.parse_args();run(a.mode,a.name,a.transformers,a.vaes,a.repeats)

import collections,json,sqlite3,statistics,bisect,hashlib
from pathlib import Path
from agent.rtx6000_benchmark.nsight_profile.analyze_traces import union_seconds,overlap_profile
P=Path(__file__).resolve().parent

def analyze(path, control_root=None):
 db=sqlite3.connect(path);db.row_factory=sqlite3.Row;strings=dict(db.execute('select id,value from StringIds'))
 ranges=[];by_pid=collections.defaultdict(list)
 for r in db.execute("select start,end,text,globalTid from NVTX_EVENTS where text like 'h3five.%repeat_1.%' and end is not null"):
  _,mode,role,rep,items=r['text'].split('.');d=dict(start=r['start'],end=r['end'],pid=r['globalTid']//(1<<24)*(1<<24),role='denoise' if role=='t' else 'decode',items=[int(v) for v in items.removeprefix('items_').split(',')]);ranges.append(d);by_pid[d['pid']].append(d)
 assert len(ranges)==10,(path,len(ranges))
 for rr in by_pid.values():rr.sort(key=lambda r:r['start'])
 lo=min(r['start'] for r in ranges);hi=max(r['end'] for r in ranges)
 def match(pid,t):
  rr=by_pid.get(pid,[]);i=bisect.bisect_right([r['start'] for r in rr],t)-1
  return rr[i] if i>=0 and t<=rr[i]['end'] else None
 rows=[];kernels=collections.defaultdict(lambda:dict(count=0,ns=0,registers=0,shared_bytes=0,max_grid_blocks=0,max_block_threads=0))
 for k in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?',(lo,hi)):
  r=match(k['globalPid'],k['start'])
  if not r:continue
  rows.append((k['start'],k['end'],k['globalPid'],r['role']))
  x=kernels[(r['role'],strings[k['demangledName']])];x['count']+=1;x['ns']+=k['end']-k['start'];x['registers']=max(x['registers'],k['registersPerThread']);x['shared_bytes']=max(x['shared_bytes'],k['staticSharedMemory']+k['dynamicSharedMemory']);x['max_grid_blocks']=max(x['max_grid_blocks'],k['gridX']*k['gridY']*k['gridZ']);x['max_block_threads']=max(x['max_block_threads'],k['blockX']*k['blockY']*k['blockZ'])
 out={'source':path.name,'ranges':[{**r,'start_s':(r['start']-lo)/1e9,'end_s':(r['end']-lo)/1e9} for r in ranges],'window_seconds':(hi-lo)/1e9,'kernel_count':len(rows),'kernel_overlap':overlap_profile(rows),'host_call_overlap':overlap_profile([(r['start'],r['end'],r['pid'],r['role']) for r in ranges]),'top_kernels':{},'metrics':{},'metric_series':{}}
 for role in ['denoise','decode']:
  out['top_kernels'][role]=[{'name':n,**x,'seconds':x['ns']/1e9} for (s,n),x in sorted(kernels.items(),key=lambda kv:-kv[1]['ns']) if s==role][:10]
 names=dict(db.execute('select metricId,metricName from TARGET_INFO_GPU_METRICS'))
 for i,n in names.items():
  if not any(t in n for t in ['SMs Active','SM Issue','Tensor Active','DRAM Read Bandwidth','DRAM Write Bandwidth','Compute Warps in Flight [Throughput']):continue
  vals=list(db.execute('select timestamp,value from GPU_METRICS where metricId=? and timestamp>=? and timestamp<=? order by timestamp',(i,lo,hi)))
  a=[r['value'] for r in vals]
  if not a:continue
  out['metrics'][n]={'mean':statistics.mean(a),'p95':sorted(a)[int(.95*(len(a)-1))],'samples':len(a)}
  buckets=collections.defaultdict(list)
  for r in vals:buckets[int((r['timestamp']-lo)/1e8)].append(r['value'])
  out['metric_series'][n]=[[k/10,statistics.mean(v)] for k,v in sorted(buckets.items())]
  singles=[];multiple=[];initial=[]
  tr=[r for r in ranges if r['role']=='denoise'];first_finish=min(r['end'] for r in tr)
  for r in vals:
   c=sum(iv['start']<=r['timestamp']<iv['end'] for iv in tr)
   if r['timestamp']<first_finish:initial.append(r['value'])
   if c==1:singles.append(r['value'])
   elif c>=2:multiple.append(r['value'])
  out['metrics'][n].update(initial_transformer_only_mean=statistics.mean(initial) if initial else None,denoising_window_mean=statistics.mean(singles+multiple) if singles or multiple else None,one_transformer_call_active_mean=statistics.mean(singles) if singles else None,multiple_transformer_calls_active_mean=statistics.mean(multiple) if multiple else None)
 # GPU timeline in 100ms bins; each process contributes kernel union, not summed duration.
 bins=collections.defaultdict(lambda:collections.defaultdict(list))
 for a,b,pid,s in rows:
  for j in range(int((a-lo)//100000000),int((b-lo)//100000000)+1):
   x=max(a,lo+j*100000000);y=min(b,lo+(j+1)*100000000)
   if y>x:bins[(s,pid)][j].append((x,y))
 out['kernel_series']=[{'role':s,'pid':pid,'bins':[[j/10,union_seconds(v)/.1*100] for j,v in sorted(bb.items())]} for (s,pid),bb in bins.items()]
 tables={x[0] for x in db.execute("select name from sqlite_master where type='table'")}
 out['memcpy']={}
 if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
  kinds=dict(db.execute('select id,label from ENUM_CUDA_MEMCPY_OPER'));copies=collections.defaultdict(lambda:{'count':0,'bytes':0,'summed_seconds':0})
  for x in db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=?',(lo,hi)):
   r=match(x['globalPid'],x['start'])
   if r:
    v=copies[(r['role'],kinds.get(x['copyKind'],str(x['copyKind'])))];v['count']+=1;v['bytes']+=x['bytes'];v['summed_seconds']+=(x['end']-x['start'])/1e9
  out['memcpy']={s+' / '+k:v for (s,k),v in copies.items()}
 out['diagnostics']=[dict(x) for x in db.execute('select * from DIAGNOSTIC_EVENT')] if 'DIAGNOSTIC_EVENT' in tables else []
 video=P/'runs'/('profile_'+path.stem)/'repeat_1/video.mp4';control=P/'runs'/(control_root or path.stem)/'repeat_1/video.mp4'
 if (control.parent/'original_assembly/video.mp4').exists():control=control.parent/'original_assembly/video.mp4'
 out['comparison_note']='SHA comparison uses the matched final MP4, or historical original assembly when present.'
 out['profile_video_sha256']=hashlib.sha256(video.read_bytes()).hexdigest();out['matches_unprofiled_video_bytes']=out['profile_video_sha256']==hashlib.sha256(control.read_bytes()).hexdigest()
 db.close();return out
if __name__=='__main__':
 out={}
 for p in (P/'profiles').glob('*.sqlite'):
  out[p.stem]=analyze(p);print(p.stem,out[p.stem]['kernel_overlap'],flush=True)
 (P/'profiles_summary.json').write_text(json.dumps(out,indent=2))

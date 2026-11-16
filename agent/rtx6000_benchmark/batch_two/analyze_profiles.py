import collections,json,sqlite3
from pathlib import Path
P=Path(__file__).resolve().parent;out={}
def union(v):
 total=0;end=-1
 for a,b in sorted(v):total+=max(0,b-max(a,end));end=max(end,b)
 return total/1e9
for mode in ['sequential','batch','batch_modulation']:
 if not (P/'profiles'/f'{mode}.sqlite').exists():continue
 db=sqlite3.connect(P/'profiles'/f'{mode}.sqlite');strings=dict(db.execute('select id,value from StringIds'))
 ranges=[{'start':a,'end':b,'label':s,'pid':t//(1<<24)*(1<<24)} for a,b,s,t in db.execute("select start,end,text,globalTid from NVTX_EVENTS where text like 'h3pair.%' and text like '%.t.repeat_1.%' and end is not null")]
 assert len(ranges)==(2 if mode=='sequential' else 1),ranges
 events=[];kernels=collections.defaultdict(lambda:{'count':0,'seconds':0});apis=collections.defaultdict(lambda:{'count':0,'seconds':0})
 for r in ranges:
  for a,b,name in db.execute('select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=? and start>=? and end<=?',(r['pid'],r['start'],r['end'])):
   events.append((a,b));k=kernels[strings[name]];k['count']+=1;k['seconds']+=(b-a)/1e9
  for a,b,tid,name in db.execute('select start,end,globalTid,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=?',(r['start'],r['end'])):
   if tid//(1<<24)*(1<<24)==r['pid']:
    q=apis[strings[name]];q['count']+=1;q['seconds']+=(b-a)/1e9
 span=(max(r['end'] for r in ranges)-min(r['start'] for r in ranges))/1e9
 launch={k:v for k,v in apis.items() if 'LaunchKernel' in k or 'LaunchCooperativeKernel' in k}
 out[mode]={'nvtx_ranges':ranges,'denoise_span_seconds':span,'kernel_count':len(events),'kernel_sum_seconds':sum(b-a for a,b in events)/1e9,'kernel_union_seconds':union(events),'kernel_busy_fraction':union(events)/span,'kernel_mean_microseconds':sum(b-a for a,b in events)/len(events)/1000,'launch_api_calls':sum(v['count'] for v in launch.values()),'summed_launch_api_seconds':sum(v['seconds'] for v in launch.values()),'launch_apis':launch,'top_kernels':[{'name':k,**v} for k,v in sorted(kernels.items(),key=lambda x:-x[1]['seconds'])[:15]],'cuda_apis':[{'name':k,**v} for k,v in sorted(apis.items(),key=lambda x:-x[1]['seconds'])[:15]]}
 db.close()
(P/'profile_summary.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:{a:b for a,b in v.items() if a not in ('nvtx_ranges','top_kernels','cuda_apis','launch_apis')} for k,v in out.items()},indent=2))

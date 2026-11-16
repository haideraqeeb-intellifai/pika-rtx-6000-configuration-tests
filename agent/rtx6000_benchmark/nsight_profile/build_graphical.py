"""Create a portable visual report from existing Nsight evidence. No GPU work."""
import base64,bisect,collections,csv,gzip,hashlib,io,json,shutil,sqlite3,zipfile
from pathlib import Path
P=Path(__file__).resolve().parent;O=P.parent/'h3-pika-flex-nsight-rtx6000';O.mkdir(exist_ok=True)
a=json.loads((P/'analysis.json').read_text());n=a['nsys']['2t2v'];cs=n['chunks'];lo=min(c['start'] for c in cs);hi=max(c['end'] for c in cs if c['role']=='denoise');span=(hi-lo)/1e9
ranges=collections.defaultdict(list)
for c in cs:ranges[c['globalPid']].append(c)
for r in ranges.values():r.sort(key=lambda c:c['start'])
db=sqlite3.connect(P/'2t2v_v2.sqlite');names=dict(db.execute('select id,value from StringIds'));kn=[];ki={};rows=[];ints=collections.defaultdict(list)
for start,end,pid,name,regs,smem,threads,grid in db.execute('select start,end,globalPid,demangledName,registersPerThread,staticSharedMemory+dynamicSharedMemory,blockX*blockY*blockZ,gridX*gridY*gridZ from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=? order by start',(lo,hi)):
 rr=ranges.get(pid,[]);ix=bisect.bisect_right([c['start'] for c in rr],start)-1
 if ix<0 or start>rr[ix]['end'] or rr[ix]['role']!='denoise':continue
 c=rr[ix];key=(name,regs,smem,threads,grid)
 if key not in ki:ki[key]=len(kn);kn.append({'name':names[name],'registersPerThread':regs,'sharedBytes':smem,'threads':threads,'gridBlocks':grid})
 # Exact relative nanoseconds; integers fit JS's safe integer range.
 rows.append([start-lo,end-lo,c['lane'],ki[key],c['index']]);ints[c['lane']].append((start-lo,end-lo))
def union(seq):
 result=[]
 for s,e in sorted(seq):
  if result and s<=result[-1][1]:result[-1][1]=max(result[-1][1],e)
  else:result.append([s,e])
 return result
u={k:union(v) for k,v in ints.items()};inter=[];i=j=0
while i<len(u[0]) and j<len(u[1]):
 s=max(u[0][i][0],u[1][j][0]);e=min(u[0][i][1],u[1][j][1])
 if e>s:inter.append([s,e])
 if u[0][i][1]<u[1][j][1]:i+=1
 else:j+=1
overlap=sum(e-s for s,e in inter)/1e9
assert abs(overlap-n['overlap']['multiple_transformer_kernel_seconds'])<1e-8
calls=[{'start':c['start']-lo,'end':c['end']-lo,'lane':c['lane'],'chunk':c['index']} for c in cs if c['role']=='denoise']
ce=[]
for c in calls:ce.extend([(c['start'],1),(c['end'],-1)])
active=0;last=0;cpu=0
for t,d in sorted(ce):
 if active>=2:cpu+=t-last
 active+=d;last=t
bin_ns=250_000_000;N=int((hi-lo)//bin_ns)+1
bins=[[0.0]*N for _ in range(3)]
for lane,seq in [(0,u[0]),(1,u[1]),(2,inter)]:
 for s,e in seq:
  for b in range(int(s//bin_ns),min(N,int(e//bin_ns)+1)):
   bins[lane][b]+=max(0,min(e,(b+1)*bin_ns)-max(s,b*bin_ns))/bin_ns
metric=db.execute("select metricId from TARGET_INFO_GPU_METRICS where metricName like 'SMs Active%'").fetchone()[0]
sm=[[] for _ in range(N)]
for t,v in db.execute('select timestamp,value from GPU_METRICS where metricId=? and timestamp>=? and timestamp<?',(metric,lo,hi)):
 sm[int((t-lo)//bin_ns)].append(v)
sm=[sum(v)/len(v) if v else None for v in sm]
epoch=db.execute('select utcEpochNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0]/1e9+lo/1e9
ownership=[]
for l in (P.parent/'runs/followup_ownership_continuous.jsonl').open():
 r=json.loads(l)
 if epoch<=r['time']<=epoch+span:
  ownership.append({'seconds':r['time']-epoch,'processes':[{'pid':p['pid'],'gib':p['used_mib']/1024} for p in r['gpu_processes'] if p['owned']]})
snapshot=min(ownership,key=lambda r:abs(r['seconds']-50));pids={c['lane']:(c['globalPid']>>24)&0xffffff for c in cs}
mem=json.loads((P.parent/'runs/followup_nsys_v2_2t2v/gpu_trace.json').read_text());mem=[[(r['time']-epoch),r['used_gib']] for r in mem if epoch<=r['time']<=epoch+span]
summary={'span':span,'kernelOverlap':overlap,'cpuOverlap':cpu/1e9,'fraction':overlap/span,'kernelEvents':len(rows),'bins':bins,'sm':sm,'binSeconds':.25,'calls':calls,'memory':mem,'processSnapshot':snapshot,'pids':pids,'overlapIntervals':inter,'source':'2t2v_v2.sqlite','timeZero':'First transformer NVTX range start','allCaptures':{k:{'stages':v['stages'],'overlap':v['overlap'],'metrics':v['metric_windows']['denoising_window']} for k,v in a['nsys'].items()},'historical':{k:{'total':v['timed_total_seconds'],'tSpan':v['stages']['denoise']['span_seconds'],'tail':v['decode_tail_after_denoising_seconds'],'peak':v['gpu_trace_summary']['peak_vram_gib']} for k,v in a['historical'].items()}}
(O/'visual_summary.json').write_text(json.dumps(summary,indent=2))
payload=json.dumps({'kernels':rows,'names':kn},separators=(',',':')).encode();packed=gzip.compress(payload,mtime=0);(O/'transformer_kernel_events.json.gz').write_bytes(packed)
for f in ['ncu_denoise.csv','ncu_decode.csv','ncu_summary.json','kernel_resources.json','analysis_validation.json','ncu_validation.json','provenance.json','historical_chunks.csv']:
 shutil.copy2(P/f,O/f)
# Publish selected numeric evidence only; raw profiler archives can contain host environment metadata.
with zipfile.ZipFile(O/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in ['visual_summary.json','transformer_kernel_events.json.gz','ncu_denoise.csv','ncu_decode.csv','kernel_resources.json','historical_chunks.csv','provenance.json'] :z.write(O/f,f)
 for f in ['build_graphical.py','graphical_template.html','analyze_traces.py']:z.write(P/f,'source/'+f)
 for f in P.glob('*_v2_status.json'):
  r=json.loads(f.read_text());r.pop('command',None);z.writestr('capture_status/'+f.name,json.dumps(r,indent=2))
validation={'source':str(P/'2t2v_v2.sqlite'),'kernel_events':len(rows),'overlap_seconds_recomputed':overlap,'overlap_matches_original_analysis':True,'cpu_call_overlap_seconds':cpu/1e9,'counts_by_lane':dict(collections.Counter(r[2] for r in rows)),'payload_sha256':hashlib.sha256(payload).hexdigest(),'privacy':'Only selected kernel events and numerical counter evidence exported; no captured host environment tables.'}
(O/'data_validation.json').write_text(json.dumps(validation,indent=2))
t=(P/'graphical_template.html').read_text();t=t.replace('__SUMMARY__',json.dumps(summary,separators=(',',':'))).replace('__KERNELS_GZIP__',base64.b64encode(packed).decode());(O/'index.html').write_text(t)
print(json.dumps(validation,indent=2));print('Report:',O,'HTML bytes',len(t))

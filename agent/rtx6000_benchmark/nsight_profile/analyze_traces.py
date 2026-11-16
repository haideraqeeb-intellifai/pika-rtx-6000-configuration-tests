"""Reproduce CPU-call and CUDA-kernel profiles without starting GPU workloads."""
import bisect,collections,csv,hashlib,json,sqlite3,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUNS=ROOT.parent/'runs'
HISTORY={'1t2v_control':'followup_capacity_1t_2v_control','2t2v':'followup_best_two_mps','3t1v':'followup_capacity_3t_1v','3t2v_staged':'followup_mps_3t_2v_staged','isolated_1t1v':'followup_isolated_20'}
def union_seconds(intervals):
    end=-1;total=0
    for a,b in sorted(intervals):
        total+=max(0,b-max(a,end));end=max(end,b)
    return total/1e9

def overlap_profile(rows):
    events=[]
    for a,b,pid,stage in rows:events.extend([(a,1,pid,stage),(b,-1,pid,stage)])
    active=collections.Counter();by_stage=collections.Counter();times=collections.Counter();last=None
    for t,delta,pid,stage in sorted(events):
        if last is not None:
            dt=(t-last)/1e9;npids=sum(v>0 for v in active.values())
            if npids:times['any_kernel_seconds']+=dt
            if npids>=2:times['multiple_process_kernel_seconds']+=dt
            if by_stage['denoise'] and by_stage['decode']:times['transformer_vae_kernel_overlap_seconds']+=dt
            if sum(v>0 for (s,p),v in active.items() if s=='denoise')>=2:times['multiple_transformer_kernel_seconds']+=dt
        active[(stage,pid)]+=delta;by_stage[stage]+=delta;last=t
    return dict(times)

def historical(name):
    r=json.loads((RUNS/name/'result.json').read_text());out={k:r.get(k) for k in ['name','transformers','vaes','schedule','mps','timed_total_seconds','parallel_generation_seconds','preparation_seconds','encoder_release_gate_seconds','assembly_and_full_validation_seconds','gpu_trace_summary','output_sha256','complete_video_audio_decode','audio_check','stage_phase_seconds']}
    out['extra_validation_seconds']=r['timed_total_seconds']-r['time_to_assembled_valid_video_seconds']
    out['stages']={};calls=[]
    for role,field,timer in [('denoise','denoise_results','sampler_intervals'),('decode','child_results','decode_intervals')]:
        children=[json.loads((RUNS/c['name']/'result.json').read_text()) for c in r[field]]
        first={}
        for c in sorted(children,key=lambda c:c['inference_started_wall_utc']):first.setdefault(c['lane'],c['index'])
        rows=[]
        for c in children:
            assert c['status']=='success'
            for iv in c[timer]:
                row={'stage':role,'index':c['index'],'lane':c['lane'],'frames':c['frames'],'first_in_lane':first[c['lane']]==c['index'],'start':iv['started_wall_utc'],'end':iv['ended_wall_utc'],'duration':iv['ended_wall_utc']-iv['started_wall_utc'],'child_total':c['total_seconds']}
                nodes=c['telemetry']['stages']['nodes'];row['max_model_offloaded_gib']=max((n['after'].get('model_offloaded_gib',0) for n in nodes),default=0)
                rows.append(row);calls.append(row)
        out['stages'][role]={'span_seconds':max(c['end'] for c in rows)-min(c['start'] for c in rows),'summed_call_seconds':sum(c['duration'] for c in rows),'call_count':len(rows),'warm_small_mean_seconds':statistics.mean(c['duration'] for c in rows if not c['first_in_lane'] and c['frames']<=56),'warm_73_mean_seconds':statistics.mean(c['duration'] for c in rows if not c['first_in_lane'] and c['frames']==73),'max_reported_offloaded_gib':max(c['max_model_offloaded_gib'] for c in rows)}
    t_end=max(c['end'] for c in calls if c['stage']=='denoise');v_end=max(c['end'] for c in calls if c['stage']=='decode')
    out['decode_tail_after_denoising_seconds']=v_end-t_end
    out['chunks_decoded_before_last_denoise']=sum(c['end']<=t_end for c in calls if c['stage']=='decode')
    out['calls']=calls
    return out

def analyze(label):
    path=ROOT/(label+'_v2.sqlite');db=sqlite3.connect(path);db.row_factory=sqlite3.Row
    strings=dict(db.execute('select id,value from StringIds'))
    ranges=[];roles={}
    for x in db.execute("select start,end,text,globalTid from NVTX_EVENTS where text like 'h3.%' and end is not null"):
        _,role,chunk,lane=x['text'].split('.');gp=x['globalTid']//(1<<24)*(1<<24)
        roles[gp]=role;ranges.append({'start':x['start'],'end':x['end'],'globalPid':gp,'role':role,'index':int(chunk.split('_')[1]),'lane':int(lane.split('_')[1])})
    assert len(ranges)==40,(label,len(ranges))
    lo=min(r['start'] for r in ranges);hi=max(r['end'] for r in ranges)
    by_pid=collections.defaultdict(list)
    for r in ranges:by_pid[r['globalPid']].append(r)
    for rr in by_pid.values():rr.sort(key=lambda r:r['start'])
    def find_range(pid,t):
        rr=by_pid.get(pid,[])
        i=bisect.bisect_right([r['start'] for r in rr],t)-1
        return rr[i] if i>=0 and t<=rr[i]['end'] else None
    rows=[];kernels=collections.defaultdict(lambda:dict(count=0,sum_ns=0,max_ns=0));perchunk=collections.defaultdict(list)
    for k in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?',(lo,hi)):
        r=find_range(k['globalPid'],k['start'])
        if r is None:continue
        role=r['role'];name=strings[k['demangledName']];dur=k['end']-k['start']
        key=(role,name);kernels[key]['count']+=1;kernels[key]['sum_ns']+=dur;kernels[key]['max_ns']=max(kernels[key]['max_ns'],dur)
        rows.append((k['start'],k['end'],k['globalPid'],role));perchunk[(role,r['index'])].append((k['start'],k['end']))
    out={'label':label,'source':path.name,'nvtx_range_count':len(ranges),'stage_window_seconds':(hi-lo)/1e9,'cuda_kernel_count_in_stage_ranges':len(rows),'overlap':overlap_profile(rows),'top_kernels':{},'stages':{},'metric_windows':{},'memcpy':{},'chunks':[]}
    for role in ['denoise','decode']:
        rr=[r for r in ranges if r['role']==role];kk=[(a,b) for a,b,p,s in rows if s==role]
        out['stages'][role]={'span_seconds':(max(r['end'] for r in rr)-min(r['start'] for r in rr))/1e9,'summed_kernel_seconds':sum(b-a for a,b in kk)/1e9,'kernel_union_seconds':union_seconds(kk)}
        out['top_kernels'][role]=[{'name':name,**v,'sum_seconds':v['sum_ns']/1e9,'mean_ms':v['sum_ns']/v['count']/1e6} for (s,name),v in sorted(kernels.items(),key=lambda x:-x[1]['sum_ns']) if s==role][:15]
    for r in ranges:
        kk=perchunk[(r['role'],r['index'])]
        out['chunks'].append({**r,'start_relative_seconds':(r['start']-lo)/1e9,'duration_seconds':(r['end']-r['start'])/1e9,'kernel_union_seconds':union_seconds(kk),'summed_kernel_seconds':sum(b-a for a,b in kk)/1e9})
    # Counter values with percentage units are integer0..100; clock fields are not used.
    metric_names=dict(db.execute('select metricId,metricName from TARGET_INFO_GPU_METRICS'))
    selected=[i for i,n in metric_names.items() if any(x in n for x in ['SMs Active','SM Issue','Tensor Active','DRAM Read Bandwidth','DRAM Write Bandwidth','PCIe RX Throughput','PCIe TX Throughput','Compute Warps in Flight [Throughput'])]
    t_end=max(r['end'] for r in ranges if r['role']=='denoise')
    windows={'all_stages':(lo,hi),'denoising_window':(min(r['start'] for r in ranges if r['role']=='denoise'),t_end),'decode_tail':(max(t_end,min(r['start'] for r in ranges if r['role']=='decode')),hi)}
    for label,(a,b) in windows.items():
        summary={}
        for metric in selected:
            vals=[x[0] for x in db.execute('select value from GPU_METRICS where metricId=? and timestamp>=? and timestamp<?',(metric,a,b))]
            if vals:
                vals.sort();summary[metric_names[metric]]={'mean':statistics.mean(vals),'p95':vals[int(.95*(len(vals)-1))],'samples':len(vals)}
        out['metric_windows'][label]=summary
    tables={x[0] for x in db.execute("select name from sqlite_master where type='table'")}
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        kinds=dict(db.execute('select id,label from ENUM_CUDA_MEMCPY_OPER'))
        for role in ['denoise','decode']:
            sums=collections.defaultdict(lambda:{'count':0,'bytes':0,'sum_seconds':0})
            for x in db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=?',(lo,hi)):
                r=find_range(x['globalPid'],x['start'])
                if r and r['role']==role:
                    v=sums[kinds.get(x['copyKind'],str(x['copyKind']))];v['count']+=1;v['bytes']+=x['bytes'];v['sum_seconds']+=(x['end']-x['start'])/1e9
            out['memcpy'][role]=dict(sums)
            warm_sums=collections.defaultdict(lambda:{'count':0,'bytes':0,'sum_seconds':0})
            for x in db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=?',(lo,hi)):
                r=find_range(x['globalPid'],x['start'])
                if r and r['role']==role and r is not by_pid[r['globalPid']][0]:
                    v=warm_sums[kinds.get(x['copyKind'],str(x['copyKind']))];v['count']+=1;v['bytes']+=x['bytes'];v['sum_seconds']+=(x['end']-x['start'])/1e9
            out['memcpy'][role+'_excluding_first_chunk_per_lane']=dict(warm_sums)
    # CPU API durations are inclusive across threads; never treat their sums as GPU execution time.
    out['cuda_api']=[{'name':strings[x[0]],'calls':x[1],'summed_seconds':x[2]/1e9} for x in db.execute('select nameId,count(*),sum(end-start) from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=? group by nameId order by sum(end-start) desc limit 12',(lo,hi))]
    out['diagnostics']=[dict(x) for x in db.execute('select * from DIAGNOSTIC_EVENT')] if 'DIAGNOSTIC_EVENT' in tables else []
    db.close();return out

def main():
    data={'status':'fresh_nsight_profiling_plus_historical_unprofiled_timings','historical':{label:historical(name) for label,name in HISTORY.items()},'nsys':{},'methodology':['Default inference mode;20 independent anchoredchunks;two steps;identical inputs/output hash.','Historical times are performance results;Nsight captures are diagnostics and have overhead.','NVTX sampler/decode boundaries synchronize their CUDA context. CUDA activity spans establish actual kernel overlap.','GPU counters are sampled at1000Hz;SM activity is not equivalent to peak compute utilization.','Kernel durations and per-process API durations overlap;their sums are not video wall time.','Targeted NCU replays characterize selected kernels;they do not preserve the full MPS pipeline schedule.']}
    for label in list(HISTORY)[:4]:
        if (ROOT/(label+'_v2.sqlite')).exists():
            try:
                probe=sqlite3.connect(ROOT/(label+'_v2.sqlite'));ok=probe.execute("select count(*) from NVTX_EVENTS where text like 'h3.%'").fetchone()[0]==40;probe.close()
            except sqlite3.Error:ok=False
            if ok:data['nsys'][label]=analyze(label)
    (ROOT/'analysis.json').write_text(json.dumps(data,indent=2))
    with (ROOT/'historical_chunks.csv').open('w') as f:
        w=None
        for name,r in data['historical'].items():
            for c in r['calls']:
                row={'configuration':name,**c}
                if w is None:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader()
                w.writerow(row)
    for name,r in data['nsys'].items():
        print(name,'overlap',r['overlap'])
        print('denoising counters',r['metric_windows']['denoising_window'])
        for role in ['denoise','decode']:print(role,'top',r['top_kernels'][role][:3])
if __name__=='__main__':main()

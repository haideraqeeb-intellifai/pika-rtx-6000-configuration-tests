"""Report capacity trials, actual overlapping calls, and GPU ownership."""
import csv,json,statistics
from pathlib import Path
from .parallel_chunk_processes import RUNS

def main():
    samples=[json.loads(s) for s in (RUNS/'followup_ownership_continuous.jsonl').read_text().splitlines()]
    rows=[]
    for p in sorted(RUNS.glob('followup_capacity_*/result.json')):
        r=json.loads(p.read_text())
        if r.get('loader')!='separate stage processes':continue
        row={k:r.get(k) for k in ('name','status','transformers','vaes','schedule','mps','timed_total_seconds','time_to_assembled_valid_video_seconds','admission_memory_envelope_gib','memory_budget_gib','output_sha256')}
        if r['status']=='success':
            events=[];lanes={'denoise':set(),'decode':set()};windows={};offload={}
            for field,stage,timer in [('denoise_results','denoise','sampler_intervals'),('child_results','decode','decode_intervals')]:
                intervals=[];offload[stage]=[]
                for child in r[field]:
                    cr=json.loads((RUNS/child['name']/'result.json').read_text());lanes[stage].add(cr['lane'])
                    intervals+=cr[timer]
                    for node in cr['telemetry']['stages']['nodes']:
                        value=node['after'].get('model_offloaded_gib')
                        if value is not None:offload[stage].append(value)
                windows[stage]=max(i['ended_wall_utc'] for i in intervals)-min(i['started_wall_utc'] for i in intervals)
                for i in intervals:events.extend([(i['started_wall_utc'],stage,1),(i['ended_wall_utc'],stage,-1)])
            active={'denoise':0,'decode':0};peaks=active.copy();last=None;overlap=0.;both_peak=0
            for when,stage,change in sorted(events):
                if last is not None and all(active.values()):overlap+=when-last
                active[stage]+=change;peaks[stage]=max(peaks[stage],active[stage]);both_peak=max(both_peak,sum(active.values()));last=when
            servers=set()
            for meta in p.parent.rglob('mps_clients.txt'):
                for line in meta.read_text().splitlines():
                    fields=line.split()
                    if len(fields)>2 and fields[0].isdigit() and fields[2].isdigit():servers.add(int(fields[2]))
            start=r['empty_gpu_gate'][0]['time'];end=start+r['total_seconds']+2
            checks=[s for s in samples if start<=s['time']<=end]
            foreign=[s for s in checks if any(not proc['owned'] and proc['pid'] not in servers for proc in s['gpu_processes'])]
            row.update(peak_vram_gib=r['gpu_trace_summary']['peak_vram_gib'],gpu_util_mean_pct=r['gpu_trace_summary']['gpu_util_mean_pct'],workers_with_completed_chunks={k:len(v) for k,v in lanes.items()},maximum_concurrent_calls=peaks,maximum_combined_calls=both_peak,denoise_decode_overlap_seconds=overlap,stage_call_windows_seconds=windows,ownership_samples=len(checks),competing_samples=len(foreign),complete_video_audio_decode=r['complete_video_audio_decode'],audio_check=r['audio_check'],matches_previous_output=r['output_sha256']=='0084bd7f5c25885c692b469c9de71c6d3c482a888cefb05e537296f01531f3d3',max_reported_offloaded_gib={k:max(v) if v else None for k,v in offload.items()})
            if foreign:row['status']='timing_excluded'
        rows.append(row)
    valid=[r for r in rows if r['status']=='success']
    groups={}
    for row in valid:groups.setdefault((row['transformers'],row['vaes']),[]).append(row)
    summary=[{'transformers':t,'vaes':v,'n':len(rs),'mean_seconds':statistics.mean(r['timed_total_seconds'] for r in rs),'peak_vram_gib':max(r['peak_vram_gib'] for r in rs)} for (t,v),rs in groups.items()]
    summary.sort(key=lambda r:r['mean_seconds'])
    data={'runs':rows,'summary':summary,'notes':['All configurations use MPS and overlapping transformer/VAE stages. Same 20 independent chunks, two steps, 24 fps.','Wall-clock call overlap is verified; simultaneous GPU kernel execution is not profiled.','Total includes GPU preparation, loading, pipeline, encoding, assembly and QA. Single samples are screens, not variance estimates.','Three-transformer admission uses the previous isolated pool whole-device peak plus isolated VAE peak. Other layouts use sum of isolated per-worker peaks. Admission ceiling 95% of 95.59 GiB.']}
    (RUNS/'capacity_matrix_summary.json').write_text(json.dumps(data,indent=2)+'\n')
    with (RUNS/'capacity_matrix_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['transformers','vaes','n','mean_seconds','peak_vram_gib']);w.writeheader();w.writerows(summary)
    print(json.dumps(data,indent=2))
if __name__=='__main__':main()

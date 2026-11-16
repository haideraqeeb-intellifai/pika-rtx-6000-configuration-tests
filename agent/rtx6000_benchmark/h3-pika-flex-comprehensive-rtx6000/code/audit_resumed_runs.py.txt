"""Resolve transient MPS server ownership and audit resumed run intervals."""
import json
from .parallel_chunk_processes import RUNS

def main():
    samples=[json.loads(line) for line in (RUNS/'followup_ownership_continuous.jsonl').read_text().splitlines()]
    names=['followup_one_mps_repeat3','followup_mps_1t_2v','followup_mps_1t_4v',
           'followup_mps_1t_2v_repeat2','followup_mps_1t_4v_repeat2',
           'followup_mps_2t_1v','followup_mps_2t_1v_repeat2','followup_best_combined_repeat3','followup_best_combined_repeat4','followup_isolated_20']
    exclusions=json.loads((RUNS/'timing_exclusions.json').read_text())
    audit=[]
    for name in names:
        path=RUNS/name/'result.json'
        if not path.exists():continue
        r=json.loads(path.read_text())
        if r.get('status')!='success':continue
        start=r['empty_gpu_gate'][0]['time'];end=path.stat().st_mtime
        servers=set()
        for metadata in (RUNS/name).rglob('mps_clients.txt'):
            for line in metadata.read_text().splitlines():
                parts=line.split()
                if len(parts)>2 and parts[0].isdigit() and parts[2].isdigit():servers.add(int(parts[2]))
        window=[s for s in samples if start<=s['time']<=end]
        foreign=[{'time':s['time'],'process':p} for s in window for p in s['gpu_processes'] if not p['owned'] and p['pid'] not in servers]
        entry={'name':name,'start_epoch':start,'end_epoch':end,'samples':len(window),
            'resolved_private_MPS_server_pids':sorted(servers),'unowned_observations':foreign,
            'status':'contention_detected' if foreign else 'no_competing_process_observed'}
        if not window:raise RuntimeError(f'Missing ownership coverage for {name}')
        entry['maximum_sample_gap_seconds']=max(b['time']-a['time'] for a,b in zip(window,window[1:]))
        audit.append(entry)
        if foreign:exclusions[name]={'reason':'Competing workload in resumed run; timings excluded.','evidence':'followup_resume_audit.json'}
    (RUNS/'timing_exclusions.json').write_text(json.dumps(exclusions,indent=2))
    (RUNS/'followup_resume_audit.json').write_text(json.dumps({'runs':audit,'note':'Host NVML sampled at approximately 1 Hz. Early MPS server sightings resolved using the run-specific attached-client evidence. Sampling does not guarantee exclusive reservation.'},indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k not in ('unowned_observations',)} for r in audit],indent=2))
if __name__=='__main__':main()

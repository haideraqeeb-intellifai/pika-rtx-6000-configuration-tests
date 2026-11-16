"""Export only CUDA kernel intervals and measure cross-process overlap."""
import csv
import gzip
import json
import sqlite3
import subprocess
from collections import defaultdict
from .parallel_chunk_processes import RUNS


def main():
    prefix=RUNS/'vae_profile_mps_four_retry_trace'
    database=prefix.with_suffix('.sqlite')
    if not database.exists():
        subprocess.run(['nsys','export','--type','sqlite','--output',str(database),str(prefix.with_suffix('.nsys-rep'))],check=True)
    connection=sqlite3.connect(database)
    table='CUPTI_ACTIVITY_KIND_KERNEL'
    columns={r[1] for r in connection.execute(f'PRAGMA table_info({table})')}
    wanted=['start','end','globalPid','contextId','streamId','deviceId']
    if not set(wanted)<=columns:
        diagnostics=[row[0] for row in connection.execute("SELECT DISTINCT text FROM DIAGNOSTIC_EVENT WHERE text LIKE '%hardware tracing%' OR text LIKE '%incomplete CUPTI%' LIMIT 12")]
        evidence={'status':'kernel_overlap_unverified','scope':'unmeasured four-chunk MPS decoder profile',
            'reason':'CUDA runtime events were captured but no GPU kernel interval table was exported. GPU timestamps were incomplete.',
            'diagnostics':diagnostics,'runtime_events':connection.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME').fetchone()[0]}
        (RUNS/'vae_mps_kernel_overlap.json').write_text(json.dumps(evidence,indent=2))
        print(json.dumps(evidence,indent=2));return

    rows=connection.execute(f'SELECT {",".join(wanted)} FROM {table} ORDER BY start').fetchall()
    if not rows: raise ValueError('No CUDA kernel intervals were captured')
    path=RUNS/'vae_mps_kernel_intervals.csv.gz'
    with gzip.open(path,'wt',newline='') as f:
        writer=csv.writer(f);writer.writerow(wanted);writer.writerows(rows)
    events=[]
    for start,end,pid,context,stream,device in rows:
        events.extend([(start,1,pid),(end,-1,pid)])
    events.sort(key=lambda e:(e[0],e[1]))
    active=defaultdict(int);last=None;overlap=busy=0;maximum=0
    for when,change,pid in events:
        processes=sum(v>0 for v in active.values())
        if last is not None:
            if processes: busy+=when-last
            if processes>=2: overlap+=when-last
        active[pid]+=change
        maximum=max(maximum,sum(v>0 for v in active.values()))
        last=when
    profile=json.loads((RUNS/'vae_profile_mps_four_retry/result.json').read_text())
    clients={int(line.split()[0]) for line in profile['phase']['mps_clients'].splitlines() if line.split() and line.split()[0].isdigit()}
    globals_={r[2] for r in rows}
    decoded={p>>24 for p in globals_}
    evidence={'scope':'unmeasured four-chunk decoder profile under MPS; excluded from timing averages',
        'kernel_intervals':len(rows),'distinct_global_process_ids':sorted(globals_),
        'decoded_client_pids':sorted(decoded),'expected_mps_client_pids':sorted(clients),
        'all_kernel_processes_match_mps_clients':decoded<=clients,
        'kernel_busy_union_seconds':busy/1e9,'cross_process_kernel_overlap_seconds':overlap/1e9,
        'fraction_kernel_busy_time_with_cross_process_overlap':overlap/busy,
        'maximum_processes_with_simultaneous_kernels':maximum,
        'exported_intervals':path.name,'nsight_report_local':str(prefix.with_suffix('.nsys-rep'))}
    if not decoded<=clients: raise ValueError(f'Kernel process identifiers need manual verification: {evidence}')
    (RUNS/'vae_mps_kernel_overlap.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence,indent=2))

if __name__=='__main__':main()

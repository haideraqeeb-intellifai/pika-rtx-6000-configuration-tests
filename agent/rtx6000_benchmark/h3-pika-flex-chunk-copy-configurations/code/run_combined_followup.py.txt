"""Combine individually confirmed residency/export changes; two full runs."""
import json
from .parallel_chunk_processes import RUNS
from .run_chunk_copy_matrix import run

def main():
    variants={}
    for name in ('followup_best_resident','followup_best_async'):
        records=[json.loads((RUNS/(name+suffix)/'result.json').read_text()) for suffix in ('','_repeat2')]
        if any(r['status']!='success' for r in records):raise RuntimeError('Individual variant did not pass')
        variants[name]=records
    resident=variants['followup_best_resident'][0]
    options=['--schedule','pipeline','--chunks',str(resident['chunks']),
        '--transformers',str(resident['transformers']),'--vaes',str(resident['vaes']),
        '--residency','highvram','--async-encoding','--reuse-calibration','followup_best_resident']
    for suffix in ('','_repeat2'):
        run('followup_best_combined'+suffix,options)

if __name__=='__main__':main()

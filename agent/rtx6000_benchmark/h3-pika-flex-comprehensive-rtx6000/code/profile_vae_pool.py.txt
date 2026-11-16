"""Unmeasured four-chunk decoder replay for CUDA kernel concurrency tracing."""
import argparse
import json
import subprocess
import time
from .parallel_chunk_processes import RUNS
from .parallel_chunk_resident import wait_for_empty_gpu
from .stage_pipeline import run_phase


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--mps',action='store_true')
    parser.add_argument('--vaes',type=int,choices=[2,4],default=4)
    parser.add_argument('--baseline-only',action='store_true')
    parser.add_argument('--profiler-baseline')
    args=parser.parse_args()
    directory=RUNS/args.name;directory.mkdir()
    source=RUNS/'vae_staged_1t_1v'
    original=json.loads((source/'result.json').read_text())
    evidence={'scope':'unmeasured profiler diagnostic','mps':args.mps,'vaes':args.vaes}
    try:
        if args.baseline_only or args.profiler_baseline:
            import pynvml
            pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByIndex(0)
            samples=[]
            try:
                for i in range(3):
                    values=subprocess.run(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,memory.reserved',
                        '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout.strip().split(', ')
                    samples.append({'uuid':values[0],'gpu_util_pct':int(values[1]),'memory_used_mib':int(values[2]),
                        'driver_reserved_mib':int(values[3]),'compute_pids':[p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)],
                        'graphics_pids':[p.pid for p in pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)]})
                    if i<2: time.sleep(1)
            finally: pynvml.nvmlShutdown()
            if any(s['gpu_util_pct'] or s['compute_pids'] or s['graphics_pids'] for s in samples):
                raise RuntimeError('Profiler baseline has GPU workload activity')
            if len({(s['uuid'],s['memory_used_mib'],s['driver_reserved_mib']) for s in samples})!=1:
                raise RuntimeError('Profiler idle baseline is not stable')
            evidence['empty_gpu_gate']=samples
            if args.baseline_only:
                evidence.update(status='baseline_recorded',note='CPU-only process under Nsight; no CUDA model or tensor workload was started.')
                (directory/'result.json').write_text(json.dumps(evidence,indent=2));return
            baseline=json.loads((RUNS/args.profiler_baseline/'result.json').read_text())
            if baseline['status']!='baseline_recorded' or samples!=baseline['empty_gpu_gate']:
                raise RuntimeError('GPU differs from independently recorded profiler idle baseline')
            evidence['profiler_baseline']=args.profiler_baseline
        else:
            evidence['empty_gpu_gate']=wait_for_empty_gpu()
    except BaseException as exc:
        evidence.update(status='preflight_failure',error=repr(exc))
        (directory/'result.json').write_text(json.dumps(evidence,indent=2));raise
    try:
        counts=original['chunk_frames'][:4]
        result=run_phase(directory,counts,[source/f'audio_{i}.wav' for i in range(4)],
            source/'prepared',source/'latents','decode',0,args.vaes,args.name,args.mps)
        evidence.update(status='success',phase=result)
    except BaseException as exc:
        evidence.update(status='failure',error=repr(exc));raise
    finally:
        (directory/'result.json').write_text(json.dumps(evidence,indent=2))
    if not args.profiler_baseline:
        evidence['post_shutdown_idle_gate']=wait_for_empty_gpu()
    (directory/'result.json').write_text(json.dumps(evidence,indent=2))

if __name__=='__main__':main()

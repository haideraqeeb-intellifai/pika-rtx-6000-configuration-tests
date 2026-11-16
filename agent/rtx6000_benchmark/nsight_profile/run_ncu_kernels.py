"""Target dominant warm H3 kernels, separately from MPS concurrency captures."""
import json,os,subprocess,sys,time
from pathlib import Path
from agent.rtx6000_benchmark.parallel_chunk_resident import wait_for_empty_gpu
from agent.rtx6000_benchmark.parallel_chunk_processes import RUNS
ROOT=Path(__file__).resolve().parent

def main():
    (RUNS/'owned_coordinator_pids.json').write_text(json.dumps([os.getpid()]))
    stop=RUNS/'followup_monitor.stop';stop.unlink(missing_ok=True)
    with (ROOT/'ncu_ownership_monitor.log').open('w') as f:
        monitor=subprocess.Popen([sys.executable,'-m','agent.rtx6000_benchmark.watch_gpu_ownership'],stdout=f,stderr=subprocess.STDOUT)
        try:
            for stage,pattern in [('denoise',r'regex:qk_int_sv_f8_attn_kernel|DefaultGemmWithVisitor|CUDAFunctor_add'),('decode',r'regex:cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8|pytorch_flash::flash_fwd_kernel|vectorized_layer_norm_kernel')]:
                wait_for_empty_gpu(max_wait_seconds=600)
                prefix=ROOT/('ncu_'+stage)
                cmd=['ncu','--config-file','off','--nvtx','--nvtx-include',f'h3.{stage}.chunk_9.lane_0/',
                    '--kernel-name-base','demangled','--kernel-name',pattern,'--filter-mode','per-launch-config','--launch-count','1',
                    '--section','SpeedOfLight','--section','Occupancy','--section','LaunchStats',
                    '--clock-control','none','--cache-control','none','--target-processes','application-only',
                    '--export',str(prefix),sys.executable,'-m','agent.rtx6000_benchmark.nsight_profile.ncu_stage',
                    '--stage',stage,'--name','followup_ncu_'+stage]
                start=time.time();record={'command':cmd,'started_wall_utc':start,'note':'Isolated kernels;not a multi-process speed measurement'}
                (ROOT/('ncu_'+stage+'_command.json')).write_text(json.dumps(record,indent=2))
                print('START NCU',stage,flush=True)
                with (ROOT/('ncu_'+stage+'.log')).open('w') as log:
                    p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
                end=time.time()
                checks=[json.loads(s) for s in (RUNS/'followup_ownership_continuous.jsonl').read_text().splitlines()]
                foreign=[s for s in checks if start<=s['time']<=end and any(not p['owned'] for p in s['gpu_processes'])]
                record.update(returncode=p.returncode,ended_wall_utc=end,competing_samples=len(foreign))
                (ROOT/('ncu_'+stage+'_status.json')).write_text(json.dumps(record,indent=2))
                print('END NCU',stage,record,flush=True)
                if p.returncode or foreign:raise RuntimeError('NCU failed or contended')
                with (ROOT/('ncu_'+stage+'.csv')).open('w') as csv:
                    subprocess.run(['ncu','--import',str(prefix)+'.ncu-rep','--csv','--page','raw'],stdout=csv,check=True,timeout=120)
        finally:stop.touch();monitor.wait(timeout=15)
if __name__=='__main__':main()

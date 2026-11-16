import json,os,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent
if __name__=='__main__':
 prefix=P/'profiles/ncu_transformer'
 cmd=['ncu','--config-file','off','--nvtx','--nvtx-push-pop-scope','process','--nvtx-include','h3five.sequential.t.repeat_1.items_0/','--kernel-name-base','demangled','--kernel-name','regex:qk_int_sv_f8_attn_kernel|compute_attn_ws|DefaultGemmWithVisitor','--filter-mode','per-launch-config','--launch-count','1','--section','SpeedOfLight','--section','Occupancy','--section','LaunchStats','--clock-control','none','--cache-control','none','--target-processes','application-only','--export',str(prefix),sys.executable,'-m','agent.rtx6000_benchmark.five_ten.ncu_stage']
 (P/'profiles/ncu_command.json').write_text(json.dumps({'command':cmd,'note':'One kernel per matching launch configuration, within the first warm measured 10-second chunk. Isolated kernel diagnostics; full 5-chunk warmup and 5-chunk replay, denoise only; no generation-time claim.'},indent=2))
 with (P/'profiles/ncu.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=2400)
 if r.returncode:raise SystemExit(r.returncode)
 with (P/'profiles/ncu.csv').open('w') as f:subprocess.run(['ncu','--import',str(prefix)+'.ncu-rep','--csv','--page','raw'],stdout=f,check=True,timeout=120)

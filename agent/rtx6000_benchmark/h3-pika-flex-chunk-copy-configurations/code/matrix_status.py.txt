"""Read-only progress for the follow-up experiment."""
from pathlib import Path
import argparse,json,time
RUNS=Path(__file__).resolve().parent/'runs'
parser=argparse.ArgumentParser();parser.add_argument('--confirmation',action='store_true');parser.add_argument('--final',action='store_true');args=parser.parse_args()
log=RUNS/('followup_final.log' if args.final else 'followup_confirmation.log' if args.confirmation else 'followup_matrix.log')
lines=log.read_text().splitlines()
for line in lines[-3:]:print(line)
starts=[line.split(' ',1)[1] for line in lines if line.startswith('Starting ')]
if starts:
    name=starts[-1]
    for role,pattern in [('denoise','_denoise_chunk_*'),('decode','_chunk_*')]:
        completed=list(RUNS.glob(name+pattern+'/result.json'))
        print(role,len(completed))
        calibrated=list(RUNS.glob(name+'_calibration'+pattern+'/result.json'))
        if calibrated:print('calibration '+role,len(calibrated))
    p=RUNS/name/'result.json'
    if p.exists():print('status',json.loads(p.read_text())['status'])

import json,time
from pathlib import Path
P=Path(__file__).resolve().parent
for d in sorted((P/'runs').glob('*')):
 if not d.is_dir():continue
 complete=d/'result.json';records=d/'results.json'
 if records.exists():print(d.name,[(r['repeat'],round(r['seconds'],2),round(r['peak_vram_gib'],2)) for r in json.loads(records.read_text())])
 else:print(d.name,'initializing/warmup')
 if not complete.exists():
  finished={str(r['repeat']) for r in json.loads(records.read_text())} if records.exists() else set()
  for rep in sorted(d.glob('repeat_*')):
   if rep.name.removeprefix('repeat_') in finished:continue
   files=list(rep.glob('result*.json'))
   if len(files)<10:
    print(' ',rep.name,[(r['role'],r['indices'],round(r['seconds'],1)) for f in files for r in [json.loads(f.read_text())]])
 if (d/'failure.json').exists():print('FAILED', (d/'failure.json').read_text()[-600:])

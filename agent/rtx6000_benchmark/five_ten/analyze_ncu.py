import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent
if __name__=='__main__':
 data=list(csv.DictReader((P/'profiles/ncu.csv').open()));out=[];units=next((r for r in data if not r.get('ID')), {})
 keys=['device__attribute_multiprocessor_count','device__attribute_max_shared_memory_per_multiprocessor','device__attribute_max_registers_per_multiprocessor','device__attribute_max_warps_per_multiprocessor','gpu__time_duration.sum','sm__throughput.avg.pct_of_peak_sustained_elapsed','gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed','sm__warps_active.avg.pct_of_peak_sustained_active','launch__occupancy_per_sm','launch__occupancy_limit_registers','launch__occupancy_limit_shared_mem','launch__registers_per_thread','launch__shared_mem_per_block','launch__grid_size','launch__block_size','launch__waves_per_multiprocessor']
 for r in data:
  if not r.get('ID','').isdigit():continue
  out.append({'id':r['ID'],'name':r['Kernel Name'],'block':r['Block Size'],'grid':r['Grid Size'],'metrics':{k:r[k] for k in keys if k in r},'units':{k:units.get(k,'') for k in keys}})
 assert out, 'No NCU kernels parsed'
 for term in ['qk_int_sv_f8_attn_kernel','compute_attn_ws','DefaultGemmWithVisitor']:
  assert any(term in r['name'] for r in out), 'Missing requested NCU kernel family: '+term
 (P/'ncu_summary.json').write_text(json.dumps(out,indent=2));print('NCU selected kernels:',len(out))

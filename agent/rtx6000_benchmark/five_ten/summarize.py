import json,statistics,csv
from pathlib import Path
P=Path(__file__).resolve().parent
out={'workload':{'chunks':5,'chunk_seconds':10,'fps':24,'delivered_frames_per_chunk':240,'native_frames_per_chunk':243,'video_seconds':50,'steps':2,'inference_mode':'fast','independent_anchors':'original portrait at native first and last frames','seeds':list(range(9901,9906))},'preparation_seconds':json.loads((P/'preparation.json').read_text())['seconds'],'configurations':[]}
for name in ['sequential_1t2v','batch_compact_1t2v','concurrent_2t1v','concurrent_2t2v']:
 d=P/'runs'/name
 if not (d/'result.json').exists():continue
 r=json.loads((d/'result.json').read_text());rs=[x for x in r['records'] if x['repeat']!='warmup'];assert len(rs)==3 and all(x['foreign_samples']==0 for x in rs)
 stages={};fast=[];offload=[]
 for role in ['t','v']:
  ev=[e for x in rs for e in x['events'] if e['role']==role]
  stages[role]={'mean_call_seconds':statistics.mean(e['seconds'] for e in ev),'summed_call_seconds_per_video':sum(e['seconds'] for e in ev)/len(rs),'max_torch_allocated_gib':max(e['torch_allocated_peak'] for e in ev)/2**30,'max_torch_reserved_gib':max(e['torch_reserved_peak'] for e in ev)/2**30,'model_loaded_gib':max(n['after'].get('model_loaded_gib',0) for e in ev for n in e['telemetry']['stages']['nodes']),'sampled_peak_model_offloaded_gib':max(e['telemetry']['gpu'].get('model_offloaded_peak_gib',0) for e in ev),'max_node_end_offloaded_gib':max(n['after'].get('model_offloaded_gib',0) for e in ev for n in e['telemetry']['stages']['nodes'])}
  offload.extend(n['after'].get('model_offloaded_gib',0) for e in ev for n in e['telemetry']['stages']['nodes'])
 for x in rs:
  counters={}
  for e in x['events']:
   for k,v in e['telemetry'].get('fast_inference',{}).items():counters[k]=counters.get(k,0)+v
  expected_samplers=3 if r['mode']=='batch' else 5
  assert counters['sampler_calls']==expected_samplers and counters['model_forward_calls']==2*expected_samplers
  assert counters['sage2_calls']==50*expected_samplers and counters['sage3_calls']==50*expected_samplers and counters['sage3_fallback_calls']==0
  assert counters['video_decode_calls']==5 and counters['nvfp4_gemm_calls']==5*56448
  fast.append(counters)
 c={'name':name,'mode':r['mode'],'transformers':r['transformers'],'vaes':r['vaes'],'n':len(rs),'seconds':[x['seconds'] for x in rs],'mean_seconds':statistics.mean(x['seconds'] for x in rs),'std_seconds':statistics.stdev(x['seconds'] for x in rs),'peak_vram_gib':max(x['peak_vram_gib'] for x in rs),'mean_gpu_util_pct':statistics.mean(x['gpu_util_mean_pct'] for x in rs),'mean_last_denoise_seconds':statistics.mean(x['last_denoise_finished_seconds'] for x in rs),'mean_decode_tail_seconds':statistics.mean(x['seconds']-x['last_denoise_finished_seconds'] for x in rs),'startup_seconds':r['startup_seconds'],'cold_first_video_seconds':r['startup_seconds']+r['records'][0]['seconds'],'warmup_seconds':r['records'][0]['seconds'],'max_node_end_model_offloaded_gib':max(offload),'stages':stages,'fast_counters':fast}
 out['configurations'].append(c)
if out['configurations']:
 base=out['configurations'][0]['mean_seconds']
 for c in out['configurations']:c.update(speedup_vs_sequential=base/c['mean_seconds'],time_reduction_pct=100*(1-c['mean_seconds']/base))
 t=out['configurations'][0]['stages']['t'];v=out['configurations'][0]['stages']['v']
 out['three_transformer_admission']={'status':'not launched; conservative sum of per-stage peaks exceeds device memory','formula':'3 * (T CUDA peak allocated + reported resident T weights) + V CUDA peak allocated + reported resident V weights','estimated_concurrent_peak_with_one_vae_gib':3*(t['max_torch_allocated_gib']+t['model_loaded_gib'])+v['max_torch_allocated_gib']+v['model_loaded_gib'],'device_capacity_gib':97887/1024,'caveat':'Conservative simultaneous-peak allocation estimate, not an observed OOM or proof that staggered allocation cannot fit. Torch allocators and model-manager weights use different allocators in this dynamic loader; no claim about redesigned lower-memory implementations.'}
initial=json.loads((P/'runs/batch_1t2v/result.json').read_text());irs=[r for r in initial['records'] if r['repeat']!='warmup']
out['initial_batch_handoff']={'name':'batch_1t2v','transformers':1,'vaes':2,'mode':'batch','n':3,'seconds':[r['seconds'] for r in irs],'mean_seconds':statistics.mean(r['seconds'] for r in irs),'peak_vram_gib':max(r['peak_vram_gib'] for r in irs),'note':'Initial saved chunk tensors retained both batch items in their backing storage. Corrected compact handoff is the primary batch comparison; initial evidence retained.'}
if (P/'timestamp_normalization.json').exists():
 normal=json.loads((P/'timestamp_normalization.json').read_text())
 for c in out['configurations']+[out['initial_batch_handoff']]:
  rr=[r for r in normal if r['configuration']==c['name']];assert len(rr)==3
  c['normalization_seconds']=[r['normalization_seconds'] for r in rr]
  c['mean_normalization_seconds']=statistics.mean(c['normalization_seconds'])
  c['delivery_seconds']=[r['combined_generation_plus_separately_timed_normalization_seconds'] for r in rr]
  c['mean_delivery_seconds']=statistics.mean(c['delivery_seconds'])
 out['delivery_timing_note']='Original measured generation plus separately timed lossless timestamp normalization. Original generation timings remain unchanged; this sum is not a new integrated pipeline run.'
(P/'summary.json').write_text(json.dumps(out,indent=2))
for c in out['configurations']:print(c['name'],round(c['mean_seconds'],2),round(c['peak_vram_gib'],2),round(c['time_reduction_pct'],2))

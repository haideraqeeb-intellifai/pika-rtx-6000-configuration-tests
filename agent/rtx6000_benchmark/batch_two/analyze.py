import json,statistics,hashlib,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;results={}
for mode in ['sequential','batch','concurrent','batch_modulation']:
 root=P/'runs'/('batch_modulation_v1' if mode=='batch_modulation' else f'{mode}_v2')
 if not (root/'result.json').exists():continue
 r=json.loads((root/'result.json').read_text());rows=[x for x in r['records'] if isinstance(x['repeat'],int)];assert len(rows)==3
 for x in rows:assert x['foreign_samples']==0
 metrics={}
 for k in ['time_to_both_saved_seconds','denoise_pair_seconds','decode_and_save_seconds','peak_vram_gib','gpu_util_mean_pct']:
  v=[x[k] for x in rows];metrics[k]={'mean':statistics.mean(v),'median':statistics.median(v),'stddev':statistics.stdev(v),'min':min(v),'max':max(v),'values':v}
 trace=json.loads((root/'gpu_trace.json').read_text());assert not any(s.get('foreign_pids') for s in trace)
 phase=[]
 for row in rows:
  cut=row['started']+row['denoise_pair_seconds'];samples=[s for s in trace if row['started']<=s['time']<cut]
  phase.append({'repeat':row['repeat'],'denoise_gpu_util_mean_pct':statistics.mean(s['util_pct'] for s in samples),'denoise_peak_gib':max(s['used_gib'] for s in samples)})
 forwards=[[x['telemetry']['fast_inference'] for x in r['denoise']] for r in rows]
 results[mode]={'n':len(rows),'metrics':metrics,'phase_metrics':phase,'fast_counters':forwards,'batch_stats':[x.get('batch_stats') for row in rows for x in row['denoise']], 'validation':json.loads((root/'validation.json').read_text()) if (root/'validation.json').exists() else None}
if 'sequential' in results:
 base=results['sequential']['metrics']
 for mode,r in results.items():
  r['speedup_vs_sequential']=base['time_to_both_saved_seconds']['mean']/r['metrics']['time_to_both_saved_seconds']['mean'];r['denoise_speedup_vs_sequential']=base['denoise_pair_seconds']['mean']/r['metrics']['denoise_pair_seconds']['mean']
validation={'same_audio_across_modes':None}
if len(results)>=3 and all(r['validation'] for r in results.values()):
 audios={}
 for mode,r in results.items():
  for v in r['validation']:audios.setdefault(Path(v['file']).name,set()).add(v['decoded_audio_sha256'])
 validation['same_audio_across_modes']=all(len(v)==1 for v in audios.values());assert validation['same_audio_across_modes']
summary={'workload':{'source':'followup_fast_comparison_fast_1','chunks':[0,1],'requested_frames_each':55,'native_frames_each':56,'fps':24,'seeds':[9901,9902],'steps':2,'mode':'fast','anchors':'Original portrait at first and last native frame; independent chunks.','text_and_audio_preparation':'Existing GPU-prepared artifacts reused identically; excluded from warm request timings.','timing_boundary':'Dispatch first denoise through completion of both decoded, encoded and saved 1080x1920 MP4s; model loading/warm-up and post-run validation excluded.','decode_policy':'One shared resident VAE; decode chunks sequentially after both denoises finish, identical for all modes.','mps':'Enabled for all configurations, 100% active-thread allowance.','repeats':'One warm-up and three measured repetitions per resident configuration. Configurations run sequential, batch, concurrent; no simultaneous unrelated tests.'},'results':results,'validation':validation}
(P/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:{'total':v['metrics']['time_to_both_saved_seconds'],'denoise':v['metrics']['denoise_pair_seconds'],'peak':v['metrics']['peak_vram_gib'],'speedup':v['speedup_vs_sequential']} for k,v in results.items()},indent=2))

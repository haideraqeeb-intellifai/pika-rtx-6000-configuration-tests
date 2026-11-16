"""Build the separate, self-contained fixed-chunk report from measured evidence."""
import csv,datetime,hashlib,html,json,shutil,subprocess,zipfile
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parent;DEST=ROOT/'h3-pika-flex-five-ten-rtx6000';DEST.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def esc(x):return html.escape(str(x))
def table(head,rows):return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'
def copy(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
def label(c):return f"{c['transformers']} transformer{'s' if c['transformers']>1 else ''} + {c['vaes']} VAE{'s' if c['vaes']>1 else ''} · "+('batch two' if c['mode']=='batch' else 'concurrent denoising' if c['mode']=='concurrent' else 'sequential denoising')+(' · initial handoff' if c['name']=='batch_1t2v' else '')
def fmt(v,n=2):return '—' if v is None else f'{v:.{n}f}'
if __name__=='__main__':
 s=read(P/'summary.json');cs=s['configurations'];assert len(cs)==4
 profiles=read(P/'profiles_summary.json');profiles={k:profiles[k] for k in ['sequential_1t2v','concurrent_2t1v','concurrent_2t2v']};quality=read(P/'quality.json');ncu=read(P/'ncu_summary.json');vals=[]
 for c in cs+[s['initial_batch_handoff']]:vals.extend(read(P/'runs'/c['name']/'validation.json'))
 assert len(vals)==90 and len(quality['latents'])==75
 (P/'validation.json').write_text(json.dumps(vals,indent=2))
 for f in ['summary.json','profiles_summary.json','quality.json','ncu_summary.json','validation.json','PLAN.md','template_provenance.json','reference_report_provenance.json','preparation.json','sage3_launch_finding.json','timestamp_normalization.json']:
  copy(P/f,DEST/f)
 with (DEST/'timings.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['configuration','repeat','video_seconds','generation_seconds','timestamp_normalization_seconds','combined_delivery_seconds','peak_vram_gib','gpu_util_mean_pct','last_denoise_finished_seconds'])
  for c in cs+[s['initial_batch_handoff']]:
   for r in read(P/'runs'/c['name']/'results.json'):
    if r['repeat']=='warmup':continue
    w.writerow([c['name'],r['repeat'],50,r['seconds'],c['normalization_seconds'][r['repeat']-1],c['delivery_seconds'][r['repeat']-1],r['peak_vram_gib'],r['gpu_util_mean_pct'],r['last_denoise_finished_seconds']])
 allvideos=[];videos=[]
 for c in cs+[s['initial_batch_handoff']]:
  for rep in range(1,4):
   filename=f"{c['name']}_repeat_{rep}.mp4";copy(P/'runs'/c['name']/f'repeat_{rep}/video.mp4',DEST/'media'/filename)
   allvideos.append(f'<li><a href="media/{filename}">{esc(label(c))} — repetition {rep}</a></li>')
  if c['name']=='batch_1t2v':continue
  poster=DEST/'media'/f"{c['name']}.jpg"
  subprocess.run(['ffmpeg','-v','error','-y','-ss','5','-i',str(DEST/'media'/f"{c['name']}_repeat_1.mp4"),'-frames:v','1','-vf','scale=270:480',str(poster)],check=True)
  videos.append(f'<div class="video-card"><h3>{esc(label(c))}</h3><p><b>{c["mean_delivery_seconds"]:.2f} s generation + remux</b> · {c["peak_vram_gib"]:.1f} GiB peak</p><video controls preload="metadata" poster="media/{poster.name}" src="media/{c["name"]}_repeat_1.mp4"></video></div>')
 base=cs[0];batch=cs[1];best=min(cs,key=lambda c:c['mean_seconds']);bchange=batch['time_reduction_pct']
 rows=[]
 for c in cs:rows.append([esc(label(c)),f'<b>{c["mean_delivery_seconds"]:.2f} s</b><br><small>{c["mean_seconds"]:.2f} s generation + {c["mean_normalization_seconds"]:.2f} s remux</small>',f'{min(c["delivery_seconds"]):.2f}–{max(c["delivery_seconds"]):.2f} s',f'{c["time_reduction_pct"]:+.1f}%',f'{c["peak_vram_gib"]:.1f} GiB',f'{c["mean_gpu_util_pct"]:.1f}%'])
 profile_rows=[]
 for name,p in profiles.items():
  def metric(term):
   return next((v['initial_transformer_only_mean'] for k,v in p['metrics'].items() if term in k),None)
  ov=p['kernel_overlap'];denoise=[r for r in p['ranges'] if r['role']=='denoise'];span=(max(r['end'] for r in denoise)-min(r['start'] for r in denoise))/1e9
  profile_rows.append([esc(name.replace('_',' ')),fmt(metric('SMs Active'))+'%',fmt(metric('SM Issue'))+'%',fmt(metric('Tensor Active'))+'%',fmt(p['host_call_overlap'].get('multiple_transformer_kernel_seconds',0))+' s',fmt(ov.get('multiple_transformer_kernel_seconds',0))+' s',fmt(ov.get('transformer_vae_kernel_overlap_seconds',0))+' s'])
 # Package evidence without large tensors/weights or native profiler environment data.
 with zipfile.ZipFile(DEST/'reproduction.zip','w',zipfile.ZIP_DEFLATED) as z:
  for folder,pattern in [(P,'*.py'),(ROOT,'*.py'),(ROOT/'batch_two','*.py')]:
   for f in folder.glob(pattern):z.write(f,'agent/rtx6000_benchmark/'+str(f.relative_to(ROOT)))
  z.write(P/'report_template.html','agent/rtx6000_benchmark/five_ten/report_template.html')
  z.write(ROOT/'nsight_profile/analyze_traces.py','agent/rtx6000_benchmark/nsight_profile/analyze_traces.py')
  for f in (P/'runs').rglob('batch_source/*.py'):z.write(f,'evidence/runs/'+str(f.relative_to(P/'runs')))
  for f in (P/'reference_source').glob('*'):z.write(f,'reference_source/'+f.name)
  for f in (P/'templates').glob('*'):z.write(f,'agent/rtx6000_benchmark/five_ten/templates/'+f.name)
  for f in (P/'inputs').glob('*.wav'):z.write(f,'agent/rtx6000_benchmark/five_ten/inputs/'+f.name)
  z.write(ROOT/'inputs/raccoon_headshot.jpg','agent/rtx6000_benchmark/inputs/raccoon_headshot.jpg')
  z.write(ROOT/'inputs/report_audio.wav','agent/rtx6000_benchmark/inputs/report_audio.wav')
  for f in (P/'prepared').glob('*.json'):z.write(f,'evidence/prepared/'+f.name)
  for f in P.glob('*.json'):z.write(f,'evidence/'+f.name)
  for f in P.glob('*.log'):z.write(f,'evidence/logs/'+f.name)
  for f in P.glob('*.md'):z.write(f,f.name)
  for f in (P/'runs').rglob('*.json'):
   if '/runtime_' not in str(f):z.write(f,'evidence/runs/'+str(f.relative_to(P/'runs')))
  for f in (P/'profiles').rglob('*'):
   if f.suffix in ('.json','.csv','.log','.txt'):z.write(f,'evidence/profiles/'+str(f.relative_to(P/'profiles')))
  for f in (ROOT/'fast_comparison_source').glob('*'):
   if f.is_file():z.write(f,'environment_reference/'+f.name)
  for f in ['runtime.py','fast_inference.py']:
   path=ROOT.parents[1]/'ml_modules/minimax_h3_turbo'/f;z.write(path,'current_source/'+f)
 ncurows=[]
 for x in ncu:
  m=x['metrics'];grid=float(m.get('launch__grid_size','0').replace(',','') or 0)
  if grid<188 and 'compute_attn_ws' not in x['name']:continue
  short='INT8 GEMM' if 'DefaultGemm' in x['name'] else 'SA2 attention' if 'qk_int' in x['name'] else 'SA3 FP4 attention' if 'compute_attn_ws' in x['name'] else 'Elementwise add'
  ncurows.append([short,esc(x['grid']),esc(m.get('launch__registers_per_thread','—')),esc(m.get('launch__shared_mem_per_block','—')+' '+x.get('units',{}).get('launch__shared_mem_per_block','')),esc(m.get('sm__warps_active.avg.pct_of_peak_sustained_active','—')),esc(m.get('sm__throughput.avg.pct_of_peak_sustained_elapsed','—')),esc(m.get('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed','—'))])
 replacements={
 'DATE':datetime.datetime.now(datetime.timezone.utc).strftime('%d %B %Y'),
 'HEADLINE':'One transformer + two VAEs remains fastest.',
 'CONCLUSION':f'Batch-two on 1T/2V took <b>{batch["mean_delivery_seconds"]:.2f} seconds</b>, versus <b>{base["mean_delivery_seconds"]:.2f} seconds</b> for sequential denoising: <b>{abs(bchange):.1f}% '+('less' if bchange>=0 else 'more')+f' time</b>. The fastest measured layout was <b>{esc(label(best))}</b> at <b>{best["mean_delivery_seconds"]:.2f} seconds</b>. These are three-run means for the complete 50-second video: original generation plus a separately timed, lossless timestamp correction of about 0.06 seconds.',
 'CARDS':''.join(f'<div class="card"><div class="big">{v}</div><div>{k}</div></div>' for k,v in [('Fixed video workload','5 × 10 s'),('Sequential 1T/2V',f'{base["mean_delivery_seconds"]:.2f} s'),('Batch-two 1T/2V',f'{batch["mean_delivery_seconds"]:.2f} s')]),
 'RESULT_TABLE':table(['Configuration','Generation + timestamp fix','Observed range (n=3)','Time reduction vs baseline','Peak device VRAM','Mean GPU busy'],rows),
 'REFINEMENT':f'<p>The initial batch implementation took <b>{s["initial_batch_handoff"]["mean_seconds"]:.2f} seconds</b> on average. Its four paired chunk handoffs each stored both batch items in their backing storage (51.1 MB per file). We compacted each saved item to 25.5 MB, verified bit-identical tensor values on CPU, and repeated the full warmup plus three videos. The primary batch row above uses the corrected compact handoff. Initial runs remain in the downloads and raw evidence.</p>',
 'REPEATS':table(['Configuration','Run 1','Run 2','Run 3','Sample standard deviation'],[[esc(label(c))]+[f'{x:.2f} s' for x in c['seconds']]+[f'{c["std_seconds"]:.2f} s'] for c in cs]),
 'STAGE_FINDING':f'<p>With two transformers and two VAEs, denoising finished <b>{base["mean_last_denoise_seconds"]-cs[3]["mean_last_denoise_seconds"]:.2f} seconds earlier</b>. The remaining decode/assembly tail took <b>{cs[3]["mean_decode_tail_seconds"]-base["mean_decode_tail_seconds"]:.2f} seconds longer</b>, leaving the complete video <b>{cs[3]["mean_seconds"]-base["mean_seconds"]:.2f} seconds slower</b>. Batching changed denoising completion by only {abs(batch["mean_last_denoise_seconds"]-base["mean_last_denoise_seconds"]):.2f} seconds and also lengthened the tail.</p>',
 'STAGES':table(['Configuration','Last denoise completes','Remaining decode/assembly tail','Sum of T call times','Sum of V call times'],[[esc(label(c)),f'{c["mean_last_denoise_seconds"]:.2f} s',f'{c["mean_decode_tail_seconds"]:.2f} s',f'{c["stages"]["t"]["summed_call_seconds_per_video"]:.2f} s',f'{c["stages"]["v"]["summed_call_seconds_per_video"]:.2f} s'] for c in cs]),
 'MEMORY':table(['Configuration','Peak device VRAM','T CUDA allocated / reserved peak per process','V CUDA allocated / reserved peak per process','T / V node-end model offload'],[[esc(label(c)),f'{c["peak_vram_gib"]:.2f} GiB',f'{c["stages"]["t"]["max_torch_allocated_gib"]:.2f} / {c["stages"]["t"]["max_torch_reserved_gib"]:.2f} GiB',f'{c["stages"]["v"]["max_torch_allocated_gib"]:.2f} / {c["stages"]["v"]["max_torch_reserved_gib"]:.2f} GiB',f'{c["stages"]["t"]["max_node_end_offloaded_gib"]:.3f} / {c["stages"]["v"]["max_node_end_offloaded_gib"]:.3f} GiB'] for c in cs])+f'<p>Whole-device memory is sampled every 200 ms; brief peaks can be missed. Dynamic-loader model weights use additional allocation accounting, so PyTorch peaks alone are not total VRAM. A conservative sum of per-stage peaks for three transformers plus one VAE is <b>{s["three_transformer_admission"]["estimated_concurrent_peak_with_one_vae_gib"]:.1f} GiB</b> against {s["three_transformer_admission"]["device_capacity_gib"]:.1f} GiB reported device memory (638 MiB is driver-reserved): that layout was not launched. This is a memory estimate, not an observed OOM.</p>',
 'COLD':f'<p>Shared GPU preparation took <b>{s["preparation_seconds"]:.2f} seconds</b> and is excluded equally from all resident timings.</p>'+table(['Configuration','Worker startup + first full video','Warmup video alone'],[[esc(label(c)),f'{c["cold_first_video_seconds"]:.2f} s',f'{c["warmup_seconds"]:.2f} s'] for c in cs]),
 'VIDEOS':''.join(videos),'ALL_VIDEOS':'<ul>'+''.join(allvideos)+'</ul>',
 'VALIDATION':f'<p class="status">All {len(vals)} measured MP4s passed integrity and frame/timestamp checks ({len({x["sha256"] for x in vals})} distinct byte sequences fully decoded; identical copies verified by SHA-256): 15 complete videos and 75 component chunks, including the initial batch handoff. All 75 measured latent pairs are finite and their video latents are bit-identical to baseline. All delivered full videos are byte-identical across the tested configurations and repeats.</p><p>Complete videos: 1,200 frames, 24 fps, 1080 × 1920, 50 seconds. Audio alignment checks found zero lag at 8 kHz, with minimum waveform correlation {min(x["audio_zero_lag_correlation"] for x in vals if "audio_zero_lag_correlation" in x):.6f}. Export uses the supplied source audio; generated audio latents are not used.</p>'+table(['Compared with sequential output','180×320 full-video SSIM (repeat 1)'],[[esc(x['configuration']),f'{x["ssim_vs_sequential"]:.6f}'] for x in quality['visual_comparisons']])+'<p class="muted">SSIM measures pixel similarity, not perceptual equivalence or lip-sync accuracy.</p>',
 'PROFILE_CONCLUSION':'One 10-second chunk already reached <b>96.2% SMs Active</b> before any VAE started. Two transformers raised this to about <b>99.7%</b>, while Tensor Active rose only from <b>60.7% to 62.5%</b>. The copies did overlap, but useful end-to-end speed did not improve.',
 'PROFILE_TABLE':table(['Profiled configuration','SMs Active before VAE','SM Issue before VAE','Tensor Active before VAE','Two T calls in flight','Two T kernels simultaneously','Transformer + VAE kernel overlap'],profile_rows),
 'NCU':table(['Selected kernel','Grid','Registers/thread','Shared memory/block','Achieved occupancy (%)','SM throughput (%)','DRAM throughput (%)'],ncurows)+'<p>These are selected isolated kernel instances, not an average across the video. The accompanying Nsight Systems launch evidence records maximum shared memory and registers for the dominant kernels in the actual pipeline.</p>',
 'METHOD':''.join('<tr><th>'+esc(k)+'</th><td>'+esc(v)+'</td></tr>' for k,v in [('GPU','One RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB reported memory / 95.59 GiB, 638 MiB driver reservation, 188 SMs, 600 W'),('Video contract','Exactly five 10-second chunks; 240 delivered frames each; 1,200 frames / 50 seconds'),('Native model frames','243 frames per chunk, trimmed by the existing export path to 240'),('Model and optimization','H3 Pika Flex; two steps; SA2 first step, SA3 FP4 second step; NVFP4 VAE projections'),('Anchors and seeds','Original portrait at both native endpoints for each independent chunk; seeds 9901 through 9905'),('Dimensions','Native 736×1280, cropped to 720×1280, exported at 1080×1920'),('Audio','Same 49.408-second reference recording, padded with 0.592 seconds of trailing silence to 50 seconds; five identical 10-second slices across configurations'),('Preparation','GPU text/image/audio conditioning prepared once and reused byte-for-byte; text encoder process exits before resident generation workers start'),('Timing boundary','Five prepared chunks dispatched through final assembled MP4; includes latent serialization, IPC, decode, encode and mux. Excludes initial preparation, warmup and post-run validation.'),('Repetitions','One complete warmup and three measured full videos; no best-run selection'),('Order','Sequential 1T/2V → initial batch 1T/2V → concurrent 2T/1V → concurrent 2T/2V → compact batch 1T/2V; no randomized blocks'),('GPU checks','Three consecutive idle host samples before each fresh workload; continuous process-ownership checks throughout'),('Profiling','Separate full-workload Nsight Systems replays; 1 kHz GPU metrics; NCU isolated kernels; never included in performance means')]),
 'COMMANDS':'# Use a fresh copy of the recorded checkout and fast overlay. Existing runs refuse overwrite.\n# Python interpreter, package versions and overlay are recorded in reproduction.zip/evidence/provenance.json.\npython -m agent.rtx6000_benchmark.five_ten.prepare\npython -m agent.rtx6000_benchmark.five_ten.experiment --mode sequential --name sequential_1t2v --transformers 1 --vaes 2 --repeats 3\nH3_BATCH_MODULATION=1 python -m agent.rtx6000_benchmark.five_ten.experiment --mode batch --name batch_1t2v --transformers 1 --vaes 2 --repeats 3\npython -m agent.rtx6000_benchmark.five_ten.experiment --mode concurrent --name concurrent_2t1v --transformers 2 --vaes 1 --repeats 3\npython -m agent.rtx6000_benchmark.five_ten.experiment --mode concurrent --name concurrent_2t2v --transformers 2 --vaes 2 --repeats 3\n# finish.py checks compact handoff equality, runs the corrected batch warmup + 3 repeats,\n# summarizes, profiles 3 layouts with NSYS, profiles selected isolated kernels with NCU,\n# then normalizes delivery timestamps and validates all videos, latents and input hashes.\npython -m agent.rtx6000_benchmark.five_ten.finish\npython -m agent.rtx6000_benchmark.five_ten.summarize\npython -m agent.rtx6000_benchmark.five_ten.build_report',
 'LIMITATIONS':'Three repetitions per layout on one fixed podcast; performance and quality may differ for other content. No eight-GPU performance claim.',
 'PROFILE_JSON':json.dumps(profiles).replace('</','<\/')}
 template=(P/'report_template.html').read_text()
 for k,v in replacements.items():template=template.replace('@@'+k+'@@',v)
 assert '@@' not in template
 (DEST/'index.html').write_text(template)
 (DEST/'README.txt').write_text('Fixed five-by-ten-second H3 Pika Flex benchmark. Open index.html. Raw timing/quality/profile summaries are JSON. Reproduction ZIP includes code, input audio/portrait, commands and per-run evidence. Native Nsight traces and model/latent tensors are not bundled. All speed means exclude profiling.\n')
 (DEST/'report-files.json').write_text(json.dumps(sorted({str(f.relative_to(DEST)) for f in DEST.rglob('*') if f.is_file()}|{'report-files.json'}),indent=2))
 print(DEST,flush=True)

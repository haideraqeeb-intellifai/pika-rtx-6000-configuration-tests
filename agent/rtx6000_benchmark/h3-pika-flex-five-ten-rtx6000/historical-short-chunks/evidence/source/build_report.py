import csv,hashlib,html,json,shutil,statistics,subprocess,zipfile
from pathlib import Path
P=Path(__file__).resolve().parent;O=P.parent/'h3-pika-flex-batch-two-rtx6000';O.mkdir(exist_ok=True)
s=json.loads((P/'summary.json').read_text());profile=json.loads((P/'profile_summary.json').read_text());latent=json.loads((P/'latent_validation.json').read_text());R=s['results'];modes=['sequential','batch','batch_modulation','concurrent'];names={'sequential':'One transformer · sequential','batch':'Batch of two · initial','batch_modulation':'Batch of two · refined','concurrent':'Two transformer copies · concurrent'}
assert len(R)==4 and s['validation']['same_audio_across_modes']
for mode in modes:assert len(R[mode]['validation'])==6
for name in ['summary.json','profile_summary.json','latent_validation.json','audio_input_alignment.json']:shutil.copy2(P/name,O/name)
media=O/'media';media.mkdir(exist_ok=True);evidence=O/'evidence';evidence.mkdir(exist_ok=True);src=evidence/'source';src.mkdir(exist_ok=True)
for f in P.glob('*.py'):shutil.copy2(f,src/f.name)
for mode in modes:
 root=P/'runs'/('batch_modulation_v1' if mode=='batch_modulation' else f'{mode}_v2');dest=evidence/mode;dest.mkdir(exist_ok=True)
 for f in ['result.json','results.json','validation.json','gpu_trace.json','empty_gpu_gate.json','workers.json']:shutil.copy2(root/f,dest/f)
 shutil.copy2(P/('batch_modulation_v1.log' if mode=='batch_modulation' else mode+'_v2.log'),dest/'execution.log')
 for r in [1,2,3]:
  for i in [0,1]:shutil.copy2(root/f'repeat_{r}/chunk_{i}.mp4',media/f'{mode}_r{r}_chunk{i}.mp4')
 for f in root.glob('repeat_1/graph_*.json'):shutil.copy2(f,dest/f.name)
 if (root/'batch_source').exists():shutil.copytree(root/'batch_source',dest/'batch_source',dirs_exist_ok=True)
for name in ['batch_compatibility_v1.log','sequential_v1.log','profiling.log']:shutil.copy2(P/name,evidence/name)
inputs=evidence/'inputs';inputs.mkdir(exist_ok=True);source=P.parent/'runs/followup_fast_comparison_fast_1'
for i in [0,1]:
 shutil.copy2(source/f'audio_{i}.wav',inputs/f'audio_{i}.wav');shutil.copy2(source/f'prepared/prepared_{i}.json',inputs/f'prepared_{i}.json')
shutil.copy2(P.parent/'inputs/raccoon_headshot.jpg',inputs/'raccoon_headshot.jpg')
provenance=json.loads((P.parent/'fast_comparison_source/provenance.json').read_text());provenance['benchmark_source_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in P.glob('*.py')};provenance['prepared_artifacts_sha256']={f'prepared_{i}.pt':hashlib.sha256((source/f'prepared/prepared_{i}.pt').read_bytes()).hexdigest() for i in [0,1]};provenance['captured_gpu']='RTX PRO 6000 Blackwell Workstation Edition, 96 GB, 600 W';provenance['scope']='Benchmark-only batch adaptation; production files unchanged by this task.';(O/'provenance.json').write_text(json.dumps(provenance,indent=2))
with (O/'timings.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['configuration','repeat','both_saved_seconds','denoise_pair_seconds','decode_save_seconds','sampled_peak_device_gib','gpu_util_mean_pct'])
 for mode in modes:
  for row in json.loads((P/'runs'/('batch_modulation_v1' if mode=='batch_modulation' else f'{mode}_v2')/'results.json').read_text()):
   if isinstance(row['repeat'],int):w.writerow([mode,row['repeat'],*[row[k] for k in ['time_to_both_saved_seconds','denoise_pair_seconds','decode_and_save_seconds','peak_vram_gib','gpu_util_mean_pct']]])
readme='''H3 Pika Flex batch-of-two experiment

Run from the script2video repository recorded in provenance.json. Reuse the existing
.h3-pika-env interpreter and fast_dependencies/site-packages PYTHONPATH overlay.
No dependency installation or production change is required.

Source under evidence/source contains the experiment, B=2 adapter, matrix runner,
profiling commands, analysis and output-validation scripts. Absolute paths in graphs
identify the original artifacts and must be rebased on another host.

The original GPU-prepared conditioning comes from followup_fast_comparison_fast_1.
The two raw audio inputs, portrait, preparation metadata and artifact hashes are
included. Model weights and prepared tensor caches are not bundled; reproduce the
cache using the repository's agent.rtx6000_benchmark.gpu_preencode workflow with the
recorded original inputs, prompt, anchors and settings, or use the retained local
prepared cache. Check hashes before replaying. The default pair is indices 0 and 1.

For the refined batch, set H3_BATCH_MODULATION=1 (initial batch leaves it unset).

Run: python -m agent.rtx6000_benchmark.batch_two.experiment --mode sequential
     --name <fresh-name> --repeats 3
Repeat with --mode batch and --mode concurrent. A private MPS session is created
and shut down for each configuration. Empty-GPU gates and ownership monitoring are
mandatory. Diagnostic profiler runs are excluded from the timing means.

Warm-up and failed attempts are preserved. sequential_v1 aborted during warm-up
because its initial monitor did not recognize its own NVENC child process; none of
that attempt's timings are in the results. The first NSYS attempt stopped at the
idle gate before model loading (profiler allocation 4 MiB versus normal 2 MiB).
Successful diagnostics used the established 4 MiB profiler idle baseline.

Media includes all 24 measured videos. Native raw .nsys-rep / .sqlite archives and
saved latent tensors remain in the local batch_two directory. Numerical profile
extracts are included; captured host environment tables are not published.
'''
(O/'README.txt').write_text(readme)
with zipfile.ZipFile(O/'reproduction_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(evidence.rglob('*')):
  if f.is_file():z.write(f,str(f.relative_to(O)))
 for name in ['summary.json','profile_summary.json','latent_validation.json','audio_input_alignment.json','timings.csv','provenance.json','README.txt']:z.write(O/name,name)

def metric(mode,key):return R[mode]['metrics'][key]['mean']
slow=(metric('batch_modulation','time_to_both_saved_seconds')/metric('sequential','time_to_both_saved_seconds')-1)*100
launch_reduction=(1-profile['batch_modulation']['kernel_count']/profile['sequential']['kernel_count'])*100
launch_api_reduction=(1-profile['batch_modulation']['launch_api_calls']/profile['sequential']['launch_api_calls'])*100
rows='';repeats='';prof='';cards='';bars='';maxval=18
for j,mode in enumerate(modes):
 r=R[mode];m=r['metrics'];d=statistics.mean(x['denoise_gpu_util_mean_pct'] for x in r['phase_metrics'])
 rows+=f'<tr><td><b>{names[mode]}</b></td><td><b>{m["time_to_both_saved_seconds"]["mean"]:.2f} s</b><small>{m["time_to_both_saved_seconds"]["min"]:.2f}–{m["time_to_both_saved_seconds"]["max"]:.2f} s</small></td><td>{m["denoise_pair_seconds"]["mean"]:.2f} s</td><td>{m["decode_and_save_seconds"]["mean"]:.2f} s</td><td>{m["peak_vram_gib"]["max"]:.1f} GiB</td><td>{d:.1f}%</td></tr>'
 for k in range(3):repeats+=f'<tr><td>{names[mode]}</td><td>{k+1}</td><td>{m["time_to_both_saved_seconds"]["values"][k]:.3f}</td><td>{m["denoise_pair_seconds"]["values"][k]:.3f}</td><td>{m["peak_vram_gib"]["values"][k]:.3f}</td><td>{m["gpu_util_mean_pct"]["values"][k]:.2f}</td><td><a href="media/{mode}_r{k+1}_chunk0.mp4">Chunk 0</a> · <a href="media/{mode}_r{k+1}_chunk1.mp4">Chunk 1</a></td></tr>'
 y=25+j*67;td=metric(mode,'denoise_pair_seconds');tv=metric(mode,'decode_and_save_seconds');scale=42
 bars+=f'<text x="0" y="{y+22}" font-size="14">{["Sequential","Batch: initial","Batch: refined","Two copies"][j]}</text><rect x="150" y="{y}" width="{td*scale}" height="36" rx="3" fill="#2768c6"/><rect x="{150+td*scale}" y="{y}" width="{tv*scale}" height="36" fill="#16857b"/><text x="162" y="{y+24}" fill="white">{td:.2f} s</text><text x="{162+td*scale}" y="{y+24}" fill="white">{tv:.2f} s</text><text x="{165+(td+tv)*scale}" y="{y+24}">{td+tv:.2f} s</text>'
 for i in ([] if mode=='batch' else [0,1]):
  v=next(v for v in r['validation'] if f'repeat_1/chunk_{i}.mp4' in v['file']);ssim=v.get('ssim_vs_sequential');q='Reference output' if mode=='sequential' else f'SSIM vs sequential: {ssim:.4f}'
  poster=media/f'{mode}_chunk{i}.jpg'
  if not poster.exists():subprocess.run(['ffmpeg','-v','error','-i',str(media/f'{mode}_r1_chunk{i}.mp4'),'-frames:v','1','-vf','scale=270:480',str(poster)],check=True,capture_output=True)
  cards+=f'<article><h3>{names[mode]}</h3><p class="muted">Chunk {i} · seed {9901+i} · {q}</p><video controls preload="none" playsinline poster="media/{mode}_chunk{i}.jpg" src="media/{mode}_r1_chunk{i}.mp4"></video><p><a href="media/{mode}_r1_chunk{i}.mp4">Download video</a></p></article>'
for mode in ['sequential','batch','batch_modulation']:
 p=profile[mode];prof+=f'<tr><td>{names[mode]}</td><td>{p["kernel_count"]:,}</td><td>{p["launch_api_calls"]:,}</td><td>{p["summed_launch_api_seconds"]*1000:.1f} ms</td><td>{p["kernel_union_seconds"]:.3f} s</td><td>{p["kernel_mean_microseconds"]:.1f} µs</td></tr>'
ssims=[v['ssim_vs_sequential'] for v in R['batch_modulation']['validation']];lat=[r['video']['cosine'] for r in latent if r['mode']=='batch_modulation']
html_page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H3 fast mode · Does batching two chunks help?</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#17314c;font:16px/1.6 system-ui,sans-serif}main{max-width:1240px;margin:auto;padding:40px 26px 70px}h1{font-size:43px;line-height:1.15;letter-spacing:-1.3px;margin:15px 0 20px}h2{font-size:26px;line-height:1.3;margin:0 0 15px}h3{font-size:17px;margin:6px 0}p{margin:12px 0 17px}a{color:#195da9}section,.card,article{background:white;border:1px solid #dce4ee;border-radius:13px;padding:24px}section{margin:23px 0}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:1.5px;color:#08796f;font-weight:750}.lead{font-size:20px;max-width:1000px}.muted,small{color:#596c80;font-size:13px}small{display:block}.cards,.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:17px}.cards{margin:25px 0}.card strong{font-size:34px;display:block}.callout{background:#edf7f4;border-left:4px solid #16857b;padding:15px 20px;border-radius:4px;margin:18px 0}.warning{background:#fff5e9;border-color:#c96a18}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:13px 10px;border-bottom:1px solid #e0e7f0}th{background:#f1f5fa}td{font-variant-numeric:tabular-nums}svg{width:100%;height:auto;font-family:system-ui,sans-serif;fill:#17314c}.legend{font-size:14px;color:#52667b}.split{display:grid;grid-template-columns:1fr 1fr;gap:25px}video{width:100%;max-height:430px;background:#111;border-radius:8px}.videos{display:grid;grid-template-columns:repeat(3,1fr);gap:17px}.videos article{padding:17px}.diagram{padding:15px;background:#f3f6fa;border-radius:8px;text-align:center;font-size:14px}.chunk{display:inline-block;background:#dfeafc;color:#17569a;padding:8px 12px;border-radius:5px;margin:5px}.transformer{background:#1b476f;color:white;padding:14px 8px;border-radius:6px;margin:10px 0}.downloads{display:flex;gap:18px;flex-wrap:wrap}details{margin-top:16px}summary{cursor:pointer;font-weight:700}code{font-size:13px}footer{font-size:13px;color:#52667b;margin-top:30px}@media(max-width:800px){h1{font-size:32px}.cards,.grid3,.split,.videos{grid-template-columns:1fr}main{padding:24px 14px}section{padding:18px}}
</style></head><body><main><div class="eyebrow">Separate experiment · H3 Pika Flex · Fast mode · One RTX PRO 6000</div>
<h1>Batching two independent chunks<br>did not make this workload faster.</h1>
<p class="lead">One resident transformer processed a real batch of two, with separate audio conditioning, anchors and random seeds. It reduced kernel launches, but the pair finished <b>__SLOW__% slower</b> than processing the chunks sequentially.</p>
<p class="muted">11 November 2026 · RTX PRO 6000 Blackwell Workstation Edition, 96 GB, 600 W · two denoising steps · SA2 → SA3 attention · NVFP4 VAE projections · three measured repetitions per implementation.</p>
<div class="cards"><div class="card"><span>One transformer, sequential</span><strong>__SEQ__ s</strong><small>Mean time until both videos are saved</small></div><div class="card"><span>One transformer, refined batch of two</span><strong>__BATCH__ s</strong><small>__SLOW__% slower · peak 38.2 GiB</small></div><div class="card"><span>Two transformer copies</span><strong>__CON__ s</strong><small>Only about 0.6% faster · peak 59.9 GiB</small></div></div>
<section><div class="eyebrow">1 · Measured performance</div><h2>Same two chunks. Same decoder. Different denoising layouts.</h2>
<p>Each chunk is 55 delivered frames (2.292 s) at 24 fps. H3 generates 56 native frames and the existing export path trims to 55. Both use the original portrait as independent first/last native-frame anchors, with distinct audio slices and seeds 9901 / 9902.</p>
<div class="scroll"><table><thead><tr><th>Configuration</th><th>Both videos saved<br>mean + observed range</th><th>Denoising pair</th><th>Decode + save</th><th>Peak device VRAM</th><th>Denoising GPU util.</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<svg viewBox="0 0 1080 300" role="img" aria-label="Denoising and decoding time for the three configurations">__BARS__</svg><p class="legend">Blue: denoising pair and latent handoff. Teal: identical sequential VAE decoding, encoding and saving. Shorter is faster.</p>
<div class="callout"><b>Recommendation:</b> keep the existing single-transformer fast setup. This batch implementation uses more memory and takes longer for the tested pair. The two-copy difference is too small to treat as a reliable practical improvement from three repetitions in one session.</div>
<p class="muted">Peak VRAM is sampled device memory, not just PyTorch allocations. NVML sampled every 200 ms; utilization is the hardware activity reading, not percent of peak SM throughput. The table’s utilization covers denoising only. Whole-pair utilization and per-process PyTorch memory peaks are in the raw results.</p>
<details><summary>All twelve measured runs</summary><div class="scroll"><table><thead><tr><th>Configuration</th><th>Repeat</th><th>Both saved (s)</th><th>Denoise (s)</th><th>Peak GiB</th><th>Whole-pair GPU util. %</th><th>Outputs</th></tr></thead><tbody>__REPEATS__</tbody></table></div></details></section>
<section><div class="eyebrow">2 · What “batch of two” means here</div><h2>This is genuine batched transformer execution.</h2><div class="grid3"><div class="diagram"><h3>Sequential</h3><div class="chunk">Chunk 0</div> → <div class="chunk">Chunk 1</div><div class="transformer">One resident transformer</div><p>Two sampler calls<br>Four main model forwards per pair</p></div><div class="diagram"><h3>Batch of two</h3><div class="chunk">[Chunk 0, Chunk 1]</div><div class="transformer">One resident transformer<br>Batch dimension = 2</div><p>One sampler call<br>Two main model forwards per pair</p></div><div class="diagram"><h3>Two copies</h3><div class="chunk">Chunk 0</div><div class="chunk">Chunk 1</div><div class="transformer">Transformer A + Transformer B</div><p>Two concurrent sampler calls<br>Four main model forwards per pair</p></div></div>
<p>All three layouts feed the same <b>one resident VAE</b>, which decodes the two chunks sequentially after both denoises complete. This holds decoding constant and isolates the transformer comparison.</p>
<p>The existing H3 wrapper and fast attention code assume batch size one. The experiment adds a local adapter: linear/MLP operations combine both chunks’ token rows, while attention keeps a separate batch axis so tokens cannot attend across chunks. Conditioning and output unpacking remain per item.</p><div class="callout"><b>Verified runtime shape:</b> attention Q/K/V = <code>[2, 56, 19563, 128]</code> for the batched main transformer. Each measured batch completed two model forwards, 50 SA2 calls, 50 SA3 calls and <b>zero attention fallbacks</b>.</div>
<p class="muted">The initial adapter kept per-item modulation operations separate. The refined adapter broadcasts matching modality/timestep operations across the two items. Per-item preparation, conditioning assembly and output unpacking still remain separate. This tests one correct experimental batching strategy, not every possible batching or kernel-fusion implementation. Production model code and dependencies were not changed.</p></section>
<section><div class="eyebrow">3 · Did fewer launches help?</div><h2>Fewer kernel launches did not translate into a faster pair.</h2>
<p>Separate Nsight Systems captures replayed one warm sequential pair and one warm pair for each batch implementation. Their instrumented timings are <b>excluded</b> from the performance averages above.</p><div class="scroll"><table><thead><tr><th>Diagnostic run</th><th>GPU kernel events</th><th>CUDA launch API calls</th><th>Summed launch API time</th><th>GPU kernel busy time</th><th>Mean kernel duration</th></tr></thead><tbody>__PROFILE__</tbody></table></div>
<div class="callout warning">The refined batched trace has <b>__LAUNCH__% fewer kernel events</b> and <b>__API__% fewer CUDA launch API calls</b>. GPU kernel busy time was 9.84 s for the refined batch versus 9.64 s sequentially. Larger batched kernels still have to process both chunks. The measured launch reduction was not sufficient to produce a warm end-to-end speedup.</div>
<p class="muted">Kernel busy time is the union of recorded CUDA execution intervals inside denoising request ranges. Summed launch API time measures time inside CUDA launch calls; it does not cover all Python scheduling overhead, nor can it be subtracted directly from total generation time. These traces do not establish whether compute, bandwidth or a particular kernel is the sole bottleneck.</p>
<p>The unprofiled single-transformer baseline already averaged about 94.7% NVML GPU utilization during denoising. The initial batching implementation averaged about 95.0%; per-run readings for the refinement are included above. That activity reading changed little and does not prove the SMs reached peak arithmetic throughput.</p></section>
<section><div class="eyebrow">4 · Output correctness and visible differences</div><h2>All 24 measured videos pass frame and audio checks.</h2><div class="split"><div><h3>Validated</h3><ul><li>55 output frames per chunk, 24 fps, 1080×1920 export.</li><li>Complete video/audio decoding and evenly spaced frame timestamps.</li><li>Audio/video durations agree within one AAC packet.</li><li>Decoded audio is identical across configurations and repeats for each input chunk.</li><li>Both soundtracks align with the correct source slice at zero-sample lag, with correlation above 0.9999 after AAC encoding.</li><li>All saved video/audio latent tensors are finite.</li></ul></div><div><h3>Not bit-identical visually</h3><p>Batching changes numerical results despite the same seeds. Batched video SSIM against sequential output ranges from <b>__SSIM__</b>. Video-latent cosine similarity ranges from <b>__COS__</b>.</p><p>The initial and refined batch videos are byte-identical. Two concurrent copies match the sequential videos exactly in these runs.</p><p class="muted">SSIM is a difference measure here, not a judgment of natural motion or lip-sync quality. Same seeds do not guarantee identical floating-point results when low-precision kernel shapes change. Exported audio is the supplied recording; generated audio latents are not used for the MP4 soundtrack.</p></div></div><p>Representative repeat 1 outputs are below. Each configuration has both chunks, and all 24 measured videos are linked in the expandable twelve-run table above. Visual smoothness and mouth-motion quality require viewing; passing timing checks does not certify perceptual lip sync.</p><div class="videos">__VIDEOS__</div></section>
<section><div class="eyebrow">5 · Scope and reproducibility</div><h2>Why there is no new full-video batching result</h2><p>The requested condition was to test full-video generation <b>if batching helps</b>. It did not improve the warm two-chunk measurements, so a full-video batch run was not launched. The existing approximately 200.3 s best full-video fast result is historical context; these 15–16 s pair timings are not a new measurement of that full video.</p>
<h3>Timing protocol</h3><ul><li>One full warm-up pair, then three measured pairs in the same resident processes for each configuration.</li><li>Existing GPU-prepared text/image/audio conditioning reused identically. Text encoding, initial process startup, lazy checkpoint loading during warm-up and post-run validation excluded from warm timings.</li><li>Measured interval: first denoise dispatch through both completed, decoded, encoded and saved MP4 files. Latent serialization and queue handoffs included.</li><li>Private CUDA MPS session for all configurations, 100% active-thread allowance. No concurrent unrelated GPU workloads were observed in the valid runs.</li><li>Three samples per implementation, in sequential → initial batch → concurrent → refined batch order; no cross-day replication or randomized block design. Small differences should not be generalized.</li><li>One short matching-length pair tested. Other lengths, larger batches and more fused implementations may behave differently.</li></ul>
<details><summary>Warm-up and excluded attempts</summary><p>The first sequential attempt stopped during warm-up because the initial ownership monitor did not recognize its own NVENC encoder subprocess. Its timings were excluded; the monitor was corrected to check descendants before the valid matrix. The first NSYS attempt stopped before loading models because profiling adds 2 MiB to the established idle allocation. The retry used the verified 4 MiB profiler idle baseline. Logs are retained in the evidence bundle.</p></details>
<h3 style="margin-top:25px">Downloads</h3><div class="downloads"><a href="reproduction_bundle.zip">Reproduction bundle + logs</a><a href="timings.csv">All measured timings CSV</a><a href="summary.json">Results + validation JSON</a><a href="profile_summary.json">Nsight launch evidence</a><a href="latent_validation.json">Latent comparisons</a><a href="audio_input_alignment.json">Source audio alignment</a><a href="provenance.json">Configuration provenance</a><a href="README.txt">Reproduction notes</a></div><p class="muted">The bundle includes benchmark code, generated batch adapters, workflow graphs, original two audio slices, portrait, prepared-input metadata and numerical evidence. Model weights, prepared tensor caches, saved latent tensors and native profiler archives remain local. Nothing was deployed or merged.</p></section><footer>Measured on one RTX PRO 6000. A negative result for this batch implementation and pair, not a claim that batching is universally slower.</footer></main></body></html>'''
for a,b in {'__SLOW__':f'{slow:.1f}','__SEQ__':f'{metric("sequential","time_to_both_saved_seconds"):.2f}','__BATCH__':f'{metric("batch_modulation","time_to_both_saved_seconds"):.2f}','__CON__':f'{metric("concurrent","time_to_both_saved_seconds"):.2f}','__ROWS__':rows,'__BARS__':bars,'__REPEATS__':repeats,'__PROFILE__':prof,'__LAUNCH__':f'{launch_reduction:.1f}','__API__':f'{launch_api_reduction:.1f}','__SSIM__':f'{min(ssims):.4f}–{max(ssims):.4f}','__COS__':f'{min(lat):.4f}–{max(lat):.4f}','__VIDEOS__':cards}.items():html_page=html_page.replace(a,b)
(O/'index.html').write_text(html_page)
files=[str(p.relative_to(O)) for p in O.rglob('*') if p.is_file()];(O/'report-files.json').write_text(json.dumps(sorted(files),indent=2));print('Created',O)

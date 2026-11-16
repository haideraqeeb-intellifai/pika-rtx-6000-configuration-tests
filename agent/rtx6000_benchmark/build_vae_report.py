"""Readable report of separate VAE pools and fixed-latent batch diagnostics."""
import csv
import html
import json
import shutil
import statistics
import zipfile
from pathlib import Path
from .parallel_chunk_processes import RUNS

DEST=RUNS.parent/'vae_configuration_report'
BASELINES=[('gpu_preencoded_cached_one','Earlier: one combined worker'),('gpu_preencoded_cached_two','Earlier: two combined workers')]
CASES=[(f'vae_{s}_{t}t_{v}v',f'{"Denoise all, then decode" if s=="staged" else "Overlap stages"} · {t} transformer / {v} VAE')
       for s,t,v in [(s,1,v) for s in ('staged','pipeline') for v in (1,2,4)]+[('pipeline',2,1),('pipeline',2,2)]]

def read(name): return json.loads((RUNS/name/'result.json').read_text())
def copy_data(name):
    src=RUNS/name;target=DEST/'data'/name;target.mkdir(parents=True,exist_ok=True)
    for file in ('result.json','gpu_trace.json','traceback.txt'):
        if (src/file).exists(): shutil.copy2(src/file,target/file)
    result=read(name)
    if result.get('status')=='failure':
        for file in ('pipeline/mps_shutdown.json','pipeline/mps_logs/server.log'):
            if (src/file).exists():
                destination=target/Path(file).name
                shutil.copy2(src/file,destination)
        for detail in list(RUNS.glob(name+'_denoise_chunk_*/result.json'))+list(RUNS.glob(name+'_chunk_*/result.json')):
            shutil.copy2(detail,target/(detail.parent.name+'.json'))
    for child in result.get('child_results',[])+result.get('denoise_results',[]):
        shutil.copy2(RUNS/child['name']/'result.json',target/f'{child["name"]}.json')
    return result

def main():
    DEST.mkdir(exist_ok=True)
    measurements=[];sections=[];rows=[];attempts=[];failed=[]
    extra=[]
    for path in sorted(RUNS.glob('vae_mps_*/result.json')):
        record=json.loads(path.read_text())
        if path.parent.name.endswith('_repeat2') or record.get('loader')!='separate stage processes': continue
        pct=record.get('mps_percentages',{})
        extra.append((path.parent.name, f'MPS{" · longest chunk first" if record.get("prime_longest") else ""} · {record.get("schedule")} · {record.get("transformers")} transformer / {record.get("vaes")} VAE · {pct.get("denoise",100)}/{pct.get("decode",100)}% thread caps'))
    for name,label in BASELINES+CASES+extra:
        if not (RUNS/name/'result.json').exists():
            attempts.append(f'{label}: pending');continue
        result=copy_data(name)
        if result['status']!='success':
            reason = 'CUDA launch failure followed by GPU lost; excluded from performance rankings.' if name.endswith('_65_35') else result.get('error', result['status'])[:180]
            attempts.append(f'{label}: {reason}')
            failed.append({'configuration':name,'label':label,'status':result['status']})
            continue
        samples=[result]
        repeated=RUNS/f'{name}_repeat2/result.json'
        if repeated.exists() and read(name+'_repeat2')['status']=='success': samples.append(copy_data(name+'_repeat2'))
        times=[r['timed_total_seconds'] for r in samples];mean=statistics.mean(times)
        peak=max(r['gpu_trace_summary']['peak_vram_gib'] for r in samples)
        util=statistics.mean(r['gpu_trace_summary']['gpu_util_mean_pct'] for r in samples)
        sums={};numbers={}
        for child in result['child_results']+result.get('denoise_results',[]):
            detail=read(child['name'])
            for stage in detail['telemetry']['stages']['stages']:
                key=stage['stage']
                if detail.get('stage') == 'decode' and key == 'generation': key='latent_handoff_load'
                sums[key]=sums.get(key,0)+stage['duration_seconds'];numbers[key]=numbers.get(key,0)+1
        averages={key:sums[key]/numbers[key] for key in sums}
        seams=result['seam_check']['seams_above_nonseam_95th_percentile']
        record={'configuration':name,'label':label,'status':'success','repetitions':len(samples),'mean_seconds':mean,
            'individual_seconds':';'.join(f'{t:.6f}' for t in times),'videos_per_hour':3600/mean,
            'peak_vram_gib':peak,'mean_gpu_util_pct':util,'mean_chunk_denoise_seconds':averages.get('generation'),
            'mean_chunk_decode_seconds':averages.get('video_decode'),'flagged_joins':seams}
        measurements.append(record)
        rows.append(f'<tr><th><a href="#{name}">{html.escape(label)}</a></th><td>{mean:.1f}</td><td>{len(samples)}</td><td>{3600/mean:.2f}</td><td>{peak:.1f}</td><td>{util:.0f}%</td><td>{averages.get("generation",0):.1f}</td><td>{averages.get("video_decode",0):.1f}</td></tr>')
        for file in ('output.mp4','contact_sheet.jpg'): shutil.copy2(RUNS/name/file,DEST/f'{name}_{file}')
        timings='; '.join(f'{t:.2f} s' for t in times)
        details=''.join(f'<li>{html.escape(k)}: {v:.2f} s summed across workers</li>' for k,v in sums.items())
        decode_overlap=result.get('decode_overlap',{})
        sections.append(f'''<section id="{name}"><h2>{html.escape(label)}</h2><p>Full-run timings: <strong>{timings}</strong>. Peak memory: {peak:.1f} GiB.
{seams}/9 joins flagged by luma screening. Audio lag: {result['audio_check']['best_lag_ms']:.0f} ms; audio duration difference: {result['audio_check']['duration_delta_ms']:.0f} ms.</p>
<details><summary>Watch this video</summary><video controls preload="none" src="{name}_output.mp4"></video></details>
<details><summary>Stage measurements and join images</summary><p>Stage sums overlap in concurrent runs; do not add them to obtain elapsed video time.</p><ul>{details}</ul>
<p>Decode call overlap: {html.escape(str(decode_overlap))}. <a href="data/{name}/result.json">Raw result</a>.</p>
<img loading="lazy" src="{name}_contact_sheet.jpg" alt="Boundary frames"></details></section>''')
    full=[r for r in measurements if r['configuration'].startswith('vae_')]
    best=min(full,key=lambda r:r['mean_seconds'])
    baseline=next(r for r in measurements if r['configuration']=='gpu_preencoded_cached_two')
    gain=(baseline['mean_seconds']/best['mean_seconds']-1)*100
    batch_rows=[];batch_data=[]
    for size in (1,2,4):
        name=f'vae_batch_{size}'
        if not (RUNS/name/'result.json').exists(): continue
        result=copy_data(name)
        if result['status']!='success':
            attempts.append(f'VAE batch {size}: {result["status"]}');continue
        batch_data.append(result)
        calls=[shape[0] for b in result['batches'] for shape in b['actual_model_input_shapes']]
        batch_rows.append(f'<tr><th>{size}</th><td>{result["decoder_phase_seconds"]:.1f}</td><td>{sum(b["decode_seconds"] for b in result["batches"]):.1f}</td><td>{result["gpu_trace_summary"]["peak_vram_gib"]:.1f}</td><td>{html.escape(str(calls))}</td></tr>')
        shutil.copy2(RUNS/name/'output.mp4',DEST/f'{name}_output.mp4')
        sections.append(f'<section><h2>One VAE · requested batch {size}</h2><p>Decoder-only fixed-latent replay. <a href="data/{name}/result.json">Raw data</a>.</p><video controls preload="metadata" src="{name}_output.mp4"></video></section>')
    with (DEST/'configuration_matrix.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(measurements[0]));writer.writeheader();writer.writerows(measurements+failed)
    (DEST/'measurements.json').write_text(json.dumps({'full_runs':measurements,'decoder_diagnostics':batch_data,'other_attempts':attempts,'failed_configurations':failed},indent=2))
    code=DEST/'code';code.mkdir(exist_ok=True)
    names=['stage_pipeline.py','latent_stage_nodes.py','batch_decode.py','parallel_chunk_resident.py',
        'parallel_chunk_processes.py','gpu_preencode.py','preencoded_parallel.py','run_case.py','async_encode_nodes.py',
        'check_audio.py','check_seams.py','make_contact_sheet.py','compare_vae_outputs.py','check_mps.py','profile_vae_pool.py','run_vae_followup.py','run_vae_final_checks.py','analyze_vae_trace.py','mps_session.py','run_vae_stage_matrix.py','VAE_RESULTS.md','VAE_EXPERIMENT_PLAN.md','README.md','build_vae_report.py']
    for name in names: shutil.copy2(RUNS.parent/name,code/(name+'.txt'))
    copy_data('vae_stages_calibration')
    for diagnostic in ('vae_mps_smoke','vae_profiler_baseline','vae_profile_mps_four','vae_profile_mps_four_retry'):
        if (RUNS/diagnostic/'result.json').exists(): copy_data(diagnostic)
    diagnostic=RUNS/'vae_mps_kernel_overlap.json'
    if diagnostic.exists(): shutil.copy2(diagnostic,DEST/'data/vae_mps_kernel_overlap.json')
    comparison=RUNS/'vae_output_comparison.json'
    if comparison.exists(): shutil.copy2(comparison,DEST/'data/vae_output_comparison.json')
    with zipfile.ZipFile(DEST/'reproducibility.zip','w',zipfile.ZIP_DEFLATED) as z:
        for file in (DEST/'data').rglob('*'):
            if file.is_file(): z.write(file,file.relative_to(DEST))
        for file in code.iterdir(): z.write(file,'agent/rtx6000_benchmark/'+file.name[:-4])
        for name in ('configuration_matrix.csv','measurements.json'): z.write(DEST/name,name)
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RTX PRO 6000: do more VAEs help?</title><style>
body{{font:17px/1.55 system-ui,sans-serif;color:#17212b;background:#f4f6fa;max-width:1200px;margin:auto;padding:28px}}
h1,h2{{line-height:1.2}}section,.note{{background:white;border:1px solid #dbe2eb;border-radius:12px;padding:22px;margin:20px 0}}
table{{border-collapse:collapse;width:100%;background:white;font-size:15px}}th,td{{border:1px solid #dbe2eb;padding:9px;text-align:left}}th{{background:#eef2f8}}
video{{width:min(100%,480px);display:block;margin:18px 0}}img{{max-width:100%}}a{{color:#0759ae}}summary{{cursor:pointer}}code{{font-size:14px}}</style>
<h1>Do more VAEs make this video faster?</h1>
<p>Measured on one RTX PRO 6000, using the same 49.4-second podcast, ten independent chunks, two denoising steps and 24 fps. Model canvas: 736 × 1280, cropped to 720 × 1280 and exported at 1080 × 1920. Fixed chunk seeds: 9901–9910. Hardware: RTX PRO 6000 Blackwell Workstation Edition, 95.59 GiB usable device capacity.</p>
<div class="note"><h2>What the tests found</h2><p>The fastest new full-run setting was <strong>{html.escape(best['label'])}</strong>, averaging <strong>{best['mean_seconds']:.1f} seconds</strong> across {best['repetitions']} run(s).
The previous paired-worker baseline averaged {baseline['mean_seconds']:.1f} seconds. The new setting changes measured throughput by <strong>{gain:+.1f}%</strong>.</p>
<p>A transformer generates a chunk's latent representation. A VAE turns it into video frames. More copies share the same GPU's compute and memory bandwidth. The table below tests both the copy count and when each stage runs.</p></div>
<div class="note"><h2>Practical choice and limits</h2><p><strong>Use one transformer feeding one VAE, with ordinary scheduling, as the provisional choice.</strong> GPU-encode conditioning once, release the encoder, and let these two stage workers overlap. It is the fastest observed setting here and uses about 39.6 GiB, compared with 72.1 GiB for the earlier paired combined workers. Differences of a few seconds need repeat confirmation.</p><p>Four VAEs plus MPS reduced the staged decoder phase from 98.2 to 83.2 seconds, but the complete video still took 251.9 seconds. One-VAE batching saved only about 3.4% of decoder-phase time. More copies helped that stage modestly; they did not produce the fastest complete-video result.</p><p><strong>Repeat confirmation is pending GPU recovery.</strong> The 65/35 MPS resource-split attempt failed with a CUDA launch error and NVML reported GPU lost. All benchmark MPS clients/controllers exited; the reset command requires administrator credentials. The fault occurred during this configuration, but its root cause is not established. This setting is excluded and is not recommended.</p></div>
<h2>Complete video runs — lower time is better</h2>
<p>All rows include GPU conditioning preparation, process startup, model release, latent handoff, final assembly and media validation. Source-audio splitting and memory calibration are excluded. Each output has 1,186 frames. Initial tests have one repetition; repeat counts are shown explicitly.</p>
<table><thead><tr><th>Configuration</th><th>Seconds (mean if repeated)</th><th>Runs</th><th>Videos/hour</th><th>Peak GiB</th><th>GPU use</th><th>Denoise s/chunk*</th><th>Decode s/chunk*</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>*Stage averages are from the first run, include work under concurrent load, and overlap. They are not an additive breakdown of video latency. NVML memory/utilization samples are taken every 0.2 seconds over preparation and workers.</p>
<div class="note"><strong>Process overlap versus GPU concurrency:</strong> The ordinary pool runs use separate CUDA contexts. NVIDIA documents that these contexts time-share the GPU; MPS permits kernels from different processes to execute concurrently. MPS rows, when present, use a temporary private controller and verify worker attachment. MPS attachment was verified. The Nsight capture lacked usable GPU kernel timestamps, so actual kernel overlap remains unverified; API-call overlap is insufficient evidence. <a href="data/vae_mps_kernel_overlap.json">Profiler limitations</a>. <a href="https://docs.nvidia.com/deploy/mps/latest/architecture.html">NVIDIA architecture reference</a>.</div>
<h2>One VAE processing batches — diagnostic only</h2>
<p>These runs reuse identical saved denoised latents, so they exclude text preparation and denoising. Compare these rows with each other; their times are not complete generation times. Chunks are grouped by equal latent shape and restored to the original order.</p>
<table><thead><tr><th>Requested batch</th><th>Decoder phase seconds</th><th>Summed VAE call seconds</th><th>Peak GiB</th><th>Actual model batch sizes</th></tr></thead><tbody>{''.join(batch_rows)}</tbody></table>
<div class="note"><h2>What “valid” means here</h2><p>Successful outputs passed complete video/audio decoding, frame-count, frame-rate and audio timing checks. All twelve successful new full-generation layouts produced identical decoded pixels; the batch-of-one replay matched too. Batch sizes two and four differed slightly (SSIM at 180 × 320: 0.998492 and 0.998115), which is technical similarity, not a human quality rating. Audio offsets are measured at 8 kHz. Join brightness changes screen for possible discontinuities; they do not establish seamless motion or lip synchronization. Review the videos below.</p>
<p>All chunks use the same portrait as first/last conditioning. They are independent; this is different from previous-last-frame chaining. Overlapping process calls do not prove simultaneous CUDA kernels. These are one-GPU measurements, with no eight-GPU speedup claim.</p></div>
{''.join(sections)}
<section><h2>Reproduce and inspect</h2><p><a href="configuration_matrix.csv">Configuration CSV</a> · <a href="measurements.json">Measurement data</a> · <a href="reproducibility.zip">Runnable source and raw measurements ZIP</a> · <a href="code/VAE_EXPERIMENT_PLAN.md.txt">Experiment design</a></p>
<pre>python -m agent.rtx6000_benchmark.stage_pipeline --name YOUR_RUN --schedule pipeline --transformers 1 --vaes 2 --run</pre>
<p>Use the pinned script2video checkout and model environment described in the bundled README. Model weights and source audio/portrait must already be available. Calibration uses two longest chunks per stage; its cost is excluded from measured run times.</p>
<p><a href="data/vae_output_comparison.json">Decoded-frame and technical-similarity checks</a>. These do not establish seamlessness.</p><p>Other attempts: {html.escape('; '.join(attempts) or 'None failed or pending in this matrix.')}</p></section></html>'''
    (DEST/'index.html').write_text(page)
    print(DEST/'index.html')

if __name__=='__main__':main()

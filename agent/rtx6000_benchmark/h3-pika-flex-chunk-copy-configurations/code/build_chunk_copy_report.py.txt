"""Build readable, self-contained follow-up report from measured run records."""
import csv
import hashlib
import html
import json
import shutil
import statistics
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from markdown_it import MarkdownIt
from datetime import datetime
from zoneinfo import ZoneInfo
from .parallel_chunk_processes import RUNS
from .run_case import PROMPT

ROOT=RUNS.parent
DEST=ROOT/'h3-pika-flex-chunk-copy-configurations'

def read(path):return json.loads(path.read_text())
def group_key(r):return (r['chunks'],r['transformers'],r['vaes'],r.get('residency','dynamic'),r.get('async_encoding',False),r.get('mps',False))
def label(key):
    c,t,v,res,enc,mps=key
    return f'{c} chunks · {t} transformer / {v} VAE'+(' · resident' if res=='highvram' else '')+(' · background export' if enc else '')+(' · MPS' if mps else '')

def main():
    DEST.mkdir(exist_ok=True)
    records=[];groups=defaultdict(list);failures=[];stages=[];media_paths={}
    exclusions=read(RUNS/'timing_exclusions.json') if (RUNS/'timing_exclusions.json').exists() else {}
    (DEST/'media').mkdir(exist_ok=True)
    for path in sorted(RUNS.glob('followup_*/result.json')):
        r=read(path)
        if r.get('schedule') == 'staged':continue
        if r.get('loader')!='separate stage processes':continue
        if r['status']=='calibrated':continue
        r['key']=group_key(r)
        if r['name'] in exclusions:
            r['recorded_status']=r['status'];r['status']='timing_excluded';r['error']=exclusions[r['name']]['reason'];r['timing_exclusion']=exclusions[r['name']]
        if r['status']=='success':groups[r['key']].append(r)
        else:failures.append(r)
        records.append(r)
        target=DEST/'data'/r['name'];target.mkdir(parents=True,exist_ok=True)
        for f in ('result.json','gpu_trace.json','audio_check.json','seam_check.json','traceback.txt'):
            if (path.parent/f).exists():shutil.copy2(path.parent/f,target/f)
        checksums={f.name:f.read_text().strip() for f in (path.parent/'latents').glob('*.sha256')}
        if checksums:(target/'latent_checksums.json').write_text(json.dumps(checksums,indent=2))
        if (path.parent/'prepared/result.json').exists():
            prep=read(path.parent/'prepared/result.json')
            shutil.copy2(path.parent/'prepared/result.json',target/'preparation.json')
            for chunk in prep['chunks']:
                for node in chunk['telemetry']['stages']['nodes']:
                    stages.append({'run':r['name'],'worker':'prepare','chunk':prep['chunks'].index(chunk),'role':'preparation',
                        'class_type':node['class_type'],'seconds':node['duration_seconds'],
                        'cuda_peak_allocated_gib':node['after']['cuda_peak_allocated_gib'],
                        'model_offloaded_gib_after':node['after']['model_offloaded_gib']})
        if (RUNS/(r['name']+'.log')).exists():shutil.copy2(RUNS/(r['name']+'.log'),target/'execution.log')
        for f in ('pipeline/mps_shutdown.json','pipeline/mps_clients.txt','pipeline/mps_logs/server.log'):
            if (path.parent/f).exists():shutil.copy2(path.parent/f,target/Path(f).name)
        for child in r.get('child_results',[])+r.get('denoise_results',[]):
            cr=read(RUNS/child['name']/'result.json')
            shutil.copy2(RUNS/child['name']/'result.json',target/(child['name']+'.json'))
            workflow=RUNS/child['name']/'workflow.json'
            if cr['index']==0 and workflow.exists():shutil.copy2(workflow,target/(cr['stage']+'_workflow.json'))
            if cr.get('encoding_interval'):
                interval=cr['encoding_interval']
                stages.append({'run':r['name'],'worker':cr['lane'],'chunk':cr['index'],'role':cr['stage'],
                    'class_type':'DeferredEncodeAndMux','seconds':interval['ended_wall_utc']-interval['started_wall_utc'],
                    'cuda_peak_allocated_gib':None,'model_offloaded_gib_after':None})
            for node in cr['telemetry']['stages']['nodes']:
                stages.append({'run':r['name'],'worker':cr['lane'],'chunk':cr['index'],'role':cr['stage'],
                    'class_type':node['class_type'],'seconds':node['duration_seconds'],
                    'cuda_peak_allocated_gib':node['after']['cuda_peak_allocated_gib'],
                    'model_offloaded_gib_after':node['after']['model_offloaded_gib']})
        if (path.parent/'output.mp4').exists():
            media=DEST/'media'/(r['output_sha256']+'.mp4')
            if not media.exists():shutil.copy2(path.parent/'output.mp4',media)
            media_paths[r['name']]=str(media.relative_to(DEST))
            shutil.copy2(path.parent/'contact_sheet.jpg',target/'contact_sheet.jpg')
    summary=[]
    for key,samples in groups.items():
        times=[r['timed_total_seconds'] for r in samples]
        ready=[r['time_to_assembled_valid_video_seconds'] for r in samples]
        mean=statistics.mean(times);valid=statistics.mean(ready)
        summary.append({'key':key,'label':label(key),'status':'success','samples':len(samples),'mean_total_seconds':mean,
            'min_total_seconds':min(times),'max_total_seconds':max(times),'mean_video_ready_seconds':valid,
            'videos_per_hour_including_qa':3600/mean,'videos_per_hour_ready':3600/valid,
            'peak_vram_gib':max(r['gpu_trace_summary']['peak_vram_gib'] for r in samples),
            'gpu_util_mean_pct':statistics.mean(r['gpu_trace_summary']['gpu_util_mean_pct'] for r in samples),
            'preparation_mean_seconds':statistics.mean(r['preparation_seconds'] for r in samples),
            'workers_mean_seconds':statistics.mean(r['parallel_generation_seconds'] for r in samples),
            'max_concurrent_sampler_calls':max(r['sampler_overlap']['maximum_concurrent_inference_calls'] for r in samples),
            'max_concurrent_decode_calls':max(r['decode_overlap']['maximum_concurrent_inference_calls'] for r in samples),
            'sampler_overlap_mean_seconds':statistics.mean(r['sampler_overlap']['two_or_more_calls_overlap_seconds'] for r in samples),
            'decode_overlap_mean_seconds':statistics.mean(r['decode_overlap']['two_or_more_calls_overlap_seconds'] for r in samples),
            'run_names':[r['name'] for r in samples]})
    summary.sort(key=lambda r:r['mean_total_seconds'])
    if not summary:raise ValueError('No successful measurements')
    for path in RUNS.glob('followup_*/cancellation.json'):
        target=DEST/'data'/path.parent.name;target.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target/'cancellation.json')
    best=summary[0]
    timeline_run=next((r for r in records if r['status']=='success' and r['name']=='followup_mps_1t_2v'), next(r for r in records if r['name']==best['run_names'][0]))
    events=[]
    for child in timeline_run['denoise_results']+timeline_run['child_results']:
        cr=read(RUNS/child['name']/'result.json');role=cr['stage'];lane=cr['lane']
        for span in cr['sampler_intervals' if role=='denoise' else 'decode_intervals']:
            events.append((role,lane,cr['index'],span['started_wall_utc'],span['ended_wall_utc']))
    origin=min(e[3] for e in events);duration=max(e[4] for e in events)-origin
    lanes=sorted({(e[0],e[1]) for e in events},key=lambda x:(x[0]!='denoise',x[1]))
    svg=[f'<svg viewBox="0 0 1120 {60+len(lanes)*48}" role="img" aria-label="Measured transformer and VAE call overlap" style="width:100%">']
    for i,lane in enumerate(lanes):
        svg.append(f'<text x="0" y="{35+i*48}" font-size="15">{lane[0]} worker {lane[1]}</text>')
        for role,l,index,start,end in events:
            if (role,l)!=lane:continue
            x=180+(start-origin)/duration*900;w=(end-start)/duration*900;y=18+i*48
            color='#2668a7' if role=='denoise' else '#17816c'
            svg.append(f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="24" fill="{color}"><title>Chunk {index+1}: {end-start:.2f} seconds</title></rect>')
    svg.append(f'<text x="180" y="{48+len(lanes)*48}" font-size="14">0 s</text><text x="1010" y="{48+len(lanes)*48}" font-size="14">{duration:.1f} s</text></svg>')
    timeline_html=''.join(svg)
    window_rows=[]
    for name in ('followup_one_mps_repeat3','followup_mps_1t_2v','followup_mps_1t_4v','followup_best_two_mps'):
        found=next((r for r in records if r['name']==name and r['status']=='success'),None)
        if not found:continue
        windows=[]
        for field,timer in [('denoise_results','sampler_intervals'),('child_results','decode_intervals')]:
            spans=[i for child in found[field] for i in read(RUNS/child['name']/'result.json')[timer]]
            windows.append(max(i['ended_wall_utc'] for i in spans)-min(i['started_wall_utc'] for i in spans))
        window_rows.append(f'<tr><td>{found["transformers"]} transformer / {found["vaes"]} VAE</td><td>{windows[0]:.1f} s</td><td>{windows[1]:.1f} s</td><td>{found["parallel_generation_seconds"]:.1f} s</td></tr>')
    best_runs=[r for r in records if r['name'] in best['run_names']]
    timing_rows=[]
    parts=[('GPU preparation, including loading','preparation_seconds'),('Wait for encoder release','encoder_release_gate_seconds'),('Transformer/VAE workers, including startup and chunk export','parallel_generation_seconds'),('Final assembly and full media validation','assembly_and_full_validation_seconds')]
    accounted=0
    for title,key in parts:
        values=[r[key] for r in best_runs if key in r]
        if values:
            value=statistics.mean(values);accounted+=value
            timing_rows.append(f'<tr><td>{title}</td><td>{value:.1f} s</td></tr>')
    qa=statistics.mean(r['timed_total_seconds']-r['time_to_assembled_valid_video_seconds'] for r in best_runs)
    timing_rows.append(f'<tr><td>Additional audio/join checks and contact sheet</td><td>{qa:.1f} s</td></tr>')
    timing_rows.append(f'<tr><th>Total including QA</th><th>{best["mean_total_seconds"]:.1f} s</th></tr>')
    baseline=next(r for r in summary if r['key']==(20,1,1,'dynamic',False,False))
    gain=(baseline['mean_total_seconds']/best['mean_total_seconds']-1)*100
    bykey={tuple(r['key']):r for r in summary}
    matrix=[]
    for t,v in [(1,1),(2,1),(1,2),(2,2),(1,4)]:
        cells=[]
        for c in (5,10,20):
            k=(c,t,v,'dynamic',False,False);s=bykey.get(k)
            failed=next((r for r in failures if tuple(r['key'])==k),None)
            cells.append(f"<td><b>{s['mean_total_seconds']:.1f} s</b><br>{s['peak_vram_gib']:.1f} GiB · n={s['samples']}</td>" if s else '<td>'+html.escape(('Rejected by memory guard' if failed['status']=='memory_guard_rejected' else failed['status']) if failed else 'Not measured')+'</td>')
        matrix.append(f'<tr><th>{t} transformer / {v} VAE</th>'+''.join(cells)+'</tr>')
    mps_rows=[]
    for t,v in sorted({(k[1],k[2]) for k in bykey if k[0]==20 and k[3]=='dynamic' and not k[4]}):
        a=bykey.get((20,t,v,'dynamic',False,False));b=bykey.get((20,t,v,'dynamic',False,True))
        ordinary=f"{a['mean_total_seconds']:.1f} s · n={a['samples']}" if a else 'Not measured'
        mps=f"{b['mean_total_seconds']:.1f} s · n={b['samples']}" if b else 'Not measured'
        memory=f"{b['peak_vram_gib']:.1f} GiB" if b else '—'
        mps_rows.append(f'<tr><th>{t} transformer / {v} VAE</th><td>{ordinary}</td><td>{mps}</td><td>{memory}</td></tr>')
    gallery=[]
    for c in (5,10,20):
        group=bykey.get((c,1,1,'dynamic',False,False))
        if group:
            name=group['run_names'][0]
            gallery.append(f'<div><h3>{c} chunks · ≈{49.408/c:.1f} s each</h3><p>{group["mean_total_seconds"]:.1f} s generation + QA · {c-1} joins</p><video controls preload="none" src="{media_paths[name]}"></video></div>')
    ranking=[];sections=[]
    quality=ROOT/'runs/followup_quality.json'
    q=read(quality) if quality.exists() else {}
    qm={r['name']:r for r in q.get('results',[])}
    for s in summary:
        ranking.append(f"<tr><th>{html.escape(s['label'])}</th><td>{s['mean_total_seconds']:.1f}</td><td>{s['mean_video_ready_seconds']:.1f}</td><td>{s['samples']}</td><td>{s['min_total_seconds']:.1f}–{s['max_total_seconds']:.1f}</td><td>{s['videos_per_hour_including_qa']:.2f}</td><td>{s['peak_vram_gib']:.1f}</td><td>{s['gpu_util_mean_pct']:.0f}%</td></tr>")
        r=next(r for r in records if r['name']==s['run_names'][0]);n=r['name'];aq=r['audio_check'];sq=r['seam_check'];qq=qm.get(n,{})
        sections.append(f'''<details><summary>{html.escape(s['label'])} — {s['mean_total_seconds']:.1f} s</summary>
<p>Runs: {html.escape(', '.join(s['run_names']))}. Audio lag {aq['best_lag_ms']:.0f} ms; duration difference {aq['duration_delta_ms']:.0f} ms at the check's 8 kHz resolution. {sq['seams_above_nonseam_95th_percentile']}/{r['chunks']-1} joins flagged by luma screening. Same-plan pixel equality: {html.escape(str(qq.get('same_plan_pixels_identical','Not checked')))}. Downsampled SSIM when pixels differ: {html.escape(str(qq.get('ssim_at_180x320','Not applicable')))}.</p>
<p>Measured maximum overlapping sampler calls: {s["max_concurrent_sampler_calls"]}; decoder calls: {s["max_concurrent_decode_calls"]}. Time with at least two calls overlapping: denoise {s["sampler_overlap_mean_seconds"]:.1f} s; decode {s["decode_overlap_mean_seconds"]:.1f} s. This measures API calls, not kernel overlap.</p>
<video controls preload="none" src="{media_paths[n]}"></video><button onclick="jump(this,{max(0,r['chunk_frames'][0]/24-1):.3f})">Watch first join</button> <button onclick="jump(this,{max(0,sum(r['chunk_frames'][:r['chunks']//2])/24-1):.3f})">Watch middle join</button><p><a href="data/{n}/result.json">Raw measurements</a></p>
<details><summary>View frames around joins</summary><img loading="lazy" src="data/{n}/contact_sheet.jpg" alt="Frames around chunk joins"></details></details>''')
    (DEST/'measurements.json').write_text(json.dumps({'summary':summary,'failed_or_rejected':failures,'runs':records},indent=2))
    csv_rows=summary+[{'key':r['key'],'label':label(r['key']),'status':r['status'],'samples':0,'run_names':[r['name']]} for r in failures]
    for filename,rows in [('configuration_matrix.csv',csv_rows),('per_node_times.csv',stages)]:
        with (DEST/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if quality.exists():shutil.copy2(quality,DEST/'quality.json')
    code=DEST/'code';code.mkdir(exist_ok=True)
    names=['stage_pipeline.py','parallel_chunk_resident.py','parallel_chunk_processes.py','preencoded_parallel.py',
        'gpu_preencode.py','latent_stage_nodes.py','run_case.py','async_encode_nodes.py','deferred_encoding.py',
        'run_chunk_copy_matrix.py','check_mps.py','mps_session.py','check_audio.py','check_seams.py','make_contact_sheet.py',
        'compare_vae_outputs.py','build_chunk_copy_report.py','CHUNK_COPY_PLAN.md','README.md']
    for name in ['followup_quality.py','run_chunk_copy_confirm.py','run_combined_followup.py','run_mps_control_followup.py','run_final_followups.py','run_mps_vae_sizing.py','CHUNK_COPY_RESULTS.md']:
        if (ROOT/name).exists():names.append(name)
    names=sorted(set(names)|{p.name for p in ROOT.glob('*.py')})
    for name in names:shutil.copy2(ROOT/name,code/(name+'.txt'))
    for name in ('manifest.json','raccoon_headshot.jpg','report_audio.wav'):
        target=DEST/'inputs'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'inputs'/name,target)
    for cal in ('vae_stages_calibration','followup_cal_5','followup_cal_20','followup_mps_smoke','followup_isolated_20','followup_mps_3t_2v_staged'):
        if (RUNS/cal/'result.json').exists():
            target=DEST/'data'/cal;target.mkdir(exist_ok=True);shutil.copy2(RUNS/cal/'result.json',target/'result.json')
    for pattern in ('followup_cal_20_calibration_*/result.json','followup_cal_5_calibration_*/result.json','followup_best_resident_calibration_*/result.json','vae_stages_calibration_calibration_*/result.json'):
        for source in RUNS.glob(pattern):
            target=DEST/'data/calibration'/source.parent.name;target.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target/'result.json')
    for name in ('followup_matrix.log','followup_provenance.json','followup_confirmation.log','followup_confirmation_initial_error.log','followup_final.log','followup_mps_sizing.log','followup_ownership_samples.jsonl','followup_ownership_continuous.jsonl','timing_exclusions.json','followup_contention_incident.json','followup_resume.log','followup_stopped_by_user.json','followup_resume_audit.json','followup_finish.log','followup_resume_environment.json','followup_report_validation.json','followup_browser_validation.json'):
        if (RUNS/name).exists():shutil.copy2(RUNS/name,DEST/'data'/name)
    for name in ('capacity_matrix_summary.json','capacity_matrix_summary.csv','capacity_scope_change.json','three_transformers_summary.json','followup_capacity_matrix.log'):
        if (RUNS/name).exists():shutil.copy2(RUNS/name,DEST/'data'/name)
    with zipfile.ZipFile(DEST/'reproducibility.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in code.iterdir():z.write(f,'agent/rtx6000_benchmark/'+f.name[:-4])
        for f in (DEST/'data').rglob('*.json'):z.write(f,f.relative_to(DEST))
        for f in (DEST/'inputs').iterdir():z.write(f,'agent/rtx6000_benchmark/inputs/'+f.name)
        for f in ('configuration_matrix.csv','per_node_times.csv','measurements.json'):z.write(DEST/f,f)
        for f in ('followup_eight_gpu_estimates.json','followup_quality.json'):
            if (RUNS/f).exists():z.write(RUNS/f,'data/'+f)
    failure_html=''.join('<li>'+html.escape(r['name']+': '+r['status']+' '+(f"estimated {r.get('admission_memory_envelope_gib',r.get('conservative_memory_envelope_gib',0)):.1f} GiB exceeds {r.get('memory_budget_gib',86.031):.1f} GiB budget" if r['status']=='memory_guard_rejected' else str(r.get('error',''))[:250]))+'</li>' for r in failures) or '<li>None in this follow-up.</li>'
    projection=RUNS/'followup_eight_gpu_estimates.json'
    projection_html=''
    if projection.exists():
        estimate=read(projection);shutil.copy2(projection,DEST/'eight_gpu_estimates.json')
        scenario_rows=[]
        for allocation in [(1,4,3),(1,5,2),(2,4,2),(2,3,3),(1,3,4),(1,6,1)]:
            cases=[r for r in estimate['stage_allocation_scenarios'] if r['GPUs_preparation_transformer_VAE']==list(allocation)]
            central=next(r for r in cases if r['stage_scenario']=='Measured stage proxy + overhead' and r['transfer_scenario']=='Moderate host-mediated')
            scenario_rows.append(f"<tr><td>{'/'.join(map(str,allocation))}</td><td>{central['estimated_videos_per_hour']:.1f}</td><td>{min(r['estimated_videos_per_hour'] for r in cases):.1f}–{max(r['estimated_videos_per_hour'] for r in cases):.1f}</td><td>{central['bottleneck']}</td></tr>")
        independent=estimate['eight_independent_worker_scenarios']
        transfers=[r['transfer_seconds_per_video'] for r in estimate['stage_allocation_scenarios']]
        projection_html=f'''<h3>Estimated throughput for this independent-chunk workload</h3>
<p><b>Eight independent workers:</b> {independent[0]['estimated_videos_per_hour']:.1f}–{independent[-1]['estimated_videos_per_hour']:.1f} videos/hour at assumed 80–100% scaling efficiency, based on video-ready time. This is an assumption range, not a confidence interval.</p>
<p><b>Stage allocations:</b> GPU counts below are preparation / transformer / VAE. Preparation includes text, image and audio conditioning. VAE GPUs also export frames locally.</p>
<table><tr><th>GPUs by stage</th><th>Central estimate, videos/hour</th><th>Sensitivity range</th><th>Central bottleneck</th></tr>{''.join(scenario_rows)}</table>
<p>Service-rate model: completion interval = max(preparation time / preparation GPUs, denoising time / transformer GPUs, decode/export time / VAE GPUs, shared-transfer time). Central scenario adds 10% scheduling overhead. Sensitivity varies measured stage times by −20% to +20% and overhead from 0% to 20%.</p>
<p>Measured isolated stage phases (including fresh process/model loading): {html.escape(str(estimate['measured_stage_seconds_including_fresh_stage_loading']))} seconds per video. These are not dedicated warm-service measurements. Serialized conditioning plus latent payload is {estimate['serialized_artifact_payload_GiB']:.3f} GiB/video. Assumed aggregate bandwidth 1–25 GiB/s, 0.1–5 ms per handoff, and one direct or two host-mediated link traversals give {min(transfers):.3f}–{max(transfers):.3f} seconds of serialized transfer service/video. Decoded frames stay with the decoder/export GPU. Model-weight startup traffic and local GPU-to-host frame/export traffic are additional resources; this payload estimate covers cross-stage conditioning and latents only. Their concurrent CPU, PCIe and storage costs must be measured on the target host.</p>
<p>Memory: the chosen independent worker inherits its measured one-GPU peak. Preparation peaked at {estimate['memory_limits']['sampled_preparation_peak_GiB']:.1f} GiB in per-operation sampling. One isolated transformer needs a calibrated {estimate['memory_limits']['isolated_transformer_calibrated_peak_GiB']:.1f} GiB and a VAE {estimate['memory_limits']['isolated_VAE_calibrated_peak_GiB']:.1f} GiB for this plan. Larger concurrent batches and queues need new activation/buffer measurements. CPU export, storage and final assembly are assumed adequate and could reduce the estimates.</p>
<p><a href="eight_gpu_estimates.json">Download all assumptions and 54 allocation/transfer scenarios</a>. More than 8× could be possible if dedicated warm stages remove enough loading/offloading or improve utilization relative to the stated one-GPU baseline; these scenarios do not establish that it happens. A warm eight-GPU service compared with a fresh-process one-GPU run changes both the lifecycle and GPU count. Validate against an equally warmed one-GPU baseline before calling a result superlinear hardware scaling.</p>
<p>For the best allocation in this model, matching eight independent workers requires about <b>{100*estimate['uniform_stage_time_reduction_to_match_eight_independent']['0']:.0f}% shorter stage service times with zero added scheduling overhead</b>, or <b>{100*estimate['uniform_stage_time_reduction_to_match_eight_independent']['0.1']:.0f}% shorter with 10% overhead</b>. Those dedicated-stage gains have not been measured. This calculation assumes all stage times scale by the same factor and transfers remain below the bottleneck.</p><p><b>Validation hardware tests:</b> actual eight-card PCIe/P2P topology, all-pairs and simultaneous bandwidth/latency for these tensor sizes, host NUMA/RAM bandwidth, isolated warm stage service distributions and activation peaks, CPU/export/storage capacity under eight jobs, then sustained eight-worker and stage-split end-to-end runs with frame/audio checks and latency percentiles.</p>'''
    explanation=(ROOT/'CHUNK_COPY_RESULTS.md').read_text() if (ROOT/'CHUNK_COPY_RESULTS.md').exists() else 'Analysis pending; this is a draft.'
    renderer=MarkdownIt('commonmark').enable('table')
    if '### Did extra model copies help?' in explanation:
        overview,detail=explanation.split('### Did extra model copies help?',1)
        explanation_html=renderer.render(overview)+'<details><summary>Why this recommendation, quality limits, and the command to run it</summary>'+renderer.render('### Did extra model copies help?'+detail)+'</details>'
    else:explanation_html=renderer.render(explanation)
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H3: chunk length and parallel model copies</title><style>
body{{font:17px/1.55 system-ui,sans-serif;color:#17212b;background:#f4f6fa;max-width:1250px;margin:auto;padding:28px}}h1,h2{{line-height:1.2}}section,.note,details{{background:white;border:1px solid #dbe2eb;border-radius:10px;padding:20px;margin:18px 0}}table{{border-collapse:collapse;width:100%;background:white;font-size:15px}}th,td{{border:1px solid #dbe2eb;padding:9px;text-align:left}}th{{background:#eef2f8}}video{{width:min(100%,430px);aspect-ratio:9/16;display:block;margin:16px 0;background:#101827}}button{{padding:9px 13px;border:1px solid #b9cbe0;border-radius:6px;background:#eef5ff;color:#164779;cursor:pointer}}button:focus-visible{{outline:3px solid #3280c9}}img{{max-width:100%}}a{{color:#0759ae}}summary{{cursor:pointer;font-weight:600}}pre{{white-space:pre-wrap;font:inherit}}.scroll{{overflow:auto}}</style>
<p style="color:#53657b">Updated {datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y, %H:%M IST")} · Measured one GPU · Eight-GPU projections clearly separated</p><h1>Which chunk length and model-copy layout helps?</h1>
<p>Measurements on <b>one RTX PRO 6000 Blackwell, 95.59 GiB</b>. Same 49.408-second podcast, two steps, 24 fps. Every chunk uses the original image as first and last conditioning and can generate independently. No claim of seamless joins.</p><p><b>Transformer</b> = the model that generates the video representation. <b>VAE</b> = the decoder that turns it into visible frames. <b>MPS</b> = NVIDIA’s service that lets separate GPU processes share execution capacity.</p>
<section><h2>Result at a glance</h2><p>Fastest mean in this follow-up: <b>{html.escape(best['label'])}</b>, <b>{best['mean_total_seconds']:.1f} seconds</b> across {best['samples']} run(s), versus {baseline['mean_total_seconds']:.1f} seconds for the matched 20-chunk, one-transformer/one-VAE control (n={baseline['samples']}). Measured throughput difference: <b>{gain:+.1f}%</b>. Peak VRAM: {best['peak_vram_gib']:.1f} GiB.</p>{explanation_html}<p><b>Quality decision still needed:</b> the 20-chunk version has 19 joins, and all 19 are flagged by the simple frame-change screen. Use the videos below to judge whether those transitions are acceptable.</p></section>
<section><h2>What runs on the GPU?</h2><p><b>1.</b> Prepare text, image and audio conditioning on the GPU once per video, reusing identical text/image inputs. Release the encoder. <b>2.</b> Keep the selected transformer and VAE worker processes alive across chunks. <b>3.</b> Each transformer takes an available independent chunk; each VAE takes a finished latent. <b>4.</b> Restore chunk order and mux the original audio.</p><p>The original image conditions both ends of each chunk. It does not force the first and last output pixels to equal the portrait, and it does not prove smooth joins.</p><details><summary>Exact prompt and source inputs</summary><p>{html.escape(PROMPT)}</p><img src="inputs/raccoon_headshot.jpg" alt="Original raccoon reference portrait" style="max-width:230px"><audio controls preload="none" src="inputs/report_audio.wav"></audio><p><a href="inputs/manifest.json">Input hashes and provenance</a></p></details></section><h2>Ordinary scheduling: chunk length × copies</h2><p><b>Lower seconds is better.</b> T means the transformer that denoises chunks. VAE means the model that turns denoised chunks into frames. All layouts release the shared GPU text encoder before these workers load. Cells show full run time including QA, sampled VRAM, and repeat count.</p>
<table><thead><tr><th>Concurrent worker layout</th><th>5 chunks<br>≈10 seconds each</th><th>10 chunks<br>≈5 seconds each</th><th>20 chunks<br>≈2.5 seconds each</th></tr></thead><tbody>{''.join(matrix)}</tbody></table>
<h2>Does MPS change the value of extra copies?</h2><p>Matched 20-chunk tests with dynamic loading and synchronous export. This separates copy count from scheduling mode.</p><table><tr><th>Copies</th><th>Ordinary scheduling</th><th>Standard MPS</th><th>MPS peak memory</th></tr>{''.join(mps_rows)}</table>
<section><h2>Where the time goes in the fastest measured layout</h2><table><tr><th>Non-overlapping elapsed phase</th><th>Mean time</th></tr>{"".join(timing_rows)}</table><p>The worker phase contains overlapping transformer and VAE work. Its individual node times must not be added together. Small timer-boundary overhead can remain outside these named phases.</p></section><section><h2>What “parallel” looks like</h2><p>Example: {html.escape(label(group_key(timeline_run)))}. Each bar is a timed denoising or decoding call for one chunk. Calls on separate rows overlap; this does not directly prove simultaneous CUDA kernel execution.</p>{timeline_html}<p>With one transformer, denoising calls remain sequential by design. Multiple VAE workers decode different completed chunks while the transformer continues. Two-transformer layouts also overlap denoising calls; their measured gains are shown separately.</p><details><summary>Why extra copies give a modest gain</summary><p>These representative clean MPS runs show each stage’s elapsed window, from its first call starting to its last call ending. The windows overlap and cannot be added. More decoders can shorten the decode window while lengthening denoising because they share the same GPU.</p><table><tr><th>Layout</th><th>Denoising window</th><th>Decoding window</th><th>Whole worker phase</th></tr>{"".join(window_rows)}</table><p>One selected clean run per row for explanation; the ranking table uses all valid repeats.</p></details></section><h2>Compare chunk lengths</h2><p>These three examples all use one transformer and one VAE. Different chunk lengths generate different motion and reset to the image more or less often.</p><section style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px">{''.join(gallery)}</section>
<h2>All configurations and repeats</h2><p><b>Resident</b> keeps complete model weights on the GPU using the conventional loading backend. <b>Background export</b> lets decoding continue while a previous chunk is encoded and checked. <b>MPS</b> is NVIDIA’s service for scheduling work from separate GPU processes concurrently. These are separate changes; combinations are measured explicitly.</p><div class="scroll"><table><thead><tr><th>Configuration</th><th>Total incl. QA (s)</th><th>Valid video ready (s)</th><th>n</th><th>Observed total range (s)</th><th>Videos/hour incl. QA</th><th>Peak GiB</th><th>Mean GPU util.</th></tr></thead><tbody>{''.join(ranking)}</tbody></table></div>
<section><h2>How to read these numbers</h2><ul><li>Fresh-process timings include GPU conditioning preparation, encoder release, stage-worker startup/load, generation, latent handoff, encoding, full media validation and assembly. Initial source-audio splitting and memory calibration are excluded. The total column additionally includes audio checks, join screening and contact-sheet generation. These are not warm-service latency measurements. OS file caches and persistent kernel/compiler caches may remain between runs; any initialization or compilation inside the recorded interval is counted.</li><li>Throughput is 3600 divided by the measured complete-video time. It describes repeated execution with this lifecycle; sustained multi-job throughput was not measured in this follow-up.</li><li>GPU sampling is about every 0.2 s during preparation and stage workers. Peak is sampled whole-device memory, not PyTorch allocated memory. GPU utilization means activity during sampling; it is not a measurement of SM occupancy or percent of peak compute throughput. Per-node allocated-memory data are in raw results. Final assembly/QA is outside that GPU sampling interval.</li><li>Each stage runs in a separate process. Overlapping worker calls are recorded; simultaneous GPU kernel execution is unverified. Ordinary CUDA contexts time-share compute according to <a href="https://docs.nvidia.com/deploy/mps/latest/architecture.html">NVIDIA’s architecture documentation</a>. A model copy adds no GPU compute units.</li><li>Per-stage/node timings overlap and must not be added to infer elapsed time. Loader-node timing omits deferred weight transfers that occur inside inference. Full node timings and measured model-loaded/offloaded memory are downloadable.</li><li>Chunk lengths are aligned to H3's 17k+5 inference grid; delivered frames total 1186 in every run (49.4167 s of video versus 49.408 s of source audio, an 8.7 ms frame-grid difference). Changing chunk count changes audio boundaries, seeds applied to each region, and anchor frequency. Compare scheduling quality within each chunk plan, then judge changed chunk lengths visually.</li><li>Original source WAV was unavailable; this source audio was decoded from the reference report MP4. Native model canvas 736×1280, crop720×1280, export1080×1920. Seeds9901+chunk index. No resolution or step reduction.</li><li>The 65/35 MPS attempt in the previous report failed with GPU lost. GPU recovery was confirmed before this follow-up. The main matrix uses ordinary scheduling. Follow-up checks use standard 100/100 MPS on the best two-transformer layout and a matched one-transformer/one-VAE control, with attached clients verified; the failed 65/35 configuration is excluded.</li></ul></section>
<section><h2>Memory guards, excluded timings and failed attempts</h2><p>Two maximum-length chunks calibrate each stage. For 20 chunks, three transformers plus one VAE would have a conservative summed envelope of 94.0 GiB and four plus one VAE 122.7 GiB, above the 86.0 GiB admission budget. These larger transformer counts were not executed; fitting weights alone does not establish enough activation memory. Initial layouts use a 90% memory admission ceiling; the capacity sweep uses 95%, leaving at least 4.8 GiB estimated headroom. Three transformers plus one VAE uses the measured isolated three-transformer pool peak plus the isolated VAE peak; other layouts sum isolated per-worker peaks. These are admission estimates; actual sampled memory is reported per run. Timings affected by another GPU workload are retained as excluded attempts and never enter the averages. A memory-rejected layout has no measured latency. This is a conservative admission decision, not a measured out-of-memory failure or proof that the layout could never fit.</p><ul>{failure_html}<li>followup_best_combined_repeat2 was stopped at your request before completion; no timing is used. Its replacement attempts have distinct names.</li></ul></section>
<h2>Watch the outputs and inspect joins</h2><p>Outputs with identical SHA-256 hashes share one video asset in this report. Every successful output passed frame-count checks and complete video/audio decoding. Audio is remuxed from the same master in original order. Pixel equality is checked within each chunk plan. Luma screening detects abrupt changes; it does not establish lip-sync or semantic quality.</p>{''.join(sections)}
<section><h2>Reproduce and inspect</h2><p><a href="reproducibility.zip">Download code, inputs and raw measurements</a> · <a href="configuration_matrix.csv">Configuration CSV</a> · <a href="per_node_times.csv">Per-node times</a> · <a href="measurements.json">All measurements</a> · <a href="quality.json">Pixel comparison</a> · <a href="code/CHUNK_COPY_PLAN.md.txt">Protocol and commands</a> · <a href="data/followup_provenance.json">Model and input provenance</a> · <a href="data/followup_resume_environment.json">Host hardware and resumed environment</a> · <a href="data/followup_resume_audit.json">GPU ownership audit</a></p><p>Run the included harness from the script2video repository root using the recorded environment and model assets. Model weights are not bundled. The manifest records input hashes. Commands and assumptions are in CHUNK_COPY_PLAN.md and the runner.</p></section>
<section><h2>Eight GPUs — estimates only</h2>{projection_html}<p>These are one-GPU measurements. Independent workers on eight identical GPUs have an ideal throughput ceiling of eight times the comparable one-GPU worker, before contention in CPU, storage and transfers. Stage-split allocations need isolated warm stage service times, real peer-to-peer bandwidth/latency, topology and memory measurements, plus an eight-GPU end-to-end test. No greater-than-eight-times result is claimed here.</p></section><script>function jump(el,t){{const v=el.parentElement.querySelector("video");const seek=()=>{{v.currentTime=t;v.play();}};if(v.readyState)seek();else{{v.addEventListener("loadedmetadata",seek,{{once:true}});v.load();}}}}</script></html>'''
    (DEST/'index.html').write_text(page)
    files=sorted(str(f.relative_to(DEST)) for f in DEST.rglob('*') if f.is_file())
    if 'report-files.json' not in files:files.append('report-files.json')
    (DEST/'report-files.json').write_text(json.dumps(files,indent=2))
    print(json.dumps({'directory':str(DEST),'successful_runs':len([r for r in records if r['status']=='success']),'groups':len(summary),'best':best},indent=2))

if __name__=='__main__':main()

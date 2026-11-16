"""Build a focused, reproducible fast-mode comparison after generation ends."""
import csv
import html
import json
import re
import shutil
import statistics
import subprocess
import zipfile
from pathlib import Path
from .parallel_chunk_processes import RUNS, file_hash
from .compare_vae_outputs import frame_hash

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'h3-pika-flex-fast-rtx6000'


def main():
    REPORT.mkdir(exist_ok=True)
    exclusions = json.loads((RUNS / 'timing_exclusions.json').read_text())
    names = [f'followup_fast_comparison_{mode}_{i}' for i in (1, 2) for mode in ('default', 'fast')]
    records, children, summaries, quality = [], {}, [], []
    reference = RUNS / names[0] / 'output.mp4'
    ref_hash = frame_hash(reference)
    for name in names:
        folder = RUNS / name
        record = json.loads((folder / 'result.json').read_text())
        assert record['status'] == 'success' and name not in exclusions, name
        assert record['complete_video_audio_decode'] and record['chunks'] == 20
        assert record['transformers'] == 1 and record['vaes'] == 2 and record['mps']
        media = folder / 'output.mp4'
        assert file_hash(media) == record['output_sha256']
        hashed = frame_hash(media)
        assert hashed['decoded_frames'] == 1186
        q = {'name': name, **hashed, 'pixels_equal_to_control': hashed == ref_hash,
             'audio': record['audio_check'], 'seams': record['seam_check']}
        if not q['pixels_equal_to_control']:
            compared = subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-i', str(reference),
                '-i', str(media), '-lavfi', '[0:v]scale=180:320[a];[1:v]scale=180:320[b];[a][b]ssim',
                '-an', '-f', 'null', '-'], capture_output=True, text=True, check=True, timeout=300)
            q['ssim_at_180x320'] = float(re.findall(r'SSIM Y:.*All:([\d.]+)', compared.stderr)[-1])
        quality.append(q)
        stage_children = {stage: [json.loads((RUNS / c['name'] / 'result.json').read_text())
            for c in record[key]] for stage, key in [('denoise', 'denoise_results'), ('decode', 'child_results')]}
        children[name] = stage_children
        counters = {}
        for stage, values in stage_children.items():
            assert len(values) == 20
            for c in values:
                if record['inference_mode'] == 'fast':
                    stats = c['telemetry']['fast_inference']
                    if stage == 'denoise':
                        assert stats['sampler_calls'] == 1 and stats['model_forward_calls'] == 2
                        assert stats['sage2_calls'] > 0 and stats['sage3_calls'] > 0 and stats['sage3_fallback_calls'] == 0
                    else:
                        assert stats['video_decode_calls'] == 1 and stats['nvfp4_gemm_calls'] > 0
                    for k, v in stats.items(): counters[k] = counters.get(k, 0) + v
        row = {'name': name, 'mode': record['inference_mode'], 'total_seconds': record['timed_total_seconds'],
               'video_ready_seconds': record['time_to_assembled_valid_video_seconds'],
               'preparation_seconds': record['preparation_seconds'],
               'encoder_release_gate_seconds': record['encoder_release_gate_seconds'],
               'pipeline_seconds': record['parallel_generation_seconds'],
               'assembly_validation_seconds': record['assembly_and_full_validation_seconds'],
               'extra_qa_seconds': record['timed_total_seconds'] - record['time_to_assembled_valid_video_seconds'],
               'peak_vram_gib': record['gpu_trace_summary']['peak_vram_gib'],
               'gpu_util_mean_pct': record['gpu_trace_summary']['gpu_util_mean_pct']}
        for stage, values in stage_children.items():
            timer = 'sampler_intervals' if stage == 'denoise' else 'decode_intervals'
            spans = [s for c in values for s in c[timer]]
            row[stage + '_summed_call_seconds'] = sum(s['ended_wall_utc'] - s['started_wall_utc'] for s in spans)
            row[stage + '_span_seconds'] = max(s['ended_wall_utc'] for s in spans) - min(s['started_wall_utc'] for s in spans)
        row['fast_counters'] = counters
        summaries.append(row)
        records.append(record)
        dest = REPORT / name
        dest.mkdir(exist_ok=True)
        for filename in ('output.mp4', 'contact_sheet.jpg', 'result.json', 'audio_check.json', 'seam_check.json', 'gpu_trace.json'):
            shutil.copy2(folder / filename, dest / filename)
    aggregates = {}
    for mode in ('default', 'fast'):
        group = [r for r in summaries if r['mode'] == mode]
        aggregates[mode] = {'n': len(group), 'min_total_seconds': min(r['total_seconds'] for r in group),
            'max_total_seconds': max(r['total_seconds'] for r in group),
            **{k: statistics.mean(r[k] for r in group) for k in summaries[0]
               if k not in ('name', 'mode', 'fast_counters')}}
    old = 223.57876813167
    new = aggregates['fast']['total_seconds']
    control = aggregates['default']['total_seconds']
    comparisons = {'historical_total_seconds': old, 'reduction_vs_historical_pct': 100 * (1-new/old),
                   'reduction_vs_control_pct': 100 * (1-new/control), 'speedup_vs_control': control/new}
    result = {'runs': summaries, 'aggregates': aggregates, 'comparisons': comparisons, 'quality': quality,
              'records': records, 'children': children}
    (REPORT / 'results.json').write_text(json.dumps(result, indent=2))
    with (REPORT / 'timings.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in summaries[0] if k != 'fast_counters'])
        writer.writeheader()
        writer.writerows({k:v for k,v in r.items() if k != 'fast_counters'} for r in summaries)
    with zipfile.ZipFile(REPORT / 'reproduce.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('README.txt', '''Reproduction
1. Check out script2video at the repo_commit in source/provenance.json.
2. Apply source/runtime.patch with git apply. Copy harness/*.py into
   agent/rtx6000_benchmark/ and inputs/* into agent/rtx6000_benchmark/inputs/.
3. Use the recorded Python/Torch/CUDA/package versions and existing H3 model
   weights under cache/minimax_h3. Model provenance is in model_provenance.json;
   inputs, workflow graphs, exact commands and seeds are in the run records.
4. Build SageAttention/sageattention3_blackwell at the recorded SageAttention
   commit, with csrc/cutlass at the recorded CUTLASS commit. Build with
   CUDA_HOME=/usr/local/cuda-13.1 and MAX_JOBS=8; use pip/uv --no-deps
   --no-build-isolation --target a private site-packages overlay.
5. Set PYTHONPATH to that overlay. Use the benchmark interpreter explicitly.
   On an idle dedicated GPU, run:
   python -m agent.rtx6000_benchmark.stage_pipeline --name followup_cal_20 --chunks 20 --calibrate-only --run
   python -m agent.rtx6000_benchmark.run_fast_comparison
   python -m agent.rtx6000_benchmark.report_fast_comparison
The comparison runner preserves existing completed runs. Use fresh run names
or a fresh benchmark directory for a new measurement. Calibration is not timed
as part of full-video latency. Do not run quality evaluation during generation.
The deployed worker is not changed by this benchmark. No model weights bundled.
''')
        z.write(RUNS / 'followup_provenance.json', 'model_provenance.json')
        for p in (ROOT / 'fast_comparison_source').iterdir(): z.write(p, 'source/' + p.name)
        for filename in ('run_fast_comparison.py', 'run_chunk_copy_matrix.py', 'stage_pipeline.py',
                         'parallel_chunk_resident.py', 'parallel_chunk_processes.py', 'run_case.py',
                         'preencoded_parallel.py', 'gpu_preencode.py', 'latent_stage_nodes.py',
                         'mps_session.py', 'watch_gpu_ownership.py', 'deferred_encoding.py',
                         'check_audio.py', 'check_seams.py', 'make_contact_sheet.py',
                         'compare_vae_outputs.py', 'report_fast_comparison.py'):
            z.write(ROOT / filename, 'harness/' + filename)
        for p in (ROOT / 'inputs').glob('*'):
            if p.is_file() and p.suffix in ('.jpg', '.wav'): z.write(p, 'inputs/' + p.name)
        for name in names + ['followup_fast_cal20']:
            for p in (RUNS / name).rglob('*.json'): z.write(p, 'runs/' + str(p.relative_to(RUNS)))
            log = RUNS / (name + '.log')
            if log.exists(): z.write(log, 'logs/' + log.name)
        z.writestr('children.json', json.dumps(children, indent=2))
        first = json.loads((RUNS / 'followup_fast_cal20/result.json').read_text())['empty_gpu_gate'][0]['time']
        samples = [json.loads(line) for line in (RUNS / 'followup_ownership_continuous.jsonl').read_text().splitlines()]
        z.writestr('ownership.json', json.dumps([s for s in samples if s['time'] >= first]))
    table = ''.join(f'<tr><td>{r["mode"]} · run {r["name"].rsplit("_",1)[1]}</td><td>{r["total_seconds"]:.2f}</td><td>{r["video_ready_seconds"]:.2f}</td><td>{r["peak_vram_gib"]:.2f}</td><td>{r["gpu_util_mean_pct"]:.1f}%</td></tr>' for r in summaries)
    stages = ''.join(f'<tr><td>{label}</td><td>{aggregates["default"][key]:.2f}</td><td>{aggregates["fast"][key]:.2f}</td></tr>' for key, label in [
        ('preparation_seconds','GPU conditioning preparation'), ('encoder_release_gate_seconds','Wait for encoder release and idle telemetry'), ('pipeline_seconds','Transformer + 2 VAE pipeline, including startup'),
        ('assembly_validation_seconds','Assembly + full decode validation'), ('extra_qa_seconds','Additional audio and join checks'),
        ('denoise_summed_call_seconds','Denoising calls: summed elapsed time (overlaps decoding)'),
        ('decode_summed_call_seconds','VAE calls: summed elapsed time (two workers; overlapping)'),
        ('decode_span_seconds','First VAE call start → last VAE call end')])
    videos = ''.join(f'<section><h3>{html.escape(r["name"])}</h3><video controls preload="metadata" src="{r["name"]}/output.mp4"></video><p><a href="{r["name"]}/contact_sheet.jpg">Inspect chunk joins</a> · <a href="{r["name"]}/result.json">Raw run</a></p></section>' for r in summaries)
    quality_rows = ''.join(f'<tr><td>{q["name"].replace("followup_fast_comparison_", "")}</td><td>{q["decoded_frames"]}</td><td>{q["audio"]["best_lag_ms"]} ms</td><td>{q.get("ssim_at_180x320", "Identical pixels")}</td><td>{q["seams"]["seams_above_nonseam_95th_percentile"]}/19</td></tr>' for q in quality)
    title = f'Fast mode: {new:.1f} seconds for the same podcast'
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{title}</title>
<style>body{{margin:0;background:#101820;color:#e5edf2;font:17px/1.6 system-ui}}main{{max-width:1180px;margin:auto;padding:48px}}h1{{font-size:42px;line-height:1.15}}h2{{margin-top:48px}}a{{color:#79d7ff}}.card{{padding:24px;background:#1b2b38;border-radius:14px}}table{{width:100%;border-collapse:collapse;margin:24px 0}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #395061}}th{{color:#89ddd0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}video{{width:100%;max-height:650px;background:black}}pre{{white-space:pre-wrap;background:#1b2b38;padding:20px}}small{{color:#b8cbd8}}</style>
<main><small>RTX PRO 6000 Blackwell · measured on one GPU · Pika Flex · two steps</small><h1>{title}</h1>
<div class="card"><b>Same configuration: one transformer + two VAEs, running under MPS.</b><p>Fast mode averaged <b>{new:.2f}s</b>; the current-code control averaged <b>{control:.2f}s</b> (two full runs each). That is <b>{comparisons['reduction_vs_control_pct']:.1f}% less time</b> relative to the control. The historical winner averaged <b>223.58s</b> across three runs.</p><p>The output is the same 49.408-second podcast workload. Independent chunks use the same image for first/last conditioning. Chunk joins are still a visual tradeoff; they are not guaranteed seamless.</p></div>
<h2>What changed?</h2><p>Step 1 uses SageAttention2; step 2 uses SageAttention3 FP4. Selected video-VAE decoder projections use NVFP4, retaining a CPU cache of quantized weights in each VAE worker. The input, seeds, chunk lengths, number of steps, export settings and model-copy counts stay fixed.</p>
<h2>Measured times</h2><p>Totals include GPU conditioning preparation, fresh worker startup, generation, encoding, assembly, full video/audio decode validation and additional QA. Calibration and dependency compilation are excluded. “Video ready” excludes the additional audio/join checks. These are fresh-worker video latencies, not sustained warm-service measurements.</p>
<table><tr><th>Run</th><th>Total seconds</th><th>Video ready seconds</th><th>Peak VRAM GiB</th><th>Mean GPU utilization</th></tr>{table}</table>
<h2>Where the time went</h2><table><tr><th>Stage</th><th>Default mean seconds</th><th>Fast mean seconds</th></tr>{stages}</table><p>Stage call totals overlap: do not add denoising and VAE times to predict total latency. GPU utilization and memory are device samples during preparation and the worker pipeline, excluding final assembly and QA. The encoder-release gate waits for three clean idle telemetry samples; its measured waiting time is included in the total.</p>
<h2>Output checks</h2><table><tr><th>Run</th><th>Frames</th><th>Audio lag</th><th>SSIM vs control at 180×320</th><th>Joins above ordinary-motion p95</th></tr>{quality_rows}</table><p>All four outputs passed full video/audio decoding: 1,186 frames at 24 fps, 1080×1920 export. SSIM measures downsampled pixel similarity; it is not a perceptual quality, lip-sync or seamlessness score. FP4 changes numerical results, so inspect both videos. Join flags indicate visible-transition risk already present in the independent-chunk approach.</p>
<h2>Watch the outputs</h2><div class="grid">{videos}</div>
<h2>Reproduce and audit</h2><p><a href="results.json">Complete timings, counters and quality checks</a> · <a href="timings.csv">Timing CSV</a> · <a href="reproduce.zip">Code, inputs, provenance and logs</a></p><p>The source archive records the exact repository commit, source hashes and package versions. SageAttention3 was built from d1a57a5 with CUTLASS 0b55a2f using CUDA 13.1 against PyTorch 2.11.0+cu130, in a private dependency overlay. Existing packages were retained. Calibration tested repeated stage calls. Every fast chunk must use SA2 and SA3 without fallback; every VAE chunk must execute NVFP4 GEMMs.</p><p>The harness refreshes request-local sampler/attention nodes while retaining model loaders. A small runtime validation adjustment permits decode-only fast graphs and verifies their NVFP4 calls. No deployment or merge was performed. GPU ownership was monitored; contended runs are excluded.</p>
<pre>PYTHONPATH=/absolute/path/to/fast_dependencies/site-packages \\
/path/to/benchmark/python -m agent.rtx6000_benchmark.run_fast_comparison</pre><p>Use a clean run directory or new run names for a fresh replay: the runner preserves completed records. Required base model weights are referenced in the archived configuration and are not bundled.</p>
<p><b>Limits:</b> two repeats per mode on one GPU and one podcast. No eight-GPU test, new scaling claim, or broad perceptual benchmark is implied.</p></main></html>'''
    (REPORT / 'index.html').write_text(page)
    (RUNS / 'fast_comparison_summary.json').write_text(json.dumps({'aggregates': aggregates, 'comparisons': comparisons, 'quality': quality}, indent=2))
    print(json.dumps({'aggregates': aggregates, 'comparisons': comparisons, 'report': str(REPORT)}, indent=2))


if __name__ == '__main__': main()

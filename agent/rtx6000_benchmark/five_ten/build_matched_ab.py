"""Build the matched simple-vs-parallel decision report from raw run records."""
import hashlib
import html
import json
import statistics
from pathlib import Path

P = Path(__file__).resolve().parent
GROUPS = {'simple': ['matched_simple_a2', 'matched_simple_a3'],
          'parallel': ['matched_parallel_b2', 'matched_parallel_b1']}


def collect(names):
    runs = [json.loads((P / 'runs' / n / 'result.json').read_text()) for n in names]
    measured = [r for run in runs for r in run['records'] if r['repeat'] != 'warmup']
    assert len(measured) == 3 and all(r['chunks'] == 5 and r['frames_per_chunk'] == 240 and r['fps'] == 24 and not r['foreign_samples'] for r in measured)
    seconds = [r['seconds'] for r in measured]
    events = [e for r in measured for e in r['events']]
    node = [n for e in events for n in e['telemetry']['stages']['nodes']]
    fast = [e['telemetry']['fast_inference'] for e in events]
    return {'runs': names, 'seconds': seconds, 'mean_seconds': statistics.mean(seconds),
            'stdev_seconds': statistics.stdev(seconds), 'min_seconds': min(seconds),
            'max_seconds': max(seconds), 'peak_vram_gib': max(r['peak_vram_gib'] for r in measured),
            'gpu_util_mean_pct': statistics.mean(r['gpu_util_mean_pct'] for r in measured),
            'startup_seconds': [r['startup_seconds'] for r in runs],
            'warmup_seconds': [r['records'][0]['seconds'] for r in runs],
            'stage_seconds_per_video': {role: sum(e['seconds'] for e in events if e['role'] == role) / 3 for role in ('t', 'v')},
            'max_node_end_offloaded_gib': max(n['after'].get('model_offloaded_gib', 0) for n in node),
            'max_model_loaded_gib_by_role': {role: max(n['after'].get('model_loaded_gib', 0) for e in events if e['role'] == role for n in e['telemetry']['stages']['nodes']) for role in ('t', 'v')},
            'fast_per_video': {key: sum(s[key] for s in fast) / 3 for key in ('sampler_calls', 'model_forward_calls', 'sage2_calls', 'sage3_calls', 'sage3_fallback_calls', 'video_decode_calls', 'nvfp4_gemm_calls')},
            'outputs': [str(Path('runs') / name / f'repeat_{rep}' / 'video.mp4') for name, run in zip(names, runs) for rep in [r['repeat'] for r in run['records'] if r['repeat'] != 'warmup']]}


def main():
    a, b = (collect(GROUPS[key]) for key in ('simple', 'parallel'))
    prep = json.loads((P / 'preparation.json').read_text())['seconds']
    saved = a['mean_seconds'] - b['mean_seconds']
    summary = {'simple': a, 'parallel': b, 'preparation_seconds': prep,
               'seconds_saved': saved, 'time_reduction_pct': 100 * saved / a['mean_seconds'],
               'speedup_factor': a['mean_seconds'] / b['mean_seconds'],
               'threshold_met': b['mean_seconds'] <= .8 * a['mean_seconds'],
               'real_time_factor': {k: v['mean_seconds'] / 50 for k, v in [('simple', a), ('parallel', b)]},
               'request_latency_with_preparation_seconds': {k: v['mean_seconds'] + prep for k, v in [('simple', a), ('parallel', b)]},
               'preparation_note': 'Phase-summed, resident request estimate: one measured GPU conditioning preparation pass plus each measured final-MP4 generation run. Startup is separate.',
               'order': 'A warmup+2 measured, B warmup+2 measured, A warmup+1 measured, B warmup+1 measured',
               'workload': 'Five 10-second chunks, 24 fps, 243 native and 240 exported frames per chunk, two-step Pika Flex fast, seeds 9901–9905, independent original portrait anchors, 49.408-second source audio padded with 0.592 seconds silence.'}
    for key, data in [('simple', a), ('parallel', b)]:
        data['video_sha256'] = [hashlib.sha256((P / x).read_bytes()).hexdigest() for x in data['outputs']]
    summary['all_output_bytes_equal'] = len(set(a['video_sha256'] + b['video_sha256'])) == 1
    (P / 'matched_ab_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (P / 'matched_ab_surya.txt').write_text(
        f'RTX PRO 6000 matched Pika Flex fast-mode test (five 10-second chunks, three measured videos each): '
        f'simple 1T+1V/no MPS {a["mean_seconds"]:.2f}s versus pipelined 1T+2V/MPS {b["mean_seconds"]:.2f}s '
        f'through the corrected final MP4. Parallel saves {saved:.2f}s ({summary["time_reduction_pct"]:.1f}%, '
        f'{summary["speedup_factor"]:.3f}x) but uses {b["peak_vram_gib"]-a["peak_vram_gib"]:.2f} GiB more peak VRAM '
        f'({a["peak_vram_gib"]:.2f} → {b["peak_vram_gib"]:.2f} GiB). '
        f'The roughly 20% lower-latency threshold is {"met" if summary["threshold_met"] else "not met"}; '
        f'{"the measured gain supports" if summary["threshold_met"] else "the measured gain does not justify"} '
        f'the extra MPS, VAE worker, and pipeline maintenance complexity for this workload.\n')
    f = lambda x: f'{x:.3f}'
    rows = ''.join(f'<tr><td>{label}</td><td>{f(d["mean_seconds"])} ± {f(d["stdev_seconds"])}</td><td>{", ".join(map(f,d["seconds"]))}</td><td>{f(d["peak_vram_gib"])}</td><td>{f(d["gpu_util_mean_pct"])}</td><td>{f(d["mean_seconds"]/50)}</td></tr>' for label,d in [('A · simple 1T+1V, no MPS',a),('B · parallel 1T+2V, MPS',b)])
    videos = ''.join(f'<article><h3>{label} repeat {i}</h3><video controls preload="metadata" width="360" src="{html.escape(path)}"></video><p><a href="{html.escape(path)}">Open MP4</a></p></article>' for label,d in [('A',a),('B',b)] for i,path in enumerate(d['outputs'],1))
    profile_path = P / 'matched_ab_profiles.json'
    profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
    profile_html = ''.join(f'<tr><td>{k}</td><td>{f(v["kernel_overlap"].get("transformer_vae_kernel_overlap_seconds",0))}</td><td>{f(v["metrics"].get("SMs Active [Throughput %]",{}).get("mean",0))}</td><td>{v["kernel_count"]}</td></tr>' for k,v in profile.items()) if profile else '<tr><td colspan="4">Profile captures pending.</td></tr>'
    decision = 'meets' if summary['threshold_met'] else 'does not meet'
    markup = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>RTX PRO 6000 Pika Flex matched A/B</title><style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#17212b}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbc8d4;padding:9px;text-align:left}}th{{background:#eaf0f5}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}}article{{border:1px solid #ccd;padding:12px}}code,pre{{background:#f3f6f8;padding:3px}}pre{{white-space:pre-wrap}}p,li{{line-height:1.5}}</style><h1>RTX PRO 6000: simple versus parallel Pika Flex fast mode</h1><p><b>Decision:</b> B saves {f(saved)} seconds ({f(summary['time_reduction_pct'])}%) per 50-second video, a {f(summary['speedup_factor'])}× speedup. It {decision} Surya's approximately 20% lower-latency threshold (B ≤ 0.8 × A), so the observed speed gain {'supports' if summary['threshold_met'] else 'does not justify'} the added MPS, second VAE, and pipeline maintenance complexity under this workload.</p><h2>Measured final MP4 latency</h2><table><tr><th>Configuration</th><th>Mean ± sample SD (s)</th><th>Every repeat (s)</th><th>Peak VRAM (GiB)</th><th>Mean GPU util. (%)</th><th>Real-time factor</th></tr>{rows}</table><p>Time starts at first chunk dispatch and ends after final 50-second MP4 assembly <em>and integrated lossless timestamp correction</em>. One full-video warmup per block is excluded. Order: A(2), B(2), A(1), B(1). Each repeat uses exactly five 10-second chunks, 24 fps, 243 native/240 exported frames per chunk, 1,200 exported frames total, seeds 9901–9905, and independent original-image anchors. Both use two-step Pika Flex fast mode, identical prepared conditioning and encoding. The 49.408-second source audio is padded with 0.592 seconds trailing silence for both.</p><p>Resident request estimate including the separately measured {f(prep)}-second GPU conditioning pass: A {f(a['mean_seconds']+prep)} s, B {f(b['mean_seconds']+prep)} s. These are phase sums, not new wall-clock end-to-end requests. Initial process startup: A {', '.join(map(f,a['startup_seconds']))} s; B {', '.join(map(f,b['startup_seconds']))} s. First full-video warmups, which include lazy model loading: A {', '.join(map(f,a['warmup_seconds']))} s; B {', '.join(map(f,b['warmup_seconds']))} s. Startup and warmup are excluded from steady-state means.</p><h2>Residency and fast path</h2><p>A has one transformer worker and one VAE worker with serial chunk dispatch, no MPS, no GPU stage overlap. B has one transformer worker and two VAE workers, MPS and cross-chunk transformer/VAE overlap. Both retain workers through warmup and repeats. Mean call time summed by stage per video: A transformer {f(a['stage_seconds_per_video']['t'])} s, VAE {f(a['stage_seconds_per_video']['v'])} s; B transformer {f(b['stage_seconds_per_video']['t'])} s, VAE {f(b['stage_seconds_per_video']['v'])} s. B stage call sums overlap and cannot be added as latency. Max node-end offloaded memory: A {f(a['max_node_end_offloaded_gib'])} GiB, B {f(b['max_node_end_offloaded_gib'])} GiB (the VAE's existing dynamic-VRAM behavior). Fast-path counters per video: A {html.escape(str(a['fast_per_video']))}; B {html.escape(str(b['fast_per_video']))}. Zero Sage3 fallback calls are required.</p><h2>Nsight Systems</h2><p>Separate full-video warmup plus one replay captures. Profiler runtimes are excluded from the table. Kernel overlap and 1 kHz SM activity come from Nsight Systems; existing Nsight Compute kernel findings are reused from <a href="ncu_summary.json">ncu_summary.json</a>.</p><table><tr><th>Capture</th><th>T/V kernel overlap (s)</th><th>SM active mean (%)</th><th>Kernels</th></tr>{profile_html}</table><h2>Output validation</h2><p>All measured MP4s are linked below. Audio/video frame counts, timestamps and waveform alignment are checked by <a href="validate.py">validate.py</a>; <a href="matched_ab_validation.json">A/B output equivalence</a> and <a href="matched_ab_summary.json">raw summary</a> are preserved. Byte-identical final MP4s across all six runs: {summary['all_output_bytes_equal']}. These checks do not rate visual motion or lip synchronization.</p><div class="grid">{videos}</div><h2>Reproduction</h2><p>From the repository root, use the Python and overlay recorded in <a href="provenance.json">provenance.json</a>. The original preparation pass is in <a href="prepare.py">prepare.py</a>; the benchmark runner is <a href="experiment.py">experiment.py</a>. Replace names with unused directories on rerun, and gate the GPU before every block.</p><pre>python -m agent.rtx6000_benchmark.five_ten.experiment --mode simple --name rerun_a1 --transformers 1 --vaes 1 --repeats 2\npython -m agent.rtx6000_benchmark.five_ten.experiment --mode sequential --name rerun_b1 --transformers 1 --vaes 2 --repeats 2\npython -m agent.rtx6000_benchmark.five_ten.experiment --mode simple --name rerun_a2 --transformers 1 --vaes 1 --repeats 1\npython -m agent.rtx6000_benchmark.five_ten.experiment --mode sequential --name rerun_b2 --transformers 1 --vaes 2 --repeats 1</pre><p>Historical 20-chunk timings were excluded. No new model-copy sweep, NCU capture, deployment, merge or Slack send was performed.</p></html>'''
    if profile:
        observed = profile['matched_parallel']
        markup = markup.replace('<h2>Output validation</h2>',
            f'<p>B shows {f(observed["host_call_overlap"]["transformer_vae_kernel_overlap_seconds"])} seconds of overlapping host stage calls, but only {f(observed["kernel_overlap"]["transformer_vae_kernel_overlap_seconds"])} seconds of actual T/V GPU kernel overlap. A has zero T/V overlap. Denoising-window SM active: A {f(profile["matched_simple"]["metrics"]["SMs Active [Throughput %]"]["denoising_window_mean"])}%, B {f(observed["metrics"]["SMs Active [Throughput %]"]["denoising_window_mean"])}%. Raw captures: <a href="profiles/matched_simple.nsys-rep">A Nsight</a>, <a href="profiles/matched_parallel.nsys-rep">B Nsight</a>; <a href="matched_ab_profiles.json">analysis JSON</a>.</p><h2>Output validation</h2>')
    markup = markup.replace('<pre>python -m agent.rtx6000_benchmark.five_ten.experiment',
        '<pre>export PYTHONPATH="$PWD/agent/rtx6000_benchmark/fast_dependencies/site-packages:$PWD"\nexport PATH="/home/raqeeb/h3/flex-2k-refinement/.h3-pika-env/bin:$PATH"\npython -m agent.rtx6000_benchmark.five_ten.experiment')
    markup = markup.replace('The original preparation pass is in <a href="prepare.py">prepare.py</a>',
        'Exact GPU, source and input hashes are in <a href="matched_ab_provenance.json">matched_ab_provenance.json</a>. The original preparation pass is in <a href="prepare.py">prepare.py</a>')
    markup = markup.replace('</html>', '<h2>Decision context and limits</h2><p><a href="https://videosoftware.slack.com/docs/TA6KDFNQ4/F0CGYU3ALSY">Surya huddle</a> sets a rough 20% latency reduction bar: B ≤ 0.8 × A, equivalent to at least 1.25× throughput at equal workload. The earlier 1T+2V result was 208.868 seconds plus a separately timed ~0.060-second timestamp correction; it is historical reference, not one of these matched measurements. Initial model weights load lazily during the first warmup, so the startup figures isolate process setup while warmup figures include cold model loading and all generation; a pure model-loading subphase is not separately timed. Visual review remains useful for motion and lip sync.</p></html>')
    raw = ''.join(f'<li>{name}: <a href="runs/{name}/result.json">all repeat records and stage telemetry</a>, <a href="runs/{name}/gpu_trace.json">GPU activity samples</a>, <a href="runs/{name}/workers.json">worker PIDs</a></li>' for name in GROUPS['simple'] + GROUPS['parallel'])
    markup = markup.replace('</html>', f'<h2>Raw measurements</h2><ul>{raw}</ul></html>')
    (P / 'matched_ab_report.html').write_text(markup)
    print(json.dumps({k: summary[k] for k in ('seconds_saved','time_reduction_pct','speedup_factor','threshold_met','all_output_bytes_equal')}, indent=2))


if __name__ == '__main__':
    main()

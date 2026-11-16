"""Build a small, media-backed report for the independent-chunk experiments."""

import html
import csv
import json
import shutil
import zipfile
from pathlib import Path

from .parallel_chunk_processes import RUNS


CASES = [
    ('gpu_preencoded_cached_one', '10 chunks · GPU encode once and release · one worker'),
    ('gpu_preencoded_cached_two', '10 chunks · GPU encode once and release · two workers'),
    ('gpu_preencoded_two', '10 chunks · GPU prepare all and release · two workers'),
    ('original_image_parallel_resident_1_default', '10 chunks · one resident model · hard anchor'),
    ('original_image_parallel_eager_1_default', '10 chunks · fresh model process per chunk · hard anchor'),
    ('original_image_parallel_resident_1_anchor_every2', '10 chunks · one resident model · anchor every 2'),
    ('original_image_parallel_resident_1_seven_chunks', '7 chunks · one resident model · hard anchor'),
    ('original_image_parallel_resident_1_fourteen_chunks', '14 chunks · one resident model · hard anchor'),
    ('original_image_parallel_resident_2_cpu_text_retry', '10 chunks · two resident models · CPU text · hard anchor'),
]
DEST = RUNS.parent / 'original_image_parallel_report'


def main():
    DEST.mkdir(exist_ok=True)
    rows = []
    sections = []
    matrix = []
    measurements = {}
    for name, label in CASES:
        source = RUNS / name
        result = json.loads((source / 'result.json').read_text())
        if result['status'] != 'success':
            raise ValueError(f'{name} did not succeed')
        measurements[name] = result
        for filename in ('output.mp4', 'contact_sheet.jpg'):
            shutil.copy2(source / filename, DEST / f'{name}_{filename}')
        data_dir = DEST / 'data' / name
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / 'result.json', data_dir / 'result.json')
        for filename in ('gpu_trace.json',):
            if (source / filename).exists():
                shutil.copy2(source / filename, data_dir / filename)
        stages = {}
        for child in result['child_results']:
            detail = json.loads((RUNS / child['name'] / 'result.json').read_text())
            shutil.copy2(RUNS / child['name'] / 'result.json', data_dir / f'{child["name"]}.json')
            for stage in detail['telemetry']['stages']['stages']:
                key = stage['stage']
                stages[key] = stages.get(key, 0) + stage['duration_seconds']
        timing = result['timed_total_seconds']
        peak = result['gpu_trace_summary']['peak_vram_gib']
        util = result['gpu_trace_summary']['gpu_util_mean_pct']
        seams = result['seam_check']
        matrix.append({'configuration': name, 'timed_seconds': timing,
            'videos_per_hour': 3600 / timing, 'peak_vram_gib': peak,
            'gpu_util_mean_pct': util, 'unusual_joins': seams['seams_above_nonseam_95th_percentile'],
            'joins': len(seams['boundaries']), 'preparation_seconds': result.get('preparation_seconds', ''),
            'sampler_overlap_seconds': result.get('sampler_overlap', {}).get('two_or_more_calls_overlap_seconds', '')})
        rows.append(f'<tr><th>{html.escape(label)}</th><td>{timing:.1f} s</td>'
                    f'<td>{3600/timing:.2f}</td><td>{peak:.1f} GiB</td>'
                    f'<td>{util:.0f}%</td><td>{seams["seams_above_nonseam_95th_percentile"]}'
                    f' / {len(seams["boundaries"])}</td></tr>')
        stage_lines = ''.join(f'<li>{html.escape(k)}: {v:.1f} s summed across chunks</li>'
                              for k, v in stages.items())
        preparation_note = ''
        if 'preparation_seconds' in result:
            preparation_note = (f'<p>Preparation, serialization and preparation-process startup took '
                f'{result["preparation_seconds"]:.1f} s, included in the total above. '
                f'Sampler calls overlapped for {result["sampler_overlap"]["two_or_more_calls_overlap_seconds"]:.1f} s. '
                f'Text cache: {html.escape(str(result["preparation"].get("text_cache")))}.</p>')
        sections.append(f'''<section id="{name}"><h2>{html.escape(label)}</h2>
<p><strong>{timing:.1f} seconds</strong> for a 49.4-second, 1,186-frame video; peak VRAM
{peak:.1f} GiB; mean sampled GPU utilization {util:.0f}%. Inference calls overlapped for
{result['inference_overlap']['two_or_more_calls_overlap_seconds']:.1f} seconds.
{seams['seams_above_nonseam_95th_percentile']} of {len(seams['boundaries'])} joins exceeded
the ordinary-frame 95th percentile. The audio check found {result['audio_check']['best_lag_ms']:.0f} ms lag
and {result['audio_check']['duration_delta_ms']:.0f} ms duration difference.</p>{preparation_note}
<p><a href="data/{name}/result.json">Raw measurement JSON</a></p>
<video controls preload="metadata" src="{name}_output.mp4"></video>
<details><summary>Boundary contact sheet and stage timings</summary>
<img loading="lazy" src="{name}_contact_sheet.jpg" alt="Frames near each chunk boundary">
<ul>{stage_lines}</ul></details></section>''')
    repeats = []
    means = {}
    for base, second, label in (
        ('original_image_parallel_resident_1_default',
         'original_image_parallel_resident_1_default_repeat2', 'Hard anchor'),
        ('original_image_parallel_resident_1_anchor_every2',
         'original_image_parallel_resident_1_anchor_every2_repeat2', 'Anchor every 2'),
        ('gpu_preencoded_cached_one', 'gpu_preencoded_cached_one_repeat2', 'GPU encode once · one worker'),
        ('gpu_preencoded_cached_two', 'gpu_preencoded_cached_two_repeat2', 'GPU encode once · two workers'),
    ):
        first_data = json.loads((RUNS / base / 'result.json').read_text())
        second_data = json.loads((RUNS / second / 'result.json').read_text())
        if second_data['status'] != 'success':
            raise ValueError(f'{second} did not succeed')
        first = first_data['timed_total_seconds']
        second_time = second_data['timed_total_seconds']
        means[base] = (first + second_time) / 2
        data_dir = DEST / 'data' / second
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RUNS / second / 'result.json', data_dir / 'result.json')
        for child in second_data['child_results']:
            shutil.copy2(RUNS / child['name'] / 'result.json', data_dir / f'{child["name"]}.json')
        matrix.append({'configuration': second, 'timed_seconds': second_time,
            'videos_per_hour': 3600 / second_time,
            'peak_vram_gib': second_data['gpu_trace_summary']['peak_vram_gib'],
            'gpu_util_mean_pct': second_data['gpu_trace_summary']['gpu_util_mean_pct'],
            'unusual_joins': second_data['seam_check']['seams_above_nonseam_95th_percentile'],
            'joins': len(second_data['seam_check']['boundaries']),
            'preparation_seconds': second_data.get('preparation_seconds', ''),
            'sampler_overlap_seconds': second_data.get('sampler_overlap', {}).get('two_or_more_calls_overlap_seconds', '')})
        repeats.append(f'<tr><th>{label}</th><td>{first:.1f} s</td>'
                       f'<td>{second_time:.1f} s</td><td>{(first+second_time)/2:.1f} s</td>'
                       f'<td>{second_data["seam_check"]["seams_above_nonseam_95th_percentile"]}/9</td></tr>')
    with (DEST / 'configuration_matrix.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matrix[0]))
        writer.writeheader()
        writer.writerows(matrix)
    code_dir = DEST / 'code'
    code_dir.mkdir(exist_ok=True)
    code_links = []
    for filename in ('preencoded_parallel.py', 'gpu_preencode.py', 'parallel_chunk_resident.py',
                     'parallel_chunk_processes.py', 'run_case.py', 'async_encode_nodes.py', 'check_audio.py',
                     'check_seams.py', 'make_contact_sheet.py', 'ORIGINAL_IMAGE_POLICY.md', 'README.md'):
        target = filename + '.txt'
        shutil.copy2(RUNS.parent / filename, code_dir / target)
        code_links.append(f'<a href="code/{target}">{filename}</a>')
    proof = RUNS / 'gpu_preencoded_cached_one/full_cache_equivalence_check.json'
    shutil.copy2(proof, DEST / 'data/cache_equivalence_check.json')
    with zipfile.ZipFile(DEST / 'reproducibility.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for folder in ('data', 'code'):
            for path in (DEST / folder).rglob('*'):
                if path.is_file():
                    archive.write(path, path.relative_to(DEST))
                    if folder == 'code' and path.name.endswith('.txt'):
                        archive.write(path, 'agent/rtx6000_benchmark/' + path.name[:-4])
        archive.write(DEST / 'configuration_matrix.csv', 'configuration_matrix.csv')
    one_mean = means['gpu_preencoded_cached_one']
    two_mean = means['gpu_preencoded_cached_two']
    gain = (one_mean / two_mean - 1) * 100
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RTX PRO 6000 · same-image parallel chunk experiment</title>
<style>body{{font:17px/1.55 system-ui,sans-serif;color:#17212b;background:#f5f7fb;margin:0 auto;padding:24px;max-width:1050px}}
h1,h2{{line-height:1.2}}.note,section{{background:white;border:1px solid #dbe2eb;border-radius:12px;padding:20px;margin:18px 0}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{border:1px solid #dbe2eb;padding:10px;text-align:left}}th{{background:#eef2f8}}
video{{width:min(100%,540px);display:block;margin:16px 0}}img{{max-width:100%;height:auto}}a{{color:#0759ae}}summary{{cursor:pointer}}
small{{color:#526173}}</style>
<h1>Same-image chunks on one RTX PRO 6000</h1>
<div class="note"><strong>What changed:</strong> Every chunk starts from the same source portrait. Hard-anchor runs also
condition each chunk's last frame on that portrait. This removes the previous-last-frame dependency and permits independent
chunk workers. The final frames are generated, not pixel-locked to the portrait, so a join can still jump.</div>
<div class="note"><strong>GPU encoding first works:</strong> Encode the shared prompt/portrait on the GPU once,
prepare each chunk's audio and latents, release that process, then run workers. Two workers fit at about 72.1 GiB.
Across two runs, one worker averaged {one_mean:.1f} seconds and two averaged {two_mean:.1f} seconds,
giving {gain:.1f}% more videos per hour for this workload. This is a small measured gain, not a doubling.
Both workers share the single GPU's compute and memory bandwidth. Two repetitions are initial evidence; they do not establish a precise long-run speedup. All 70 cached preparation tensors were exactly equal
to the uncached GPU preparation in the <a href="data/cache_equivalence_check.json">equivalence check</a>.</div>
<div class="note"><strong>Practical choice:</strong> GPU encode once, then release the encoder. Use one worker for the lower-memory option (36.4 GiB); use two workers for the fastest measured hard-anchor option (72.1 GiB), accepting nearly twice the memory for this modest gain. More worker copies do not fit the measured safety margin. Shared-VAE scheduling remains untested.</div>
<h2>Measured results</h2><p>All rows use the same 49.4-second source audio, 24 fps, two denoising steps, 1080 × 1920
output and a single RTX PRO 6000. “Videos/hour” is 3600 divided by measured timed seconds for one video.
Worker totals include all lanes, final assembly, and CPU media validation. Calibration and initial source-audio preparation/splitting are excluded. These are timed benchmark runs, not service request-to-response measurements.
GPU preparation rows also include preparation, serialization, model release, and worker startup time.</p>
<table><thead><tr><th>Configuration</th><th>Time</th><th>Videos/hour</th><th>Peak VRAM</th><th>Mean GPU use</th><th>Unusual joins</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Repeat check for the leading settings</h2>
<table><thead><tr><th>Ten-chunk policy</th><th>Run 1</th><th>Run 2</th><th>Mean time</th><th>Flagged joins on run 2</th></tr></thead>
<tbody>{''.join(repeats)}</tbody></table>
<p>Compared with the earlier fully resident hard-anchor baseline, the periodic-anchor setting averaged 8.7 seconds faster across these two runs, about 3.3%.
This is a small sample and it changes the end-image condition on alternating chunks.</p>
<div class="note"><strong>Correction to the initial full-copy experiment:</strong> CPU text encoding is not required to
fit two denoising workers. Preparing conditioning on the GPU first and releasing the encoder lets two workers fit.
The first such test peaked at 72.1 GiB and took 271.8 seconds, including 38.3 seconds of preparation. The new cache
encodes identical prompt/image inputs once and reuses that result; audio and latent preparation remain specific to each chunk.
Workers each retain a transformer and video VAE. Concurrent shared-VAE decoding is not tested here. Sampler-call overlap
is measured with synchronized wall intervals; proving simultaneous CUDA kernels requires a GPU profiler.</div>
<div class="note"><strong>Quality:</strong> Every final MP4 decoded fully, contained exactly 1,186 video frames at 24 fps,
and matched source audio to 0 ms at the checker’s 8 kHz resolution. The luma join statistic is a screening signal,
not a human seam rating. Watch the videos below before deciding whether the pose resets are acceptable.</div>
{''.join(sections)}
<section><h2>Reproduce and inspect</h2><p><a href="configuration_matrix.csv">Configuration matrix CSV</a> ·
<a href="reproducibility.zip">Source and raw measurements ZIP</a></p>
<p>{' · '.join(code_links)}</p><pre>python -m agent.rtx6000_benchmark.preencoded_parallel --name YOUR_RUN --copies 2 --cache-text --run</pre>
<p>Run from the pinned script2video checkout with the benchmark virtual environment described in README.
Extract the ZIP into the pinned checkout: its agent/rtx6000_benchmark directory contains runnable source. Model files and the input audio/portrait are required; this upload contains results and source, not model weights.</p></section>
<div class="note"><strong>Scope:</strong> The original leading two settings and both GPU-encode-once settings have two complete runs each; other rows have one
run each. GPU preparation workers calibrate two longest chunks to check warm memory use; older workers calibrated one chunk.
This report measures only one GPU and makes no eight-GPU speedup claim. The runnable
supervisors and raw JSON are in the local script2video benchmark directory.</div>
</html>'''
    (DEST / 'index.html').write_text(page)
    print(DEST / 'index.html')


if __name__ == '__main__':
    main()

"""Prepare H3 conditioning on GPU, release the encoder, then run resident lanes."""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from ml_modules.minimax_h3_turbo.chunking import prepare_audio
from workers.image2video_h3_turbo.schema import build_chunk_plan
from .parallel_chunk_processes import GPUTrace, RUNS, anchor_at, assemble, file_hash, inference_overlap, split_audio
from .parallel_chunk_resident import one_pool_run, wait_for_empty_gpu
from .run_case import INPUTS, ROOT, chunk_counts


def preencode(directory, counts, audios, anchor_mode, cache_text=False):
    directory.mkdir(parents=True)
    manifest = [{'frames': frames, 'audio': str(audios[i]),
                 'anchor': anchor_at(i, anchor_mode)} for i, frames in enumerate(counts)]
    path = directory / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2))
    started = time.perf_counter()
    with (directory / 'prepare.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'agent.rtx6000_benchmark.gpu_preencode',
                        '--manifest', str(path), *(['--cache-text'] if cache_text else [])], stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=9000, cwd=ROOT.parents[1])
    result = json.loads((directory / 'result.json').read_text())
    if result['status'] != 'success':
        raise RuntimeError('Preparation did not succeed')
    return time.perf_counter() - started, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--copies', type=int, choices=[1, 2, 3], default=2)
    parser.add_argument('--chunks', type=int, default=10)
    parser.add_argument('--anchor-mode', choices=['hard', 'every-2'], default='hard')
    parser.add_argument('--cache-text', action='store_true')
    parser.add_argument('--reuse-calibration', help='Successful same-policy run to reuse for a timing repeat')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    args.clip_device = 'default'
    args.residency = 'dynamic'
    counts = chunk_counts(args.chunks, 'aligned')
    if not args.run:
        print(json.dumps({'copies': args.copies, 'chunk_frames': counts,
            'policy': 'GPU prepare all chunks; preparation process exits; resident denoise/decode workers'}, indent=2))
        return
    directory = RUNS / args.name
    directory.mkdir()
    overall = time.perf_counter()
    evidence = {'name': args.name, 'status': 'running', 'loader': 'GPU-preencoded resident workers',
        'parallel_processes': args.copies, 'reference': 'original', 'anchor_mode': args.anchor_mode,
        'chunk_frames': counts, 'chunks': args.chunks, 'chunk_plan': 'aligned',
        'clip_device': 'gpu preparation process only', 'residency': 'dynamic',
        'cache_text': args.cache_text}
    try:
        evidence['empty_gpu_gate'] = wait_for_empty_gpu()
        master, duration = prepare_audio(INPUTS / 'report_audio.wav', directory, time.monotonic() + 9000)
        if round(duration * 24) != sum(counts):
            raise ValueError('Source duration does not match workload')
        audios = split_audio(master, counts, directory)
        # Two maximum-sized chunks capture both initial and warm-lane allocations.
        longest = max(range(len(counts)), key=lambda i: counts[i])
        cal_counts = [counts[longest]] * 2
        cal_audio = [audios[longest]] * 2
        if args.reuse_calibration:
            source = RUNS / args.reuse_calibration / 'result.json'
            prior = json.loads(source.read_text())
            if (prior['status'] != 'success' or prior['loader'] != evidence['loader'] or
                    prior['cache_text'] != args.cache_text or prior['anchor_mode'] != args.anchor_mode or
                    prior['chunk_frames'] != counts or prior['calibration_chunk_frames'] != cal_counts):
                raise ValueError('Calibration source must be a successful identical preparation policy and chunk plan')
            peak = prior['calibration_peak_gib']
            evidence['calibration_reused_from'] = str(source)
            evidence['calibration_source_sha256'] = file_hash(source)
        else:
            cal_prep = directory / 'calibration_prepared'
            _, cal_meta = preencode(cal_prep, cal_counts, cal_audio, args.anchor_mode, args.cache_text)
            wait_for_empty_gpu()
            args.prepared_dir = str(cal_prep)
            _, calibration = one_pool_run(args, directory / 'calibration', cal_counts, cal_audio,
                                          1, f'{args.name}_calibration')
            peak = max(child['telemetry']['gpu']['memory_used_peak_gib'] for child in calibration)
        evidence['calibration_peak_gib'] = peak
        evidence['calibration_chunk_frames'] = cal_counts
        if args.copies * peak > 95.59 * 0.90:
            raise MemoryError(f'{args.copies} copies × {peak:.2f} GiB exceeds 90% of 95.59 GiB')
        evidence['timed_run_empty_gpu_gate'] = wait_for_empty_gpu()
        trace = GPUTrace()
        trace.start()
        timed = time.perf_counter()
        try:
            prep_dir = directory / 'prepared'
            seconds, preparation = preencode(prep_dir, counts, audios, args.anchor_mode, args.cache_text)
            evidence['preparation_seconds'] = seconds
            evidence['preparation'] = preparation
            evidence['encoder_released_gate'] = wait_for_empty_gpu()
            args.prepared_dir = str(prep_dir)
            generation_started = time.perf_counter()
            startup, children = one_pool_run(args, directory, counts, audios, args.copies, args.name)
            evidence['parallel_generation_seconds'] = time.perf_counter() - generation_started
            evidence['lane_startup'] = startup
        finally:
            trace.shutdown()
            (directory / 'gpu_trace.json').write_text(json.dumps(trace.samples))
        evidence['gpu_trace_summary'] = {'samples': len(trace.samples),
            'peak_vram_gib': max(s['used_gib'] for s in trace.samples),
            'gpu_util_mean_pct': sum(s['util_pct'] for s in trace.samples) / len(trace.samples)}
        evidence['child_results'] = [{k: child[k] for k in ('name', 'lane', 'total_seconds',
            'inference_started_wall_utc', 'inference_ended_wall_utc', 'sampler_intervals')} for child in children]
        evidence['inference_overlap'] = inference_overlap(children)
        spans = [{'inference_started_wall_utc': interval['started_wall_utc'],
                  'inference_ended_wall_utc': interval['ended_wall_utc']}
                 for child in children for interval in child['sampler_intervals']]
        evidence['sampler_overlap'] = inference_overlap(spans)
        if len(spans) != len(counts):
            raise ValueError('Missing sampler timing intervals')
        output, parsed = assemble(directory, [c['name'] for c in children], master, sum(counts))
        evidence['time_to_assembled_valid_video_seconds'] = time.perf_counter() - timed
        evidence['output_probe'] = parsed
        evidence['output_sha256'] = file_hash(output)
        evidence['complete_video_audio_decode'] = True
        evidence['plan'] = build_chunk_plan([{'duration_seconds': frames / 24,
            'reference': 'original', 'anchor': anchor_at(i, args.anchor_mode)}
            for i, frames in enumerate(counts)], sum(counts))
        (directory / 'result.json').write_text(json.dumps(evidence, indent=2))
        for checker in ('check_audio', 'check_seams', 'make_contact_sheet'):
            subprocess.run([sys.executable, '-m', f'agent.rtx6000_benchmark.{checker}', str(directory)],
                           cwd=ROOT.parents[1], check=True, timeout=300, capture_output=True)
        evidence['audio_check'] = json.loads((directory / 'audio_check.json').read_text())
        evidence['seam_check'] = json.loads((directory / 'seam_check.json').read_text())
        evidence['timed_total_seconds'] = time.perf_counter() - timed
        evidence['status'] = 'success'
    except BaseException as exc:
        evidence['status'] = 'failure'
        evidence['error'] = repr(exc)
        (directory / 'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        evidence['total_seconds'] = time.perf_counter() - overall
        (directory / 'result.json').write_text(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()

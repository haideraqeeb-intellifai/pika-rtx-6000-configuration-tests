"""Experimental independent-chunk worker with one complete model copy per process.

Each chunk uses the same portrait as its first image. Hard-anchor mode also uses
it as the last image. A supervisor checks GPU idle, calibrates one chunk's peak
memory, and runs at most the requested number of isolated Comfy processes.
The final MP4 uses the source master audio, not concatenated AAC segments.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
import wave
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .run_case import INPUTS, ROOT, chunk_counts, empty_gpu_gate, probe
from ml_modules.minimax_h3_turbo.chunking import SAMPLE_RATE, prepare_audio
from workers.image2video_h3_turbo.schema import build_chunk_plan


RUNS = ROOT / 'runs'


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inference_overlap(children):
    """Wall-time overlap of child inference calls; GPU kernel overlap needs a profiler."""
    events = []
    for child in children:
        events.extend(((child['inference_started_wall_utc'], 1),
                       (child['inference_ended_wall_utc'], -1)))
    events.sort(key=lambda item: (item[0], item[1]))
    active = maximum = 0
    last = None
    overlap = 0.0
    for when, change in events:
        if last is not None and active >= 2:
            overlap += when - last
        active += change
        maximum = max(maximum, active)
        last = when
    return {'maximum_concurrent_inference_calls': maximum,
            'two_or_more_calls_overlap_seconds': round(overlap, 3)}


def split_audio(master, counts, directory):
    paths = []
    with wave.open(str(master), 'rb') as source:
        if source.getframerate() != SAMPLE_RATE or source.getnchannels() != 2 or source.getsampwidth() != 2:
            raise ValueError('Expected stereo 48 kHz 16-bit master audio')
        for index, frames in enumerate(counts):
            samples = frames * SAMPLE_RATE // 24
            pcm = source.readframes(samples)
            pcm += bytes(max(0, samples * 4 - len(pcm)))
            path = directory / f'audio_{index}.wav'
            with wave.open(str(path), 'wb') as out:
                out.setparams((2, 2, SAMPLE_RATE, 0, 'NONE', 'not compressed'))
                out.writeframes(pcm)
            paths.append(path)
    return paths


def anchor_at(index, mode):
    return ('hard' if mode == 'hard' or
            (mode == 'every-2' and index % 2 == 0) or
            (mode == 'every-3' and index % 3 == 0) or
            (mode == 'first-only' and index == 0) else 'none')


def child_command(name, index, frames, audio, args):
    return [sys.executable, '-m', 'agent.rtx6000_benchmark.run_case',
            '--name', name, '--reference', 'original', '--chunks', '1',
            '--chunk-plan', 'uniform', '--anchor-mode', anchor_at(index, args.anchor_mode),
            '--source-audio', str(audio), '--total-frames', str(frames),
            '--seed-base', str(9901 + index), '--clip-device', args.clip_device,
            '--residency', args.residency, '--supervised-parallel-child']


def run_child(name, index, frames, audio, args):
    path = RUNS / name
    if path.exists():
        raise FileExistsError(path)
    command = child_command(name, index, frames, audio, args)
    started = time.time()
    with (RUNS / f'{name}.log').open('w') as log:
        completed = subprocess.run(command, cwd=ROOT.parents[1], env={**os.environ,
                                   'H3_PARALLEL_SUPERVISOR': '1'}, stdout=log,
                                   stderr=subprocess.STDOUT)
    result_path = path / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    result.update({'process_started_wall_utc': started,
                   'process_ended_wall_utc': time.time(), 'return_code': completed.returncode})
    if completed.returncode or result.get('status') != 'success':
        raise RuntimeError(f'Chunk {index} failed: {result.get("error", "see child log")}')
    return result


class GPUTrace:
    def __init__(self):
        import threading
        self.stop_event = threading.Event()
        self.samples = []
        self.thread = None

    def start(self):
        import pynvml
        import threading
        pynvml.nvmlInit()
        self.pynvml = pynvml
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        def sample():
            while not self.stop_event.is_set():
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                self.samples.append({'time': time.time(),
                                     'used_gib': round(info.used / 1024**3, 3),
                                     'util_pct': util.gpu})
                self.stop_event.wait(0.2)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()

    def shutdown(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        if hasattr(self, 'pynvml'):
            self.pynvml.nvmlShutdown()


def assemble(case_dir, child_names, master, total_frames):
    playlist = case_dir / 'video_concat.ffconcat'
    lines = ['ffconcat version 1.0']
    for name in child_names:
        path = (RUNS / name / 'output.mp4').resolve()
        if "'" in str(path):
            raise ValueError('Apostrophes in run paths are not supported by ffconcat')
        lines.append(f"file '{path}'")
    playlist.write_text('\n'.join(lines) + '\n')
    video_only = case_dir / 'video_only.mp4'
    output = case_dir / 'output.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                    '-f', 'concat', '-safe', '0', '-i', str(playlist), '-map', '0:v:0',
                    '-c:v', 'copy', '-an', str(video_only)], check=True, timeout=300)
    subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', str(video_only), '-i', str(master), '-map', '0:v:0',
                    '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                    '-movflags', '+faststart', str(output)], check=True, timeout=300)
    parsed = probe(output)
    video = next(s for s in parsed['streams'] if s['codec_type'] == 'video')
    audio = next(s for s in parsed['streams'] if s['codec_type'] == 'audio')
    if int(video.get('nb_read_frames', -1)) != total_frames or video['codec_name'] != 'h264':
        raise ValueError('Assembled video has wrong frame count or codec')
    rate_num, rate_den = map(int, video['avg_frame_rate'].split('/'))
    if video['width'] != 1080 or video['height'] != 1920 or rate_den <= 0 or rate_num / rate_den != 24:
        raise ValueError('Assembled video has wrong size or frame rate')
    if audio['codec_name'] != 'aac' or int(audio['channels']) != 2:
        raise ValueError('Assembled audio has wrong codec or channels')
    with wave.open(str(master), 'rb') as source:
        audio_duration = source.getnframes() / source.getframerate()
    if abs(float(audio['duration']) - audio_duration) > 0.05:
        raise ValueError('Assembled audio duration differs from the source master')
    subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
                    '-i', str(output), '-map', '0:v:0', '-map', '0:a:0',
                    '-f', 'null', '-'], check=True, timeout=300, capture_output=True)
    return output, parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--chunks', type=int, default=10)
    parser.add_argument('--chunk-plan', choices=['report', 'uniform', 'aligned'], default='aligned')
    parser.add_argument('--copies', type=int, choices=[1, 2, 3], default=2)
    parser.add_argument('--clip-device', choices=['default', 'cpu'], default='default')
    parser.add_argument('--residency', choices=['dynamic', 'highvram'], default='dynamic')
    parser.add_argument('--anchor-mode', choices=['hard', 'every-2', 'every-3', 'first-only'], default='hard')
    parser.add_argument('--run', action='store_true', help='Execute GPU work; otherwise print the plan')
    args = parser.parse_args()
    counts = chunk_counts(args.chunks, args.chunk_plan)
    if not args.run:
        print(json.dumps({'name': args.name, 'copies': args.copies,
                          'clip_device': args.clip_device, 'residency': args.residency,
                          'anchor_mode': args.anchor_mode, 'chunk_frames': counts}, indent=2))
        return
    case_dir = RUNS / args.name
    if case_dir.exists():
        raise FileExistsError(case_dir)
    case_dir.mkdir(parents=True)
    started = time.perf_counter()
    evidence = {'name': args.name, 'status': 'running', 'parallel_processes': args.copies,
                'reference': 'original', 'anchor_mode': args.anchor_mode,
                'chunks': args.chunks, 'chunk_plan': args.chunk_plan,
                'clip_device': args.clip_device, 'residency': args.residency,
                'total_frames': sum(counts), 'chunk_frames': counts}
    try:
        evidence['empty_gpu_gate'] = empty_gpu_gate()
        master, duration = prepare_audio(INPUTS / 'report_audio.wav', case_dir,
                                         time.monotonic() + 9000)
        if round(duration * 24) != sum(counts):
            raise ValueError('Source duration changed; expected the original 1,186-frame workload')
        audios = split_audio(master, counts, case_dir)
        # Calibrate the chosen model-placement policy before admitting multiple copies.
        calibration_name = f'{args.name}_calibration_{uuid.uuid4().hex[:8]}'
        calibration = run_child(calibration_name, 0, counts[0], audios[0], args)
        peak = calibration['telemetry']['gpu']['memory_used_peak_gib']
        capacity = 95.59
        evidence['calibration_peak_gib'] = peak
        if args.copies * peak > capacity * 0.90:
            raise MemoryError(f'{args.copies} copies × {peak:.2f} GiB exceeds 90% of {capacity} GiB')
        evidence['timed_run_empty_gpu_gate'] = empty_gpu_gate()
        names = [f'{args.name}_chunk_{index}_{uuid.uuid4().hex[:6]}' for index in range(args.chunks)]
        trace = GPUTrace()
        trace.start()
        timed_started = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=args.copies) as pool:
                futures = [pool.submit(run_child, names[i], i, counts[i], audios[i], args)
                           for i in range(args.chunks)]
                children = [future.result() for future in futures]
        finally:
            trace.shutdown()
        evidence['parallel_generation_seconds'] = time.perf_counter() - timed_started
        evidence['child_results'] = [{'name': child['name'],
                                     'total_seconds': child['total_seconds'],
                                     'peak_vram_gib': child['telemetry']['gpu']['memory_used_peak_gib'],
                                     'inference_started_wall_utc': child['inference_started_wall_utc'],
                                     'inference_ended_wall_utc': child['inference_ended_wall_utc']}
                                    for child in children]
        evidence['inference_overlap'] = inference_overlap(children)
        evidence['gpu_trace_summary'] = {
            'samples': len(trace.samples),
            'peak_vram_gib': max((s['used_gib'] for s in trace.samples), default=0),
            'gpu_util_mean_pct': round(sum(s['util_pct'] for s in trace.samples) / len(trace.samples), 2)
            if trace.samples else None,
        }
        output, parsed = assemble(case_dir, names, master, sum(counts))
        evidence['output_probe'] = parsed
        evidence['output_sha256'] = file_hash(output)
        evidence['complete_video_audio_decode'] = True
        evidence['plan'] = build_chunk_plan([
            {'duration_seconds': count / 24, 'reference': 'original',
             'anchor': anchor_at(index, args.anchor_mode)}
            for index, count in enumerate(counts)], sum(counts))
        (case_dir / 'result.json').write_text(json.dumps(evidence, indent=2) + '\n')
        for checker in ('check_audio', 'check_seams', 'make_contact_sheet'):
            subprocess.run([sys.executable, '-m', f'agent.rtx6000_benchmark.{checker}',
                            str(case_dir)], cwd=ROOT.parents[1], check=True,
                           timeout=300, capture_output=True)
        evidence['audio_check'] = json.loads((case_dir / 'audio_check.json').read_text())
        evidence['seam_check'] = json.loads((case_dir / 'seam_check.json').read_text())
        evidence['timed_total_seconds'] = time.perf_counter() - timed_started
        evidence['status'] = 'success'
    except BaseException as exc:
        evidence['status'] = 'failure'
        evidence['error'] = repr(exc)
        raise
    finally:
        evidence['total_seconds'] = time.perf_counter() - started
        (case_dir / 'result.json').write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__':
    main()

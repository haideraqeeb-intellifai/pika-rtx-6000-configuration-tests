"""Compare decoded AAC timing against the fixed source waveform."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def decode(path):
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(path),
               '-map', '0:a:0', '-ac', '1', '-ar', '8000', '-f', 'f32le', 'pipe:1']
    return np.frombuffer(subprocess.run(command, check=True, capture_output=True).stdout,
                         dtype='<f4').copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    source = decode(Path(__file__).resolve().parent / 'inputs' / 'report_audio.wav')
    output = decode(args.run_dir / 'output.mp4')
    # Search encoder delay over +/- 250 ms at 8 kHz. Subsample the correlation
    # work while retaining full-rate lag resolution.
    stride = 8
    reference = source[2000:-2000:stride]
    reference -= reference.mean()
    best = (-float('inf'), 0)
    for lag in range(-2000, 2001):
        start = 2000 + lag
        candidate = output[start:start + len(reference) * stride:stride]
        if len(candidate) != len(reference):
            continue
        candidate = candidate - candidate.mean()
        denominator = np.linalg.norm(reference) * np.linalg.norm(candidate)
        correlation = float(np.dot(reference, candidate) / denominator) if denominator else 0.0
        if correlation > best[0]:
            best = (correlation, lag)
    result = {'source_samples_8khz': len(source), 'output_samples_8khz': len(output),
              'duration_delta_ms': round((len(output) - len(source)) / 8, 3),
              'best_lag_ms': round(best[1] / 8, 3), 'waveform_correlation': round(best[0], 6)}
    (args.run_dir / 'audio_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

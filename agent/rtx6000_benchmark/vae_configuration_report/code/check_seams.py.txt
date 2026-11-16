"""Screen chunk boundaries for abrupt changes; not a semantic quality score."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    result = json.loads((args.run_dir / 'result.json').read_text())
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i',
               str(args.run_dir / 'output.mp4'), '-map', '0:v:0',
               '-vf', 'scale=180:320,format=gray', '-f', 'rawvideo', 'pipe:1']
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 320, 180)
    difference = np.abs(np.diff(frames.astype(np.int16), axis=0)).mean(axis=(1, 2))
    boundaries = [sum(item['frames'] for item in result['plan'][:i])
                  for i in range(1, len(result['plan']))]
    seams = [float(difference[boundary - 1]) for boundary in boundaries]
    nonseam = np.delete(difference, [boundary - 1 for boundary in boundaries])
    median = float(np.median(nonseam))
    metrics = {'frame_count': len(frames), 'boundaries': boundaries,
               'seam_mean_absolute_luma_difference': [round(value, 4) for value in seams],
               'nonseam_median_difference': round(median, 4),
               'seam_median_to_nonseam_median_ratio': round(float(np.median(seams)) / median, 3)
               if median else None,
               'seams_above_nonseam_95th_percentile': int(sum(value > np.percentile(nonseam, 95)
                                                               for value in seams))}
    (args.run_dir / 'seam_check.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()

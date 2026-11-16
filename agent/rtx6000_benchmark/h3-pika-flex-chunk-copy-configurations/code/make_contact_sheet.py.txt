"""Create a small seam inspection sheet from a validated output."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--boundaries')
    args = parser.parse_args()
    video = args.run_dir / 'output.mp4'
    result = json.loads((args.run_dir / 'result.json').read_text())
    counts = [item['frames'] for item in result['plan']]
    boundaries = ([int(x) for x in args.boundaries.split(',')] if args.boundaries else
                  [sum(counts[:i]) for i in range(1, len(counts))])
    total = sum(counts)
    frames = sorted({0, 100, total // 2, total - 1, *(x for b in boundaries for x in (b - 1, b))})
    temp = args.run_dir / 'inspection_frames'
    temp.mkdir(exist_ok=True)
    expression = '+'.join(f'eq(n\\,{n})' for n in frames)
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video),
                    '-vf', f'select={expression}', '-vsync', '0', str(temp / '%03d.png')], check=True)
    decoded = {n: np.asarray(Image.open(temp / f'{i + 1:03d}.png').convert('RGB'))
               for i, n in enumerate(frames)}
    width, height = 200, 360
    columns = 5
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new('RGB', (columns * width, rows * height), '#101319')
    draw = ImageDraw.Draw(sheet)
    for i, n in enumerate(frames):
        image = Image.fromarray(decoded[n])
        image.thumbnail((width - 8, height - 34))
        x, y = i % columns * width, i // columns * height
        sheet.paste(image, (x + (width - image.width) // 2, y + 24))
        draw.text((x + 8, y + 5), f'frame {n}', fill='white')
    sheet.save(args.run_dir / 'contact_sheet.jpg', quality=90)
    for b in boundaries:
        first = decoded[b - 1].astype(np.float32)
        second = decoded[b].astype(np.float32)
        mse = np.mean((first - second) ** 2)
        psnr = 10 * math.log10(255 ** 2 / mse) if mse else math.inf
        print(f'boundary {b}: adjacent-frame PSNR {psnr:.3f} dB')


if __name__ == '__main__':
    main()

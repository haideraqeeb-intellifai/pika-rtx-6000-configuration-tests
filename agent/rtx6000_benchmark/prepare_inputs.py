"""Prepare a documented, report-comparable local podcast input."""

import hashlib
import json
import re
import subprocess
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / 'inputs'
INPUTS.mkdir(parents=True, exist_ok=True)
REPORT_HTML = INPUTS / 'source_report.html'


def fetch(url, output):
    if output.exists():
        return
    temporary = output.with_suffix(output.suffix + '.part')
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as target:
        while block := response.read(1024 * 1024):
            target.write(block)
    temporary.replace(output)


report = REPORT_HTML.read_text()
match = re.search(r'https://s3\.renderplatform\.com/[^" ]*raccoon_full_nvenc_chained_assembled\.mp4\?[^" ]+', report)
if not match:
    raise RuntimeError('Chained report video URL is unavailable')
video_url = match.group(0).replace('&amp;', '&')
video = INPUTS / 'report_chained.mp4'
fetch(video_url, video)

image_url = ('https://cdn.podcastflow.ai/user-assets/staging/2026/11/04/avatars/headshots/'
             'rhea-the-raccoon-01a1a544-a3ae-729c-863a-5393d315c56e-headshot.jpg')
image = INPUTS / 'raccoon_headshot.jpg'
fetch(image_url, image)

audio = INPUTS / 'report_audio.wav'
if not audio.exists():
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video),
                    '-vn', '-c:a', 'pcm_s16le', str(audio)], check=True)

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

manifest = {
    'source_report_video': 'raccoon_full_nvenc_chained_assembled.mp4',
    'source_image_url': image_url,
    'audio_note': 'PCM decoded from the report chained MP4 AAC track; original WAV was unavailable.',
    'files': {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in (video, image, audio)},
}
(INPUTS / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))

"""Decode every matched output, check AV alignment, and compare exact outputs."""
import hashlib
import json
import subprocess
from pathlib import Path

from .validate import validate

P = Path(__file__).resolve().parent
NAMES = ['matched_simple_a2', 'matched_parallel_b2',
         'matched_simple_a3', 'matched_parallel_b1']


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


if __name__ == '__main__':
    rows = []
    for name in NAMES:
        rows += validate(P / 'runs' / name)
    full = [r for r in rows if r['frames'] == 1200]
    chunks = [r for r in rows if r['frames'] == 240]
    assert len(full) == 6 and len(chunks) == 30
    assert all(r['full_av_decode'] == 'passed' and r['constant_frame_timestamps'] and
               abs(r['audio_start']) < 1e-6 and r['audio_lag_ms'] == 0 and
               r['audio_zero_lag_correlation'] > .99 for r in full)
    remux = []
    for row in full:
        final = P / row['file']
        original = final.with_name('original_assembly.mp4')
        assert original.exists()
        same = {}
        for kind in ('v', 'a'):
            command = lambda path: subprocess.check_output(
                ['ffmpeg', '-v', 'error', '-i', str(path), '-map', f'0:{kind}:0',
                 '-c', 'copy', '-f', 'hash', '-'], timeout=120)
            same[kind] = command(original) == command(final)
        assert all(same.values())
        remux.append({'final': row['file'], 'original': str(original.relative_to(P)),
                      'compressed_stream_payloads_unchanged': same})
    latent_hashes = {}
    for i in range(5):
        latent_hashes[i] = [sha(P / 'runs' / name / f'repeat_{rep}' / f'latent_{i}.pt')
                            for name, count in zip(NAMES, [2, 2, 1, 1])
                            for rep in range(1, count + 1)]
    result = {'full_videos': full, 'chunks': chunks,
              'lossless_timestamp_remux': remux,
              'all_final_mp4_bytes_equal': len({r['sha256'] for r in full}) == 1,
              'all_latent_bytes_equal_per_chunk': all(len(set(h)) == 1 for h in latent_hashes.values()),
              'latent_sha256_by_chunk': latent_hashes,
              'note': 'Full AV decode and zero-start timestamp/audio alignment checked for unique content; byte-identical files reuse the validated SHA256 cache.'}
    assert result['all_latent_bytes_equal_per_chunk']
    (P / 'matched_ab_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('all_final_mp4_bytes_equal', 'all_latent_bytes_equal_per_chunk')}, indent=2))

"""Print the reproducible one-GPU configuration matrix from raw result files."""

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def row(path):
    d = json.loads(path.read_text())
    graph_path = path.parent / 'workflow.json'
    graph = json.loads(graph_path.read_text()) if graph_path.exists() else {}
    dependency_valid = all(
        graph.get(f'c{i}_5', {}).get('inputs', {}).get('first_frame') == [f'buffer_{i - 1}', 2]
        for i in range(1, d['chunks'])
    ) if d['reference'] == 'chained' else False
    stage = {}
    for node in d.get('telemetry', {}).get('stages', {}).get('nodes', []):
        key = node['class_type']
        stage[key] = stage.get(key, 0) + node['duration_seconds']
    gpu = d.get('telemetry', {}).get('gpu', {})
    audio_path = path.parent / 'audio_check.json'
    seam_path = path.parent / 'seam_check.json'
    audio = json.loads(audio_path.read_text()) if audio_path.exists() else {}
    seam = json.loads(seam_path.read_text()) if seam_path.exists() else {}
    return {
        'name': d['name'], 'status': d['status'], 'chunks': d['chunks'],
        'chunk_plan': d.get('chunk_plan', 'report'), 'reference': d['reference'],
        'anchor_mode': d.get('anchor_mode', 'hard'),
        'previous_last_frame_dependency_valid': dependency_valid,
        'transformers': d.get('transformers', 1), 'vaes': d.get('vaes', 1),
        'texts': d.get('texts', 1), 'residency': d.get('residency', 'dynamic'),
        'clip_device': d.get('clip_device', 'default'),
        'vae_decode': d.get('vae_decode', 'default'),
        'frame_storage': d.get('frame_storage', 'memory'),
        'encoding_overlap': d.get('encoding_overlap', False),
        'inferred_frames': sum(x['inference_frames'] for x in d.get('plan', [])),
        'delivered_frames': sum(x['frames'] for x in d.get('plan', [])),
        'end_to_end_s': round(d['total_seconds'], 3),
        'condition_s': round(stage.get('MiniMaxH3ImageToVideo', 0), 3),
        'denoise_s': round(stage.get('SamplerCustomAdvanced', 0), 3),
        'vae_decode_s': round(stage.get('VAEDecode', 0) + stage.get('VAEDecodeTiled', 0) +
                              stage.get('H3NestedVAEDecodeTiled', 0), 3),
        'frame_store_s': round(stage.get('H3StoreChunk', 0) +
                               stage.get('H3AsyncStoreChunk', 0), 3),
        'encode_s': d.get('encoding', {}).get('encoding_seconds', ''),
        'encode_final_wait_s': d.get('encoding', {}).get('final_wait_seconds', ''),
        'peak_vram_gib': gpu.get('memory_used_peak_gib', ''),
        'peak_process_rss_gib': d.get('peak_process_rss_gib', ''),
        'gpu_util_mean_pct': gpu.get('gpu_util_mean_pct', ''),
        'gpu_energy_wh_est': gpu.get('energy_wh_estimate', ''),
        'output_sha256': d.get('output_sha256', ''),
        'audio_lag_ms': audio.get('best_lag_ms', ''),
        'audio_correlation': audio.get('waveform_correlation', ''),
        'seam_ratio': seam.get('seam_median_to_nonseam_median_ratio', ''),
    }


def main():
    rows = [row(path) for path in sorted((ROOT / 'runs').glob('*/result.json'))
            if 'name' in json.loads(path.read_text())]
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


if __name__ == '__main__':
    main()

"""Transparent service-rate estimates from measured one-GPU H3 stage timings."""

import argparse
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAGE_CLASSES = {
    'text_conditioning': {'MiniMaxH3ImageToVideo'},
    'denoising': {'SamplerCustomAdvanced'},
    'vae_decode': {'VAEDecode', 'VAEDecodeTiled', 'H3NestedVAEDecodeTiled'},
}
ALLOCATIONS = [(1, 4, 3), (1, 5, 2), (2, 4, 2), (2, 3, 3), (1, 3, 4), (1, 6, 1)]


def service_times(name):
    result = json.loads((ROOT / 'runs' / name / 'result.json').read_text())
    nodes = result['telemetry']['stages']['nodes']
    stages = {stage: sum(node['duration_seconds'] for node in nodes
                         if node['class_type'] in classes)
              for stage, classes in STAGE_CLASSES.items()}
    return result['total_seconds'], stages


def project(names, *, bandwidth_gbs, payload_gib, transfer_latency_ms,
            denoising_speedup=0.0, stage_overhead=0.0):
    runs = [service_times(name) for name in names]
    wall = statistics.median(item[0] for item in runs)
    measured_stages = {key: statistics.median(item[1][key] for item in runs)
                       for key in STAGE_CLASSES}
    stages = dict(measured_stages)
    stages['denoising'] *= 1 - denoising_speedup
    single_per_hour = 3600 / wall
    ideal_independent = 8 * single_per_hour
    # Host-mediated transfer is deliberately modeled as a serialized resource.
    # Payload includes conditioning, latents, and the previous-last-frame return.
    transfer_service = payload_gib / bandwidth_gbs
    transfer_latency = transfer_service + len(json.loads(
        (ROOT / 'runs' / names[0] / 'result.json').read_text())['plan']) * 2 * transfer_latency_ms / 1000
    estimates = []
    for count_text, count_denoise, count_vae in ALLOCATIONS:
        services = {
            'text_conditioning': stages['text_conditioning'] * (1 + stage_overhead) / count_text,
            'denoising': stages['denoising'] * (1 + stage_overhead) / count_denoise,
            'vae_decode': stages['vae_decode'] * (1 + stage_overhead) / count_vae,
            'shared_transfer': transfer_service,
        }
        interval = max(services.values())
        estimates.append({'allocation': f'{count_text}/{count_denoise}/{count_vae}',
                          'interval_seconds': round(interval, 3),
                          'videos_per_hour_upper_bound': round(3600 / interval, 2),
                          'bottleneck': max(services, key=services.get),
                          'unloaded_latency_floor_seconds': round(
                              wall - measured_stages['denoising'] * denoising_speedup +
                              transfer_latency + stage_overhead * sum(stages.values()), 3),
                          'added_per_video_transfer_latency_seconds': round(transfer_latency, 3)})
    return {'source_runs': names, 'measured_one_gpu_wall_seconds_median': round(wall, 3),
            'measured_stage_seconds_median': {key: round(value, 3) for key, value in measured_stages.items()},
            'one_gpu_videos_per_hour': round(single_per_hour, 2),
            'eight_independent_ideal_videos_per_hour': round(ideal_independent, 2),
            'assumptions': {'payload_gib_per_video': payload_gib,
                            'effective_shared_bandwidth_gib_per_second': bandwidth_gbs,
                            'latency_ms_per_stage_transfer': transfer_latency_ms,
                            'denoising_speedup_on_dedicated_gpu': denoising_speedup,
                            'fractional_stage_scheduling_overhead': stage_overhead,
                            'no_cluster_cpu_or_encoder_limit': True},
            'stage_allocations': estimates}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', help='Names of repeated one-GPU runs')
    parser.add_argument('--bandwidth-gbs', type=float, default=5.0)
    parser.add_argument('--payload-gib', type=float, default=1.0)
    parser.add_argument('--transfer-latency-ms', type=float, default=2.0)
    parser.add_argument('--denoising-speedup', type=float, default=0.0)
    parser.add_argument('--stage-overhead', type=float, default=0.0)
    args = parser.parse_args()
    if (args.bandwidth_gbs <= 0 or args.payload_gib < 0 or
            not 0 <= args.denoising_speedup < 1 or args.stage_overhead < 0):
        parser.error('Invalid bandwidth, payload, speedup, or overhead')
    print(json.dumps(project(args.runs, bandwidth_gbs=args.bandwidth_gbs,
                             payload_gib=args.payload_gib,
                             transfer_latency_ms=args.transfer_latency_ms,
                             denoising_speedup=args.denoising_speedup,
                             stage_overhead=args.stage_overhead), indent=2))


if __name__ == '__main__':
    main()

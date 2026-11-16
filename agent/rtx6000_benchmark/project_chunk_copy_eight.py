"""Scenario model; never labels eight-GPU rates as measurements."""
import json, statistics
from .parallel_chunk_processes import RUNS

def main():
    profile=json.loads((RUNS/'followup_isolated_20'/'result.json').read_text())
    records=json.loads((RUNS.parent/'h3-pika-flex-chunk-copy-configurations/measurements.json').read_text())
    best=records['summary'][0]
    services={'preparation':profile['preparation_seconds'], **profile['stage_phase_seconds']}
    # This is deliberately a fresh-stage service model, including process startup/load.
    # Warm dedicated-worker gains are unknown and are only represented by sensitivity.
    directory=RUNS/profile['name']
    prepared=sum(p.stat().st_size for p in (directory/'prepared').glob('*.pt'))
    latent=profile['latent_handoff_bytes']
    payload=(prepared+latent)/1024**3
    allocations=[(1,4,3),(1,5,2),(2,4,2),(2,3,3),(1,3,4),(1,6,1)]
    scenarios=[]
    for speed,overhead,label in [(1.2,.20,'Slower stages + overhead'),(1.0,.10,'Measured stage proxy + overhead'),(.8,0,'Optimistic warm-stage improvement')]:
      for bandwidth,latency_ms,hops,transfer in [(1,5,2,'Slow host-mediated'),(5,2,2,'Moderate host-mediated'),(25,.1,1,'Fast direct transfer')]:
        transfer_seconds=hops*payload/bandwidth+40*latency_ms/1000
        for e,t,v in allocations:
          service={'preparation':services['preparation']*speed*(1+overhead)/e,
                   'denoising':services['denoise']*speed*(1+overhead)/t,
                   'VAE/export':services['decode']*speed*(1+overhead)/v,
                   'shared interconnect':transfer_seconds}
          interval=max(service.values())
          scenarios.append({'GPUs_preparation_transformer_VAE':[e,t,v],'stage_scenario':label,
              'stage_time_multiplier':speed,'stage_overhead_fraction':overhead,'transfer_scenario':transfer,
              'effective_bandwidth_GiB_s':bandwidth,'latency_ms_per_handoff':latency_ms,'payload_link_traversals':hops,
              'transfer_seconds_per_video':transfer_seconds,'interval_seconds':interval,
              'estimated_videos_per_hour':3600/interval,'bottleneck':max(service,key=service.get)})
    independent=[{'efficiency_assumption':e,'estimated_videos_per_hour':8*3600/best['mean_video_ready_seconds']*e} for e in (.8,.9,1)]
    best_stage_interval=min(max(services['preparation']/e,services['denoise']/t,services['decode']/v) for e,t,v in allocations)
    break_even={str(overhead):1-(best['mean_video_ready_seconds']/8)/(best_stage_interval*(1+overhead)) for overhead in (0,.1)}
    result={'uniform_stage_time_reduction_to_match_eight_independent':break_even,'status':'ESTIMATES_NOT_EIGHT_GPU_MEASUREMENTS','measured_stage_profile_run':profile['name'],
      'measured_stage_seconds_including_fresh_stage_loading':services,
      'measured_prepared_artifact_bytes':prepared,'measured_latent_artifact_bytes':latent,
      'serialized_artifact_payload_GiB':payload,'single_gpu_baseline':best,
      'eight_independent_worker_scenarios':independent,'stage_allocation_scenarios':scenarios,
      'assumptions':['Stage service includes fresh process/model load and local validation; it is a conservative proxy for dedicated warm workers, not a warm measurement.',
        'The preparation GPU handles text/image/audio conditioning; the VAE GPU also exports locally. Full decoded frames do not cross GPUs.',
        'Enough independent videos/chunks are queued to keep stages fed. Steady-state throughput is not single-video latency.',
        'No shared CPU, RAM, storage or final assembly bottleneck is assumed. Actual rates can be lower.',
        'Bandwidth is aggregate effective shared bandwidth in GiB/s, not a link specification; latency charged for 40 logical handoffs per video.',
        'File sizes include serialization overhead and duplicated conditioning. Direct tensor transfer and exact cache reuse may need fewer bytes.',
        'Scenario ranges are assumptions, not confidence intervals. No measured superlinear scaling is claimed.'],
      'memory_limits':{'GPU_GiB':95.59,'independent_peak_GiB':best['peak_vram_gib'],
        'sampled_preparation_peak_GiB':max(c['telemetry']['gpu']['memory_used_peak_gib'] for c in profile['preparation']['chunks']),
        'isolated_transformer_calibrated_peak_GiB':profile['calibration_peak_gib']['denoise'],
        'isolated_VAE_calibrated_peak_GiB':profile['calibration_peak_gib']['decode'],
        'note':'One stage worker per allocated GPU; preparation peak is a sampled per-operation maximum; no eight-GPU memory fit verified. Queued tensors and in-flight activations need headroom.'}}
    (RUNS/'followup_eight_gpu_estimates.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'services':services,'payload_GiB':payload,'independent':independent},indent=2))
if __name__=='__main__':main()

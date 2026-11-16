"""Plain-language findings derived from the final uncontended summary."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    data=json.loads((ROOT/'h3-pika-flex-chunk-copy-configurations/measurements.json').read_text())
    groups=data['summary'];best=groups[0]
    eligible=[r for r in groups if r['samples']>=2 and r['mean_total_seconds']<=best['mean_total_seconds']*1.02]
    recommended=min(eligible,key=lambda r:r['peak_vram_gib']) if eligible else best
    def find(t,v,mps=False,enc=False,res='dynamic'):
        return next(r for r in groups if r['key']==[20,t,v,res,enc,mps])
    ordinary=find(1,1);one_mps=find(1,1,mps=True);async_one=find(1,1,enc=True);two=find(1,2,mps=True);four=find(1,4,mps=True);tt=find(2,2,mps=True)
    def row(r):return f"| {r['label']} | {r['mean_total_seconds']:.1f} s | {r['min_total_seconds']:.1f}–{r['max_total_seconds']:.1f} s | {r['peak_vram_gib']:.1f} GiB | {r['samples']} |"
    capacity=''
    capacity_path=ROOT/'runs/capacity_matrix_summary.json'
    if capacity_path.exists():
        capacity_data=json.loads(capacity_path.read_text())
        capacity_rows='\n'.join(f"| {r['transformers']} transformers + {r['vaes']} VAEs | {r['mean_seconds']:.1f} s | {r['peak_vram_gib']:.1f} GiB | {r['n']} |" for r in capacity_data['summary'] if r['vaes']<=4)
        capacity=f'''### Filling more of the GPU: latest capacity sweep

These runs keep transformer and VAE workers active together under MPS. The current-day 1-transformer/2-VAE control is shown separately here from its earlier repetitions.

| Simultaneous stage pools | Mean total including QA | Peak VRAM | Clean runs in this sweep |
|---|---:|---:|---:|
{capacity_rows}

The user narrowed the sweep to practical component balances. The already-completed 1-transformer/8-VAE screen is retained in the evidence and full ranking; no 1T6V, 1T9V or 2T5V runs were launched. None of the new layouts beat the contemporaneous 1T2V control in this single-run screen; small gaps are not statistically established.

The 3-transformer/1-VAE pipeline uses the previously measured isolated transformer-pool memory to estimate admission. The ceiling is 95% of GPU memory. The earlier 3-transformer/2-VAE **staged** run (244.6 seconds) released transformers before decoding and is excluded from this pipeline ranking. Three transformers plus two overlapping VAEs were rejected by the conservative memory estimate, not tested to an allocation failure.

Allocated workers can wait for latents or share a saturated GPU. The [capacity evidence](data/capacity_matrix_summary.json) records workers used, overlapping calls, memory and contention checks. A single screening sample does not establish a small speed difference or a global optimum.

'''
    text=f'''## My recommendation

**Use {recommended['label']}** for the best measured speed/memory tradeoff in this workload: **{recommended['mean_total_seconds']:.1f} seconds** including checks, **{recommended['mean_video_ready_seconds']:.1f} seconds** until the assembled, fully decoded video is ready, and **{recommended['peak_vram_gib']:.1f} GiB** peak memory.

The selection rule is practical: among configurations with at least two clean runs and within 2% of the fastest observed mean, choose the smallest memory footprint. This is a tradeoff rule, not a statistical proof that their speeds are equal. The smallest mean was **{best['mean_total_seconds']:.1f} seconds** with **{best['label']}**.

| Setup, all using 20 independent chunks | Average including QA | Observed range | Peak VRAM | Clean runs |
|---|---:|---:|---:|---:|
{chr(10).join(row(r) for r in [ordinary,async_one,one_mps,two,four,tt])}

{capacity}

### Did extra model copies help?

**Extra VAE decoder copies helped under MPS.** Holding MPS and the chunk plan fixed, one transformer with one VAE averaged {one_mps['mean_total_seconds']:.1f} seconds; one transformer with two VAEs averaged {two['mean_total_seconds']:.1f} seconds, a **{(one_mps['mean_total_seconds']/two['mean_total_seconds']-1)*100:.1f}% throughput improvement**. This is the direct decoder-copy comparison.

Compared with ordinary scheduling, one transformer with two VAEs under MPS took {two['mean_total_seconds']:.1f} seconds, versus {ordinary['mean_total_seconds']:.1f} seconds for the ordinary one-transformer/one-VAE pipeline: **{(ordinary['mean_total_seconds']/two['mean_total_seconds']-1)*100:.1f}% more throughput** for the same video and chunk plan. Compared with the improved background-export one-copy option, the difference is **{(async_one['mean_total_seconds']/two['mean_total_seconds']-1)*100:.1f}%**.

Four VAEs took {four['mean_total_seconds']:.1f} seconds and {four['peak_vram_gib']:.1f} GiB, while two transformers plus two VAEs took {tt['mean_total_seconds']:.1f} seconds and {tt['peak_vram_gib']:.1f} GiB. The table below contains the remaining allocations. More copies do not add compute units; they change how stages compete for the same GPU.

This updates the earlier full-model-copy result. Here, text encoding happens on the GPU once and is released before independent stage workers load. The previous CPU-text-encoding penalty is absent. With one transformer, denoising stays sequential, while multiple VAEs decode completed chunks concurrently. Two-transformer tests also overlap denoising calls, but extra transformer copies have not produced a large full-video gain.

### The speed/quality tradeoff

All these runs produce the same **49.408-second podcast audio** and **1,186 video frames at 24 fps**. Shorter chunks create more joins: 20 chunks means 19 joins. Every 20-chunk join was flagged by the simple luma-change screen. The reference portrait remains recognizable in the inspected contact sheet, but this does **not** establish smooth motion or lip sync. Watch the output before accepting this chunking policy.

Ordinary loading and MPS variants with the same chunk plan produce identical output hashes in the completed checks. The alternative resident loading backend changes pixels; its technical similarity is reported separately below. Neither a small pixel difference nor matching frame counts establishes equal perceptual quality.

### How confident is this?

These are measurements from **one GPU, one prompt, one portrait and one audio recording**. Most initial screens have one sample; promising settings have repeats. Run order was adaptive, not randomized. Gaps below a few seconds should be treated cautiously. The two earlier timings affected by competing jobs are excluded, and the user-cancelled trial has no timing. The resumed ownership audit is bundled.

The measured lifecycle starts fresh preparation and worker processes for each video. It includes model loading, encoding and validation. Source-audio splitting and calibration are excluded. Sustained warm-service throughput, production concurrency and multi-GPU speed are not measured here.

### Run the recommended configuration

Use the pinned checkout, environment and weights from the reproduction package, on an idle GPU. For a fresh replication, let this command perform calibration:

```bash
python -m agent.rtx6000_benchmark.stage_pipeline --name my_replication --schedule pipeline --chunks {recommended['key'][0]} --transformers {recommended['key'][1]} --vaes {recommended['key'][2]}{' --mps' if recommended['key'][5] else ''}{' --async-encoding' if recommended['key'][4] else ''}{' --residency highvram' if recommended['key'][3]=='highvram' else ''} --run
```

This is benchmark code, not a production deployment. No worker was deployed, no PR was merged, and no Slack message was sent.
'''
    (ROOT/'CHUNK_COPY_RESULTS.md').write_text(text)
    (ROOT/'runs/followup_recommendation.json').write_text(json.dumps({'recommended':recommended,'fastest_mean':best,'rule':'Smallest memory footprint among n>=2 configurations within 2% of fastest mean.'},indent=2))
    print(json.dumps({'recommended':recommended['label'],'seconds':recommended['mean_total_seconds'],'fastest':best['label'],'fastest_seconds':best['mean_total_seconds']},indent=2))
if __name__=='__main__':main()

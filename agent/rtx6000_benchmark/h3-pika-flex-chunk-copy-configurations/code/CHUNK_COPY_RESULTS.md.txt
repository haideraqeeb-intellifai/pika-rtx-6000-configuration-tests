## My recommendation

**Use 20 chunks · 1 transformer / 2 VAE · MPS** for the best measured speed/memory tradeoff in this workload: **223.6 seconds** including checks, **219.6 seconds** until the assembled, fully decoded video is ready, and **41.3 GiB** peak memory.

The selection rule is practical: among configurations with at least two clean runs and within 2% of the fastest observed mean, choose the smallest memory footprint. This is a tradeoff rule, not a statistical proof that their speeds are equal. The smallest mean was **223.6 seconds** with **20 chunks · 1 transformer / 2 VAE · MPS**.

| Setup, all using 20 independent chunks | Average including QA | Observed range | Peak VRAM | Clean runs |
|---|---:|---:|---:|---:|
| 20 chunks · 1 transformer / 1 VAE | 237.3 s | 235.1–239.5 s | 35.5 GiB | 2 |
| 20 chunks · 1 transformer / 1 VAE · background export | 232.9 s | 232.8–233.0 s | 35.5 GiB | 2 |
| 20 chunks · 1 transformer / 1 VAE · MPS | 239.8 s | 237.4–242.3 s | 35.0 GiB | 2 |
| 20 chunks · 1 transformer / 2 VAE · MPS | 223.6 s | 221.9–225.5 s | 41.3 GiB | 3 |
| 20 chunks · 1 transformer / 4 VAE · MPS | 224.8 s | 224.8–224.8 s | 53.7 GiB | 2 |
| 20 chunks · 2 transformer / 2 VAE · MPS | 225.6 s | 225.3–225.9 s | 68.1 GiB | 2 |

### Filling more of the GPU: latest capacity sweep

These runs keep transformer and VAE workers active together under MPS. The current-day 1-transformer/2-VAE control is shown separately here from its earlier repetitions.

| Simultaneous stage pools | Mean total including QA | Peak VRAM | Clean runs in this sweep |
|---|---:|---:|---:|
| 1 transformers + 2 VAEs | 223.3 s | 41.3 GiB | 1 |
| 2 transformers + 4 VAEs | 227.0 s | 80.6 GiB | 1 |
| 2 transformers + 3 VAEs | 227.3 s | 74.4 GiB | 1 |
| 3 transformers + 1 VAEs | 249.1 s | 88.8 GiB | 1 |

The user narrowed the sweep to practical component balances. The already-completed 1-transformer/8-VAE screen is retained in the evidence and full ranking; no 1T6V, 1T9V or 2T5V runs were launched. None of the new layouts beat the contemporaneous 1T2V control in this single-run screen; small gaps are not statistically established.

The 3-transformer/1-VAE pipeline uses the previously measured isolated transformer-pool memory to estimate admission. The ceiling is 95% of GPU memory. The earlier 3-transformer/2-VAE **staged** run (244.6 seconds) released transformers before decoding and is excluded from this pipeline ranking. Three transformers plus two overlapping VAEs were rejected by the conservative memory estimate, not tested to an allocation failure.

Allocated workers can wait for latents or share a saturated GPU. The [capacity evidence](data/capacity_matrix_summary.json) records workers used, overlapping calls, memory and contention checks. A single screening sample does not establish a small speed difference or a global optimum.



### Did extra model copies help?

**Extra VAE decoder copies helped under MPS.** Holding MPS and the chunk plan fixed, one transformer with one VAE averaged 239.8 seconds; one transformer with two VAEs averaged 223.6 seconds, a **7.3% throughput improvement**. This is the direct decoder-copy comparison.

Compared with ordinary scheduling, one transformer with two VAEs under MPS took 223.6 seconds, versus 237.3 seconds for the ordinary one-transformer/one-VAE pipeline: **6.1% more throughput** for the same video and chunk plan. Compared with the improved background-export one-copy option, the difference is **4.2%**.

Four VAEs took 224.8 seconds and 53.7 GiB, while two transformers plus two VAEs took 225.6 seconds and 68.1 GiB. The table below contains the remaining allocations. More copies do not add compute units; they change how stages compete for the same GPU.

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
python -m agent.rtx6000_benchmark.stage_pipeline --name my_replication --schedule pipeline --chunks 20 --transformers 1 --vaes 2 --mps --run
```

This is benchmark code, not a production deployment. No worker was deployed, no PR was merged, and no Slack message was sent.

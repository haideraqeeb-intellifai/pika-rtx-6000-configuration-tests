# More VAE copies: measured results on one RTX PRO 6000

## Current conclusion

The fastest observed complete-video configuration is one transformer feeding one
VAE in separate ordinary processes: **248.39 seconds, 39.60 GiB**. The previous
paired combined-worker baseline averaged 252.21 seconds over two runs, so this
is a small 1.5% throughput difference with a substantial memory reduction. The
new setting currently has one run; its confirmation repeat is pending GPU recovery.

| Full-video setting | Seconds | Peak GiB |
|---|---:|---:|
| Denoise all, then 1 VAE | 269.24 | 31.73 |
| Denoise all, then 2 VAEs | 260.87 | 31.73 |
| Denoise all, then 4 VAEs | 257.67 | 31.73 |
| Overlap 1 transformer + 1 VAE | 248.39 | 39.60 |
| Overlap 1 transformer + 2 VAEs | 249.00 | 45.01 |
| Overlap 1 transformer + 4 VAEs | 249.55 | 55.82 |
| Overlap 2 transformers + 1 VAE | 253.98 | 70.70 |
| Overlap 2 transformers + 2 VAEs | 249.81 | 77.01 |
| MPS: denoise all, then 4 VAEs | 251.85 | 31.78 |
| MPS: overlap 1 transformer + 1 VAE | 254.02 | 38.28 |
| MPS: overlap 2 transformers + 1 VAE | 252.28 | 69.38 |
| MPS: process longest chunk through both stages first, then overlap 1 + 1 | 253.95 | 38.25 |
| MPS: 65% transformer / 35% VAE thread caps | Failed | Excluded |

More VAEs helped the staged decoder phase: one VAE took 98.20 seconds versus
83.19 seconds for four VAEs under MPS. This did not produce the fastest full
video. With an overlapping one-transformer pipeline, one/two/four VAEs were
within about one second, so there is no demonstrated benefit from extra VAEs
in that arrangement. More copies share the same physical GPU resources.

## One-VAE batching diagnostic

These reuse identical denoised latents and **exclude preparation and denoising**.
They are not complete-generation timings. Each has one run.

| Requested batch | Decoder phase | Summed VAE calls | Peak GiB | Actual model batches |
|---|---:|---:|---:|---|
| 1 | 93.04 s | 76.37 s | 9.19 | ten batches of 1 |
| 2 | 90.93 s | 74.45 s | 11.68 | 2, 1, 2, 2, 2, 1 |
| 4 | 89.85 s | 73.62 s | 16.65 | 3, 4, 3 |

Batch four improved decoder-phase throughput by about 3.6% (3.4% less elapsed
time). Only equal latent shapes were grouped; output order was restored.
No end-to-end generation speedup is claimed for batching.

## Correctness and concurrency evidence

All 12 successful new full-generation layouts delivered 1,186 frames at 24 fps,
passed full video/audio decoding, and produced **identical decoded pixels**.
The single-item VAE replay also matched. Batch-two/four outputs had small pixel
differences: downsampled 180×320 SSIM 0.998492 / 0.998115. All audio checks found
0 ms lag and duration difference at 8 kHz, correlation 0.999957. All nine joins
remain above the ordinary-frame luma p95; this is not a seamlessness claim.

MPS worker attachment was verified through a private controller's client list.
An unmeasured four-VAE Nsight replay completed, but the capture exported runtime
API events without usable GPU kernel timestamps. Diagnostics report unsupported
hardware tracing in MPS mode and incomplete CUPTI events. Actual kernel overlap
therefore remains unverified. The raw trace is local; only relevant diagnostics
are published because trace files include process/environment metadata.

## Failure and unfinished repeat

The 65/35 MPS trial failed during VAE decoding after four denoised and two decoded
chunks. CUDA reported an unspecified launch failure, then NVML reported GPU lost.
All benchmark clients and the private MPS controller exited; cleanup is recorded.
`nvidia-smi` subsequently could not obtain GPU 0's device handle. The authorized
reset attempt with `sudo -n /usr/bin/nvidia-smi --gpu-reset -i 0` required a password.
Privileged kernel logs were unavailable, so the root cause is not established.
Do not rank or recommend the failed configuration. Administrator recovery is
needed before confirming the provisional winner with another hardware run.

The ordinary worker configuration is the provisional practical recommendation.
No production service was deployed or merged, and no Slack message was sent.
All measurements here are one-GPU data, with no eight-GPU achievement claim.

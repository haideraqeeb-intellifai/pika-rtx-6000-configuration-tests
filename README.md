# H3 Pika Flex on one RTX PRO 6000

This repository preserves the code, measurements, videos, and GPU profiles from the RTX PRO 6000 configuration study. **Start with the fixed 50-second workload**: five independent 10-second chunks, two denoising steps, 24 fps, 240 exported frames per chunk, identical seeds and original-image anchors. The 49.408-second source audio was padded with 0.592 seconds of silence for every configuration. Older experiments changed chunking or chaining and are historical evidence, not matched controls for this workload.

## Current answer

The fastest measured fixed-workload layout kept **one transformer and two VAEs resident**, with MPS and overlapping denoising and decoding. In the repeated configuration matrix it finished in **208.93 seconds** through the corrected final MP4 (three runs; 56.76 GiB peak VRAM). In a separate interleaved comparison against the simple repo worker, the same parallel layout averaged **210.27 seconds** versus **223.14 seconds** (5.8% less time, 11.71 GiB more peak VRAM). These are distinct experiments; do not divide a time from one by a time from the other. The interleaved comparison did not meet the study's roughly 20% improvement threshold for the extra worker complexity.

| Fixed five-chunk configuration | Mean generation, 3 runs | Corrected MP4 delivery estimate | Peak VRAM |
| --- | ---: | ---: | ---: |
| 1 transformer + 2 VAEs, pipelined | 208.87 s | 208.93 s | 56.76 GiB |
| 2 transformers + 2 VAEs, concurrent | 211.87 s | 211.93 s | 92.84 GiB |
| 1 transformer + 2 VAEs, batch of two (2+2+1) | 212.62 s | 212.68 s | 74.09 GiB |
| 2 transformers + 1 VAE, concurrent | 218.98 s | 219.04 s | 85.74 GiB |

The delivery estimate adds a separately timed lossless timestamp correction to the measured generation time; it is **not** a new integrated timing run. Three transformers were not launched for this fixed workload: a conservative simultaneous-peak memory estimate was 106.10 GiB against 95.59 GiB usable device capacity. This is an admission estimate, not a measured OOM. Source: [fixed-workload summary](agent/rtx6000_benchmark/five_ten/summary.json) and [matched worker comparison](agent/rtx6000_benchmark/five_ten/matched_ab_summary.json).

Nsight Systems shows why extra transformer copies did not halve latency. The two-transformer workers overlapped in host calls for about 134 seconds, but their GPU kernels actually overlapped for only about 39–40 seconds. Compared with the one-transformer layout, transformer/VAE kernel overlap dropped from about 38.7 to 10–11.1 seconds. The Nsight Compute capture found a dominant GEMM near 95% SM throughput and another kernel near 90% DRAM throughput; a SageAttention kernel launched only 170 blocks for 188 SMs and used about 101 KiB shared memory per block. These are **kernel-specific** observations; they do not mean every SM was fully occupied for the entire video. Source: [Nsight Systems analysis](agent/rtx6000_benchmark/five_ten/profiles_summary.json), [Nsight Compute results](agent/rtx6000_benchmark/five_ten/ncu_summary.json), and [graphical profile](agent/rtx6000_benchmark/h3-pika-flex-five-ten-rtx6000/index.html).

## Browse the experiments

| Study | Entry point | Scope |
| --- | --- | --- |
| Fixed five-by-ten workload, batch-two, multi-transformer, Nsight | [interactive report](agent/rtx6000_benchmark/h3-pika-flex-five-ten-rtx6000/index.html) | Primary matched configuration matrix and 50-second video |
| Simple worker versus 1T+2V/MPS | [matched comparison](agent/rtx6000_benchmark/five_ten/matched_ab_share/index.html) | Interleaved A/B, three measured videos per arm |
| Comprehensive earlier matrix | [interactive report](agent/rtx6000_benchmark/h3-pika-flex-comprehensive-rtx6000/index.html) | Historical shorter and varied chunk experiments |
| VAE pools and MPS | [interactive report](agent/rtx6000_benchmark/vae_configuration_report/index.html) | Historical stage-pool configurations |
| Same-image independent chunks | [interactive report](agent/rtx6000_benchmark/original_image_parallel_report/index.html) | Historical parallel chunk policy |
| Fast-mode comparison | [interactive report](agent/rtx6000_benchmark/h3-pika-flex-fast-rtx6000/index.html) | Earlier fast-path study |
| Nsight transformer-copy analysis | [interactive report](agent/rtx6000_benchmark/h3-pika-flex-nsight-rtx6000/index.html) | Earlier multi-transformer profiling |
| Detailed configuration results | [report](agent/rtx6000_benchmark/h3-pika-flex-chunk-copy-configurations/index.html) | Historical configurations and eight-GPU estimates |

Open HTML files locally with a browser; their media paths are relative to this checkout. The earlier `batch-two` report was updated to the same fixed-workload report, so the duplicate directory is not copied here; its original short-chunk report remains under `h3-pika-flex-five-ten-rtx6000/historical-short-chunks/`.

## Layout and reproduction

- `agent/rtx6000_benchmark/` contains the experiment harness, effective graph templates, configuration runners, raw JSON/CSV/log evidence, reports, and output media.
- `agent/rtx6000_benchmark/five_ten/` contains the fixed-workload code and data. [PLAN.md](agent/rtx6000_benchmark/five_ten/PLAN.md) specifies the protocol; `summary.json`, `matched_ab_summary.json`, `validation.json`, and `quality.json` are the headline evidence. `runs/` keeps the timing, GPU telemetry, per-stage events, and graph records.
- `agent/rtx6000_benchmark/five_ten/profiles/` and `agent/rtx6000_benchmark/nsight_profile/` contain native `.nsys-rep` and `.ncu-rep` captures as well as exported analysis. Install NVIDIA Nsight Systems and Nsight Compute to open the native captures. SQLite exports can be regenerated from `.nsys-rep` with `nsys export`.
- `ml_modules/minimax_h3_turbo/` is the pinned inference implementation used by the experiments. The benchmark source was taken from `strontiumcorp/script2video` commit `7a1bbf944c8e2b99d175ec094768c22a945e75cb`, with the local modified `runtime.py` included and documented in [PUBLISHING_LOG.md](PUBLISHING_LOG.md).
- [MODEL_REFERENCES.md](MODEL_REFERENCES.md) lists required model assets. No model weights are committed.

On an idle RTX PRO 6000 with the pinned dependencies and model assets, place this repository's `agent/rtx6000_benchmark` and `ml_modules/minimax_h3_turbo` packages on `PYTHONPATH`. Follow [REPRODUCE.md](REPRODUCE.md) for exact commands and the recorded environment. The benchmark requires the matching ComfyUI/torch/SageAttention environment used by the source project; hardware and dependency versions are recorded in `five_ten/provenance.json` and per-run result files. Expect the scripts to enforce an idle-GPU gate and write substantial new artifacts locally.

The published report media and raw measurements are sufficient to inspect the completed tests without GPU access. Re-running generation requires the listed large model weights, CUDA dependencies, FFmpeg, and a matching GPU. The original 5090 reference and eight-GPU scenarios are comparison or projection material; **no eight-GPU benchmark was run**.

## Archive policy

All measured configuration variants and profiling summaries are retained. Repeated generated videos are available in the browser reports. Native Nsight captures are tracked with Git LFS. The roughly 105 GiB source working directory also held temporary per-run `.pt` conditioning/latent tensors, uncompressed `.rgb` frame buffers, duplicate MP4/WAV copies, installed dependencies, and exported `.sqlite` databases. These are regenerable from the published source, input media, output reports, and native traces; they are intentionally omitted. Expired signed S3 links and upload logs are omitted. See [tools/assemble_publication.py](tools/assemble_publication.py) for the exact selection rules. No license is implied for the source code, model assets, or media beyond their respective owners' terms.

# Same portrait at the start and end of every chunk: follow-up status

## What this changes

Every chunk uses the original portrait as its first-frame image and its hard
last-frame image. The chunks no longer need the previous chunk's decoded last
frame as input. The existing graph groups their preparation, denoising, and
decoding stages, but its sampler still executes chunks serially. This is a
different generation policy, not a speed result for simultaneous denoising.
An image supplied as a condition does not guarantee the generated endpoint
pixels match that image exactly.

## Evidence already available

An earlier **four-chunk** run used exactly this first/last image policy and
produced a fully decoded 1,186-frame, 24 fps MP4. The audio matched the source
with 0 ms lag and duration difference at 8 kHz, waveform correlation 0.999957.

| Four-chunk policy | Complete video time | Peak VRAM | Join luma changes | Joins above ordinary-frame p95 |
|---|---:|---:|---|---:|
| Previous-last-frame chaining, hard end image | 305.92 s | 63.96 GiB | 7.704, 7.700, 7.975 | 3/3 |
| Original portrait at both ends of every chunk | 306.59 s | 59.58 GiB | 3.647, 4.264, 3.946 | 3/3 |

The existing same-image run has roughly half the boundary luma jump, but all
three joins remain unusual relative to motion inside the chunks. The saved
`runs/original_4_grouped/contact_sheet.jpg` shows a stable studio and a visible
reset of face, paw, and mouth pose at some joins. These visual observations
need human review in the finished video. One run per policy is insufficient to
rank their timing; the measured wall times differ by less than one second.

## Measured follow-up on the RTX PRO 6000

All eight earlier full-copy runs delivered 1,186 frames at 24 fps and 1080 × 1920,
decoded fully, and matched the source audio with 0 ms lag and duration
difference at the checker's 8 kHz resolution. Each timed run followed a
separate one-chunk calibration. The leading two settings were each repeated;
the other table rows have one complete run each.

| Same-image configuration | Timed video | Peak VRAM | Mean sampled GPU use | Joins above ordinary-frame p95 |
|---|---:|---:|---:|---:|
| 10 aligned chunks, one resident full model, hard anchor | 265.5 s | 51.72 GiB | 83.6% | 9/9 |
| 10 aligned chunks, fresh full model process per chunk, hard anchor | 413.9 s | 47.37 GiB | 53.8% | 9/9 |
| 10 aligned chunks, one resident full model, anchor every second chunk | 256.4 s | 51.72 GiB | 83.8% | 4/9 |
| 7 aligned chunks, one resident full model, hard anchor | 273.4 s | 55.26 GiB | 86.5% | 6/6 |
| 14 aligned chunks, one resident full model, hard anchor | 271.1 s | 49.60 GiB | 82.7% | 13/13 |
| 10 aligned chunks, two resident full models, CPU text, hard anchor | 754.1 s | 73.49 GiB | 27.0% | 9/9 |

The two-lane run overlapped inference calls for 723.5 seconds, but the first
pair of chunks each spent about 107 seconds in CPU conditioning. It was 2.84×
slower than the one-lane GPU-text run. The memory calibration for one GPU-text
resident process was 46.31 GiB on the first chunk, and the complete run peaked
at 51.72 GiB. Keeping both text encoders and full model sets resident therefore
lacks safe headroom on a 95.59 GiB card. This does not require CPU text encoding:
the follow-up GPU-preparation design below releases the encoder before the
two workers run. Moving text to CPU in the earlier experiment lowered the single-process calibration
peak to 31.64 GiB and let two full processes run, but three would exceed the
supervisor's 90% capacity guard. This does not test the huddle's proposed
shared text encoder and shared VAE with two transformer copies; that requires
a stage-split implementation and its own measurement.

The hard-anchor repeats measured 265.5 and 265.7 seconds; the periodic-anchor
repeats measured 256.4 and 257.5 seconds. The latter averaged 8.7 seconds
(3.3%) faster and had 4 of 9 unusual luma joins on both runs, versus 9 of 9
for the hard-anchor setting. It also changes the generation policy: odd
chunks have the original portrait at the start but no hard last-frame image.
Two runs give initial repeatability evidence, but are not a broad performance
distribution or a human seam rating.

The full videos and boundary contact sheets are in
`original_image_parallel_report/`. A human should judge pose resets and mouth
continuity in the playable videos; a luma statistic alone cannot establish
seamlessness.

## GPU preparation first, then one or two workers

The user's proposed sequence works: GPU-encode the identical prompt/portrait
once, prepare chunk-specific audio and latents, save CPU tensor artifacts,
exit the preparation process to release the encoder, then launch resident
workers. Each worker retains one transformer and one video VAE. The workers
have no text encoder. This corrects the earlier claim that two copies require
CPU text encoding.

| Configuration, ten independent hard-anchor chunks | Run 1 | Run 2 | Mean | Peak VRAM |
|---|---:|---:|---:|---:|
| GPU preparation with exact-input encoding cache, one worker | 257.70 s | 257.25 s | 257.47 s | 36.37 GiB |
| Same preparation, two workers | 251.94 s | 252.48 s | 252.21 s | 72.11 GiB |
| GPU preparation without encoding cache, two workers | 271.78 s | — | — | 72.11 GiB |

The paired configuration delivered 2.1% more videos per hour than the matched
one-worker configuration. This is a modest measured gain from two repetitions,
not evidence of a precise long-run speedup. Both versions take about 18–19
seconds for cached GPU preparation, included in the totals. Initial source-audio
preparation/splitting and memory calibration are excluded; assembly, full decode,
audio checks, seam checks and contact-sheet generation are included.

Mean per-chunk generation time rose from approximately 12.7 seconds with one
worker to 26.0 seconds with two. Video decode similarly rose from about 7.7 to
14.9 seconds. Sampler calls overlapped for 118.0 and 116.2 seconds in the paired
runs; synchronized call intervals do not establish concurrent CUDA kernels.
The timing behavior is consistent with contention for one GPU's resources.

All five new complete videos have 1,186 frames at 24 fps and passed full decoding.
Audio lag and duration difference were 0 ms at 8 kHz, correlation 0.999957.
All nine joins remained above the ordinary-frame luma p95. The cache comparison
verified exact equality of all 70 preparation tensors across all ten chunks
against uncached GPU preparation, including distinct audio conditioning.

Recommendation for this workload: use GPU preparation with exact-input caching
and encoder release. One worker uses substantially less VRAM; two workers are
the fastest measured hard-anchor option, at nearly twice the memory for a 2.1%
throughput gain. Three workers exceed the measured 90% memory guard. A shared
video VAE with two transformer workers remains unmeasured, as do other hardware
and multi-GPU arrangements. These results do not establish the global optimum.

## Configuration space

`run_original_image_matrix.py` specifies 26 practical one-GPU runs: repeated
four- and ten-chunk baselines; chunk counts and grid alignment; final versus
overlapped encoding; hard, periodic, and first-only end anchors; one versus two
transformers, one/two/four VAEs, shared versus repeated loaders; the high-VRAM
flag, CPU text encoding, the H3-compatible tiled VAE entry point, and disk-backed
frames. It refuses to continue if the benchmark's idle GPU check fails. Four
resident transformers and two complete processes keeping GPU text encoders resident remain outside the
measured safe memory envelope; two CPU-text processes were tested above.
GPU preparation followed by encoder release also fits two workers, as measured
in the subsequent GPU-preencoding runs.

The existing executor cannot test simultaneous denoising merely by adding model
copies. Two experimental process supervisors now address this: 
`parallel_chunk_processes.py` starts a fresh complete model process for each
chunk while limiting concurrent processes; `parallel_chunk_resident.py` keeps
one model set loaded in each lane and assigns several chunks to it. Process
isolation avoids the Comfy executor's process-local inference lock and mutable
global model manager. Both calibrate one chunk's peak VRAM, refuse a requested
copy count above 90% of the measured 95.59 GiB capacity, assemble the chunks
in order, use the original full-length audio, and require a fully decoded MP4.
The 11-case `run_parallel_loading_matrix.py` compares one/two/three copies,
resident versus repeated loading, GPU versus CPU text encoding, dynamic versus
high-VRAM placement, an every-second anchor, and three chunk counts.

The resident and eager full-copy workers have now produced valid complete
videos. Their separate inference intervals establish process-call overlap,
but a GPU profiler would be needed to prove simultaneous kernel execution.
The copy-count memory check is conservative and cannot guarantee fit for every
future chunk shape or model revision.

## Periodic anchoring

The earlier 241.30 s every-second-anchor result used the **chained** start
frame and cannot be transferred to the same-image policy. The same-image
every-second variant above measured 256.4 s with 4 of 9 unusual joins. It
changes FL2V/I2V generation modes and should be judged for pose reset, scene
stability, and audio synchronization in the full video. Other anchor schedules
remain unmeasured under this independent-chunk policy.

## Hardware status

The user reset the dedicated GPU. It returned to the observed idle baseline,
and all measurements above passed the three-sample idle gate. A short
post-calibration utilization tail caused one aborted CPU-text attempt; the
resident supervisor now retries for up to 30 seconds while still requiring
three clean idle samples before the timed run. The aborted attempt is retained
under `runs/original_image_parallel_resident_2_cpu_text/` and excluded from
the table.

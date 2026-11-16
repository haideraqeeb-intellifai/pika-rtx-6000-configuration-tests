# H3 Pika Flex on an RTX PRO 6000

**A plain-language guide to what we tested, what won, and what remains uncertain.**

## The answer in 30 seconds

We generated the same roughly 49-second podcast video many times on **one RTX PRO
6000**, changing how the work was divided and how the model was kept in memory.
For the original generation policy, the best **repeated** result was **251 seconds
per finished video** (4 minutes 11 seconds), or **14.3 videos per hour** if jobs run
one after another. The earlier four-chunk setup took **306 seconds** in its one
measured run. This is about an **18% shorter wait**; the four-chunk comparison has
only one run, so treat that percentage as indicative.

The winning setup uses **ten carefully sized chunks**, **one copy each** of the
text encoder, transformer, and video decoder, and an encoder that starts writing the
video while later chunks are still being generated. The encoder overlap is a
benchmark prototype; the current production worker still uses final encoding.

**There is a quality tradeoff.** Ten chunks create nine joins in the video; four
chunks create three. Every inspected join showed a larger visual jump than almost
all ordinary neighboring frames. The faster setup therefore needs a human review of
the finished video before it becomes the default.

**Eight GPUs were not tested.** Eight independent copies of the measured worker
would ideally produce **about 115 videos/hour**. Splitting eight GPUs by stage did
not beat that estimate at the measured stage speeds. Any claim above 8× remains a
possibility to test, not a result.

## What “chunks” and “chaining” mean

The podcast is made in pieces. To keep the video continuous, chunk 2 starts from
the **last frame actually shown** in chunk 1; chunk 3 starts from the last frame of
chunk 2, and so on. We kept this dependency in every main comparison. It means a
single video's next chunk must wait for its preceding chunk to be decoded.

For each chunk, the GPU roughly does: **understand the text and audio → generate
latent video with the transformer → decode visible frames with the VAE → pass the
last frame to the next chunk**. The final encoder packages those frames with audio
into an MP4. “VRAM” in the tables is the GPU memory needed at the busiest point.

## Which one-GPU setup should we choose?

| Choice | Finished-video time | Joins | What it means |
|---|---:|---:|---|
| Original four chunks | 306 s, one run | 3 | Fewer opportunities for a visible join; slowest of these choices |
| Seven aligned chunks + encoding overlap | 260 s, one run | 6 | Middle ground worth reviewing |
| **Ten aligned chunks + encoding overlap** | **251 s median of two runs** | **9** | Fastest repeated result with the original hard end anchor on every chunk |
| Ten aligned chunks, hard anchor every second chunk | 241 s median of two runs | 9 | Faster, but changes how the scene is generated; needs separate visual acceptance |

“Aligned” means sizing chunks to H3's internal frame grid so it generates fewer
frames that will never appear in the final video. Ten aligned chunks generate
**1,189** frames to deliver **1,186**; ten evenly sized chunks generate **1,240**
for the same delivered video. This is why chunk sizing matters, even with the same
number of chunks.

The “hard anchor” is the reference portrait used to hold the scene in place at the
end of a chunk. Using it every second chunk made the video faster, but it also
changed some chunks from the original generation mode. The studio remained
recognizable in our contact sheet, yet we did **not** establish equal visual
quality or lip sync. Using the anchor only on the first chunk was faster still,
but the background and body drifted visibly by the end.

## Did more model copies help?

No in the current worker. Keeping **two transformers** raised peak GPU memory from
about **52 to 73 GiB** without speeding up the complete video. Keeping **two or
four VAEs** also failed to improve latency. The execution path serializes jobs,
and one video's chunks depend on each other. Two incoming jobs submitted together
finished one after the other, at essentially the same overall rate as one job.
Four resident transformers are beyond the measured 96 GB memory budget.

Loading a fresh full model for every chunk was slower and nearly filled the card.
Moving text encoding to the CPU freed memory but made the video roughly 3.6 times
slower. The useful single-GPU changes were **chunk sizing** and **overlapping
encoding**, not extra model copies.

## What can we say about eight GPUs?

These numbers are **projections from one GPU**, assuming eight equally fast cards,
enough queued videos, and no new CPU, storage, or interconnect bottleneck.

| Proposed eight-GPU layout | Ideal videos/hour | Main concern |
|---|---:|---|
| Eight independent workers, one complete video pipeline per card | **114.65** | Shared host resources may lower the real rate |
| One text GPU, four transformer GPUs, three VAE GPUs | **114.14** | Transformer stage is the bottleneck; data movement and scheduling add cost |
| One text GPU, five transformer GPUs, two VAE GPUs | 93.98 | VAE stage becomes the bottleneck |

The stage-split estimate is slightly **below** eight independent workers before
real-world overhead. We assumed 1 GiB of transfers per video at 5 GiB/s, adding
about 0.24 seconds to an unloaded video's latency; the report below shows faster
and slower interconnect cases. A dedicated-stage speedup of a few percent *could*
change the ranking, but no eight-card test has established it.

**Practical next step:** review the four-, seven-, and ten-chunk output joins with
the podcast team. If ten chunks are acceptable, integrate and validate the
overlapped encoder in the worker. On an eight-GPU host, measure eight independent
workers first, then compare a 1/4/3 stage split under a full queue, including
transfer speed, memory, CPU, encoder, and output quality.

## How to read the detailed evidence

The section below preserves the exact timings, peak VRAM, GPU use, audio/frame
checks, tested model layouts, and sensitivity assumptions. The full configuration
matrix and boundary-frame contact sheets follow it. **Measured** always means a
run on the one available GPU; **estimated** means a model for a future eight-GPU
machine.

## Detailed evidence

**Scope.** One [RTX PRO 6000 Blackwell Workstation Edition](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/) (96 GB), two H3 Pika Flex
denoising steps, 24 fps, 1,186 delivered frames (49.417 seconds), 720 × 1280 requested
generation, 1080 × 1920 H.264/AAC final video. The main workload chains every chunk
from the previous chunk's last **delivered** frame. The original reference portrait
remains the hard end anchor. There is one fixed prompt, source audio, and seed sequence
across all comparable runs. The output audio is 49.408 seconds. Each case starts after
three one-second samples establish an idle, uncontested GPU.

**Input provenance.** The four-chunk schedule matches the reference report's delivered
frame counts (293/295/290/308). The report's source WAV was unavailable, so the audio
here is PCM decoded from its chained MP4. This is a comparable replay rather than an
exact source-byte reproduction. `inputs/manifest.json` records hashes and URLs.
The code revision and full effective graph are in every `runs/*/result.json` and
`workflow.json`. The local media and model weights remain outside Git.

**Configuration selection.** The RTX 5090 reference report supplied candidates:
shared versus repeated loaders, stage grouping, paired VAE decode, asynchronous
saving, and final NVENC. Its fastest 5090 variant was not assumed to win here.
The [podcast huddle](https://videosoftware.slack.com/docs/TA6KDFNQ4/F0CF4CFK153)
suggested sharing one text encoder and one VAE with two transformers; this study
tests that ratio and additional VAE counts. Strict previous-last-frame chaining
rules out the report's all-conditioning-first graph and deferred paired full VAE
decode as equivalent candidates: each next chunk needs the prior decoded last
frame. The report's asynchronous-save idea is tested here as one overlapped NVENC
stream. The VAE path already has internal temporal tiling.

## One-GPU measurements

All times below are wall-clock seconds from request preparation through full MP4/audio
decode validation. Each row is a complete valid run unless stated otherwise.
`summarize_runs.py` produces the full machine-readable configuration matrix.

| Chunks | Schedule | Repeats | Wall seconds | Peak VRAM GiB | Inferred frames | Join count |
|---:|---|---:|---:|---:|---:|---:|
| 4 | Report | 1 | 305.92 | 63.96 | 1,210 | 3 |
| 5 | Uniform | 1 | 288.57 | 59.61 | 1,215 | 4 |
| 6 | Uniform | 1 | 286.07 | 57.01 | 1,254 | 5 |
| 7 | Uniform | 2 | 270.52, 269.71 | 55.26 | 1,225 | 6 |
| 8 | Uniform | 2 | 273.20, 272.20 | 53.84 | 1,264 | 7 |
| 9 | Uniform | 1 | 271.34 | 53.07 | 1,269 | 8 |
| 10 | Uniform | 1 | 262.87 | 51.72 | 1,240 | 9 |
| 12 | Uniform | 1 | 271.48 | 50.66 | 1,284 | 11 |
| 10 | Grid-aligned | 2 | 255.14, 253.78 | 51.72 | 1,189 | 9 |
| 14 | Grid-aligned | 1 | 255.58 | 49.60 | 1,192 | 13 |
| 4 | Grid-aligned + overlapped encode | 1 | 299.84 | 64.65 | 1,193 | 3 |
| 7 | Grid-aligned + overlapped encode | 1 | 260.48 | 55.95 | 1,191 | 6 |
| 10 | Grid-aligned + overlapped encode | 2 | 250.54, 251.87 | 52.41 | 1,189 | 9 |

H3 rounds each chunk's inference length up to `5 + 17k` frames. The grid-aligned
schedule places segment lengths near those boundaries; it does **not** discard output
frames or relax chaining. Ten aligned chunks save 51 inferred frames and about eight
seconds versus ten uniform chunks. Fourteen aligned chunks are no faster within the
observed single-run precision and have four extra joins.

For ten aligned chunks, the two repeat runs have median wall time **254.46 seconds**,
or **14.15 videos/hour** with one serial worker. The output hashes match. Median stage
times are 29.72 s conditioning, 126.80 s transformer denoising, and 76.51 s video VAE
decoding. Final NVENC encoding takes about 7.3 s. Stage times do not sum exactly to
wall time because loading, audio preparation, cropping, buffering, muxing, and validation
also run. Mean GPU use is about 90% during inference. This is a GPU-dominated workload.
Adding the measured overlapped encoder gives **251.20 seconds** median, or **14.33
videos/hour**, with the same output hash.
NVML power sampling estimates about 37.1 Wh of GPU energy per ten-chunk video,
versus 46.0 Wh for the four-chunk report schedule. These are GPU-only estimates,
not full machine or cloud billing costs.

## Output quality and validity

Every completed case is probed and fully decoded: 1,186 video frames at 24 fps,
1080 × 1920 H.264 and stereo AAC with the expected duration. Decoded AAC compared
with the source audio has a 0 ms best lag, 0 ms duration difference at 8 kHz, and
0.999957 waveform correlation for the inspected runs. This checks audio timing and
content integrity; it is not a lip-sync score.

At each inspected chunk join, the mean absolute luma change is about 7.5–8.0 on an
8-bit scale. All joins exceed the 95th percentile of non-join frame differences.
This is a visible-continuity risk with the current hard-anchor generation behavior:
four chunks create 3 such joins, ten create 9, and fourteen create 13. The contact
sheets in the run directories show the exact before/after frames. Shorter chunks
increase join frequency even when speed improves. No objective semantic or lip-sync
benchmark was available for this input, so do not interpret the integrity checks as
proof of equivalent perceptual quality.
Seven aligned chunks with overlap take 260.48 s and produce six joins; all six
also exceed the 95th-percentile non-join frame difference. This is a measured middle
ground: roughly 9 s slower than the fastest ten-chunk result with three fewer joins.
An exploratory ten-chunk run used the hard end anchor only on chunk 0, while still
starting every later chunk from the prior last delivered frame. It finished in
233.32 s (input modes: FL2V then nine I2V chunks), with exact audio timing and
somewhat smaller join differences. The contact sheet shows a changed studio
background and body appearance by the end of the video. It is a **different
generation policy**, not evidence that the fixed-anchor setup became 18 s faster
at equal quality. Periodic-anchor variants are evaluated separately below.

| Ten aligned chunks; hard end anchor on... | Wall s | Peak VRAM GiB | Join findings | Scene finding |
|---|---:|---:|---|---|
| Every chunk (report-comparable) | 250.54, 251.87 | 52.41 | 9/9 above non-join p95 | Studio stays fixed |
| Every second chunk | 241.38, 241.22 | 52.42 | 9/9 above non-join p95 | Studio recognizable in contact sheet |
| Every third chunk | 239.16 | 52.41 | 9/9 above non-join p95 | Studio recognizable in contact sheet |
| First chunk only | 233.32 | 51.85 | 9/9 above non-join p95 | Background and body drift by end |

All rows preserve the previous-last-frame **start** dependency and the same delivered
frame/audio contract. The periodic policies switch some chunks from FL2V to I2V,
so output content is not byte-equivalent to the all-hard workflow. The visible
scene judgment is from the saved boundary and final-frame sheets, not a formal
perceptual score.
The every-second policy repeated within 0.16 s and produced the same output hash.
Its median 241.30 s is 14.92 videos/hour on one GPU. If this alternative quality
policy is accepted, eight independent workers estimate to 119.35/hour, while the
same 1/4/3 stage split estimates to 118.14/hour before overhead. This remains an
eight-GPU **estimate**, not a measured throughput result.
The every-third policy has one valid run. Its 2.1-second apparent gain over the
every-second median is too small to select it without a completed repeat and
perceptual review. Its extra unanchored chunk also increases the chance of scene drift.
A follow-up run was interrupted, and the next attempt was refused by the idle-GPU
gate when the driver reported 100% utilization. Neither attempt is counted as a
valid timing or quality observation; both failure records remain in the matrix.
The study uses one fixed podcast prompt, portrait, and audio track. The ranking may
change with different content lengths, resolutions, or model revisions; the
repeat runs quantify timing stability for this workload only.

## Model placement and instances

The candidate ratios are written as **text encoders / transformers / video VAEs**.
The audio VAE stays shared. Extra instance graphs reuse each instance across chunks;
they do not create parallel execution in the current Comfy executor. The latter has
an inference lock, and one video's chunks also require the previous last frame.

| Ratio / placement | Wall s | Peak VRAM GiB | Result |
|---|---:|---:|---|
| 1 / 1 / 1, dynamic default | 255.14, 253.78 | 51.72 | Reference for model variants |
| 1 / 2 / 1 | 256.50 | 72.95 | Identical output; no serial speed gain |
| 1 / 1 / 2 | 256.57 | 56.61 | Identical output; no serial speed gain |
| 1 / 1 / 4 | 259.59 | 66.38 | Identical output; slower |
| 1 / 1 / 1, high-VRAM flag | 254.26 | 51.72 | Within repeat noise; offloading telemetry remained nonzero |
| 1 / 1 / 1, CPU text placement | 909.16 | 37.06 | Conditioning 692 s; different output hash |
| 1 / 1 / 1, H3-compatible generic tiled VAE node | 252.43 | 51.72 | Same output hash; this entry point calls the default internal decoder |
| 1 / 1 / 1, disk-backed raw frames | 256.03 | 51.72 | 3.28 GB written; same output hash |
| 1 / 1 / 1, overlapped single-stream NVENC | 250.54, 251.87 | 52.41 | Same output hash; no fallback |
| 4 chunks, separate text/transformer/VAE loader nodes per chunk | 334.26 | 95.35 | Shared four-chunk baseline: 305.92 s, 63.96 GiB |

Four resident transformers would exceed the 96 GB device limit if each adds roughly
the 21 GB observed for the second transformer. Dynamic offloading could make a
four-transformer graph *run*, but it cannot make four copies resident at the measured
working-set size. More transformer or VAE copies can only help throughput if a scheduler
executes multiple jobs concurrently and the GPU has spare compute or transfer capacity.
That scheduler and its multi-job behavior are not established by these serial runs.
Repeating all loader nodes for the four-chunk workload increases latency by 28.34 s
(9.3%) and peak VRAM by 31.39 GiB versus sharing them. The output hash is identical.
Dynamic model offloading made the graph run, but it reached 95.35 GiB VRAM and more
than 107 GiB of modeled offloaded weights at peak. That left only about 0.25 GiB of
the device's reported 95.59 GiB usable memory. It is not a practical default on
the one available card.
CPU text placement frees about 14.7 GiB but makes this workload about 3.6× slower;
it is a poor way to create room for concurrent jobs on this card.
The first generic `VAEDecodeTiled` attempt failed on H3's nested latent tensor before
producing a video; its traceback is retained. A benchmark adapter that unwraps the
latent succeeded. H3's `decode_tiled` calls its default `decode` and ignores the
generic tile settings, so the identical output and similar timing do not establish
a separate VAE decode optimization.
Disk-backed frame storage takes 1.42 s of store time versus 0.80 s in memory and
does not improve end-to-end speed. It may be useful if host RAM is constrained.
Overlapped single-stream NVENC begins after the first chunk and feeds later chunks
as they complete; it needs only 3.94–4.01 s of final waiting and saves about 3.3 s
versus the default median. Its `encoding_seconds` field records the whole stream
lifetime, not additional wall time. The prototype is in `async_encode_nodes.py` and
retains a replayable fallback; it has not been integrated into the production worker.

Two incoming ten-chunk jobs were also submitted at the same instant to one shared
in-process executor. Their inference completions were 249.54 s and 493.02 s after
model initialization began. Both complete videos passed full decode and matched the
single-job output hash. Total time including both validations was 501.54 s for two
videos, or 14.36 videos/hour; VRAM peaked at 51.72 GiB. The second request waited
behind the first because the current executor holds a global inference lock. This
tests concurrent *admission*, not concurrent GPU execution. Two fully independent
processes at the measured ~52 GiB peak each would exceed 96 GB before allowing
headroom. A shared-model multi-job scheduler would require new code and an explicit
same-card contention test.

Stage grouping from the reference report was tested on four chunks with every chunk
starting from the original portrait: 306.59 s versus 305.92 s for the chained
four-chunk graph. These are different semantic workloads, so this is **not** a valid
speed comparison for chained output. Fully grouping all conditioning before
denoising would violate the previous-last-frame dependency. The chained graph keeps
the required order: condition → denoise → decode → keep last delivered frame → next
chunk. Parallelism can occur across separate videos, not within those dependent
chunk transitions.

## Eight-GPU projections — estimates, not measurements

`estimate_eight.py` uses the measured medians from the two overlapped-encoder runs.
It assumes each of eight
future GPUs has the same speed and memory, there are enough incoming jobs, and CPU
preparation, NVENC, host memory, and storage do not bottleneck. Per-stage GPU counts
sum to eight. The model's steady-state service interval is the slowest of
conditioning/text, denoising, and VAE time divided by the GPU count for that stage.
It respects the previous-last-frame dependency *within* each video; different videos
can occupy different stages simultaneously. The independent-worker projection uses
eight times the measured single-worker rate.

| Allocation (text / denoise / VAE) | Ideal interval s/video | Ideal videos/hour | Bottleneck |
|---|---:|---:|---|
| Eight independent complete workers | 31.40 aggregate | 114.65 | Whole worker |
| 1 / 4 / 3 | 31.54 | 114.14 | Denoising |
| 1 / 5 / 2 | 38.31 | 93.98 | VAE |
| 2 / 4 / 2 | 38.31 | 93.98 | VAE |
| 2 / 3 / 3 | 42.05 | 85.61 | Denoising |
| 1 / 3 / 4 | 42.05 | 85.61 | Denoising |
| 1 / 6 / 1 | 76.61 | 46.99 | VAE |

Using the existing final-encode worker instead, the corresponding ideal figures are
113.18/hour for eight independent workers and 113.56/hour for 1/4/3. That apparent
0.3% stage-split edge is also too small to be credible without real eight-GPU data.

Even with zero stage overhead, 1/4/3 is about 0.4% *below* eight independent workers.
This narrow difference is below the precision of these measurements and ignores
scheduler, transfer, and stage-placement costs. It is **not** an observed
greater-than-8× result. The projection script can
vary transfer payload, bandwidth, latency, and dedicated-stage denoising speed.
The default assumes 1 GiB of total host-mediated stage transfers per video, 5 GiB/s
effective shared bandwidth, and 2 ms per stage handoff: about 0.24 s added to one
video's unloaded latency. A pessimistic 4 GiB over 1 GiB/s adds about 4.04 s; a
0.1 GiB payload over 25 GiB/s adds about 0.044 s. In these ranges the **stage GPU**
remains the throughput bottleneck, but the actual eight-GPU topology and payloads
are unknown. The final decoded raw frame stream is about 3 GiB per video and should
stay local to the VAE/encoder GPU rather than cross devices. Each standalone stage's
model weights are well below 96 GB, while eight independent workers inherit the
measured 52.41 GiB peak per card for the overlapped setup.

The current single-GPU results do **not** project more-than-8× at unchanged stage
speeds. A real dedicated-stage speedup of several percent could change that, as
could a redesigned text-embedding cache with sufficient host capacity; neither has been validated on
an eight-GPU system. Strong scaling should be assumed to be at or below 8× until
hardware testing says otherwise.

For the 1/4/3 split, the sensitivity to unmeasured stage speed and scheduling is:

| Dedicated denoising speedup | Added stage scheduling overhead | Estimated videos/hour |
|---:|---:|---:|
| 0% | 0% | 114.14 |
| 0% | 1% | 113.01 |
| 2% | 1% | 115.32 |
| 5% | 5% | 114.33 |

Eight independent workers are 114.65/hour under the same ideal hardware assumption.
This table shows why greater-than-8× is plausible under favorable speedups but not
established: a small change in unmeasured overhead reverses the result. It also
assumes enough jobs to fill the pipeline and no new CPU or storage bottleneck.

To validate the projection, measure the actual eight-card PCIe/NVLink/P2P topology
and effective bandwidth/latency, GPU-to-host copies of conditioning and latents,
isolated stage times and memory on dedicated cards, sustained eight-worker throughput,
the same 1/4/3 pipeline under a full job queue, CPU/NUMA/storage and NVENC saturation,
per-video latency at steady state, temperature/power, and the same frame/audio/quality
checks on assembled outputs. These measurements must preserve the chained-frame
dependency and use repeated runs.

## Recommendation

For a speed-oriented single-GPU deployment, start with one shared text encoder,
one transformer, and one video VAE, ten grid-aligned chained chunks, dynamic model
management, and a single NVENC stream fed as chunks complete. The overlapped stream
is a benchmark prototype; the existing final NVENC encode is the production-ready
choice until it is integrated and tested in the worker. For quality-sensitive podcast work,
retain the four-chunk schedule until the larger number of join changes is reviewed
in the finished video. Seven aligned chunks are an intermediate option with a
measured six joins and only about 9 s extra latency versus ten. The repeated
every-second anchor policy cuts median latency to 241.30 s, but changes the generated
scene and needs human acceptance before use. For a future
eight-GPU machine, start benchmarking eight
independent workers; test 1/4/3 as a controlled pipeline experiment. Do not promise
greater-than-8× throughput without that eight-GPU measurement.

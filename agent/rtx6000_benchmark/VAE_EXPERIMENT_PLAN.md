# Separate transformer and VAE workers

## Question and fixed workload

Does more decoder capacity improve time to a complete video on the single RTX
PRO 6000? Preserve the current independent-chunk workload: the original portrait
conditions both ends, ten aligned chunks, fixed chunk seeds 9901–9910, Pika Flex,
two denoising steps, 1,186 delivered frames at 24 fps, source master audio,
1080 × 1920 exported H264/AAC video. No chained-dependency speedup is claimed.

## Experiments

| Schedule | Transformer copies | Video VAE copies | Purpose |
|---|---:|---:|---|
| Denoise all, release transformer, decode all | 1 | 1 | Stage separation control |
| Same | 1 | 2 | Concurrent independent decoders |
| Same | 1 | 4 | Wider concurrent decoding |
| Feed decoded work as denoising completes | 1 | 1 | Shared decoder pipeline |
| Same | 1 | 2 | Extra decoder capacity |
| Same | 1 | 4 | Test excess decoder capacity |
| Same | 2 | 1 | Two denoisers feeding one shared decoder |
| Same | 2 | 2 | Balanced separate pools |

Each configuration first prepares conditioning on the GPU with exact-input text
caching, saves CPU tensor artifacts, releases the preparation process, and passes
the idle gate. Process isolation gives each stage its own Comfy model manager;
transformer workers load no VAE, and VAE workers load no transformer or text
encoder. A checksum-verified CPU file hands off each chunk's denoised latents.
Pipeline workers run under one supervisor and are intentionally concurrent.

Memory calibration ran two longest chunks consecutively per stage, covering
first-use and warm allocation. Peaks: denoise 31.757 GiB, decode 8.503 GiB.
Admission uses the sum of replicated stage peaks for pipelines, or the maximum
stage-pool peak for sequential stages, with a 90% guard on 95.59 GiB. This is a
conservative admission estimate, not a guarantee for other shapes or revisions.
Actual aggregate NVML memory and utilization are sampled at 0.2 seconds.

## Additional decoder diagnostic

Replay the exact same saved latents through one VAE with batch sizes 1, 2, and 4.
Only equal latent shapes are grouped; frame counts and ordering are restored
before final assembly. The H3 VAE already tiles internally; batching differs
from separate VAE copies. Record actual model batch calls, since a requested
batch size alone does not establish batched execution.

These are decoder-only timings; preparation and denoising are excluded. Compare
them with each other, not directly with full-video times. A promising batch
setting requires a complete generation run before claiming overall speedup.

## Measurement and validation

Full-run timing includes fresh GPU conditioning, serialization, encoder release,
worker startup, stage work, CPU latent handoff, worker shutdown, assembly, full
media decoding, audio timing and seam checks, and contact-sheet generation.
Initial source-audio splitting and memory calibration are excluded consistently
with the earlier GPU-preencoding runs. Record time to assembled/validated video
separately from the additional quality-check overhead.

Per-stage wall timers synchronize their process's CUDA context. Overlapping
sampler/decode calls do not prove simultaneous kernel execution; that requires
a GPU profiler. Summed stage times across processes overlap and are not an
additive decomposition of elapsed video time.

Keep every attempted configuration and error. Repeat the most promising complete
configuration and a matched control. Validate all 1,186 frames, 24 fps, full A/V
decode, source audio length and offset, and boundary luma screening. Numerical
or visual checks do not substitute for the user's review of seamlessness.

## MPS follow-up

NVIDIA's CUDA/MPS documentation states that work from separate CUDA contexts
cannot execute concurrently without MPS; ordinary process pools time-share the
GPU. This materially limits how their results can be interpreted:
https://docs.nvidia.com/deploy/mps/latest/architecture.html

After the ordinary pool matrix, test selected configurations with a private
MPS controller, unique temporary pipe directory, and benchmark-owned logs.
Verify each worker PID through the controller's `ps` command, retain attachment
evidence, and shut down that controller after all its clients exit. Do not
change global GPU compute mode or affect any other controller. MPS permits
kernel overlap but does not prove it occurred in a particular H3 run; retain
that distinction unless a CUDA profiler is used. Optional active-thread limits
are resource caps, not dedicated SM reservations.

# RTX PRO 6000 H3 Pika Flex configuration benchmark

This directory contains the local, reproducible benchmark harness and raw results for a
two-step, 24 fps, 49.408-second raccoon podcast video. The main runs start each chunk
from the last **delivered** frame of the prior chunk and keep the hard final-image anchor.
Every run writes its request, executable Comfy graph, runtime telemetry, media probe,
complete decode result, and output hash under `runs/<name>/`.

## Reproduce

Run on an otherwise idle RTX PRO 6000 with the pinned model assets installed in
`script2video/cache/minimax_h3`. Each run checks three consecutive one-second idle GPU
samples before loading models. The benchmark environment used here is
`/home/raqeeb/h3/flex-2k-refinement/.h3-pika-env` and the source revision is recorded
in each result. From the `script2video` directory:

```bash
/home/raqeeb/h3/flex-2k-refinement/.h3-pika-env/bin/python \
  -m agent.rtx6000_benchmark.run_case --name example --chunks 10 --chunk-plan aligned
```

The original source WAV was unavailable. `inputs/report_audio.wav` is PCM decoded from
the reference report's chained MP4. `inputs/manifest.json` records the source and
checksums. This is a comparable replay, not a byte-for-byte reproduction of its inputs.
The supplied portrait and fixed prompt are in `run_case.py`.

`--chunk-plan uniform` divides the 1,186 delivered frames almost evenly. `aligned`
chooses segment lengths near H3's 17-frame inference grid to avoid generating frames
that are discarded. `report` uses the report's 293/295/290/308 delivered frame counts
for four chunks and uniform counts otherwise.
`--anchor-mode first-only` keeps the original hard end anchor on the first chunk
and omits it later while preserving the previous-last-frame start dependency.
`--anchor-mode every-2` and `every-3` keep that anchor periodically. These modes
change the generation policy and must be judged for scene continuity separately.

The other switches are `--transformers`, `--vaes`, `--texts`, `--residency`,
`--clip-device`, `--vae-decode`, `--frame-storage`, and `--encoding-overlap`.
Model instance counts alter
Comfy loader and model graph nodes; the current Comfy executor still serializes graph
execution, so these variants measure loading and residency costs but do not establish
concurrent throughput. `--reference original` relaxes the previous-last-frame dependency
and is a separate semantic workload.

## Independent chunks and experimental model-copy workers

`ORIGINAL_IMAGE_POLICY.md` records the existing four-chunk same-image result and
the planned follow-up. Every chunk in these experiments starts from the same
portrait. With `--anchor-mode hard`, it also uses that portrait as its end-image
condition. The model can still produce different pixels at chunk boundaries.

`run_original_image_matrix.py` replays the practical serial configurations under
that policy. `parallel_chunk_processes.py` starts a new full model process for
each chunk, admitting up to `--copies` at once. `parallel_chunk_resident.py`
keeps one model set loaded per process and assigns multiple chunks to each lane.
`run_parallel_loading_matrix.py` compares both strategies, one to three copies,
text placement, residency, chunk counts, and periodic end anchoring.

All supervisors print their plan by default. Add `--run` only on an idle GPU;
the supervisor performs the existing three-sample idle gate, measures one
chunk's peak VRAM, refuses unsafe copy counts, and fully decodes the assembled
MP4. A child may skip its own idle check only when launched by a supervisor,
because its sibling processes intentionally use the GPU concurrently. For
example:

```bash
/home/raqeeb/h3/flex-2k-refinement/.h3-pika-env/bin/python \
  -m agent.rtx6000_benchmark.parallel_chunk_resident \
  --name original_image_resident_two --copies 2 --chunks 10 --run
```

These workers are benchmark prototypes. The measured runs and media validation
are summarized in `ORIGINAL_IMAGE_POLICY.md`; further configurations still
need hardware runs. The old one-process Comfy graph remains serial even if it
contains multiple loader nodes.

The completed same-image worker measurements, playable outputs, and boundary
contact sheets can be rebuilt into `original_image_parallel_report/index.html`
with:

```bash
/home/raqeeb/h3/flex-2k-refinement/.h3-pika-env/bin/python \
  -m agent.rtx6000_benchmark.build_original_image_report
```

`preencoded_parallel.py` implements GPU preparation followed by encoder release
and independent resident workers. `gpu_preencode.py` stores conditioning and
latents as CPU tensor artifacts (~50 MiB per longest chunk), then exits its
process. The supervisor verifies GPU idle before launching one or two workers,
whose graphs contain no text encoder. Each worker has a transformer and video
VAE; this is not concurrent decoding through a single shared VAE.

`--cache-text` reuses the GPU encoding only for exactly identical prompt and
resized first/last image inputs. It still prepares each chunk's own audio and
latents. The full tensor comparison in
`runs/gpu_preencoded_cached_one/full_cache_equivalence_check.json` verified all
70 tensors against uncached GPU preparation. Sampler calls have synchronized
start/end wall timestamps; these measure call overlap, not simultaneous CUDA
kernels. Timed totals include preparation, CPU handoff, worker startup, final
assembly and media checks. Initial source-audio preparation/splitting and calibration are excluded from timed totals. Calibration covers two longest chunks to measure
both first-use and warm VRAM peaks.

```bash
/home/raqeeb/h3/flex-2k-refinement/.h3-pika-env/bin/python \
  -m agent.rtx6000_benchmark.preencoded_parallel \
  --name example_gpu_prepare_two --copies 2 --cache-text --run
```

For timing repeats, `--reuse-calibration SUCCESSFUL_RUN_NAME` reuses the measured
memory guard only for an identical chunk plan and preparation policy. GPU
preparation and all delivered video work are still rerun and timed.

Eight-GPU projections come from `estimate_eight.py` and are **estimates only**:

```bash
python agent/rtx6000_benchmark/estimate_eight.py \
  chained_10_aligned_overlap_encode chained_10_aligned_overlap_encode_repeat2
```

`run_two_jobs.py` submits two simultaneous incoming
requests to the current locked in-process executor. Generate the local HTML report
and CSV matrix with:

```bash
/home/raqeeb/h3/flex-2k-refinement/.h3-pika-env/bin/python \
  agent/rtx6000_benchmark/build_report_html.py
```

The source videos and full local output media are intentionally kept out of Git.

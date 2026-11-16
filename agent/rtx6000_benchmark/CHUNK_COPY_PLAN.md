# Chunk length × concurrent component copies

## Hypotheses (written before new matrix results)

1. Shorter chunks reduce transformer sequence length and allow smaller worker activations; extra transformer copies may then use the GPU more efficiently. Conversely, more chunks repeat preparation, keyframe anchoring, serialization and export overhead.
2. Longer chunks reduce repeated setup and anchors but increase attention work and activation memory. Two transformers may fit less safely or compete more heavily.
3. Extra VAE copies can only improve completion if decoding limits the pipeline or overlaps otherwise idle work; they have no separate physical compute capacity.
4. Keeping stage models resident could avoid transfers, but the dynamic loader may already do this. Verify actual memory/timing before claiming a change.
5. Background encoding can let a decoder start its next chunk earlier. Bound the queue, retain CPU frame buffers until export completes, and include encoding plus validation in end-to-end time.

## Fixed workload and timing boundary

Pika Flex, two steps, source podcast audio 49.408 s, 1186 delivered frames, 24 fps, original raccoon portrait as first and last conditioning of every independent chunk. 736×1280 model canvas, 720×1280 crop, 1080×1920 export. Seeds 9901 + chunk index. No changed resolution/steps/weights. Changing chunk count changes audio boundaries, random draws and anchor frequency, so output equality is only expected across layouts with the same chunk plan.

Counts: 5, 10, 20 chunks (approximately 10, 5, 2.5 seconds). Align inference lengths to 17k+5. Preserve exact delivered total and original source audio. Layouts for each: (transformers, VAEs) = (1,1), (2,1), (1,2), (2,2), (1,4). All ordinary CUDA processes, no MPS caps. Text/image conditioning prepared on GPU with exact-input caching; encoder process exits before stage workers launch. Stage workers reuse their loaded model across chunks.

A fresh baseline repeat is already running. Reuse it as the 10-chunk 1T/1V cell. Recalibrate each different chunk plan using two maximum-length chunks per stage; reject layouts whose summed stage memory envelopes exceed 90% of 95.59 GiB. Calibration excluded from timings. Reuse calibration only with matching chunk plan and residency. Cases execute serially, with idle-device checks. Stop matrix on execution failure rather than continuing on a possibly unhealthy GPU.

Measured interval includes fresh GPU preparation, encoder release gate, worker startup/model loading, generation, CPU latent handoff, per-chunk encoding/verification, shutdown, final assembly and audio/seam/contact-sheet checks. Report time to assembled valid video separately from total including additional QA. Initial source-audio splitting and memory calibration are excluded. No claim of warm-service request latency.

After matrix screening, repeat the most promising layouts and compare resident-model / deferred-encoding variants. Compare decoded pixel hashes within a chunk plan; inspect changed-chunk samples separately. Retain all successful and failed results. Publish raw data and commands with a readable report. One-GPU measurements only; no inference of measured eight-GPU performance.

## Residency and concurrency implementation checks

The production loader forces dynamic VRAM on. Merely setting `highvram` does not
select a conventional resident patcher. The follow-up residency branch therefore
selects Comfy's ordinary ModelPatcher in the benchmark process before checkpoint
creation, with HIGH_VRAM enabled. Runtime patcher and VRAM state are recorded.
This changes the loading backend and requires fresh calibration and pixel checks.

NVIDIA documents time-slicing between ordinary separate CUDA contexts:
https://docs.nvidia.com/deploy/mps/latest/architecture.html
MPS can permit kernels from different processes to execute together when each
process leaves capacity unused:
https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html
Worker-call overlap alone does not prove that happened. Earlier uncapped MPS
completed, while the 65/35 cap trial failed. Any follow-up MPS check must use
ordinary uncapped settings, verify attached clients, retain failure records and
run only after the ordinary matrix/confirmation trials. Never retry 65/35 here.

## Reproduce from a fresh results directory

Use the Python environment and assets recorded in data/followup_provenance.json.
Extract the included agent/rtx6000_benchmark files into that same script2video
revision. Run from the repository root, with no other GPU workload:

```bash
python -m agent.rtx6000_benchmark.stage_pipeline --name followup_baseline_10 --schedule pipeline --chunks 10 --run
python -m agent.rtx6000_benchmark.run_chunk_copy_matrix
python -m agent.rtx6000_benchmark.run_chunk_copy_confirm
python -m agent.rtx6000_benchmark.run_mps_control_followup
python -m agent.rtx6000_benchmark.run_combined_followup
python -m agent.rtx6000_benchmark.run_mps_vae_sizing
python -m agent.rtx6000_benchmark.followup_quality
python -m agent.rtx6000_benchmark.build_chunk_copy_report
```

The first command performs a fresh memory calibration; subsequent 10-chunk
cases reuse that baseline's calibration. Our original run reused the earlier
vae_stages_calibration record, included in the evidence. The matrix runner
resumes completed cases rather than overwriting them. Use a separate results
location or new run names for an independent replication; never discard or
replace old measurements silently. The confirmation runner includes a private,
standard 100/100 MPS check after all ordinary trials. No model weights are included
in the reproduction download.

The combined follow-up runs two full videos with both static resident loading and background export, after the individual variants have completed their confirmation runs. No additive speedup is assumed.

After the two-transformer MPS run improved, add a matched one-transformer/one-VAE MPS control with two repeats. This separates the effect of copy count from scheduling mode.

The first matched 1T/1V MPS control was slower than 2T/2V. Add 1T/2V and 1T/4V MPS screens to identify whether decoder copies deliver the gain without a second transformer; repeat the best lower-memory alternative if within 2% of the 2T/2V control or faster.

## Resumed checks after the user pause

The cancelled combined trial is preserved as `followup_best_combined_repeat2/cancellation.json`; new names `repeat3` and `repeat4` replace it and the contention-affected first attempt. `followup_one_mps_repeat3` replaces the contaminated one-copy MPS repeat. GPU idle baseline was rechecked before resuming.

Test 1T/2V and 1T/4V under standard MPS, repeating any option within 2% of the earlier 2T/2V video-ready time. This keeps promising memory/speed alternatives visible even if one wins the initial screen by less than a second. Add the missing 2T/1V MPS allocation with the same repeat rule. These are adaptive follow-ups selected from previous results, not a randomized factorial experiment.

Run a full 20-chunk staged 1T/1V video with isolated denoising and decoding phases for the eight-GPU service model. It is a diagnostic profile, excluded from the pipelined configuration ranking. Its phases include startup/loading; dedicated warm-stage speed is unknown.

```bash
python -m agent.rtx6000_benchmark.run_resume_followup
python -m agent.rtx6000_benchmark.run_resume_finish
python -m agent.rtx6000_benchmark.audit_resumed_runs
python -m agent.rtx6000_benchmark.followup_quality
python -m agent.rtx6000_benchmark.build_chunk_copy_report
python -m agent.rtx6000_benchmark.project_chunk_copy_eight
python -m agent.rtx6000_benchmark.write_chunk_copy_findings
python -m agent.rtx6000_benchmark.build_chunk_copy_report
python -m agent.rtx6000_benchmark.verify_chunk_copy_report
```

The resumed runners start a host-NVML process-ownership monitor. Run only one coordinator at a time. The audit resolves transient MPS server sightings using each run's attached-client list and keeps original ownership samples. Exclude contention-affected timings; never stop another user's work to force a benchmark.

# Reproduce the fixed five-chunk study

Use a clean copy of this repository on an idle RTX PRO 6000 Blackwell Workstation Edition. The measured host used Python 3.12.13, Linux 6.8, NVIDIA driver 595.84, PyTorch 2.11.0+cu130, SageAttention 2.2.0, SageAttention3 1.0.0, comfy-kitchen 0.2.35, and comfy-aimdo 0.5.5. Exact source hashes and GPU identifiers are in [`five_ten/provenance.json`](agent/rtx6000_benchmark/five_ten/provenance.json). A matching ComfyUI checkout with the required custom nodes and CUDA kernels is also required; it is not vendored here. The benchmark code is included, but this is not a turnkey environment installer.

Fetch the pinned model assets in [MODEL_REFERENCES.md](MODEL_REFERENCES.md) into `cache/minimax_h3/`. The source portrait and 49.408-second audio are in `agent/rtx6000_benchmark/inputs/`. All commands below run from the repository root. Use your prepared Python environment in place of `python`.

```bash
export PYTHONPATH="$PWD"
python -m agent.rtx6000_benchmark.five_ten.prepare
```

The preparation step pads the audio by 0.592 seconds, creates five 10-second slices, and writes five conditioning tensors. It is measured separately from the full-video runs. Confirm the GPU idle gate succeeds before each experiment. Each run below performs a complete warmup video followed by three measured 50-second videos; replace each `rerun_...` name with a fresh unused name when repeating.

```bash
python -m agent.rtx6000_benchmark.five_ten.experiment --mode sequential --name rerun_1t2v --transformers 1 --vaes 2 --repeats 3
H3_BATCH_COMPACT=1 python -m agent.rtx6000_benchmark.five_ten.experiment --mode batch --name rerun_batch_1t2v --transformers 1 --vaes 2 --repeats 3
python -m agent.rtx6000_benchmark.five_ten.experiment --mode concurrent --name rerun_2t1v --transformers 2 --vaes 1 --repeats 3
python -m agent.rtx6000_benchmark.five_ten.experiment --mode concurrent --name rerun_2t2v --transformers 2 --vaes 2 --repeats 3
```

For the matched simple-worker comparison, run in the recorded A/B/A/B order. The four names below mirror the original raw result layout; use new names in an archive checkout because those result directories already exist.

```bash
python -m agent.rtx6000_benchmark.five_ten.experiment --mode simple --name rerun_simple_a1 --transformers 1 --vaes 1 --repeats 2
python -m agent.rtx6000_benchmark.five_ten.experiment --mode sequential --name rerun_parallel_b1 --transformers 1 --vaes 2 --repeats 2
python -m agent.rtx6000_benchmark.five_ten.experiment --mode simple --name rerun_simple_a2 --transformers 1 --vaes 1 --repeats 1
python -m agent.rtx6000_benchmark.five_ten.experiment --mode sequential --name rerun_parallel_b2 --transformers 1 --vaes 2 --repeats 1
```

`five_ten/validate.py`, `validate_matched_ab.py`, `quality.py`, and `summarize.py` document the checks and reduction. Some summary scripts use the original fixed run names; point them at your new names or use a clean scratch copy. Validate 1,200 exported frames at 24 fps, 50 seconds of synchronized audio, and output hashes before interpreting speed. The output pixel and audio equivalence checks do not establish visual smoothness or lip sync.

For profiling, [`five_ten/profile.py`](agent/rtx6000_benchmark/five_ten/profile.py) records a full warmup plus one replay for the 1T2V, 2T1V, and 2T2V layouts with Nsight Systems. [`five_ten/profile_ncu.py`](agent/rtx6000_benchmark/five_ten/profile_ncu.py) captures selected kernels with Nsight Compute. Run these only after uncontended timing tests; profiler-instrumented durations are not performance means. The published `.nsys-rep` and `.ncu-rep` files can be opened directly, and `.sqlite` exports can be regenerated with `nsys export --type=sqlite`.

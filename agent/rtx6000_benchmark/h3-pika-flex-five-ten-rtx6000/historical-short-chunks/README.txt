H3 Pika Flex batch-of-two experiment

Run from the script2video repository recorded in provenance.json. Reuse the existing
.h3-pika-env interpreter and fast_dependencies/site-packages PYTHONPATH overlay.
No dependency installation or production change is required.

Source under evidence/source contains the experiment, B=2 adapter, matrix runner,
profiling commands, analysis and output-validation scripts. Absolute paths in graphs
identify the original artifacts and must be rebased on another host.

The original GPU-prepared conditioning comes from followup_fast_comparison_fast_1.
The two raw audio inputs, portrait, preparation metadata and artifact hashes are
included. Model weights and prepared tensor caches are not bundled; reproduce the
cache using the repository's agent.rtx6000_benchmark.gpu_preencode workflow with the
recorded original inputs, prompt, anchors and settings, or use the retained local
prepared cache. Check hashes before replaying. The default pair is indices 0 and 1.

For the refined batch, set H3_BATCH_MODULATION=1 (initial batch leaves it unset).

Run: python -m agent.rtx6000_benchmark.batch_two.experiment --mode sequential
     --name <fresh-name> --repeats 3
Repeat with --mode batch and --mode concurrent. A private MPS session is created
and shut down for each configuration. Empty-GPU gates and ownership monitoring are
mandatory. Diagnostic profiler runs are excluded from the timing means.

Warm-up and failed attempts are preserved. sequential_v1 aborted during warm-up
because its initial monitor did not recognize its own NVENC child process; none of
that attempt's timings are in the results. The first NSYS attempt stopped at the
idle gate before model loading (profiler allocation 4 MiB versus normal 2 MiB).
Successful diagnostics used the established 4 MiB profiler idle baseline.

Media includes all 24 measured videos. Native raw .nsys-rep / .sqlite archives and
saved latent tensors remain in the local batch_two directory. Numerical profile
extracts are included; captured host environment tables are not published.

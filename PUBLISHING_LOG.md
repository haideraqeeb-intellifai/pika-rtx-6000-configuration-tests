# Publication log

## 2026-11-16 15:06 UTC — initial public archive

- Destination: `haideraqeeb-intellifai/pika-rtx-6000-configuration-tests`.
- Source: local `strontiumcorp/script2video` checkout at commit `7a1bbf944c8e2b99d175ec094768c22a945e75cb`, plus a pre-existing local modification to `ml_modules/minimax_h3_turbo/runtime.py`. The copied source contains that working-tree version; the original checkout was not changed for this publication.
- Included: fixed five-by-ten-second benchmark, interleaved simple-versus-pipelined comparison, historical chunk and model-copy matrices, VAE pool and fast-mode tests, effective graphs, per-run measurement and validation JSON/logs, self-contained HTML reports with output videos and inputs, inference source, and native Nsight Systems/Compute captures.
- Inventory: approximately 13,500 files; 19 Nsight Systems native traces, three Nsight Compute native reports, the complete browser reports, and 50-second and historical sample videos. Media and native traces use Git LFS.
- Omitted: model weights and symlinks to local model caches (see `MODEL_REFERENCES.md`); generated `.pt` tensor and `.rgb` frame intermediates; duplicate raw-run MP4/WAV/PNG copies already represented in reports; regenerable Nsight SQLite exports; local installed dependencies/build caches; expiring S3 links and upload logs. One expired signed reference HTML member was removed from the comprehensive reproduction ZIP. The updated fixed-workload batch-two report duplicate was omitted; its historical short-chunk content is present in the primary five-ten report.
- The fixed workload is five 10-second chunks at 24 fps, and historical variable-chunk results are not matched controls.

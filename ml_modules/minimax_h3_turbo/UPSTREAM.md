Runtime-only copy of https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo at commit
`4274783a23afcfdbea3b4876cb79effd6c510785` (Apache-2.0).

`comfyui_h3_turbo/__init__.py` is an unchanged copy of that commit's runtime implementation,
SHA-256 `a22f222f11dfc337552371f7f288bf5928cb1fa80a6afb3f24075a8315d02f06`.
It preserves the benchmark's dual-schedule sampler, bypass LoRA adapters,
INT8 fc2 merge handling, and pruned AdaLN injection.

The wrapper preloads `h3_silu_temb_grid.safetensors` from the local model cache
into `_EGRID`. The binary is downloaded from the same immutable source commit
and hash-checked; it is not stored in git. The adapter checkpoint and grid
hashes are recorded in `assets.py`.
`infer.py` and `workflow.py` are the repository adapters; upstream UI, examples,
and packaging are excluded.

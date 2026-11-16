# Model references

Model weights, adapters, checkpoints, and learned grids are excluded from this repository. Put the following files below `cache/minimax_h3/` before rerunning. The authoritative version, sizes, and available SHA-256 values come from the included [`assets.py`](ml_modules/minimax_h3_turbo/assets.py). Obtain access and follow each upstream model's license before downloading.

| Local path | Source / revision | Bytes | SHA-256 if pinned |
| --- | --- | ---: | --- |
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) @ `0bd506d2e895983a9663037febda27aa3948cf48` | 20,970,379,616 | size checked |
| `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | Same model revision | 20,970,379,616 | size checked |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | Same model revision | 15,687,142,551 | size checked |
| `vae/minimax_h3_video_vae_fp16.safetensors` | Same model revision | 5,207,808,496 | size checked |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | Same model revision | 605,254,808 | size checked |
| `loras/pika_h3_flex_fl2va_comfy_bf16.safetensors` | [Pika-Labs/Pika-H3-Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex) @ `3ff166447d5edb1406107ab8370c42000ce173e4` | 1,956,172,176 | `68783e6447696ec365d13c0554941e070bee4cfea13c37da52498ec4f9a53f89` |
| `loras/pika_h3_flex_ref2va_comfy_bf16.safetensors` | Same Pika model revision | 1,956,172,176 | `e4d058fbd027377b925a0ee607b514ddd9037873fee3afa05d73b302e6f7f8e1` |
| `loras/minimax_h3_turbo_v4_step600_ema.safetensors` | [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) @ `43a74557ac3f6539db8e0f2a959d03feb7a81480` | 779,849,816 | `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3` |
| `loras/h3_silu_temb_grid.safetensors` | Same Turbo LoRA revision / source project cache | 5,510,600 | `30eb3c2cc7fb6b470d9717ff840d359313ac27cd64b705e32da1baa10f72d6a8` |
| `loras/h3_character_swap_pro4500_1000.safetensors` | [akatz-ai/MiniMax-H3-Character-Swap-LoRA](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA) @ `62407e0cc8089c363abd9ce4b0b27662abb237af` | 155,110,320 | `4b2a3f420ae804c0aa3422761ff84dbd1bf52eef6900ffab6d2e66df63cb4e79` |

The checked-out benchmark used symlinks to local caches and mirrors. Those symlinks are not portable and are not included. `ml_modules/minimax_h3_turbo/infer.py` validates required file sizes and the adapter hashes before loading. The two-stage benchmark uses the FL2VA path with Pika Flex; the broader module also lists REF2VA assets. If an upstream artifact moves or access changes, the source revision and checksums above are the retrieval reference.

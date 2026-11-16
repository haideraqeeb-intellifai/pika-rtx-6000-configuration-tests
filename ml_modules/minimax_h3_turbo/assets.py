"""Immutable assets from the Ref2VA v4-600 / four-step Sage benchmark."""

MODEL_REPO_ID = "Comfy-Org/MiniMax-H3"
MODEL_REVISION = "0bd506d2e895983a9663037febda27aa3948cf48"
LORA_REPO = "larryvrh/MiniMax-H3-Turbo-Lora"
LORA_REVISION = "43a74557ac3f6539db8e0f2a959d03feb7a81480"
TURBO_COMMIT = "4274783a23afcfdbea3b4876cb79effd6c510785"
DIFFUSION = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
LORA = "minimax_h3_turbo_v4_step600_ema.safetensors"
GRID = "h3_silu_temb_grid.safetensors"
# akatz-ai/MiniMax-H3-Character-Swap-LoRA@62407e0c, mirrored to Boto; size and SHA-256 are checked below.
CHARACTER_SWAP = "h3_character_swap_pro4500_1000.safetensors"
CHARACTER_SWAP_BOTO_KEY = f"minimax_h3/loras/{CHARACTER_SWAP}"
EXTRA_LORAS = {"character_swap": (CHARACTER_SWAP, 1.0)}
# Pika-Labs/Pika-H3-Flex@3ff16644 comfy/loras/ official ComfyUI LoRAs (model="pika_flex"), mirrored to Boto;
# size and SHA-256 are checked below.
PIKA_REPO = "Pika-Labs/Pika-H3-Flex"
PIKA_REVISION = "3ff166447d5edb1406107ab8370c42000ce173e4"
PIKA_FL2VA = "pika_h3_flex_fl2va_comfy_bf16.safetensors"
PIKA_REF2VA = "pika_h3_flex_ref2va_comfy_bf16.safetensors"
PIKA_BOTO_KEYS = {name: f"minimax_h3/loras/{name}" for name in (PIKA_FL2VA, PIKA_REF2VA)}
REQUIRED_MODELS = {
    f"diffusion_models/{FL2VA}": 20970379616,
    f"diffusion_models/{DIFFUSION}": 20970379616,
    f"text_encoders/{TEXT_ENCODER}": 15687142551,
    f"vae/{VIDEO_VAE}": 5207808496,
    f"vae/{AUDIO_VAE}": 605254808,
}
ADAPTER_SIZES = {f"loras/{LORA}": 779849816, f"loras/{GRID}": 5510600, f"loras/{CHARACTER_SWAP}": 155110320,
                 f"loras/{PIKA_FL2VA}": 1956172176, f"loras/{PIKA_REF2VA}": 1956172176}
ADAPTER_HASHES = {
    f"loras/{LORA}": "5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3",
    f"loras/{GRID}": "30eb3c2cc7fb6b470d9717ff840d359313ac27cd64b705e32da1baa10f72d6a8",
    f"loras/{CHARACTER_SWAP}": "4b2a3f420ae804c0aa3422761ff84dbd1bf52eef6900ffab6d2e66df63cb4e79",
    f"loras/{PIKA_FL2VA}": "68783e6447696ec365d13c0554941e070bee4cfea13c37da52498ec4f9a53f89",
    f"loras/{PIKA_REF2VA}": "e4d058fbd027377b925a0ee607b514ddd9037873fee3afa05d73b302e6f7f8e1",
}

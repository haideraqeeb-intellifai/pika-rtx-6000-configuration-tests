"""Fetch only the pinned H3 assets absent from the local benchmark cache."""

import hashlib
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2] / "cache" / "minimax_h3"
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-home"))

ASSETS = (
    ("Comfy-Org/MiniMax-H3", "0bd506d2e895983a9663037febda27aa3948cf48",
     "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
     "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", 20970379616, None),
    ("Pika-Labs/Pika-H3-Flex", "3ff166447d5edb1406107ab8370c42000ce173e4",
     "comfy/loras/pika_h3_flex_fl2va_comfy_bf16.safetensors",
     "loras/pika_h3_flex_fl2va_comfy_bf16.safetensors", 1956172176,
     "68783e6447696ec365d13c0554941e070bee4cfea13c37da52498ec4f9a53f89"),
    ("akatz-ai/MiniMax-H3-Character-Swap-LoRA", "62407e0cc8089c363abd9ce4b0b27662abb237af",
     "h3_character_swap_pro4500_1000.safetensors",
     "loras/h3_character_swap_pro4500_1000.safetensors", 155110320,
     "4b2a3f420ae804c0aa3422761ff84dbd1bf52eef6900ffab6d2e66df63cb4e79"),
)

for repo, revision, remote, local, size, sha256 in ASSETS:
    target = ROOT / local
    if target.is_file() and target.stat().st_size == size:
        print(f"present {local}", flush=True)
    else:
        print(f"fetching {local}", flush=True)
        source = Path(hf_hub_download(repo_id=repo, revision=revision, filename=remote))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    if target.stat().st_size != size:
        raise ValueError(f"Size mismatch: {local}")
    if sha256:
        with target.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != sha256:
            raise ValueError(f"SHA-256 mismatch: {local}")
    print(f"verified {local}", flush=True)

"""Local-only Turbo loading and in-process execution; downloads belong to the worker."""

import hashlib
import importlib.metadata
import os
import sys
from pathlib import Path

from . import runtime

from .assets import ADAPTER_HASHES, ADAPTER_SIZES, GRID, REQUIRED_MODELS, TURBO_COMMIT


def validate_assets(model_root):
    model_root = Path(model_root)
    for name, size in {**REQUIRED_MODELS, **ADAPTER_SIZES}.items():
        path = model_root / name
        if not path.is_file() or path.stat().st_size != size:
            raise FileNotFoundError(f"Missing or incomplete H3 Turbo asset: {path}")
    for name, expected in ADAPTER_HASHES.items():
        with (model_root / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"H3 Turbo asset SHA-256 mismatch: {model_root / name}")


def load_models(model_root, runtime_root):
    """Load the pinned Turbo nodes and local AdaLN grid; checkpoints load lazily."""
    validate_assets(model_root)
    if importlib.metadata.version("sageattention") != "2.2.0":
        raise RuntimeError("H3 Turbo requires the benchmark's SageAttention 2.2.0")
    if os.fspath(runtime.COMFY_ROOT) not in sys.path:
        sys.path.insert(0, os.fspath(runtime.COMFY_ROOT))
    from comfy.cli_args import args

    args.use_pytorch_cross_attention = True
    args.enable_dynamic_vram = True
    models = runtime.load_models(model_root, runtime_root)
    import comfy.memory_management
    import comfy.utils

    from . import chunk_nodes, comfyui_h3_turbo, pika_nodes

    if not comfy.memory_management.aimdo_enabled:
        raise RuntimeError("H3 Turbo requires working dynamic VRAM support")
    # Keep the upstream implementation unchanged while storing its learned grid
    # in the model cache instead of committing a binary alongside the source.
    comfyui_h3_turbo._EGRID = comfy.utils.load_torch_file(
        os.fspath(Path(model_root) / "loras" / GRID), safe_load=True,
    )["silu_t_emb_grid"]
    if tuple(comfyui_h3_turbo._EGRID.shape) != (1025, 2688):
        raise ValueError("Unexpected H3 Turbo time-embedding grid shape")
    models["main"].nodes.NODE_CLASS_MAPPINGS.update(comfyui_h3_turbo.NODE_CLASS_MAPPINGS)
    models["main"].nodes.NODE_CLASS_MAPPINGS.update(chunk_nodes.NODE_CLASS_MAPPINGS)
    models["main"].nodes.NODE_CLASS_MAPPINGS.update(pika_nodes.NODE_CLASS_MAPPINGS)
    models["turbo_commit"] = TURBO_COMMIT
    return models


def infer(models, workflow, *, prompt_id, timeout_seconds, inference_mode="default"):
    """Run the Turbo graph with the Turbo timeout, locking, and memory telemetry."""
    return runtime.infer(models, workflow, prompt_id, timeout_seconds, inference_mode=inference_mode)

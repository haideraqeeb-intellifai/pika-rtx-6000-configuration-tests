"""ComfyUI nodes for model=\"pika_flex\": Pika LoRA application and the audio-exact re-noise sampler (see pika.py)."""

import math

import torch

import comfy.samplers
from comfy.k_diffusion.sampling import default_noise_sampler

from . import comfyui_h3_turbo as turbo
from .pika import SHIFT_A, SHIFT_V


@torch.no_grad()
def sample_pika(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
    extra_args = {} if extra_args is None else extra_args
    shapes = turbo._latent_shapes(model)
    if not shapes or len(shapes) < 2:
        raise RuntimeError("Pika sampler expects the MiniMax-H3 video+audio latent")
    v = math.prod(shapes[0][1:])
    mask = extra_args.get("denoise_mask")
    noise = default_noise_sampler(x, seed=extra_args.get("seed"))
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sv, svn = float(sigmas[i]), float(sigmas[i + 1])
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"i": i, "denoised": denoised, "x": x, "sigma": sigmas[i], "sigma_hat": sigmas[i]})
        velocity_a = ((x[..., v:] - denoised[..., v:]) / sv) / turbo._time_shift_slope(max(sv, 1e-6), SHIFT_V, SHIFT_A)
        x0 = torch.cat([denoised[..., :v], x[..., v:] - turbo._time_shift_sigma(sv, SHIFT_V, SHIFT_A) * velocity_a], dim=-1)
        if mask is not None and mask.shape[-1] == x.shape[-1]:
            x0 = denoised + mask * (x0 - denoised)
        if svn > 0:
            n, san = noise(sigmas[i], sigmas[i + 1]), turbo._time_shift_sigma(svn, SHIFT_V, SHIFT_A)
            x = torch.cat([(1 - svn) * x0[..., :v] + svn * n[..., :v], (1 - san) * x0[..., v:] + san * n[..., v:]], dim=-1)
        else:
            x = x0
    return x


class MiniMaxH3PikaSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "MiniMaxH3Turbo"

    def get_sampler(self):
        return (comfy.samplers.KSAMPLER(sample_pika),)


class MiniMaxH3PikaLoRA:
    """Apply the Pika LoRA like the Turbo LoRA on this base: run-time bypass, merge only for INT8-fused fc2."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "lora_name": (turbo.folder_paths.get_filename_list("loras"),)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "MiniMaxH3Turbo"

    def apply(self, model, lora_name):
        raw = turbo.comfy.utils.load_torch_file(turbo.folder_paths.get_full_path("loras", lora_name), safe_load=True)
        lora = {k.removeprefix("diffusion_model."): t for k, t in raw.items()}
        modules = sorted({k.rsplit(".lora_", 1)[0] for k in lora if ".lora_" in k})
        if not modules or any("adaln" in m for m in modules):
            raise ValueError(f"Unexpected Pika LoRA layout: {lora_name}")
        patched = model.clone()
        fused = set(turbo._int8_fused_fc2(model.model.diffusion_model, modules))
        n = turbo._apply_bypass_lora(patched, lora, [m for m in modules if m not in fused], 1.0)
        if fused:
            n += turbo._apply_merge_lora(patched, lora, sorted(fused), 1.0)
        if n != len(modules):
            raise RuntimeError(f"Pika LoRA applied {n} of {len(modules)} modules")
        return (patched,)


NODE_CLASS_MAPPINGS = {"MiniMaxH3PikaSampler": MiniMaxH3PikaSampler, "MiniMaxH3PikaLoRA": MiniMaxH3PikaLoRA}

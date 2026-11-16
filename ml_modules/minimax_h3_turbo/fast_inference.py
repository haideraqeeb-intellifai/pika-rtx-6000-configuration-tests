"""Request-scoped two-step Pika optimization for the RTX 5090 H3 environment."""

import math
import types
from contextlib import contextmanager
from contextvars import ContextVar


@contextmanager
def pika_fast_inference(runtime):
    """Use SA2 then SA3 FP4, and NVFP4 projections only during this request.

    The Comfy executor is single-flight. Install and restore the three class
    methods in its execution thread so timed-out requests retain their own
    patches until the thread actually stops.
    """
    import comfy.ops
    import comfy_kitchen as ck
    import torch
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout
    from sageattn3 import sageattn3_blackwell

    from .runtime import MiniMaxH3SageAttentionPatch
    from . import pika_nodes

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Pika fast inference requires an RTX 5090 (SM120)")

    original_forward = MiniMaxH3Model._forward
    original_apply = MiniMaxH3SageAttentionPatch.apply
    original_decode = MiniMaxH3VideoVAE.decode
    original_sample = pika_nodes.sample_pika
    sample_context = ContextVar("pika_fast_sample", default=None)
    step_context = ContextVar("pika_fast_step", default=None)
    cpu_cache = runtime.setdefault("pika_nvfp4_vae_cpu_cache", {})
    stats = {
        "sampler_calls": 0,
        "model_forward_calls": 0,
        "sage2_calls": 0,
        "sage3_calls": 0,
        "sage3_fallback_calls": 0,
        "video_decode_calls": 0,
        "nvfp4_gemm_calls": 0,
        "weight_quantizations": 0,
        "weight_cache_hits": 0,
    }

    def sample(*args, **kwargs):
        state = {"forwards": 0}
        token = sample_context.set(state)
        before = {key: stats[key] for key in ("sage2_calls", "sage3_calls", "sage3_fallback_calls")}
        try:
            result = original_sample(*args, **kwargs)
            if (state["forwards"] != 2 or
                    stats["sage2_calls"] == before["sage2_calls"] or
                    stats["sage3_calls"] == before["sage3_calls"] or
                    stats["sage3_fallback_calls"] != before["sage3_fallback_calls"]):
                raise RuntimeError("Pika fast chunk did not complete two SA2/SA3 steps")
            stats["sampler_calls"] += 1
            return result
        finally:
            sample_context.reset(token)

    def model_forward(
        self,
        x,
        timestep,
        context,
        transformer_options=None,
        minimax_payload=None,
        denoise_mask=None,
        **kwargs,
    ):
        state = sample_context.get()
        if state is None:
            raise RuntimeError("Pika fast model ran outside its sampler")
        step = state["forwards"]
        if step >= 2:
            raise RuntimeError(
                "Pika fast inference expected exactly two model forwards per chunk"
            )
        state["forwards"] += 1
        stats["model_forward_calls"] += 1
        token = step_context.set(step)
        try:
            if transformer_options is None:
                transformer_options = {}
            return original_forward(
                self,
                x,
                timestep,
                context,
                transformer_options=transformer_options,
                minimax_payload=minimax_payload,
                denoise_mask=denoise_mask,
                **kwargs,
            )
        finally:
            step_context.reset(token)

    def apply(self, model):
        (patched,) = original_apply(self, model)
        sage2 = patched.model_options["transformer_options"][
            "optimized_attention_override"
        ]

        def attention(
            func,
            q,
            k,
            v,
            heads,
            mask=None,
            attn_precision=None,
            skip_reshape=False,
            skip_output_reshape=False,
            **kwargs,
        ):
            step = step_context.get()
            if step == 0:
                stats["sage2_calls"] += 1
                return sage2(
                    func,
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    **kwargs,
                )
            if step != 1:
                raise RuntimeError("Pika fast attention ran outside a denoising step")
            if (
                not skip_reshape
                or skip_output_reshape
                or mask is not None
                or q.shape != k.shape
                or k.shape != v.shape
            ):
                stats["sage3_fallback_calls"] += 1
                return sage2(
                    func,
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    **kwargs,
                )
            if (
                q.shape[0] != 1
                or q.shape[1] != heads
                or q.shape[-1] != 128
                or q.dtype != torch.bfloat16
            ):
                raise RuntimeError("Unexpected Pika H3 attention shape or dtype")
            scale = kwargs.get("scale")
            if scale is not None and not math.isclose(scale, 128**-0.5, rel_tol=1e-7):
                raise RuntimeError(
                    "SageAttention3 does not support the requested scale"
                )
            # Sage3 centers the temporary projected keys in place. A copy
            # would push the full-length H3 request over 5090 VRAM.
            output = sageattn3_blackwell(q, k, v, is_causal=False, per_block_mean=False)
            torch.cuda.empty_cache()
            stats["sage3_calls"] += 1
            return output.transpose(1, 2).reshape(1, -1, heads * q.shape[-1])

        patched.model_options["transformer_options"]["optimized_attention_override"] = (
            attention
        )
        return (patched,)

    def decode(self, z):
        gpu_cache = {}
        layer_keys = {}
        restored = []

        def linear(layer, x):
            if x.dtype != torch.float16 or not x.is_cuda:
                raise RuntimeError(
                    f"Expected CUDA FP16 VAE activations, got {x.device}/{x.dtype}"
                )
            key = id(layer)
            if key not in gpu_cache:
                stable_key = layer_keys[key]
                if stable_key in cpu_cache:
                    q_cpu, s_cpu, bs_cpu, b_cpu = cpu_cache[stable_key]
                    if q_cpu.shape[0] < layer.weight.shape[0]:
                        raise RuntimeError("Cached NVFP4 VAE projection shape changed")
                    gpu_cache[key] = (
                        q_cpu.to(x.device),
                        s_cpu.to(x.device),
                        bs_cpu.to(x.device),
                        b_cpu.to(x.device) if b_cpu is not None else None,
                    )
                    stats["weight_cache_hits"] += 1
                else:
                    weight, bias, stream = comfy.ops.cast_bias_weight(
                        layer, x, offloadable=True
                    )
                    try:
                        quantized, params = TensorCoreNVFP4Layout.quantize(weight)
                        prepared_bias = bias.clone() if bias is not None else None
                        gpu_cache[key] = (
                            quantized,
                            params.scale,
                            params.block_scale,
                            prepared_bias,
                        )
                        cpu_cache[stable_key] = (
                            quantized.cpu(),
                            params.scale.cpu(),
                            params.block_scale.cpu(),
                            prepared_bias.cpu() if prepared_bias is not None else None,
                        )
                        stats["weight_quantizations"] += 1
                    finally:
                        comfy.ops.uncast_bias_weight(layer, weight, bias, stream)
            weight, sw, bsw, bias = gpu_cache[key]
            flat = x.reshape(-1, x.shape[-1]).contiguous()
            rows = flat.shape[0]
            qx, params = TensorCoreNVFP4Layout.quantize(flat)
            out = ck.scaled_mm_nvfp4(
                qx,
                weight,
                params.scale,
                sw,
                params.block_scale,
                bsw,
                bias=bias,
                out_dtype=x.dtype,
            )
            stats["nvfp4_gemm_calls"] += 1
            return out[:rows, : layer.weight.shape[0]].reshape(
                *x.shape[:-1], layer.weight.shape[0]
            )

        try:
            for block_index, block in enumerate(self.decoder.transformer_blocks):
                for name, layer in (
                    ("attn.to_qkv", block.attn.to_qkv),
                    ("attn.to_out", block.attn.to_out),
                    ("ff.w1", block.ff.w1),
                    ("ff.w2", block.ff.w2),
                ):
                    layer_keys[id(layer)] = (block_index, name)
                    restored.append((layer, layer.__dict__.get("forward")))
                    layer.forward = types.MethodType(linear, layer)
            result = original_decode(self, z)
            stats["video_decode_calls"] += 1
            return result
        finally:
            for layer, previous in restored:
                if previous is None:
                    del layer.forward
                else:
                    layer.forward = previous
            gpu_cache.clear()

    MiniMaxH3Model._forward = model_forward
    MiniMaxH3SageAttentionPatch.apply = apply
    MiniMaxH3VideoVAE.decode = decode
    pika_nodes.sample_pika = sample
    try:
        yield stats
    finally:
        MiniMaxH3Model._forward = original_forward
        MiniMaxH3SageAttentionPatch.apply = original_apply
        MiniMaxH3VideoVAE.decode = original_decode
        pika_nodes.sample_pika = original_sample

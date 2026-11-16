"""Pika H3 Flex on the Turbo base: the official Comfy LoRA plus its six-knot re-noise sampler.

Pika's release contract (fl2va/ref2va manifest.json): knots 1000/750/500/250/100/40/0, video shift 12, audio
shift 3, x0 prediction re-noised with fresh noise each step. Fewer steps use subsets of the same trained knots.
This ComfyUI has no ModelSamplingAV: the model returns the audio velocity scaled by d(sigma_a)/d(sigma_v), so the
sampler recovers audio x0 on the audio clock and re-noises audio at its own sigma. Latents the conditioning holds
fixed (denoise_mask == 0, e.g. lip sync audio and plate) are returned unchanged.
"""

KNOTS = {2: (1000, 500), 3: (1000, 500, 100), 6: (1000, 750, 500, 250, 100, 40)}
SHIFT_V, SHIFT_A = 12.0, 3.0


def sigmas(steps):
    """Video sigmas for a supported step count, terminal zero included."""
    ts = [k / 1000 for k in KNOTS[steps]] + [0.0]
    return [SHIFT_V * t / (1 + (SHIFT_V - 1) * t) for t in ts]

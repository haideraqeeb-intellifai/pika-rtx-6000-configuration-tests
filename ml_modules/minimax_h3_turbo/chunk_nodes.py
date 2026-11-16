"""H3-owned graph nodes for stage barriers, retained frames, and final encoding."""

import json
from pathlib import Path

from .chunking import encode_chunks, store_chunk


class H3AppendPreparation:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",), "latent": ("LATENT",)},
                "optional": {"previous": ("H3_PREPARATIONS",)}}

    RETURN_TYPES = ("H3_PREPARATIONS",)
    FUNCTION = "append"
    CATEGORY = "H3/chunks"

    def append(self, conditioning, latent, previous=None):
        return ([*(previous or []), (conditioning, latent)],)


class H3SelectPreparation:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prepared": ("H3_PREPARATIONS",), "index": ("INT",)}}

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    FUNCTION = "select"
    CATEGORY = "H3/chunks"

    def select(self, prepared, index):
        return prepared[index]


class H3AppendLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)}, "optional": {"previous": ("H3_LATENTS",)}}

    RETURN_TYPES = ("H3_LATENTS",)
    FUNCTION = "append"
    CATEGORY = "H3/chunks"

    def append(self, samples, previous=None):
        return ([*(previous or []), {**samples, "samples": samples["samples"].cpu()}],)


class H3SelectLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latents": ("H3_LATENTS",), "index": ("INT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "select"
    CATEGORY = "H3/chunks"

    def select(self, latents, index):
        return (latents[index],)


class H3StoreChunk:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "index": ("INT",), "frames": ("INT",),
                             "bit_depth": ("INT",), "in_memory": ("BOOLEAN",),
                             "directory": ("STRING",), "deadline": ("FLOAT",)},
                "optional": {"audio": ("AUDIO",), "previous": ("H3_CHUNKS",)}}

    RETURN_TYPES = ("H3_CHUNKS", "IMAGE", "IMAGE")
    FUNCTION = "store"
    CATEGORY = "H3/chunks"

    def store(self, previous=None, **kwargs):
        chunk, first, last = store_chunk(**kwargs)
        return ([*(previous or []), chunk], first, last)


class H3SaveChunks:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"chunks": ("H3_CHUNKS",), "output": ("STRING",), "audio_master": ("STRING",),
                             "width": ("INT",), "height": ("INT",), "crf": ("INT",), "deadline": ("FLOAT",)}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "H3/chunks"

    def save(self, **kwargs):
        details = encode_chunks(**kwargs)
        Path(kwargs["output"]).with_suffix(".json").write_text(json.dumps(details, indent=2) + "\n")
        return {"ui": {"text": [json.dumps(details)]}, "result": ()}


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    H3AppendPreparation, H3SelectPreparation, H3AppendLatent, H3SelectLatent,
    H3StoreChunk, H3SaveChunks,
)}

"""CPU latent handoff between independently owned H3 stage workers."""
import hashlib
from pathlib import Path
from .gpu_preencode import pack, unpack

class H3SaveLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'samples': ('LATENT',), 'path': ('STRING',)}}
    RETURN_TYPES = ()
    FUNCTION = 'save'
    OUTPUT_NODE = True
    def save(self, samples, path):
        import torch
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.partial')
        torch.save(pack(samples), temporary)
        temporary.replace(target)
        target.with_suffix('.sha256').write_text(hashlib.sha256(target.read_bytes()).hexdigest())
        return ()

class H3LoadLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'path': ('STRING',)}}
    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'load'
    def load(self, path):
        import torch
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != path.with_suffix('.sha256').read_text():
            raise ValueError('Latent handoff checksum mismatch')
        return (unpack(torch.load(path, map_location='cpu', weights_only=True)),)

NODE_CLASS_MAPPINGS = {c.__name__: c for c in (H3SaveLatent, H3LoadLatent)}

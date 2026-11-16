"""GPU preparation artifacts for independent H3 chunks, without model objects."""

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
import uuid
from pathlib import Path


def pack(value):
    import torch
    from comfy.nested_tensor import NestedTensor
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, NestedTensor):
        return {'__h3_nested_tensors__': [pack(t) for t in value.tensors]}
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [pack(v) for v in value]
    if isinstance(value, tuple):
        return tuple(pack(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported preparation value: {type(value)}')


def unpack(value):
    from comfy.nested_tensor import NestedTensor
    if isinstance(value, dict):
        if set(value) == {'__h3_nested_tensors__'}:
            return NestedTensor(value['__h3_nested_tensors__'])
        return {k: unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(v) for v in value]
    if isinstance(value, tuple):
        return tuple(unpack(v) for v in value)
    return value


class H3SavePrepared:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'conditioning': ('CONDITIONING',), 'latent': ('LATENT',),
                             'path': ('STRING',)}}
    RETURN_TYPES = ()
    FUNCTION = 'save'
    OUTPUT_NODE = True
    CATEGORY = 'H3/benchmark'

    def save(self, conditioning, latent, path):
        import torch
        torch.save(pack({'conditioning': conditioning, 'latent': latent}), path)
        return ()


class H3LoadPrepared:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'path': ('STRING',)}}
    RETURN_TYPES = ('CONDITIONING', 'LATENT')
    FUNCTION = 'load'
    CATEGORY = 'H3/benchmark'

    def load(self, path):
        import torch
        data = unpack(torch.load(path, map_location='cpu', weights_only=True))
        return data['conditioning'], data['latent']


def register(models):
    models['main'].nodes.NODE_CLASS_MAPPINGS.update(
        {c.__name__: c for c in (H3SavePrepared, H3LoadPrepared)})


def reachable(graph, outputs):
    required = set()
    def visit(node):
        if node in required:
            return
        required.add(node)
        for value in graph[node]['inputs'].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in graph:
                visit(value[0])
    for node in outputs:
        visit(node)
    return {key: value for key, value in graph.items() if key in required}


def cache_identical_text(models):
    """Cache only exact prompt+image inputs to the encoder, not chunk audio/latents."""
    stats = {'encoder_calls': 0, 'cache_hits': 0}
    proxies = {}
    class CachedClip:
        def __init__(self, clip):
            self.clip, self.cache = clip, {}
        def tokenize(self, prompt, images):
            digest = hashlib.sha256(prompt.encode())
            for image in images:
                digest.update(str((tuple(image.shape), image.dtype)).encode())
                digest.update(image.detach().cpu().contiguous().numpy().tobytes())
            key = digest.hexdigest()
            return key, None if key in self.cache else self.clip.tokenize(prompt, images=images)
        def encode_from_tokens_scheduled(self, token_pair):
            key, tokens = token_pair
            if key not in self.cache:
                self.cache[key] = self.clip.encode_from_tokens_scheduled(tokens)
                stats['encoder_calls'] += 1
            else:
                stats['cache_hits'] += 1
            return copy.deepcopy(self.cache[key])
    cls = models['main'].nodes.NODE_CLASS_MAPPINGS['MiniMaxH3ImageToVideo']
    original = cls.execute
    def execute(node_cls, clip, vae, prompt, width, height, length,
                first_frame=None, last_frame=None, audio_vae=None, audio=None):
        if id(clip) not in proxies:
            proxies[id(clip)] = CachedClip(clip)
        return original(proxies[id(clip)], vae, prompt, width, height, length,
                        first_frame, last_frame, audio_vae, audio)
    cls.execute = classmethod(execute)
    return stats


def prepare_batch(manifest_path, cache_text=False):
    from ml_modules.minimax_h3_turbo.infer import infer, load_models
    from ml_modules.minimax_h3_turbo.workflow import build_chunked_workflow
    from workers.image2video_h3_turbo.schema import build_chunk_plan, schema_image2video_h3_turbo
    from .parallel_chunk_processes import file_hash
    from .run_case import INPUTS, MODEL_ROOT, PROMPT
    directory = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    runtime = directory / 'runtime'
    inputs = runtime / 'input'
    inputs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INPUTS / 'raccoon_headshot.jpg', inputs / 'portrait.jpg')
    models = load_models(MODEL_ROOT, runtime)
    register(models)
    cache_stats = cache_identical_text(models) if cache_text else None
    main = models['main']
    models['executor'] = main.execution.PromptExecutor(models['progress'],
        cache_type=main.execution.CacheType.CLASSIC,
        cache_args={'lru': 0, 'ram': 0, 'ram_inactive': 0})
    results = []
    for index, chunk in enumerate(manifest):
        audio_name = f'audio_{index}.wav'
        shutil.copy2(chunk['audio'], inputs / audio_name)
        schedule = [{'duration_seconds': chunk['frames'] / 24,
                     'reference': 'original', 'anchor': chunk['anchor']}]
        message = schema_image2video_h3_turbo.model_validate({
            'uid': f'preencode_{index}', 'webhook_url': 'https://example.test/h3-benchmark',
            'prompt': PROMPT, 'model': 'pika_flex', 'num_inference_steps': 2,
            'chunked_mode': True, 'image_path': str(inputs / 'portrait.jpg'),
            'audio_path': str(inputs / audio_name), 'chunk_schedule': schedule,
            'width': 720, 'height': 1280, 'final_width': 1080, 'final_height': 1920,
            'fps': 24, 'seed': 9901 + index, 'timeout_seconds': 9000,
            'clip_device': 'default'}).model_dump()
        plan = build_chunk_plan(schedule, chunk['frames'])
        graph, effective = build_chunked_workflow(message, plan,
            first_frame_name='portrait.jpg', audio_names=[audio_name],
            audio_master=chunk['audio'], directory=directory,
            output=directory / 'unused.mp4', deadline=time.monotonic() + 9000)
        artifact = directory / f'prepared_{index}.pt'
        graph['save_prepared'] = {'class_type': 'H3SavePrepared', 'inputs': {
            'conditioning': ['c0_5', 0], 'latent': ['c0_5', 1], 'path': str(artifact)}}
        graph = reachable(graph, ['save_prepared'])
        (directory / f'workflow_{index}.json').write_text(json.dumps(graph, indent=2))
        started = time.time()
        result = infer(models, graph, prompt_id=uuid.uuid4().hex, timeout_seconds=9000)
        metadata = {**chunk, 'artifact_sha256': file_hash(artifact),
                    'audio_sha256': file_hash(Path(chunk['audio'])),
                    'portrait_sha256': file_hash(inputs / 'portrait.jpg'),
                    'prompt': PROMPT, 'effective': effective,
                    'started_wall_utc': started, 'ended_wall_utc': time.time(),
                    'artifact_bytes': artifact.stat().st_size,
                    'telemetry': result['telemetry']}
        (directory / f'prepared_{index}.json').write_text(json.dumps(metadata, indent=2))
        results.append(metadata)
    (directory / 'result.json').write_text(json.dumps({'status': 'success',
        'chunks': results, 'text_encoder_device': 'gpu',
        'text_cache': cache_stats,
        'model_release': 'preparation process exits before worker launch'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--cache-text', action='store_true')
    args = parser.parse_args()
    prepare_batch(args.manifest, args.cache_text)

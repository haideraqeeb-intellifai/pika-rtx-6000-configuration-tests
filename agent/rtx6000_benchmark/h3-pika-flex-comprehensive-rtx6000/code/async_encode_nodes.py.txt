"""Experimental single-stream NVENC feed while later H3 chunks generate."""

import atexit
import json
import queue
import subprocess
import threading
import time
import wave
from pathlib import Path

from ml_modules.minimax_h3_turbo.chunk_nodes import H3SaveChunks, H3StoreChunk
from ml_modules.minimax_h3_turbo.chunking import (FPS, NVENC_ARGS, encode_chunks,
                                                 remaining, validate_output)


_SESSIONS = {}


class _Encoder:
    def __init__(self, chunk, output, audio_master, width, height, deadline):
        self.started = time.monotonic()
        self.output = Path(output)
        self.audio_master = Path(audio_master)
        self.width, self.height = width, height
        self.deadline = deadline
        self.chunks = queue.Queue()
        self.error = None
        self.temporary = self.output.with_name(self.output.stem + '.overlap.encoding.mp4')
        raw_format = 'rgb48le' if chunk.bit_depth == 10 else 'rgb24'
        pixel_format = 'yuv420p10le' if chunk.bit_depth == 10 else 'yuv420p'
        command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                   '-f', 'rawvideo', '-pix_fmt', raw_format, '-s:v', f'{chunk.width}x{chunk.height}',
                   '-r', str(FPS), '-i', 'pipe:0', '-i', str(self.audio_master),
                   '-map', '0:v:0', '-map', '1:a:0', '-vf', f'scale={width}:{height},setsar=1',
                   '-c:v', 'h264_nvenc', *NVENC_ARGS, '-pix_fmt', pixel_format,
                   '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(self.temporary)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.PIPE, bufsize=0)
        self.thread = threading.Thread(target=self._feed, daemon=True)
        self.thread.start()

    def _feed(self):
        try:
            while (chunk := self.chunks.get()) is not None:
                for block in chunk.blocks():
                    view = memoryview(block)
                    while view:
                        written = self.process.stdin.write(view)
                        if not written:
                            raise RuntimeError('Overlapped encoder stopped accepting frames')
                        view = view[written:]
            self.process.stdin.close()
        except BaseException as exc:
            self.error = repr(exc)
            if self.process.poll() is None:
                self.process.kill()

    def submit(self, chunk):
        if self.error:
            raise RuntimeError(self.error)
        self.chunks.put(chunk)

    def finish(self, chunks):
        final_wait_start = time.monotonic()
        self.chunks.put(None)
        self.thread.join(timeout=remaining(self.deadline))
        if self.thread.is_alive():
            raise TimeoutError('Overlapped encoder feed did not finish')
        if self.error:
            raise RuntimeError(self.error)
        if self.process.wait(timeout=remaining(self.deadline)):
            raise RuntimeError(self.process.stderr.read().decode(errors='replace')[-4096:])
        frames = sum(chunk.frames for chunk in chunks)
        with wave.open(str(self.audio_master), 'rb') as stream:
            audio_duration = stream.getnframes() / stream.getframerate()
        validate_output(self.temporary, width=self.width, height=self.height, frames=frames,
                        bit_depth=chunks[0].bit_depth, audio_duration=audio_duration,
                        deadline=self.deadline)
        self.temporary.replace(self.output)
        return {'encoder': 'h264_nvenc', 'fallback_reason': None,
                'encoding_seconds': round(time.monotonic() - self.started, 3),
                'final_wait_seconds': round(time.monotonic() - final_wait_start, 3),
                'output_frames': frames, 'output_width': self.width,
                'output_height': self.height, 'audio_duration_seconds': audio_duration,
                'bytes': self.output.stat().st_size}

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)
        self.temporary.unlink(missing_ok=True)


@atexit.register
def _close_all():
    for encoder in _SESSIONS.values():
        encoder.close()


class H3AsyncStoreChunk(H3StoreChunk):
    @classmethod
    def INPUT_TYPES(cls):
        schema = H3StoreChunk.INPUT_TYPES()
        schema['required'].update({'output': ('STRING',), 'audio_master': ('STRING',),
                                   'output_width': ('INT',), 'output_height': ('INT',)})
        return schema

    def store(self, output, audio_master, output_width, output_height, previous=None, **kwargs):
        value = super().store(previous=previous, **kwargs)
        chunk = value[0][-1]
        encoder = _SESSIONS.get(output)
        if encoder is None:
            encoder = _Encoder(chunk, output, audio_master, output_width, output_height,
                               kwargs['deadline'])
            _SESSIONS[output] = encoder
        encoder.submit(chunk)
        return value


class H3AsyncSaveChunks(H3SaveChunks):
    def save(self, **kwargs):
        output = kwargs['output']
        encoder = _SESSIONS.pop(output)
        try:
            try:
                details = encoder.finish(kwargs['chunks'])
            except Exception as exc:
                encoder.close()
                details = encode_chunks(**kwargs)
                details['fallback_reason'] = f'overlapped NVENC: {exc}; ' + str(details.get('fallback_reason'))
        finally:
            encoder.close()
        Path(output).with_suffix('.json').write_text(json.dumps(details, indent=2) + '\n')
        return {'ui': {'text': [json.dumps(details)]}, 'result': ()}


NODE_CLASS_MAPPINGS = {'H3AsyncStoreChunk': H3AsyncStoreChunk,
                       'H3AsyncSaveChunks': H3AsyncSaveChunks}

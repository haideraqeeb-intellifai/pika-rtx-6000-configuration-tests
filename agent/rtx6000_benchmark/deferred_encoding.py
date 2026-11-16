"""Bounded CPU/NVENC encoding and verification while the next chunk decodes.

Only detached CPU frame buffers enter the background thread. One encoder thread
per VAE serializes its exports; at most two chunks may be in flight. Results are
announced only after encode, full A/V decode, frame count and hash succeed.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import time
import traceback
from .run_case import probe
from .parallel_chunk_processes import file_hash

class DeferredEncoding:
    def __init__(self, node, lane, results):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='h3-encode')
        self.pending = []
        self.latest = None
        self.lane, self.results = lane, results
        original = node.save
        def defer(instance, **kwargs):
            def encode():
                start = time.time()
                original(instance, **kwargs)
                return {'started_wall_utc': start, 'ended_wall_utc': time.time()}
            self.latest = self.pool.submit(encode)
            return {'ui': {'text': ['Encoding queued; completion checked before publishing result']}, 'result': ()}
        node.save = defer

    def limit_pending(self):
        while len(self.pending) >= 2:
            self.pending.pop(0).result(timeout=300)

    def finish_later(self, result, directory, output, frames, started):
        encoding = self.latest
        if encoding is None:
            raise RuntimeError('No encode task was submitted')
        self.latest = None
        def finish():
            try:
                interval = encoding.result(timeout=300)
                parsed = probe(output)
                video = next(s for s in parsed['streams'] if s['codec_type'] == 'video')
                if int(video.get('nb_read_frames', -1)) != frames:
                    raise ValueError('Deferred output has wrong frame count')
                subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',str(output),
                    '-map','0:v:0','-map','0:a:0','-f','null','-'],check=True,capture_output=True,timeout=300)
                result.update(output_probe=parsed, output_sha256=file_hash(output),
                    total_seconds=time.perf_counter()-started, process_ended_wall_utc=time.time(),
                    encoding_interval=interval, complete_video_audio_decode=True)
                (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
                self.results.put({'lane':self.lane,'kind':'result','index':result['index'],'result':result})
            except BaseException as exc:
                (directory/'traceback.txt').write_text(traceback.format_exc())
                self.results.put({'lane':self.lane,'kind':'error','index':result['index'],
                    'error':repr(exc),'traceback':traceback.format_exc()})
                raise
        self.pending.append(self.pool.submit(finish))

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=False)
        for future in self.pending:
            future.result()

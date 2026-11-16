"""Replayable raw chunk buffers and one deadline-bounded H.264/AAC save."""

import json
import logging
import os
import selectors
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path


FPS = 24
SAMPLE_RATE = 48000
FRAME_MEMORY_LIMIT = 4 * 1024**3
NVENC_ARGS = ["-preset", "p5", "-rc", "vbr", "-b:v", "3700k",
              "-maxrate", "4800k", "-bufsize", "7400k", "-multipass", "fullres",
              "-spatial-aq", "1", "-temporal-aq", "1"]


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("H3 chunked request exceeded its whole-job timeout")
    return seconds


def run_media(command, deadline):
    return subprocess.run(command, check=True, timeout=remaining(deadline), capture_output=True)


def prepare_audio(source, directory, deadline):
    """Decode once, retaining the original duration and mono level for final mux."""
    probe = json.loads(run_media([
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
        "stream=channels", "-of", "json", str(source),
    ], deadline).stdout)
    channels = int(probe["streams"][0]["channels"])
    channel_args = ["-af", "pan=stereo|c0=c0|c1=c0"] if channels == 1 else ["-ac", "2"]
    master = Path(directory) / "audio_master.wav"
    # Decode a little beyond the cap to detect overlong media without reading
    # an arbitrarily long recording into RAM or silently truncating it.
    run_media(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-i", str(source), "-map", "0:a:0", "-vn", "-t", "420.05",
               "-ar", str(SAMPLE_RATE), *channel_args, "-c:a", "pcm_s16le", str(master)], deadline)
    with wave.open(str(master), "rb") as stream:
        samples = stream.getnframes()
    if not 0 < samples <= 420 * SAMPLE_RATE:
        raise ValueError("chunked audio duration must be positive and at most 420 seconds")
    return master, samples / SAMPLE_RATE


def slice_audio(master, plan, directory, deadline):
    paths = []
    with wave.open(str(master), "rb") as source:
        for chunk in plan:
            remaining(deadline)
            source.setpos(min(source.getnframes(), chunk["start_frame"] * SAMPLE_RATE // FPS))
            pcm = source.readframes(chunk["frames"] * SAMPLE_RATE // FPS)
            target_bytes = chunk["inference_frames"] * SAMPLE_RATE // FPS * 4
            path = Path(directory) / f"audio_{chunk['index']}.wav"
            with wave.open(str(path), "wb") as out:
                out.setparams((2, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
                out.writeframes(pcm)
                out.writeframes(bytes(target_bytes - len(pcm)))
            paths.append(path)
    return paths


@dataclass
class ChunkFrames:
    index: int
    frames: int
    width: int
    height: int
    bit_depth: int
    buffers: list[bytes] | None
    path: Path | None
    audio_path: Path | None = None

    def blocks(self):
        if self.buffers is not None:
            yield from self.buffers
        else:
            with self.path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    yield block


def store_chunk(images, *, index, frames, bit_depth, in_memory, directory, deadline, audio=None):
    import numpy as np
    import torch

    if images.ndim != 4 or images.shape[-1] != 3 or images.shape[0] < frames:
        raise ValueError(f"Chunk {index} has invalid decoded frames: {tuple(images.shape)}, need {frames}")
    # Clone small reference tensors: a slice alone retains the entire video.
    first = images[:1].detach().cpu().clone()
    last = images[frames - 1:frames].detach().cpu().clone()
    path = None if in_memory else Path(directory) / f"frames_{index}.rgb"
    buffers = [] if in_memory else None
    stream = path.open("wb") if path else None
    try:
        for frame in images[:frames]:
            remaining(deadline)
            scale = 65535 if bit_depth == 10 else 255
            dtype = torch.int32 if bit_depth == 10 else torch.uint8
            data = (frame * scale).clamp(0, scale).to(device="cpu", dtype=dtype).contiguous().numpy()
            block = data.astype("<u2", copy=False).tobytes() if bit_depth == 10 else data.tobytes()
            if buffers is not None:
                buffers.append(block)
            else:
                stream.write(block)
    finally:
        if stream:
            stream.close()
    audio_path = None
    if audio is not None:
        import torchaudio

        waveform = audio["waveform"][0].detach().float().cpu()
        if int(audio["sample_rate"]) != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, int(audio["sample_rate"]), SAMPLE_RATE)
        if waveform.shape[0] == 1:
            waveform = waveform.expand(2, -1)
        waveform = waveform[:2, :frames * SAMPLE_RATE // FPS]
        if waveform.shape[0] != 2:
            raise ValueError("Generated H3 audio must have at least one channel")
        target = frames * SAMPLE_RATE // FPS
        waveform = torch.nn.functional.pad(waveform, (0, max(0, target - waveform.shape[-1])))
        pcm = (waveform.clamp(-1, 1).T.numpy() * 32767).astype(np.dtype("<i2"))
        audio_path = Path(directory) / f"generated_audio_{index}.wav"
        with wave.open(str(audio_path), "wb") as out:
            out.setparams((2, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            out.writeframes(pcm.tobytes())
    return ChunkFrames(index, frames, images.shape[2], images.shape[1], bit_depth,
                       buffers, path, audio_path), first, last


def validate_output(path, *, width, height, frames, bit_depth, audio_duration, deadline):
    probe = json.loads(run_media([
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path),
    ], deadline).stdout)
    video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in probe["streams"] if stream["codec_type"] == "audio")
    pixel_format = "yuv420p10le" if bit_depth == 10 else "yuv420p"
    actual = (video["width"], video["height"], int(video.get("nb_read_frames", -1)),
              video["codec_name"], video["pix_fmt"])
    if actual != (width, height, frames, "h264", pixel_format):
        raise ValueError(f"Encoded H3 output differs from requested dimensions/frames/bit depth: {actual}")
    num, den = map(int, video["avg_frame_rate"].split("/"))
    if den <= 0 or num / den != FPS or abs(float(video["duration"]) - frames / FPS) > 1 / FPS + 0.002:
        raise ValueError("Encoded H3 video has incorrect frame rate or duration")
    if audio["codec_name"] != "aac" or int(audio["channels"]) != 2 or abs(float(audio["duration"]) - audio_duration) > 0.05:
        raise ValueError("Encoded H3 audio differs from the source duration or AAC stereo contract")
    return probe


def encode_chunks(chunks, output, *, audio_master, width, height, crf, deadline):
    """Try NVENC once, then replay retained frames through libx264 once."""
    started = time.monotonic()
    chunks = sorted(chunks, key=lambda chunk: chunk.index)
    if not chunks or [chunk.index for chunk in chunks] != list(range(len(chunks))):
        raise ValueError("Expected one decoded buffer per chunk, in consecutive index order")
    source_width, source_height, bit_depth = chunks[0].width, chunks[0].height, chunks[0].bit_depth
    if any((chunk.width, chunk.height, chunk.bit_depth) != (source_width, source_height, bit_depth) for chunk in chunks):
        raise ValueError("All decoded chunks must have the same dimensions and bit depth")
    output = Path(output)
    frames = sum(chunk.frames for chunk in chunks)
    if not audio_master:
        audio_master = output.parent / "generated_master.wav"
        with wave.open(str(audio_master), "wb") as master:
            master.setparams((2, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            for chunk in chunks:
                remaining(deadline)
                with wave.open(str(chunk.audio_path), "rb") as source:
                    while block := source.readframes(SAMPLE_RATE):
                        master.writeframes(block)
    with wave.open(str(audio_master), "rb") as source:
        audio_duration = source.getnframes() / source.getframerate()
    pixel_format = "yuv420p10le" if bit_depth == 10 else "yuv420p"
    raw_format = "rgb48le" if bit_depth == 10 else "rgb24"
    temporary = output.with_name(output.stem + ".encoding.mp4")
    failures = []
    for encoder in ("h264_nvenc", "libx264"):
        remaining(deadline)
        # A stuck hardware encoder must leave time for CPU fallback.
        attempt_deadline = min(deadline, time.monotonic() + max(60, frames / FPS * 2)) if encoder == "h264_nvenc" else deadline
        encoder_args = NVENC_ARGS if encoder == "h264_nvenc" else ["-preset", "veryfast", "-crf", str(crf)]
        command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", raw_format, "-s:v", f"{source_width}x{source_height}",
                   "-r", str(FPS), "-i", "pipe:0", "-i", str(audio_master),
                   "-map", "0:v:0", "-map", "1:a:0", "-vf", f"scale={width}:{height},setsar=1",
                   "-c:v", encoder, *encoder_args, "-pix_fmt", pixel_format,
                   "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(temporary)]
        process = None
        with tempfile.TemporaryFile(dir=output.parent) as error_log:
            try:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                           stderr=error_log, bufsize=0)
                os.set_blocking(process.stdin.fileno(), False)
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdin, selectors.EVENT_WRITE)
                    for chunk in chunks:
                        for block in chunk.blocks():
                            data = memoryview(block)
                            while data:
                                timeout = remaining(attempt_deadline)
                                if process.poll() is not None:
                                    raise RuntimeError(f"{encoder} exited before all frames were written")
                                if not selector.select(min(timeout, 0.25)):
                                    continue
                                try:
                                    written = os.write(process.stdin.fileno(), data[:1024 * 1024])
                                except BlockingIOError:
                                    continue
                                data = data[written:]
                process.stdin.close()
                if process.wait(timeout=remaining(attempt_deadline)):
                    raise RuntimeError(f"{encoder} failed with status {process.returncode}")
                validate_output(temporary, width=width, height=height, frames=frames,
                                bit_depth=bit_depth, audio_duration=audio_duration, deadline=attempt_deadline)
                temporary.replace(output)
                return {"encoder": encoder, "fallback_reason": failures[0] if failures else None,
                        "encoding_seconds": round(time.monotonic() - started, 3),
                        "output_frames": frames, "output_width": width, "output_height": height,
                        "audio_duration_seconds": audio_duration, "bytes": output.stat().st_size}
            except Exception as exc:
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                error_log.seek(0, os.SEEK_END)
                size = error_log.tell()
                error_log.seek(max(0, size - 4096))
                detail = error_log.read().decode(errors="replace").strip()
                failures.append(f"{encoder}: {exc}; {detail}".rstrip("; "))
                logging.warning("H3 chunk encode failed: %s", failures[-1])
                temporary.unlink(missing_ok=True)
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=5)
                    if process.stdin:
                        process.stdin.close()
        # A whole-job timeout cannot be reset by the retry.
        remaining(deadline)
    raise RuntimeError("H3 NVENC and CPU encoding both failed: " + " | ".join(failures))

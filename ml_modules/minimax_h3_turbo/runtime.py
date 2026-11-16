"""Turbo-owned ComfyUI execution, timeouts, locking, and memory telemetry."""

import asyncio
import json
import os
import re
import sys
import threading
import time
from pathlib import Path


COMFY_ROOT = Path(__file__).resolve().parents[1] / "comfy_ui" / "comfy_v14"
PINNED_COMFY_COMMIT = "e377e263049f9338b4d12a3dd417b36ae62948ff"
PINNED_COMFY_VERSION = "0.30.0"
REQUIRED_NODES = {
    "UNETLoader", "CLIPLoader", "VAELoader", "LoadImage", "LoadVideo", "GetVideoComponents",
    "LoadAudio", "MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo",
    "MiniMaxH3SageAttentionPatch", "RandomNoise", "BasicGuider",
    "BasicScheduler", "SamplerCustomAdvanced", "VAEDecode", "VAEDecodeAudio", "ImageCrop",
    "CreateVideo", "SaveVideo", "LoraLoaderModelOnly",
}
_INFER_LOCK = threading.Lock()


def _gib(value):
    return round(float(value) / (1024 ** 3), 3)


class MiniMaxH3SageAttentionPatch:
    """Request-scoped SageAttention 2.2 patch for the Turbo diffusion model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "advanced/model"

    def apply(self, model):
        from comfy.ldm.modules.attention import SAGE_ATTENTION_IS_AVAILABLE, attention_sage

        if not SAGE_ATTENTION_IS_AVAILABLE:
            raise RuntimeError("SageAttention 2.2 is not installed in the MiniMax H3 Turbo environment")
        patched = model.clone()

        def sage_override(func, q, k, v, heads, mask=None, attn_precision=None,
                          skip_reshape=False, skip_output_reshape=False, **kwargs):
            return attention_sage(
                q, k, v, heads, mask=mask, attn_precision=attn_precision,
                skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs,
            )

        patched.model_options["transformer_options"]["optimized_attention_override"] = sage_override
        return (patched,)


class _ProgressServer:
    client_id = None
    last_node_id = None

    def __init__(self):
        self._lock = threading.Lock()
        self._prompts = {}
        self._model_management = None

    def configure(self, model_management):
        self._model_management = model_management

    def begin(self, prompt_id, workflow):
        with self._lock:
            self._prompts[prompt_id] = {
                "workflow": workflow, "active": {}, "nodes": [], "started": time.perf_counter(),
            }

    def _snapshot(self):
        import torch

        snapshot = {
            "cuda_allocated_gib": 0.0, "cuda_reserved_gib": 0.0,
            "cuda_peak_allocated_gib": 0.0, "model_total_gib": 0.0,
            "model_loaded_gib": 0.0, "model_offloaded_gib": 0.0,
        }
        if torch.cuda.is_available():
            snapshot.update({
                "cuda_allocated_gib": _gib(torch.cuda.memory_allocated()),
                "cuda_reserved_gib": _gib(torch.cuda.memory_reserved()),
                "cuda_peak_allocated_gib": _gib(torch.cuda.max_memory_allocated()),
            })
        if self._model_management is not None:
            models = list(self._model_management.current_loaded_models)
            total = sum(item.model_memory() for item in models if item.model is not None)
            loaded = sum(item.model_loaded_memory() for item in models if item.model is not None)
            snapshot.update({
                "model_total_gib": _gib(total),
                "model_loaded_gib": _gib(loaded),
                "model_offloaded_gib": _gib(max(0, total - loaded)),
            })
        return snapshot

    def send_sync(self, event, data, sid=None):
        prompt_id = data.get("prompt_id") if isinstance(data, dict) else None
        node_id = data.get("node") if isinstance(data, dict) else None
        if not prompt_id or node_id is None:
            return
        now = time.perf_counter()
        with self._lock:
            state = self._prompts.get(prompt_id)
            if state is None:
                return
            if event == "executing":
                # Comfy emits `executing` for every node, but `executed` only for
                # nodes with UI payloads. The next start is therefore the reliable
                # completion boundary for ordinary model/sampler nodes.
                for previous_id, started in list(state["active"].items()):
                    class_type = state["workflow"].get(previous_id, {}).get("class_type", "unknown")
                    state["nodes"].append({
                        "node_id": previous_id, "class_type": class_type,
                        "duration_seconds": round(now - started[0], 4),
                        "_started_monotonic": started[0], "_ended_monotonic": now,
                        "before": started[1], "after": self._snapshot(),
                    })
                    state["active"].pop(previous_id, None)
                state["active"][str(node_id)] = (now, self._snapshot())
            elif event == "executed":
                started = state["active"].pop(str(node_id), None)
                if started is not None:
                    class_type = state["workflow"].get(str(node_id), {}).get("class_type", "unknown")
                    state["nodes"].append({
                        "node_id": str(node_id), "class_type": class_type,
                        "duration_seconds": round(now - started[0], 4),
                        "_started_monotonic": started[0], "_ended_monotonic": now,
                        "before": started[1], "after": self._snapshot(),
                    })

    @staticmethod
    def _stage(class_type):
        if class_type in {"UNETLoader", "CLIPLoader", "VAELoader"}:
            return "model_load"
        if class_type in {"LoadImage", "LoadVideo", "GetVideoComponents", "LoadAudio",
                          "MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo",
                          "H3AppendPreparation", "H3SelectPreparation"}:
            return "condition_pack"
        if class_type == "MiniMaxH3SageAttentionPatch":
            return "optimizer_patch"
        if class_type == "VAEDecodeAudio":
            return "audio_decode"
        if class_type == "VAEDecode":
            return "video_decode"
        if class_type in {"ImageCrop", "CreateVideo", "SaveVideo", "H3StoreChunk", "H3SaveChunks"}:
            return "crop_mux_save"
        return "generation"

    def finish(self, prompt_id):
        with self._lock:
            state = self._prompts.pop(prompt_id, None)
        if state is None:
            return {"nodes": [], "stages": []}
        now = time.perf_counter()
        for node_id, started in state["active"].items():
            class_type = state["workflow"].get(node_id, {}).get("class_type", "unknown")
            state["nodes"].append({
                "node_id": node_id, "class_type": class_type,
                "duration_seconds": round(now - started[0], 4),
                "_started_monotonic": started[0], "_ended_monotonic": now,
                "before": started[1], "after": self._snapshot(),
            })
        stages = {}
        for record in state["nodes"]:
            stage = self._stage(record["class_type"])
            summary = stages.setdefault(stage, {
                "stage": stage, "duration_seconds": 0.0, "node_count": 0,
                "peak_cuda_allocated_gib": 0.0, "peak_model_offloaded_gib": 0.0,
                "mean_model_offloaded_gib": 0.0, "_offload_samples": [],
                "_intervals": [],
            })
            summary["duration_seconds"] += record["duration_seconds"]
            summary["node_count"] += 1
            summary["_intervals"].append(
                (record["_started_monotonic"], record["_ended_monotonic"]),
            )
            for snapshot in (record["before"], record["after"]):
                summary["peak_cuda_allocated_gib"] = max(
                    summary["peak_cuda_allocated_gib"], snapshot["cuda_peak_allocated_gib"]
                )
                summary["peak_model_offloaded_gib"] = max(
                    summary["peak_model_offloaded_gib"], snapshot["model_offloaded_gib"]
                )
                summary["_offload_samples"].append(snapshot["model_offloaded_gib"])
        for summary in stages.values():
            samples = summary.pop("_offload_samples")
            summary["duration_seconds"] = round(summary["duration_seconds"], 4)
            summary["mean_model_offloaded_gib"] = round(sum(samples) / len(samples), 3) if samples else 0.0
        for record in state["nodes"]:
            record.pop("_started_monotonic", None)
            record.pop("_ended_monotonic", None)
        return {
            "nodes": state["nodes"],
            "stages": [stages[name] for name in (
                "model_load", "condition_pack", "optimizer_patch", "generation",
                "audio_decode", "video_decode", "crop_mux_save",
            ) if name in stages],
            "execution_event_seconds": round(time.perf_counter() - state["started"], 4),
            "final_memory": self._snapshot(),
        }


class _NVMLSampler:
    """Compact 5 Hz GPU telemetry; raw samples are summarized then discarded."""

    def __init__(self, model_management, interval=0.2):
        self.model_management = model_management
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._pynvml = None

    def start(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return

        def sample_loop():
            while not self._stop.is_set():
                sample_started = time.perf_counter()
                try:
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    power = self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                    temperature = self._pynvml.nvmlDeviceGetTemperature(
                        self._handle, self._pynvml.NVML_TEMPERATURE_GPU,
                    )
                    rx = self._pynvml.nvmlDeviceGetPcieThroughput(
                        self._handle, self._pynvml.NVML_PCIE_UTIL_RX_BYTES,
                    ) / (1024 ** 2)
                    tx = self._pynvml.nvmlDeviceGetPcieThroughput(
                        self._handle, self._pynvml.NVML_PCIE_UTIL_TX_BYTES,
                    ) / (1024 ** 2)
                    models = list(self.model_management.current_loaded_models)
                    model_total = sum(
                        item.model_memory() for item in models if item.model is not None
                    )
                    model_loaded = sum(
                        item.model_loaded_memory() for item in models if item.model is not None
                    )
                    self.samples.append({
                        "monotonic": time.perf_counter(), "gpu_util_pct": util.gpu,
                        "memory_used_gib": _gib(memory.used), "power_w": power,
                        "temperature_c": temperature, "pcie_rx_gib_s": rx,
                        "pcie_tx_gib_s": tx,
                        "model_total_gib": _gib(model_total),
                        "model_loaded_gib": _gib(model_loaded),
                        "model_offloaded_gib": _gib(max(0, model_total - model_loaded)),
                    })
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                elapsed = time.perf_counter() - sample_started
                self._stop.wait(max(0.0, self.interval - elapsed))

        self._thread = threading.Thread(target=sample_loop, name="h3-nvml", daemon=True)
        self._thread.start()

    def stop(self):
        if self._stop.is_set():
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    @staticmethod
    def _p95(values):
        if not values:
            return 0.0
        return sorted(values)[min(len(values) - 1, int(0.95 * len(values)))]

    def _summary(self, samples):
        if not samples:
            return {"sample_count": 0, "error": getattr(self, "error", "no samples")}
        duration = max(0.0, samples[-1]["monotonic"] - samples[0]["monotonic"])
        def values(name):
            return [float(sample[name]) for sample in samples]
        rx = values("pcie_rx_gib_s")
        tx = values("pcie_tx_gib_s")
        return {
            "sample_count": len(samples),
            "observed_sample_hz": round((len(samples) - 1) / duration, 2) if duration else 0.0,
            "gpu_util_mean_pct": round(sum(values("gpu_util_pct")) / len(samples), 2),
            "gpu_util_p95_pct": round(self._p95(values("gpu_util_pct")), 2),
            "memory_used_peak_gib": round(max(values("memory_used_gib")), 3),
            "model_offloaded_mean_gib": round(
                sum(values("model_offloaded_gib")) / len(samples), 3
            ),
            "model_offloaded_peak_gib": round(max(values("model_offloaded_gib")), 3),
            "power_mean_w": round(sum(values("power_w")) / len(samples), 2),
            "power_peak_w": round(max(values("power_w")), 2),
            "energy_wh_estimate": round(
                (sum(values("power_w")) / len(samples)) * duration / 3600.0, 3
            ),
            "temperature_peak_c": round(max(values("temperature_c")), 1),
            "pcie_rx_gib_estimate": round((sum(rx) / len(samples)) * duration, 3),
            "pcie_tx_gib_estimate": round((sum(tx) / len(samples)) * duration, 3),
        }

    def summarize(self, stage_telemetry):
        result = self._summary(self.samples)
        for stage in stage_telemetry.get("stages", []):
            intervals = stage.pop("_intervals", [])
            selected = [
                sample for sample in self.samples
                if any(start <= sample["monotonic"] <= end for start, end in intervals)
            ]
            stage["gpu"] = self._summary(selected)
            if selected:
                offloaded = [sample["model_offloaded_gib"] for sample in selected]
                stage["mean_model_offloaded_gib"] = round(sum(offloaded) / len(offloaded), 3)
                stage["peak_model_offloaded_gib"] = round(max(offloaded), 3)
                stage["model_offload_sample_count"] = len(offloaded)
        return result


def load_models(model_root, runtime_root):
    """Initialize ComfyUI in this process; checkpoint nodes load lazily on first inference."""
    commit = (COMFY_ROOT / "UPSTREAM_COMMIT").read_text().strip()
    version = (COMFY_ROOT / "UPSTREAM_VERSION").read_text().strip()
    source_match = re.search(
        r'__version__\s*=\s*"([^"]+)"', (COMFY_ROOT / "comfyui_version.py").read_text(),
    )
    source_version = source_match.group(1) if source_match else None
    if (commit, version, source_version) != (
        PINNED_COMFY_COMMIT, PINNED_COMFY_VERSION, PINNED_COMFY_VERSION,
    ):
        raise RuntimeError(
            f"ComfyUI must be {PINNED_COMFY_VERSION} at {PINNED_COMFY_COMMIT}; "
            f"found marker_version={version}, source_version={source_version}, commit={commit}"
        )

    model_root = Path(model_root).resolve()
    runtime_root = Path(runtime_root).resolve()
    paths = {
        "input": runtime_root / "input", "output": runtime_root / "output",
        "temp": runtime_root / "temp", "user": runtime_root / "user",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    if os.fspath(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, os.fspath(COMFY_ROOT))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")

    from comfy.cli_args import args
    args.models_directory = os.fspath(model_root)
    args.input_directory = os.fspath(paths["input"])
    args.output_directory = os.fspath(paths["output"])
    args.temp_directory = os.fspath(paths["temp"])
    args.user_directory = os.fspath(paths["user"])
    args.cache_none = True
    args.cache_classic = False
    args.cache_lru = 0
    args.use_sage_attention = False
    args.disable_all_custom_nodes = True

    import main as comfy_main

    async def initialize_nodes():
        await comfy_main.nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
        comfy_main.nodes.NODE_CLASS_MAPPINGS["MiniMaxH3SageAttentionPatch"] = MiniMaxH3SageAttentionPatch

    asyncio.run(initialize_nodes())
    missing = sorted(REQUIRED_NODES - set(comfy_main.nodes.NODE_CLASS_MAPPINGS))
    if missing:
        raise RuntimeError(f"Pinned ComfyUI is missing H3 nodes: {missing}")

    progress = _ProgressServer()
    progress.configure(comfy_main.comfy.model_management)
    executor = comfy_main.execution.PromptExecutor(
        progress, cache_type=comfy_main.execution.CacheType.NONE,
        cache_args={"lru": 0, "ram": 0, "ram_inactive": 0},
    )
    return {
        "main": comfy_main, "executor": executor, "progress": progress,
    }


def infer(runtime, workflow, prompt_id, timeout_seconds, inference_mode="default"):
    """Validate and execute one ComfyUI graph synchronously inside the worker process."""
    with _INFER_LOCK:
        if runtime.get("poisoned"):
            raise RuntimeError(f"MiniMax H3 Turbo runtime must be restarted: {runtime['poisoned']}")
        comfy_main = runtime["main"]
        executor = runtime["executor"]
        valid, error, outputs, node_errors = asyncio.run(
            comfy_main.execution.validate_prompt(prompt_id, workflow, None)
        )
        if not valid:
            raise RuntimeError(f"ComfyUI rejected workflow: {error}; node_errors={node_errors}")

        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        runtime["progress"].begin(prompt_id, workflow)
        nvml_sampler = _NVMLSampler(comfy_main.comfy.model_management)
        nvml_sampler.start()
        thread_errors = []
        fast_stats_result = None

        def execute():
            nonlocal fast_stats_result
            try:
                # A non-null client id makes Comfy emit per-node lifecycle events
                # to our in-process progress collector; no network client exists.
                if inference_mode == "fast":
                    from .fast_inference import pika_fast_inference
                    with pika_fast_inference(runtime) as fast_stats:
                        executor.execute(
                            workflow, prompt_id, extra_data={"client_id": "h3-inprocess"},
                            execute_outputs=outputs,
                        )
                        samples = sum(node["class_type"] == "SamplerCustomAdvanced" for node in workflow.values())
                        decodes = sum(node["class_type"] == "VAEDecode" for node in workflow.values())
                        if executor.success and (fast_stats["sampler_calls"] != samples or
                                                 fast_stats["model_forward_calls"] != 2 * samples or
                                                 (samples > 0 and fast_stats["sage2_calls"] == 0) or
                                                 (samples > 0 and fast_stats["sage3_calls"] == 0) or
                                                 fast_stats["sage3_fallback_calls"] != 0 or
                                                 fast_stats["video_decode_calls"] != decodes or
                                                 (decodes > 0 and fast_stats["nvfp4_gemm_calls"] == 0)):
                            raise RuntimeError(f"Pika fast inference did not complete the expected two-step path: {fast_stats}")
                        fast_stats_result = fast_stats
                else:
                    executor.execute(
                        workflow, prompt_id, extra_data={"client_id": "h3-inprocess"},
                        execute_outputs=outputs,
                    )
            except BaseException as exc:
                thread_errors.append(exc)

        thread = threading.Thread(target=execute, name=f"comfy-{prompt_id}", daemon=True)
        try:
            thread.start()
            thread.join(timeout_seconds)
            if thread.is_alive():
                comfy_main.nodes.interrupt_processing(True)
                thread.join(30)
                if thread.is_alive():
                    runtime["poisoned"] = (
                        f"request {prompt_id} did not stop within the 30 second interrupt grace period"
                    )
                    raise RuntimeError(
                        f"ComfyUI request {prompt_id} exceeded {timeout_seconds}s and poisoned the worker; restart required"
                    )
                raise TimeoutError(f"ComfyUI request {prompt_id} exceeded {timeout_seconds}s")
            if thread_errors:
                raise thread_errors[0]
            if not executor.success:
                failures = [data for event, data in executor.status_messages if event == "execution_error"]
                raise RuntimeError("ComfyUI execution failed: " + json.dumps(failures[-1] if failures else {}))

            nvml_sampler.stop()
            stages = runtime["progress"].finish(prompt_id)
            return {
                "history": executor.history_result,
                "telemetry": {
                    "stages": stages,
                    "gpu": nvml_sampler.summarize(stages),
                    "easycache": None,
                    "spectrum": None,
                    "sol": None,
                    **({"fast_inference": fast_stats_result} if inference_mode == "fast" else {}),
                },
            }
        finally:
            nvml_sampler.stop()
            comfy_main.hook_breaker_ac10a0.restore_functions()
            if prompt_id in runtime["progress"]._prompts:
                runtime["progress"].finish(prompt_id)

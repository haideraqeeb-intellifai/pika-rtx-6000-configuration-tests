"""Self-contained H3 Turbo graph construction and multimodal input packing."""

from workers.image2video_h3_turbo.schema import infer_input_mode

from .assets import (AUDIO_VAE, DIFFUSION, EXTRA_LORAS, FL2VA, LORA, LORA_REVISION, MODEL_REVISION, PIKA_FL2VA, PIKA_REF2VA,
                     PIKA_REPO, PIKA_REVISION, TEXT_ENCODER, TURBO_COMMIT, VIDEO_VAE)
from .pika import sigmas as pika_sigmas


def build_workflow(message, *, first_frame_name=None, last_frame_name=None,
                   reference_image_names=None, reference_video_names=None,
                   reference_audio_names=None, audio_name=None, output_prefix, seed):
    reference_image_names = reference_image_names or []
    reference_video_names = reference_video_names or []
    reference_audio_names = reference_audio_names or []
    input_mode = infer_input_mode(
        first_frame_name, last_frame_name, reference_image_names,
        reference_video_names, reference_audio_names,
    )
    requested_width = int(message["width"])
    requested_height = int(message["height"])
    native_width = ((requested_width + 31) // 32) * 32
    native_height = ((requested_height + 31) // 32) * 32
    requested_frames = int(message["num_frames"])
    effective_frames = requested_frames + (5 - requested_frames) % 17
    checkpoint = DIFFUSION if input_mode == "r2v" else FL2VA
    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": checkpoint, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": message["clip_device"]}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
    }
    conditioning = {
        "clip": ["2", 0], "vae": ["3", 0], "prompt": message["prompt"],
        "width": native_width, "height": native_height, "length": effective_frames,
    }
    if input_mode == "r2v":
        conditioning.update({"audio_vae": ["4", 0], "ref_image_size": message.get("ref_image_size", "match")})
        for index, name in enumerate(reference_image_names):
            node_id = str(20 + index)
            workflow[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
            conditioning[f"ref_images.ref_image_{index}"] = [node_id, 0]
        for index, name in enumerate(reference_video_names):
            load_id = str(40 + 2 * index)
            components_id = str(41 + 2 * index)
            workflow[load_id] = {"class_type": "LoadVideo", "inputs": {"file": name}}
            workflow[components_id] = {"class_type": "GetVideoComponents", "inputs": {"video": [load_id, 0]}}
            conditioning[f"ref_videos.ref_video_{index}"] = [components_id, 0]
            conditioning[f"ref_video_audios.ref_video_audio_{index}"] = [components_id, 1]
        for index, name in enumerate(reference_audio_names):
            node_id = str(60 + index)
            workflow[node_id] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
            conditioning[f"ref_audios.ref_audio_{index}"] = [node_id, 0]
        conditioning_node = "MiniMaxH3ReferenceToVideo"
    else:
        if first_frame_name:
            workflow["20"] = {"class_type": "LoadImage", "inputs": {"image": first_frame_name}}
            conditioning["first_frame"] = ["20", 0]
        if last_frame_name:
            workflow["21"] = {"class_type": "LoadImage", "inputs": {"image": last_frame_name}}
            conditioning["last_frame"] = ["21", 0]
        if audio_name:
            workflow["22"] = {"class_type": "LoadAudio", "inputs": {"audio": audio_name}}
            conditioning.update({"audio_vae": ["4", 0], "audio": ["22", 0]})
        conditioning_node = "MiniMaxH3ImageToVideo"
    workflow["5"] = {"class_type": conditioning_node, "inputs": conditioning}
    base_ref = ["1", 0]
    if message.get("lora", "none") != "none":
        # Weight patches compose with the Turbo node's bypass adapters; a second bypass LoRA would replace them.
        lora_name, strength = EXTRA_LORAS[message["lora"]]
        workflow["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": base_ref, "lora_name": lora_name, "strength_model": strength}}
        base_ref = ["lora", 0]
    pika = message.get("model", "turbo") == "pika_flex"
    steps = message["num_inference_steps"]
    fast = message.get("inference_mode", "default") == "fast"
    if fast and (not pika or steps != 2):
        raise ValueError("inference_mode='fast' requires two-step Pika Flex")
    if pika:
        adapter = "pika"
        workflow["pika"] = {"class_type": "MiniMaxH3PikaLoRA", "inputs": {"model": base_ref, "lora_name": PIKA_REF2VA if input_mode == "r2v" else PIKA_FL2VA}}
    else:
        adapter = "turbo"
        workflow["turbo"] = {"class_type": "MiniMaxH3TurboLoRA", "inputs": {
            "model": base_ref, "lora_name": LORA, "strength": 1.0, "low_vram": False,
        }}
    # Named patch nodes cannot collide with the ninth reference image (node 28).
    workflow["sage"] = {"class_type": "MiniMaxH3SageAttentionPatch", "inputs": {"model": [adapter, 0]}}
    model_ref = ["sage", 0]
    workflow.update({
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "7": {"class_type": "BasicGuider", "inputs": {"model": model_ref, "conditioning": ["5", 0]}},
        "8": {"class_type": "MiniMaxH3PikaSampler" if pika else "MiniMaxH3TurboSampler", "inputs": {}},
        "9": ({"class_type": "ManualSigmas", "inputs": {"sigmas": ", ".join(f"{x:.9f}" for x in pika_sigmas(steps))}} if pika else
              {"class_type": "BasicScheduler", "inputs": {"model": model_ref, "scheduler": "simple", "steps": steps, "denoise": 1.0}}),
        "10": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["6", 0], "guider": ["7", 0], "sampler": ["8", 0], "sigmas": ["9", 0], "latent_image": ["5", 1]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
    })
    images_ref = ["11", 0]
    if native_width != requested_width or native_height != requested_height:
        workflow["15"] = {"class_type": "ImageCrop", "inputs": {
            "image": images_ref, "width": requested_width, "height": requested_height,
            "x": (native_width - requested_width) // 2, "y": (native_height - requested_height) // 2,
        }}
        images_ref = ["15", 0]
    save_inputs = {
        "video": ["16", 0], "filename_prefix": output_prefix,
        "format": message["container_format"], "codec": message["codec"],
    }
    if message["codec"] == "h264":
        # V3 DynamicCombo inputs use flattened live-input keys.
        save_inputs["codec.encoding"] = "re-encode"
        save_inputs["codec.encoding.crf"] = float(message["crf"])
    workflow.update({
        "16": {"class_type": "CreateVideo", "inputs": {"images": images_ref, "audio": ["22" if audio_name else "12", 0], "fps": message["fps"], "bit_depth": message["bit_depth"]}},
        "17": {"class_type": "SaveVideo", "inputs": save_inputs},
    })
    return workflow, {
        "input_mode": input_mode,
        "requested_width": requested_width, "requested_height": requested_height,
        "native_width": native_width, "native_height": native_height,
        "requested_frames": requested_frames, "effective_frames": effective_frames,
        "reference_images": len(reference_image_names), "reference_videos": len(reference_video_names),
        "reference_audios": len(reference_audio_names), "seed": seed,
        "optimization": {"profile": "pika_flex_fast" if fast else ("pika_flex" if pika else "turbo"),
                         "attention_backend": "sage2_then_sage3_fp4" if fast else "sage2",
                         "video_decoder": "nvfp4" if fast else "native",
                         "step_accelerator": "none", "approximate": True},
        "optimizer_parameters": {"easycache": None, "sol": None, "spectrum": None},
        "diffusion_checkpoint": checkpoint, "model_revision": MODEL_REVISION,
        "lora": workflow["pika"]["inputs"]["lora_name"] if pika else LORA,
        "lora_revision": f"{PIKA_REPO}@{PIKA_REVISION}" if pika else LORA_REVISION, "lora_strength": 1.0,
        "lora_low_vram": False, "turbo_commit": None if pika else TURBO_COMMIT,
        "sampler": "MiniMaxH3PikaSampler" if pika else "MiniMaxH3TurboSampler", "steps": steps, "denoise": 1.0,
        "scheduler": "pika_knots" if pika else "simple",
        "sigma_shift_video": 12.0, "sigma_shift_audio": 3.0,
        "output_frames": effective_frames, "fps": message["fps"],
    }


def build_chunked_workflow(message, plan, *, first_frame_name=None, last_frame_name=None,
                           audio_names=None, audio_master="", directory, output, deadline):
    """Share loaders, honor frame dependencies, retain raw frames, then save once."""
    from .chunking import FRAME_MEMORY_LIMIT

    shared = {"1", "2", "3", "4", "pika", "turbo", "sage", "8", "9"}
    prompts = message.get("chunk_prompts") or [message["prompt"]] * len(plan)
    graph, seeds, input_modes = {}, [], []
    total_frames = sum(chunk["frames"] for chunk in plan)
    in_memory = total_frames * message["width"] * message["height"] * 3 * (2 if message["bit_depth"] == 10 else 1) <= FRAME_MEMORY_LIMIT
    grouped = all(chunk["reference"] == "original" for chunk in plan) and (bool(first_frame_name) or len(plan) == 1)
    for index, chunk in enumerate(plan):
        seed = (message["seed"] + index) % (1 << 64)
        seeds.append(seed)
        controls = {**message, "prompt": prompts[index], "num_frames": chunk["inference_frames"]}
        source, effective = build_workflow(
            controls, first_frame_name=first_frame_name, last_frame_name=last_frame_name,
            audio_name=audio_names[index] if audio_names else None, output_prefix="unused", seed=seed,
        )
        prefix = f"c{index}_"

        def remap(value):
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in source:
                return [value[0] if value[0] in shared else prefix + value[0], value[1]]
            if isinstance(value, dict):
                return {key: remap(item) for key, item in value.items()}
            return value

        for node_id, node in source.items():
            if node_id in {"16", "17"} or (node_id == "12" and audio_names):
                continue
            if node_id in shared:
                if index == 0:
                    graph[node_id] = node
            else:
                graph[prefix + node_id] = remap(node)
        condition = graph[prefix + "5"]["inputs"]
        if index > 0:
            if chunk["reference"] == "previous_last_frame":
                condition["first_frame"] = [f"buffer_{index - 1}", 2]
            elif not first_frame_name:
                condition["first_frame"] = ["buffer_0", 1]
        if chunk["anchor"] == "none":
            condition.pop("last_frame", None)
        elif not last_frame_name:
            condition["last_frame"] = [prefix + "20", 0] if first_frame_name else ["buffer_0", 1]
        input_modes.append(infer_input_mode(condition.get("first_frame"), condition.get("last_frame")))

        if grouped:
            prepared = {"conditioning": [prefix + "5", 0], "latent": [prefix + "5", 1]}
            sampled = {"samples": [prefix + "10", 0]}
            if index:
                prepared["previous"] = [f"prepare_{index - 1}", 0]
                sampled["previous"] = [f"sample_{index - 1}", 0]
            graph[f"prepare_{index}"] = {"class_type": "H3AppendPreparation", "inputs": prepared}
            graph[f"prepared_{index}"] = {"class_type": "H3SelectPreparation", "inputs": {
                "prepared": [f"prepare_{len(plan) - 1}", 0], "index": index}}
            graph[prefix + "7"]["inputs"]["conditioning"] = [f"prepared_{index}", 0]
            graph[prefix + "10"]["inputs"]["latent_image"] = [f"prepared_{index}", 1]
            graph[f"sample_{index}"] = {"class_type": "H3AppendLatent", "inputs": sampled}
            graph[f"latent_{index}"] = {"class_type": "H3SelectLatent", "inputs": {
                "latents": [f"sample_{len(plan) - 1}", 0], "index": index}}
            for node_id in ("11", "12"):
                if prefix + node_id in graph:
                    graph[prefix + node_id]["inputs"]["samples"] = [f"latent_{index}", 0]

        buffer_inputs = {
            "images": [prefix + ("15" if "15" in source else "11"), 0],
            "index": index, "frames": chunk["frames"], "bit_depth": message["bit_depth"],
            "in_memory": in_memory, "directory": str(directory), "deadline": deadline,
        }
        if index:
            buffer_inputs["previous"] = [f"buffer_{index - 1}", 0]
        if not audio_names:
            buffer_inputs["audio"] = [prefix + "12", 0]
        graph[f"buffer_{index}"] = {"class_type": "H3StoreChunk", "inputs": buffer_inputs}

    width, height = message["width"], message["height"]
    final_width, final_height = message.get("final_width", -1), message.get("final_height", -1)
    if final_width != -1 and final_height == -1:
        height = max(2, round(height * final_width / width / 2) * 2)
    if final_height != -1 and final_width == -1:
        width = max(2, round(width * final_height / height / 2) * 2)
    width = final_width if final_width != -1 else width
    height = final_height if final_height != -1 else height
    graph["save_chunks"] = {"class_type": "H3SaveChunks", "inputs": {
        "chunks": [f"buffer_{len(plan) - 1}", 0], "output": str(output), "audio_master": str(audio_master),
        "width": width, "height": height, "crf": message["crf"], "deadline": deadline}}
    effective.update({"chunked_mode": True, "input_mode": input_modes[0], "chunk_input_modes": input_modes,
                      "seed": message["seed"], "requested_frames": total_frames,
                      "effective_frames": sum(chunk["inference_frames"] for chunk in plan),
                      "output_frames": total_frames, "output_width": width, "output_height": height,
                      "chunk_count": len(plan), "chunk_seeds": seeds,
                      "chunk_frames": [chunk["frames"] for chunk in plan],
                      "chunk_inference_frames": [chunk["inference_frames"] for chunk in plan],
                      "chunk_schedule": [{key: chunk[key] for key in ("duration_seconds", "reference", "anchor")} for chunk in plan],
                      "chunk_execution": "stage_grouped" if grouped else "dependent",
                      "frame_storage": "memory" if in_memory else "disk"})
    return graph, effective

"""Replay fixed denoised latents through one VAE in batches of equal shape.

Decoder-only diagnostic; its time excludes preparation and denoising and must
not be compared directly with full-video benchmark time.
"""
import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from .parallel_chunk_processes import RUNS, GPUTrace, file_hash, assemble
from .parallel_chunk_resident import wait_for_empty_gpu
from .run_case import MODEL_ROOT
from .latent_stage_nodes import H3LoadLatent

class H3BatchDecodeSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'vae': ('VAE',), 'jobs': ('STRING',), 'result_path': ('STRING',)}}
    RETURN_TYPES = ()
    FUNCTION = 'run'
    OUTPUT_NODE = True
    def run(self, vae, jobs, result_path):
        import torch
        from ml_modules.minimax_h3_turbo.chunking import store_chunk, encode_chunks
        jobs = json.loads(jobs)
        latents = []
        for job in jobs:
            latent = H3LoadLatent().load(job['latent'])[0]['samples']
            latents.append(latent.unbind()[0] if latent.is_nested else latent)
        if len({tuple(t.shape) for t in latents}) != 1:
            raise ValueError('Batch requires equal latent shapes')
        samples = torch.cat(latents, dim=0)
        torch.cuda.synchronize(); started = time.time()
        model_calls = []
        original_decode = vae.first_stage_model.decode
        def traced_model_decode(z, *args, **kwargs):
            model_calls.append(list(z.shape))
            return original_decode(z, *args, **kwargs)
        vae.first_stage_model.decode = traced_model_decode
        try:
            images = vae.decode(samples)
        finally:
            vae.first_stage_model.decode = original_decode
        torch.cuda.synchronize(); ended = time.time()
        if images.ndim != 5 or images.shape[0] != len(jobs):
            raise ValueError(f'Expected B,T,H,W,C output, got {images.shape}')
        stats = {'batch_size': len(jobs), 'latent_shape': list(samples.shape),
                 'decoded_shape': list(images.shape), 'actual_model_input_shapes': model_calls, 'decode_seconds': ended-started,
                 'decode_started': started, 'decode_ended': ended, 'chunks': []}
        for i, job in enumerate(jobs):
            directory = RUNS / job['name']; directory.mkdir()
            begin = time.perf_counter()
            cropped = images[i, :, :1280, 8:728, :]
            chunk, _, _ = store_chunk(cropped, index=0, frames=job['frames'], bit_depth=8,
                in_memory=True, directory=directory, deadline=time.monotonic()+9000)
            encoded = encode_chunks([chunk], directory/'output.mp4', audio_master=job['audio'],
                width=1080,height=1920,crf=23,deadline=time.monotonic()+9000)
            stats['chunks'].append({**job, 'encoding': encoded, 'crop_encode_seconds': time.perf_counter()-begin,
                                    'output_sha256': file_hash(directory/'output.mp4')})
        Path(result_path).write_text(json.dumps(stats, indent=2))
        return ()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--batch-size',type=int,choices=[1,2,4],required=True)
    args=parser.parse_args()
    directory=RUNS/args.name;directory.mkdir()
    source=RUNS/args.source_run
    original=json.loads((source/'result.json').read_text())
    if original['status']!='success': raise ValueError('Replay requires successful source run')
    evidence={'name':args.name,'scope':'decoder-only fixed-latent replay','source_run':args.source_run,
              'source_result_sha256':file_hash(source/'result.json'), 'batch_size':args.batch_size,'status':'running'}
    (directory/'result.json').write_text(json.dumps(evidence,indent=2))
    evidence['empty_gpu_gate']=wait_for_empty_gpu()
    # The four-copy calibration is a conservative bound for four activations
    # with a single set of weights; record actual peak and check OOM fallback.
    calibration=json.loads((RUNS/'vae_stages_calibration/result.json').read_text())
    if args.batch_size*calibration['calibration_peak_gib']['decode'] > 95.59*.9:
        raise MemoryError('Batch exceeds calibrated capacity guard')
    trace=GPUTrace();trace.start();started=time.perf_counter()
    try:
        from ml_modules.minimax_h3_turbo.infer import load_models,infer
        models=load_models(MODEL_ROOT,directory/'runtime')
        main=models['main'];main.nodes.NODE_CLASS_MAPPINGS['H3BatchDecodeSave']=H3BatchDecodeSave
        models['executor']=main.execution.PromptExecutor(models['progress'],cache_type=main.execution.CacheType.CLASSIC,
            cache_args={'lru':0,'ram':0,'ram_inactive':0})
        groups={}
        for i,frames in enumerate(original['chunk_frames']):
            path=source/'latents'/f'latent_{i}.pt'
            value=H3LoadLatent().load(str(path))[0]['samples']
            video=value.unbind()[0] if value.is_nested else value
            groups.setdefault(tuple(video.shape),[]).append({'index':i,'name':f'{args.name}_chunk_{i}',
                'latent':str(path),'latent_sha256':file_hash(path),'frames':frames,'audio':str(source/f'audio_{i}.wav')})
        batches=[]
        for jobs in groups.values():
            for offset in range(0,len(jobs),args.batch_size): batches.append(jobs[offset:offset+args.batch_size])
        results=[]
        for i,jobs in enumerate(batches):
            path=directory/f'batch_{i}.json'
            graph={'vae':{'class_type':'VAELoader','inputs':{'vae_name':'minimax_h3_video_vae_fp16.safetensors'}},
                'decode':{'class_type':'H3BatchDecodeSave','inputs':{'vae':['vae',0], 'jobs':json.dumps(jobs),'result_path':str(path)}}}
            measured=infer(models,graph,prompt_id=uuid.uuid4().hex,timeout_seconds=9000)
            result=json.loads(path.read_text());result['telemetry']=measured['telemetry'];results.append(result)
        evidence['decoder_phase_seconds']=time.perf_counter()-started
        evidence['batches']=results
    finally:
        trace.shutdown();(directory/'gpu_trace.json').write_text(json.dumps(trace.samples))
    evidence['gpu_trace_summary']={'peak_vram_gib':max(x['used_gib'] for x in trace.samples),
        'gpu_util_mean_pct':sum(x['util_pct'] for x in trace.samples)/len(trace.samples)}
    names=[f'{args.name}_chunk_{i}' for i in range(original['chunks'])]
    output,probe=assemble(directory,names,source/'audio_master.wav',sum(original['chunk_frames']))
    evidence.update(output_probe=probe,output_sha256=file_hash(output),plan=original['plan'],complete_video_audio_decode=True)
    (directory/'result.json').write_text(json.dumps(evidence,indent=2))
    for checker in ('check_audio','check_seams','make_contact_sheet'):
        subprocess.run([sys.executable,'-m',f'agent.rtx6000_benchmark.{checker}',str(directory)],check=True,timeout=300,capture_output=True)
    evidence['audio_check']=json.loads((directory/'audio_check.json').read_text())
    evidence['seam_check']=json.loads((directory/'seam_check.json').read_text())
    evidence.update(status='success',decoder_plus_validation_seconds=time.perf_counter()-started)
    (directory/'result.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps({k:evidence[k] for k in ('name','status','decoder_phase_seconds','gpu_trace_summary')}),flush=True)

if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        import traceback
        if '--name' in sys.argv:
            folder=RUNS/sys.argv[sys.argv.index('--name')+1]
            if folder.exists():
                path=folder/'result.json'
                result=json.loads(path.read_text()) if path.exists() else {}
                result.update(status='failure',error=repr(exc))
                path.write_text(json.dumps(result,indent=2))
                (folder/'traceback.txt').write_text(traceback.format_exc())
        raise

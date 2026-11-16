"""Replay original prepared chunks in one resident lane for targeted NCU kernels.

Indices 0 and 4 warm the model and both frame sizes; index9 repeats73frames.
Existing conditioning/latent artifacts are read-only. No video-level timing claim.
"""
import argparse,json,os,queue
from pathlib import Path
from agent.rtx6000_benchmark.parallel_chunk_resident import lane_main
from agent.rtx6000_benchmark.parallel_chunk_processes import RUNS

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['denoise','decode'],required=True);p.add_argument('--name',required=True)
    a=p.parse_args();source=RUNS/'followup_capacity_1t_2v_control';meta=json.loads((source/'result.json').read_text())
    root=RUNS/a.name;root.mkdir()
    tasks=queue.SimpleQueue();results=queue.SimpleQueue()
    for i in (0,4,9):
        tasks.put((i,meta['chunk_frames'][i],str(source/f'audio_{i}.wav'),a.name+f'_chunk_{i}','hard'))
    tasks.put(None)
    os.environ['H3_PROFILE_NVTX']='1'
    latents=root/'latents' if a.stage=='denoise' else source/'latents';latents.mkdir(exist_ok=True)
    lane_main(0,str(root/'lane'),tasks,results,'default','dynamic',str(source/'prepared'),a.stage,str(latents))
    messages=[]
    while not results.empty():messages.append(results.get())
    (root/'messages.json').write_text(json.dumps(messages,indent=2))
    errors=[m for m in messages if m['kind']=='error']
    if errors:raise RuntimeError(str(errors))
    assert len([m for m in messages if m['kind']=='result'])==3
    print('Targeted stage replay complete',a.stage,flush=True)
if __name__=='__main__':main()

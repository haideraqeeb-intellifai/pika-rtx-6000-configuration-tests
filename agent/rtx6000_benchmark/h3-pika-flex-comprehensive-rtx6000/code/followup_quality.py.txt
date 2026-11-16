"""CPU-only full-frame pixel comparisons, run after timed GPU cases finish."""
import json
import re
import subprocess
from .parallel_chunk_processes import RUNS, file_hash
from .compare_vae_outputs import frame_hash

def main():
    runs=[]
    for path in sorted(RUNS.glob('followup_*/result.json')):
        r=json.loads(path.read_text())
        if r.get('loader')=='separate stage processes' and r['status']=='success':runs.append(r)
    references={}
    for c in (5,10,20):
        reference=next(r for r in runs if r['chunks']==c and r['transformers']==r['vaes']==1
            and r.get('residency','dynamic')=='dynamic' and not r.get('async_encoding',False))
        references[c]=reference
    hashes={};results=[];identical_files={};ssim_cache={}
    for r in runs:
        name=r['name'];output=RUNS/name/'output.mp4';sha=file_hash(output)
        if sha!=r['output_sha256']:raise ValueError(f'Output changed since measurement: {name}')
        if sha in identical_files:
            source=identical_files[sha]
            current={**hashes[source],'full_frame_hash_reused_from_identical_sha256':source}
        else:
            current=frame_hash(output);identical_files[sha]=name
        current['verified_output_sha256']=sha
        hashes[name]=current
        if current['decoded_frames']!=1186:raise ValueError(f'Wrong frame count: {name}')
        print(f'Hashed {name}',flush=True)
    for r in runs:
        name=r['name'];ref=references[r['chunks']]['name'];current=hashes[name]
        latents=list((RUNS/name/'latents').glob('*.sha256'))
        entry={'name':name,'chunk_plan_reference':ref,**current,
            'same_plan_pixels_identical':current['frame_md5_list_sha256']==hashes[ref]['frame_md5_list_sha256'],
            'latent_files':len(latents),'identical_latent_files':sum(p.read_text()==(RUNS/ref/'latents'/p.name).read_text() for p in latents),
            'audio_check':r['audio_check'],'seam_check':r['seam_check']}
        pair=(hashes[ref]['verified_output_sha256'],current['verified_output_sha256'])
        if not entry['same_plan_pixels_identical'] and pair not in ssim_cache:
            compared=subprocess.run(['ffmpeg','-nostdin','-hide_banner','-i',str(RUNS/ref/'output.mp4'),
                '-i',str(RUNS/name/'output.mp4'),'-lavfi','[0:v]scale=180:320[a];[1:v]scale=180:320[b];[a][b]ssim',
                '-an','-f','null','-'],capture_output=True,text=True,check=True,timeout=300)
            match=re.findall(r'SSIM Y:.*All:([\d.]+)',compared.stderr)
            if not match:raise ValueError('SSIM output missing')
            ssim_cache[pair]=float(match[-1])
        if pair in ssim_cache:entry['ssim_at_180x320']=ssim_cache[pair]
        results.append(entry)
        print(json.dumps({k:v for k,v in entry.items() if k not in ('audio_check','seam_check')}),flush=True)
    (RUNS/'followup_quality.json').write_text(json.dumps({'results':results,
        'note':'Full-frame decoded pixel equality only within a chunk plan. Changed chunk plans intentionally generate different motion; luma join screening is not a lip-sync or seamlessness guarantee.'},indent=2))

if __name__=='__main__':main()

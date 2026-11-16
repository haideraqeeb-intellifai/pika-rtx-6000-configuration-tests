"""CPU-only comparison after all timed GPU experiments finish."""
import hashlib
import json
import re
import subprocess
from .parallel_chunk_processes import RUNS

BASE='vae_staged_1t_1v'

def frame_hash(path):
    result=subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',str(path),
        '-map','0:v:0','-f','framemd5','-'],check=True,capture_output=True,timeout=300)
    lines=[x for x in result.stdout.splitlines() if not x.startswith(b'#')]
    return {'decoded_frames':len(lines),'frame_md5_list_sha256':hashlib.sha256(b'\n'.join(lines)).hexdigest()}

def main():
    reference=RUNS/BASE/'output.mp4';base=frame_hash(reference)
    results=[]
    for path in sorted(RUNS.glob('vae_*/result.json')):
        r=json.loads(path.read_text())
        if r.get('status')!='success' or not r.get('complete_video_audio_decode') or not (path.parent/'output.mp4').exists(): continue
        current=frame_hash(path.parent/'output.mp4')
        if current['decoded_frames'] != 1186: raise ValueError(f'Full video has wrong frame count: {path.parent.name}')
        identical=current['frame_md5_list_sha256']==base['frame_md5_list_sha256']
        entry={'name':path.parent.name,**current,'decoded_pixels_identical_to_reference':identical}
        if not identical:
            check=subprocess.run(['ffmpeg','-nostdin','-hide_banner','-i',str(reference),'-i',str(path.parent/'output.mp4'),
                '-lavfi','[0:v]scale=180:320[a];[1:v]scale=180:320[b];[a][b]ssim',
                '-an','-f','null','-'],capture_output=True,text=True,check=True,timeout=300)
            match=re.findall(r'SSIM Y:.*All:([\d.]+)',check.stderr)
            if not match: raise ValueError('No SSIM result')
            entry['ssim_at_180x320']=float(match[-1])
        latent_paths=list((path.parent/'latents').glob('*.sha256'))
        if latent_paths:
            entry['latent_files']=len(latent_paths)
            entry['identical_latent_checksums']=sum(p.read_text()==(RUNS/BASE/'latents'/p.name).read_text() for p in latent_paths)
        results.append(entry)
        print(json.dumps(entry),flush=True)
    (RUNS/'vae_output_comparison.json').write_text(json.dumps({'reference':BASE,'reference_frames':base,
        'ssim_note':'Downsampled technical similarity only; no human seamlessness or lip-sync rating.', 'results':results},indent=2))

if __name__=='__main__':main()

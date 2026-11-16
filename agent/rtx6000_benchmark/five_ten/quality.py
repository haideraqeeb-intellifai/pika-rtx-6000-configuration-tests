"""Additional CPU checks after measurement; no quality-score claim."""
import json,subprocess,re,hashlib
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
if __name__=='__main__':
 torch.set_num_threads(4);out={'latents':[],'visual_comparisons':[]}
 out['prepared_input_integrity']=[]
 for i in range(5):
  meta=json.loads((P/f'prepared/prepared_{i}.json').read_text());blob=P/f'prepared/prepared_{i}.pt'
  assert meta['frames']==240 and meta['effective']['effective_frames']==243 and meta['effective']['seed']==9901+i
  assert hashlib.sha256(blob.read_bytes()).hexdigest()==meta['artifact_sha256']
  out['prepared_input_integrity'].append({'chunk':i,'sha256':meta['artifact_sha256'],'native_frames':243,'output_frames':240,'seed':9901+i})
 base=P/'runs/sequential_1t2v'
 for name in ['sequential_1t2v','batch_compact_1t2v','concurrent_2t1v','concurrent_2t2v','batch_1t2v']:
  root=P/'runs'/name
  for rep in range(1,4):
   for i in range(5):
    p=root/f'repeat_{rep}/latent_{i}.pt';x=torch.load(p,map_location='cpu',weights_only=True)['samples']['__h3_nested_tensors__'];y=torch.load(base/f'repeat_{rep}/latent_{i}.pt',map_location='cpu',weights_only=True)['samples']['__h3_nested_tensors__'];ms=[]
    for a,b in zip(x,y):
     a=a.float();b=b.float();assert a.shape==b.shape and bool(a.isfinite().all());ms.append({'shape':list(a.shape),'finite':True,'bit_equal':torch.equal(a,b),'rmse':float((a-b).square().mean().sqrt()),'cosine':float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0))})
    out['latents'].append({'configuration':name,'repeat':rep,'chunk':i,'video':ms[0],'unused_generated_audio':ms[1]})
  if name!='sequential_1t2v':
   cmd=['ffmpeg','-hide_banner','-i',str(base/'repeat_1/video.mp4'),'-i',str(root/'repeat_1/video.mp4'),'-filter_complex','[0:v]scale=180:320[a];[1:v]scale=180:320[b];[a][b]ssim','-an','-f','null','-']
   r=subprocess.run(cmd,check=True,capture_output=True,timeout=300);m=re.search(r'All:([\d.]+)',r.stderr.decode());assert m;out['visual_comparisons'].append({'configuration':name,'ssim_vs_sequential':float(m[1]),'resolution':[180,320],'frames':1200,'repeat':1})
 out['compact_vs_initial_video_bytes']=[{'repeat':rep,'equal':hashlib.sha256((P/f'runs/batch_compact_1t2v/repeat_{rep}/video.mp4').read_bytes()).digest()==hashlib.sha256((P/f'runs/batch_1t2v/repeat_{rep}/video.mp4').read_bytes()).digest()} for rep in range(1,4)]
 (P/'quality.json').write_text(json.dumps(out,indent=2));print('Finite latents and visual comparisons complete',flush=True)

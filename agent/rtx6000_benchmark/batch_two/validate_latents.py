"""Read saved tensors on CPU, preserving all experiment artifacts."""
import hashlib,json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
out=[]
for mode in ['sequential','batch','concurrent','batch_modulation']:
 for repeat in [1,2,3]:
  for i in [0,1]:
   p=P/'runs'/('batch_modulation_v1' if mode=='batch_modulation' else f'{mode}_v2')/f'repeat_{repeat}'/f'latent_{i}.pt'
   base=P/'runs/sequential_v2'/f'repeat_{repeat}'/f'latent_{i}.pt'
   x=torch.load(p,map_location='cpu',weights_only=True)['samples']['__h3_nested_tensors__'];y=torch.load(base,map_location='cpu',weights_only=True)['samples']['__h3_nested_tensors__']
   metrics=[]
   for a,b in zip(x,y):
    a=a.float();b=b.float();assert bool(a.isfinite().all()) and a.shape==b.shape
    metrics.append({'shape':list(a.shape),'finite':True,'bit_equal':bool(torch.equal(a,b)),'rmse':float((a-b).square().mean().sqrt()),'max_abs_difference':float((a-b).abs().max()),'cosine':float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0))})
   out.append({'mode':mode,'repeat':repeat,'chunk':i,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'video':metrics[0],'audio_latent_unused_for_export':metrics[1]})
(P/'latent_validation.json').write_text(json.dumps(out,indent=2));print('All saved chunk latent outputs finite; comparisons saved.')

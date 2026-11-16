"""Experiment-only B=2 support. Production files are never modified.

Linear/MLP kernels operate on concatenated rows; attention uses a separate batch
axis. Each item retains its conditioning, layout, RNG stream and output slice.
"""
import copy,inspect,json,textwrap
from pathlib import Path
SEEDS=[]
STATS={'batch_forwards':0,'block_calls':0,'attention_batch_shapes':[]}

def install(models,source_dir):
 import torch
 import comfy.model_base as mb
 import comfy.conds
 import comfy.ldm.minimax.model as mm
 import comfy.model_prefetch
 from ml_modules.minimax_h3_turbo import fast_inference,pika_nodes
 source_dir=Path(source_dir);source_dir.mkdir(exist_ok=True,parents=True)
 original=mm.MiniMaxH3Model._forward
 src=textwrap.dedent(inspect.getsource(original))
 a=src.index('    # blocks\n');b=src.index('    # target streams',a)
 generated=src[:a]+'    h = yield locals()\n\n'+src[b:]
 generated=generated.replace('def _forward(', 'def prepare_and_finish(' ,1)
 (source_dir/'prepare_and_finish.py').write_text(generated)
 namespace=dict(mm.__dict__);exec(compile(generated,str(source_dir/'prepare_and_finish.py'),'exec'),namespace)
 prepare=namespace['prepare_and_finish']
 def forward(self,x,timestep,context,transformer_options=None,minimax_payload=None,denoise_mask=None,**kwargs):
  opts=transformer_options or {};payload=minimax_payload or {}
  if 'batch_payloads' not in payload:return original(self,x,timestep,context,transformer_options=opts,minimax_payload=payload,denoise_mask=denoise_mask,**kwargs)
  B=len(payload['batch_payloads']);assert B==2 and x[0].shape[0]==B
  generators=[];states=[]
  for i in range(B):
   gen=prepare(self,[z[i:i+1] for z in x],timestep[i:i+1],context[i:i+1],transformer_options=opts,minimax_payload=payload['batch_payloads'][i],denoise_mask=denoise_mask[i:i+1] if denoise_mask is not None else None,**kwargs)
   states.append(next(gen));generators.append(gen)
  lengths=[s['h'].shape[0] for s in states];assert lengths[0]==lengths[1],lengths
  h=torch.cat([s['h'] for s in states],dim=0);emb=torch.cat([s['t_emb'] for s in states],dim=0)
  mods=[];offset=0;emb_offset=0
  for s in states:
   mods.extend((a+offset,b+offset,row+emb_offset*3) for a,b,row in s['mod_segments'])
   offset+=s['h'].shape[0];emb_offset+=s['t_emb'].shape[0]
  rope=torch.cat([s['rope_freqs'] for s in states],dim=1)
  opts={**opts,'_h3_batch_size':B}
  assert not opts.get('patches_replace',{}).get('dit',{}),'Unexpected block replacement'
  prefetch=comfy.model_prefetch.make_prefetch_queue(list(self.blocks),h.device,opts)
  for block in self.blocks:
   comfy.model_prefetch.prefetch_queue_pop(prefetch,h.device,block)
   h=block(h,emb,mods,rope,transformer_options=opts);STATS['block_calls']+=1
  if prefetch is not None:comfy.model_prefetch.prefetch_queue_pop(prefetch,h.device,None)
  outputs=[]
  for gen,chunk in zip(generators,h.split(lengths,dim=0)):
   try:gen.send(chunk)
   except StopIteration as done:outputs.append(done.value)
   else:raise AssertionError('Expected one yield')
  STATS['batch_forwards']+=1
  return [torch.cat([r[j] for r in outputs],dim=0) for j in range(2)]
 mm.MiniMaxH3Model._forward=forward
 attn=textwrap.dedent(inspect.getsource(mm.Attention.forward))
 attn=attn.replace('s = x.shape[0]','s = x.shape[0]\n    batch = transformer_options.get("_h3_batch_size", 1)\n    assert s % batch == 0')
 for v in ['q','k','v']:
  attn=attn.replace(f'{v} = {v}.transpose(0, 1).unsqueeze(0)',f'{v} = {v}.reshape(batch, s // batch, self.heads, self.head_dim).transpose(1, 2)')
 attn=attn.replace('out = optimized_attention(', 'if list(q.shape) not in STATS["attention_batch_shapes"]: STATS["attention_batch_shapes"].append(list(q.shape))\n    out = optimized_attention(')
 attn=attn.replace('return self.out_proj(out.squeeze(0))','return self.out_proj(out.reshape(s, self.heads * self.head_dim))')
 (source_dir/'attention.py').write_text(attn);ns={**mm.__dict__,'STATS':STATS};exec(compile(attn,str(source_dir/'attention.py'),'exec'),ns);mm.Attention.forward=ns['forward']
 fast=inspect.getsource(fast_inference)
 assert 'q.shape[0] != 1' in fast and 'reshape(1, -1, heads * q.shape[-1])' in fast
 fast=fast.replace('q.shape[0] != 1','q.shape[0] not in (1, 2)').replace('reshape(1, -1, heads * q.shape[-1])','reshape(q.shape[0], -1, heads * q.shape[-1])')
 (source_dir/'fast_inference_batch.py').write_text(fast);ns=dict(fast_inference.__dict__);exec(compile(fast,str(source_dir/'fast_inference_batch.py'),'exec'),ns);fast_inference.pika_fast_inference=ns['pika_fast_inference']
 orig_extra=mb.MiniMaxH3.extra_conds
 def extra(self,**kw):
  items=kw.get('h3_batch_items')
  if items is None:return orig_extra(self,**kw)
  conds=[]
  for i,item in enumerate(items):
   local={**kw,**item['metadata'],'cross_attn':item['context'],'seed':item['seed']}
   if local.get('denoise_mask') is not None:local['denoise_mask']=kw['denoise_mask'][i:i+1]
   if local.get('noise') is not None:local['noise']=kw['noise'][i:i+1]
   if local.get('latent_image') is not None:local['latent_image']=kw['latent_image'][i:i+1]
   local['latent_shapes']=[(1,*s[1:]) for s in kw['latent_shapes']]
   conds.append(orig_extra(self,**local))
  out=dict(conds[0]);out['latent_shapes']=comfy.conds.CONDConstant(kw['latent_shapes'])
  out['c_crossattn']=comfy.conds.CONDRegular(torch.cat([c['c_crossattn'].cond for c in conds],dim=0))
  out['minimax_payload']=comfy.conds.CONDConstant({'batch_payloads':[c['minimax_payload'].cond for c in conds]})
  if 'denoise_mask' in out:out['denoise_mask']=comfy.conds.CONDRegular(torch.cat([c['denoise_mask'].cond for c in conds],dim=0))
  return out
 mb.MiniMaxH3.extra_conds=extra
 original_noise=pika_nodes.default_noise_sampler
 def noise(x,seed=None):
  if x.shape[0]!=2:return original_noise(x,seed)
  assert len(SEEDS)==2
  samplers=[original_noise(x[i:i+1],s) for i,s in enumerate(SEEDS)]
  return lambda a,b:torch.cat([f(a,b) for f in samplers],dim=0)
 pika_nodes.default_noise_sampler=noise
 models['main'].nodes.NODE_CLASS_MAPPINGS.update({c.__name__:c for c in [H3BatchPrepared,H3BatchNoise,H3BatchSave]})

class H3BatchPrepared:
 @classmethod
 def INPUT_TYPES(cls):return {'required':{'paths':('STRING',),'seeds':('STRING',)}}
 RETURN_TYPES=('CONDITIONING','LATENT');FUNCTION='load';CATEGORY='H3/benchmark'
 def load(self,paths,seeds):
  import torch
  from comfy.nested_tensor import cat_nested
  from agent.rtx6000_benchmark.gpu_preencode import unpack
  items=[unpack(torch.load(p,map_location='cpu',weights_only=True)) for p in json.loads(paths)];seeds=json.loads(seeds)
  assert len(items)==len(seeds)==2
  for d in items:assert len(d['conditioning'])==1
  context=torch.cat([d['conditioning'][0][0] for d in items],dim=0)
  meta=copy.deepcopy(items[0]['conditioning'][0][1]);meta['h3_batch_items']=[{'context':d['conditioning'][0][0],'metadata':d['conditioning'][0][1],'seed':s} for d,s in zip(items,seeds)]
  lat=copy.copy(items[0]['latent'])
  for k in ['samples','noise_mask']:
   if k not in lat:continue
   vs=[d['latent'][k] for d in items]
   lat[k]=cat_nested(vs,dim=0) if getattr(vs[0],'is_nested',False) else torch.cat(vs,dim=0)
  return ([[context,meta]],lat)

class H3BatchNoise:
 @classmethod
 def INPUT_TYPES(cls):return {'required':{'seeds':('STRING',)}}
 RETURN_TYPES=('NOISE',);FUNCTION='get';CATEGORY='H3/benchmark'
 def get(self,seeds):
  values=json.loads(seeds)
  class Noise:
   seed=values[0]
   def generate_noise(self,latent):
    import torch,comfy.sample
    from comfy.nested_tensor import cat_nested
    SEEDS[:]=values
    values_noise=[comfy.sample.prepare_noise(latent['samples'][i:i+1],s) for i,s in enumerate(values)]
    return cat_nested(values_noise,dim=0) if getattr(values_noise[0],'is_nested',False) else torch.cat(values_noise,dim=0)
  return (Noise(),)

class H3BatchSave:
 @classmethod
 def INPUT_TYPES(cls):return {'required':{'samples':('LATENT',),'paths':('STRING',)}}
 RETURN_TYPES=();FUNCTION='save';OUTPUT_NODE=True
 def save(self,samples,paths):
  from agent.rtx6000_benchmark.latent_stage_nodes import H3SaveLatent
  for i,p in enumerate(json.loads(paths)):
   result={k:v[i:i+1] if k in ('samples','noise_mask') else v for k,v in samples.items()}
   H3SaveLatent().save(result,p)
  return ()

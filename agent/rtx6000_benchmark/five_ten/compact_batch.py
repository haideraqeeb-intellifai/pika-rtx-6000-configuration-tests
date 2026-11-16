"""Remove unrelated batch-item storage from serialized per-chunk handoffs."""
from agent.rtx6000_benchmark.batch_two.batch_support import H3BatchSave

def compact(value):
 import torch
 if isinstance(value,torch.Tensor):return value.clone(memory_format=torch.contiguous_format)
 if isinstance(value,dict):return {k:compact(v) for k,v in value.items()}
 if isinstance(value,list):return [compact(v) for v in value]
 if isinstance(value,tuple):return tuple(compact(v) for v in value)
 return value

class H3CompactBatchSave(H3BatchSave):
 def save(self,samples,paths):
  import json
  from agent.rtx6000_benchmark.gpu_preencode import pack,unpack
  from agent.rtx6000_benchmark.latent_stage_nodes import H3SaveLatent
  for i,p in enumerate(json.loads(paths)):
   result={k:v[i:i+1] if k in ('samples','noise_mask') else v for k,v in samples.items()}
   H3SaveLatent().save(unpack(compact(pack(result))),p)
  return ()

def install(models):models['main'].nodes.NODE_CLASS_MAPPINGS['H3BatchSave']=H3CompactBatchSave

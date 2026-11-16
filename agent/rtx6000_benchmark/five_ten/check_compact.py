"""CPU check: compact serialization must preserve every tensor exactly."""
import json,tempfile
from pathlib import Path
import torch
from agent.rtx6000_benchmark.five_ten.compact_batch import compact
P=Path(__file__).resolve().parent

def equal(a,b):
 if isinstance(a,torch.Tensor):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
if __name__=='__main__':
 torch.set_num_threads(1);rows=[]
 for i in range(5):
  src=P/f'runs/batch_1t2v/repeat_1/latent_{i}.pt';original=torch.load(src,map_location='cpu',weights_only=True);out=P/f'compact_check_{i}.pt'
  torch.save(compact(original),out);loaded=torch.load(out,map_location='cpu',weights_only=True);assert equal(original,loaded)
  rows.append({'chunk':i,'tensor_values_bit_equal':True,'original_bytes':src.stat().st_size,'compact_bytes':out.stat().st_size});out.unlink()
 (P/'compact_storage_validation.json').write_text(json.dumps(rows,indent=2));print(rows)

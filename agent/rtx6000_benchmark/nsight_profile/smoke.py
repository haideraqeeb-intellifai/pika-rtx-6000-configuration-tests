import torch
x=torch.randn((2048,2048),device='cuda')
for _ in range(8):
    with torch.cuda.nvtx.range('profiler_smoke'):
        y=x@x
        torch.cuda.synchronize()
print('profiler smoke passed')

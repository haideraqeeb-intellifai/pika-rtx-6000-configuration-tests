import subprocess,sys
for name in ['sequential_1t2v','batch_compact_1t2v','concurrent_2t1v','concurrent_2t2v','batch_1t2v']:
 subprocess.run([sys.executable,'-m','agent.rtx6000_benchmark.five_ten.validate',name],check=True,timeout=1800)
subprocess.run([sys.executable,'-m','agent.rtx6000_benchmark.five_ten.quality'],check=True,timeout=1800)

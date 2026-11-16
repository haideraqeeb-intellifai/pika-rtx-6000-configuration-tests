import json,subprocess,time,os
from pathlib import Path
from agent.rtx6000_benchmark.run_case import empty_gpu_gate,INPUTS
from agent.rtx6000_benchmark.preencoded_parallel import preencode
from agent.rtx6000_benchmark.batch_two.experiment import Monitor
P=Path(__file__).resolve().parent
if __name__=='__main__':
 (P/'inputs').mkdir(exist_ok=True)
 source=INPUTS/'report_audio.wav';master=P/'inputs/audio_master.wav'
 subprocess.run(['ffmpeg','-v','error','-y','-i',str(source),'-af','apad','-t','50','-ar','48000','-ac','2','-c:a','pcm_s16le',str(master)],check=True)
 audios=[]
 for i in range(5):
  out=P/f'inputs/audio_{i}.wav';subprocess.run(['ffmpeg','-v','error','-y','-i',str(master),'-ss',str(10*i),'-t','10','-c:a','pcm_s16le',str(out)],check=True);audios.append(out)
 (P/'preparation_gate.json').write_text(json.dumps(empty_gpu_gate(),indent=2))
 mon=Monitor({os.getpid()});mon.thread.start()
 try:
  seconds,result=preencode(P/'prepared',[240]*5,audios,'hard',True);mon.check()
  (P/'preparation.json').write_text(json.dumps({'seconds':seconds,'result':result,'delivered_frames':[240]*5,'fps':24,'source_seconds':49.408,'padding_silence_seconds':0.592,'counts_fixed':True},indent=2))
 finally:
  mon.stop.set();mon.thread.join();(P/'preparation_gpu.json').write_text(json.dumps(mon.samples))

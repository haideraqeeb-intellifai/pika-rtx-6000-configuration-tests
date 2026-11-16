"""CPU-only integrity, timing, audio equality and visual-difference checks."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def run(cmd):return subprocess.run(cmd,check=True,capture_output=True,timeout=300)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate(root):
 root=Path(root).resolve()
 output=[]
 for p in sorted(root.glob('repeat_[123]/chunk_*.mp4')):
  probe=json.loads(run(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(p)]).stdout)
  v=next(s for s in probe['streams'] if s['codec_type']=='video');a=next(s for s in probe['streams'] if s['codec_type']=='audio')
  assert int(v['nb_read_frames'])==55 and v['r_frame_rate']=='24/1' and (v['width'],v['height'])==(1080,1920)
  assert abs(float(v['duration'])-55/24)<1e-5
  assert abs(float(a['duration'])-55/24)<1024/48000+.001
  frames=json.loads(run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(p)]).stdout)['frames'];ts=np.array([float(f['best_effort_timestamp_time']) for f in frames]);assert abs(ts[0])<1e-6 and np.max(abs(np.diff(ts)-1/24))<2e-6
  run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',str(p),'-map','0:v:0','-map','0:a:0','-f','null','-'])
  pcm=run(['ffmpeg','-nostdin','-v','error','-i',str(p),'-map','0:a:0','-f','s16le','-ac','2','-ar','48000','-']).stdout
  row={'file':str(p.relative_to(ROOT)),'frames':55,'fps':24,'dimensions':[v['width'],v['height']],'video_duration':float(v['duration']),'audio_duration':float(a['duration']),'audio_start':float(a.get('start_time',0)),'video_start':float(v.get('start_time',0)),'constant_frame_timestamps':True,'full_av_decode':'passed','file_sha256':sha(p),'decoded_audio_sha256':hashlib.sha256(pcm).hexdigest(),'decoded_audio_samples':len(pcm)//4}
  baseline=ROOT/'runs/sequential_v2'/p.parent.name/p.name
  if baseline.exists() and baseline!=p:
   q=run(['ffmpeg','-nostdin','-hide_banner','-i',str(baseline),'-i',str(p),'-filter_complex','[0:v][1:v]ssim','-an','-f','null','-']).stderr.decode()
   import re
   m=re.search(r'All:([\d.]+)',q);row['ssim_vs_sequential']=float(m[1]) if m else None
  output.append(row)
 (root/'validation.json').write_text(json.dumps(output,indent=2));print(json.dumps({'run':root.name,'validated':len(output),'ssim':[x.get('ssim_vs_sequential') for x in output]}))
 return output
if __name__=='__main__':
 import sys
 validate(Path(sys.argv[1]))

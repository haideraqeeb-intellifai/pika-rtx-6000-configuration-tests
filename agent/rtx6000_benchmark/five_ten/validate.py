"""CPU-only full-video integrity and timing verification after GPU tests."""
import hashlib,json,subprocess,re
from pathlib import Path
import numpy as np
from scipy.signal import correlate,correlation_lags
P=Path(__file__).resolve().parent

def call(cmd):return subprocess.run(cmd,check=True,capture_output=True,timeout=300)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pcm(p):return np.frombuffer(call(['ffmpeg','-v','error','-i',str(p),'-vn','-ac','1','-ar','8000','-f','f32le','-']).stdout,dtype=np.float32)
def validate(root):
 rows=[];ref=pcm(P/'inputs/audio_master.wav');cache_path=P/'validation_cache.json';cache=json.loads(cache_path.read_text()) if cache_path.exists() else {};reference_sha=sha(P/'inputs/audio_master.wav')
 for d in sorted(root.glob('repeat_[123]')):
  for p in sorted(d.glob('*.mp4')):
   if p.name=='original_assembly.mp4':continue
   full=p.name=='video.mp4';expected=1200 if full else 240
   digest=sha(p);key=reference_sha+':'+digest
   if key in cache:
    row=dict(cache[key]);assert row['frames']==expected
    row['validation_reused_from_identical_sha256']=row['file'];row['file']=str(p.relative_to(P));rows.append(row);continue
   probe=json.loads(call(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(p)]).stdout)
   v=next(s for s in probe['streams'] if s['codec_type']=='video');a=next(s for s in probe['streams'] if s['codec_type']=='audio')
   assert int(v['nb_read_frames'])==expected and v['r_frame_rate']=='24/1' and (v['width'],v['height'])==(1080,1920),p
   assert abs(float(v['duration'])-expected/24)<1e-5 and abs(float(a['duration'])-expected/24)<1024/48000+.001,p
   frames=json.loads(call(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(p)]).stdout)['frames'];ts=np.array([float(f['best_effort_timestamp_time']) for f in frames]);assert abs(ts[0])<1e-6 and np.max(abs(np.diff(ts)-1/24))<2e-6
   call(['ffmpeg','-v','error','-i',str(p),'-map','0:v:0','-map','0:a:0','-f','null','-'])
   row={'file':str(p.relative_to(P)),'frames':expected,'fps':24,'video_duration':float(v['duration']),'audio_duration':float(a['duration']),'audio_start':float(a.get('start_time',0)),'constant_frame_timestamps':True,'full_av_decode':'passed','sha256':digest}
   x=pcm(p);row['decoded_audio_sha256']=hashlib.sha256(x.tobytes()).hexdigest()
   if full:
    n=min(len(x),len(ref));c=correlate(x[:n],ref[:n],mode='full',method='fft');lags=correlation_lags(n,n);valid=abs(lags)<=8000;lag=int(lags[valid][np.argmax(c[valid])]);row.update(audio_lag_ms=lag/8,audio_zero_lag_correlation=float(np.corrcoef(x[:n],ref[:n])[0,1]));assert lag==0 and row['audio_zero_lag_correlation']>.99
   cache[key]=dict(row);cache_path.write_text(json.dumps(cache,indent=2));rows.append(row)
  print(root.name,d.name,'validated',flush=True)
 (root/'validation.json').write_text(json.dumps(rows,indent=2));return rows
if __name__=='__main__':
 import sys
 validate(P/'runs'/sys.argv[1])

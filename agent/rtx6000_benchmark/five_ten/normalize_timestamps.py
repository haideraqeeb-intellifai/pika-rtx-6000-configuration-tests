"""Lossless delivery repair after timing; preserve original measured MP4s."""
import hashlib,json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def streams(p):return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(p)]))['streams']
def payload(p,kind):return subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-map','0:'+kind+':0','-c','copy','-f','hash','-']).decode().strip()
if __name__=='__main__':
 out=[]
 for name in ['sequential_1t2v','batch_compact_1t2v','concurrent_2t1v','concurrent_2t2v','batch_1t2v']:
  root=P/'runs'/name;timings={x['repeat']:x['seconds'] for x in json.loads((root/'results.json').read_text())}
  for rep in range(1,4):
   d=root/f'repeat_{rep}';video=d/'video.mp4';original=d/'original_assembly/video.mp4';record=d/'timestamp_normalization.json'
   if record.exists():out.append(json.loads(record.read_text()));continue
   assert not original.exists();original.parent.mkdir();video.rename(original)
   before=streams(original);cmd=['ffmpeg','-v','error','-y','-i',str(original),'-map','0:v:0','-map','0:a:0','-c','copy','-bsf:v','setts=pts=PTS-STARTPTS:dts=DTS-STARTPTS','-avoid_negative_ts','disabled','-movflags','+faststart',str(video)]
   t=time.perf_counter();subprocess.run(cmd,check=True);elapsed=time.perf_counter()-t
   after=streams(video)
   for stream in after:assert abs(float(stream['start_time']))<1e-6 and abs(float(stream['duration'])-50)<1e-6
   same={k:payload(original,k)==payload(video,k) for k in ['v','a']};assert all(same.values())
   row={'configuration':name,'repeat':rep,'original_video_start_seconds':float(next(x for x in before if x['codec_type']=='video')['start_time']),'normalization_seconds':elapsed,'measured_generation_seconds':timings[rep],'combined_generation_plus_separately_timed_normalization_seconds':timings[rep]+elapsed,'original_sha256':sha(original),'delivered_sha256':sha(video),'compressed_stream_payloads_unchanged':same,'command':cmd,'note':'Timestamp-only remux after measurement. Video/audio packets preserved, no inference or encoding repeated. Verification cost excluded.'}
   record.write_text(json.dumps(row,indent=2));out.append(row);print(name,rep,round(elapsed,4),flush=True)
 (P/'timestamp_normalization.json').write_text(json.dumps(out,indent=2))

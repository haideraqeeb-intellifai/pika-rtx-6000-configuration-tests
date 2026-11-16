"""Publish only report content; keep source code and execution logs local."""
import csv,html,json,re,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'h3-pika-flex-chunk-copy-configurations'
DEST=ROOT/'publication'/'h3-pika-flex-chunk-copy-configurations'

def write(relative,value):
 p=DEST/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2))
def main():
 DEST.mkdir(parents=True,exist_ok=True)
 for folder in ('media','inputs'):
  (DEST/folder).mkdir(exist_ok=True)
 for p in (SOURCE/'media').glob('*.mp4'):shutil.copy2(p,DEST/'media'/p.name)
 for name in ('raccoon_headshot.jpg','report_audio.wav'):shutil.copy2(SOURCE/'inputs'/name,DEST/'inputs'/name)
 manifest=json.loads((SOURCE/'inputs/manifest.json').read_text())
 write('inputs/manifest.json',{k:v for k,v in manifest.items() if k in ('audio_note','files')})
 for name in ('configuration_matrix.csv','per_node_times.csv','quality.json','eight_gpu_estimates.json'):
  shutil.copy2(SOURCE/name,DEST/name)
 full=json.loads((SOURCE/'measurements.json').read_text())
 fields=['name','status','recorded_status','error','schedule','transformers','vaes','chunks','chunk_frames',
 'reference','anchor_mode','residency','async_encoding','cache_text','mps','mps_percentages',
 'preparation_seconds','encoder_release_gate_seconds','parallel_generation_seconds',
 'assembly_and_full_validation_seconds','time_to_assembled_valid_video_seconds','timed_total_seconds',
 'gpu_trace_summary','calibration_peak_gib','conservative_memory_envelope_gib','sampler_overlap',
 'decode_overlap','inference_overlap','background_encode_decode_overlap_seconds','output_sha256',
 'complete_video_audio_decode','output_probe','audio_check','seam_check','latent_handoff_bytes']
 runs=[{k:r[k] for k in fields if k in r} for r in full['runs']]
 write('measurements.json',{'summary':full['summary'],'runs':runs,'failed_or_rejected':[r for r in runs if r['status']!='success'],
 'publication_note':'Source code, workflow graphs, detailed execution logs and process identifiers are retained in the local reproducibility bundle.'})
 for r in runs:
  write('data/'+r['name']+'/result.json',r)
  image=SOURCE/'data'/r['name']/'contact_sheet.jpg'
  if image.exists():shutil.copy2(image,DEST/'data'/r['name']/'contact_sheet.jpg')
 provenance=json.loads((SOURCE/'data/followup_provenance.json').read_text())
 write('data/followup_provenance.json',{k:v for k,v in provenance.items() if k in ('git_revision','protocol','hardware','packages','assets')})
 env=json.loads((SOURCE/'data/followup_resume_environment.json').read_text())
 write('data/followup_resume_environment.json',{k:env[k] for k in ('recorded_utc','python','platform','cpu','memory','gpu_hardware','ffmpeg_version','source_revision')})
 audit=json.loads((SOURCE/'data/followup_resume_audit.json').read_text())
 write('data/followup_resume_audit.json',{'note':audit['note'],'runs':[{k:r[k] for k in ('name','samples','status','maximum_sample_gap_seconds')} for r in audit['runs']]})
 (DEST/'protocol.txt').write_text((ROOT/'CHUNK_COPY_PLAN.md').read_text())
 page=(SOURCE/'index.html').read_text().replace('href="reproducibility.zip"','href="results.zip"').replace('Download code, inputs and raw measurements','Download results and configuration')
 page=page.replace('href="code/CHUNK_COPY_PLAN.md.txt"','href="protocol.txt"')
 page=page.replace('Run the included harness from the script2video repository root using the recorded environment and model assets.', 'The full runnable harness and execution logs are in the local reproducibility bundle linked separately in the chat. This shared report contains measured results, media and configuration details; source code was not uploaded.')
 (DEST/'index.html').write_text(page)
 with zipfile.ZipFile(DEST/'results.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in DEST.rglob('*'):
   if f.is_file() and f.suffix in ('.json','.csv','.txt'):z.write(f,f.relative_to(DEST))
 files=sorted(str(f.relative_to(DEST)) for f in DEST.rglob('*') if f.is_file())
 if 'report-files.json' not in files:files.append('report-files.json')
 write('report-files.json',files)
 print(json.dumps({'directory':str(DEST),'files':len(files),'source_code_files':0,'execution_logs':0,'bytes':sum((DEST/f).stat().st_size for f in files)},indent=2))
if __name__=='__main__':main()

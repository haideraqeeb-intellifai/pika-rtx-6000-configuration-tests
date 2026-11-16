"""Reuse established full-frame QA only after verifying byte-identical artifacts."""
import json
from .parallel_chunk_processes import RUNS,file_hash

def main():
    path=RUNS/'followup_quality.json';q=json.loads(path.read_text())
    existing={r['name'] for r in q['results']};bysha={r['verified_output_sha256']:r for r in q['results']}
    paths=list(RUNS.glob('followup_capacity_*/result.json'))+[RUNS/'followup_mps_3t_2v_staged/result.json']
    for p in paths:
        r=json.loads(p.read_text())
        if r.get('loader')!='separate stage processes' or r['status']!='success' or r['name'] in existing:continue
        sha=file_hash(p.parent/'output.mp4');assert sha==r['output_sha256']
        reference=bysha[sha];assert reference['decoded_frames']==1186 and r['complete_video_audio_decode']
        latents=list((p.parent/'latents').glob('*.sha256'))
        entry={**reference,'name':r['name'],'full_frame_hash_reused_from_identical_sha256':reference['name'],'audio_check':r['audio_check'],'seam_check':r['seam_check'],'latent_files':len(latents),'identical_latent_files':sum(f.read_text()==(RUNS/reference['chunk_plan_reference']/'latents'/f.name).read_text() for f in latents)}
        q['results'].append(entry)
        print('Verified byte-identical video and reused full-frame QA:',r['name'])
    path.write_text(json.dumps(q,indent=2)+'\n')
if __name__=='__main__':main()

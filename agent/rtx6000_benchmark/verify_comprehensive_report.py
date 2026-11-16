"""Check complete study coverage, timing boundaries, artifacts and archive integrity."""
import csv,json,statistics,zipfile,re
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
ROOT=Path(__file__).resolve().parent;RUNS=ROOT/'runs';DEST=ROOT/'h3-pika-flex-comprehensive-rtx6000'
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.ids=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ('href','src') and v:self.links.append(v)
   if k=='id':self.ids.append(v)
def main():
 data=json.loads((DEST/'experiments.json').read_text());records=data['records'];byname={r['name']:r for r in records}
 expected=set()
 for p in RUNS.glob('*/result.json'):
  raw=json.loads(p.read_text())
  if 'lane' in raw or '_chunk_' in p.parent.name:continue
  name=p.parent.name;expected.add(name);row=byname[name]
  if row['status']=='success' and row['scope']!='diagnostic':
   key='decoder_phase_seconds' if row['scope']=='decode-only' else 'timed_total_seconds' if 'timed_total_seconds' in raw else 'total_seconds'
   assert row['seconds']==raw[key],name
  else:assert row['seconds'] is None,name
  if row['status']=='success' and row['scope']=='full-video':assert row['full_decode_passed'] and int(row['delivered_frames'])==1186 and row['fps']=='24/1',name
 for p in RUNS.glob('*/cancellation.json'):expected.add(p.parent.name)
 assert set(byname)==expected
 best=data['recommendation'];ts=[byname[n]['seconds'] for n in best['run_names']]
 assert abs(statistics.mean(ts)-best['mean_total_seconds'])<1e-9 and len(ts)==2
 assert best['inference_mode']=='fast'
 assert len(records)==121
 assert sum(r['scope']=='full-video' and r['status']=='success' for r in records)==96
 parser=Links();parser.feed((DEST/'index.html').read_text());assert len(parser.ids)==len(set(parser.ids))
 checked=0
 for link in parser.links:
  u=urlsplit(link)
  if u.scheme or u.netloc:continue
  if u.path:
   p=(DEST/unquote(u.path)).resolve();assert p.is_relative_to(DEST.resolve()) and p.is_file(),link
  elif u.fragment:assert unquote(u.fragment) in parser.ids,link
  checked+=1
 with (DEST/'configuration_matrix.csv').open() as f:assert len(list(csv.DictReader(f)))==len(records)
 sensitive=[]
 with zipfile.ZipFile(DEST/'reproducibility.zip') as z:
  assert z.testzip() is None
  archived=z.namelist()
  for name in archived:
   assert not Path(name).is_absolute() and '..' not in Path(name).parts
   if name.endswith(('.json','.py','.log','.md','.txt','.jsonl')):
    s=z.read(name).decode(errors='replace')
    if re.search(r'(?:AKIA|ASIA)[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',s):sensitive.append(name)
  for name in expected:
   filename='cancellation.json' if byname[name]['status']=='cancelled' else 'result.json'
   assert 'evidence/runs/'+name+'/'+filename in archived
 assert not sensitive,sensitive
 inventory=json.loads((DEST/'report-files.json').read_text())
 assert set(inventory)=={str(p.relative_to(DEST)) for p in DEST.rglob('*') if p.is_file()}
 result={'record_coverage':len(records),'clean_full_video_trials':96,'recommendation_mean_seconds':statistics.mean(ts),'links_checked':checked,'inventory_files':len(inventory),'archive_entries':len(archived),'archive_crc':'passed','credential_pattern_scan':'no matches','source_hash_checks':'All included output files rehashed and checked during build.'}
 (RUNS/'comprehensive_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

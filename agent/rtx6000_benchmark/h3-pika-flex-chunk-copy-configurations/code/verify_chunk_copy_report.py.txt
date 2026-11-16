"""CPU-only report arithmetic, asset and inventory verification."""
import csv,hashlib,json,statistics,zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit,unquote
ROOT=Path(__file__).resolve().parent
DEST=ROOT/'h3-pika-flex-chunk-copy-configurations'
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('href','src') and value:self.links.append(value)
def main():
    parser=Links();parser.feed((DEST/'index.html').read_text())
    local=[]
    for link in parser.links:
        uri=urlsplit(link)
        if uri.scheme or uri.netloc or not uri.path:continue
        f=(DEST/unquote(uri.path)).resolve()
        assert f.is_relative_to(DEST.resolve()) and f.is_file(),link
        local.append(str(f.relative_to(DEST)))
    data=json.loads((DEST/'measurements.json').read_text());runs={r['name']:r for r in data['runs']}
    for group in data['summary']:
        samples=[runs[n] for n in group['run_names']]
        assert all(r['status']=='success' for r in samples)
        assert abs(statistics.mean(r['timed_total_seconds'] for r in samples)-group['mean_total_seconds'])<1e-8
        assert abs(statistics.mean(r['time_to_assembled_valid_video_seconds'] for r in samples)-group['mean_video_ready_seconds'])<1e-8
        assert group['samples']==len(samples)
    inventory=json.loads((DEST/'report-files.json').read_text())
    assert set(inventory)=={str(p.relative_to(DEST)) for p in DEST.rglob('*') if p.is_file()}
    with zipfile.ZipFile(DEST/'reproducibility.zip') as z:
        assert z.testzip() is None
        for name in z.namelist():assert '..' not in Path(name).parts and not Path(name).is_absolute()
    result={'local_links_checked':len(local),'summary_groups_checked':len(data['summary']),
       'inventory_files_checked':len(inventory),'zip_integrity':'passed',
       'excluded_runs':[r['name'] for r in data['failed_or_rejected']]}
    (ROOT/'runs/followup_report_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()

"""Check the published comprehensive report and representative signed assets."""
import html
from html.parser import HTMLParser
import json
import re
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs'
log = (RUNS / 'comprehensive_fast_update_upload.log').read_text()
url = re.findall(r'SHARE_URL\tindex.html\t(\S+)', log)[-1]
expiry = re.findall(r'SHARE_EXPIRES_AT_UTC\t(\S+)', log)[-1]
with urlopen(url, timeout=60) as response:
    assert response.status == 200
    page = response.read().decode()
assert '200.3 seconds' in page and '121 recorded attempts' in page and 'id="fast"' in page

class Links(HTMLParser):
    urls = []
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src') and value.startswith('https://'):
                self.urls.append(value)

links = Links(); links.feed(page)
def asset(suffix):
    return next(u for u in links.urls if urlsplit(u).path.endswith(suffix))
with urlopen(asset('/experiments.json'), timeout=60) as response:
    data = json.load(response)
assert len(data['records']) == 121
assert data['recommendation']['inference_mode'] == 'fast'
assert abs(data['recommendation']['mean_total_seconds'] - 200.34568004599714) < 1e-8
checks = {}
for suffix in ('cf3a71d5750aa952febbb5673aab3dae49456410bee29d5f72529af2f4ed8a67.mp4', '/fast_generation_reproduce.zip'):
    with urlopen(Request(asset(suffix), headers={'Range': 'bytes=0-1023'}), timeout=60) as response:
        assert response.status == 206
        assert len(response.read()) == 1024
        checks[suffix] = 'HTTP 206, 1024-byte range verified'
result = {'status': 'published_and_verified', 'url': url, 'expires_at_utc': expiry,
          'destination': 's3://agent-assets/experiments/raqeeb/h3-pika-flex-comprehensive-rtx6000/',
          'records': 121, 'recommendation_mean_seconds': data['recommendation']['mean_total_seconds'],
          'assets': checks}
(RUNS / 'comprehensive_publication_status.json').write_text(json.dumps(result, indent=2))
(RUNS / 'latest_comprehensive_report_url.txt').write_text(url + '\n')
print(json.dumps(result, indent=2))

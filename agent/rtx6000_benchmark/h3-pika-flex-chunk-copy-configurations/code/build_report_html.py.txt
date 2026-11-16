"""Build a local, self-contained HTML report; no upload or deployment."""

import base64
import csv
import html
import io
import subprocess
import sys
from pathlib import Path

from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parent


def main():
    markdown = (ROOT / 'REPORT.md').read_text()
    marker = '## Detailed evidence\n'
    if markdown.count(marker) != 1:
        raise ValueError('REPORT.md must contain one detailed-evidence divider')
    overview, evidence = markdown.split(marker, 1)
    renderer = MarkdownIt('commonmark').enable('table')
    overview_html = renderer.render(overview)
    evidence_html = renderer.render(marker + evidence)
    matrix = subprocess.run([sys.executable, str(ROOT / 'summarize_runs.py')],
                            check=True, capture_output=True, text=True).stdout
    rows = list(csv.DictReader(io.StringIO(matrix)))
    preferred = ['name', 'status', 'chunks', 'chunk_plan', 'reference', 'anchor_mode',
                 'previous_last_frame_dependency_valid', 'transformers', 'vaes',
                 'texts', 'clip_device', 'residency', 'vae_decode', 'frame_storage',
                 'encoding_overlap', 'end_to_end_s', 'condition_s', 'denoise_s',
                 'vae_decode_s', 'frame_store_s', 'encode_s', 'encode_final_wait_s',
                 'peak_vram_gib', 'gpu_util_mean_pct', 'gpu_energy_wh_est',
                 'inferred_frames', 'delivered_frames']
    table = '<table><thead><tr>' + ''.join(f'<th>{html.escape(key)}</th>' for key in preferred) + '</tr></thead><tbody>'
    for row in rows:
        table += '<tr>' + ''.join(f'<td>{html.escape(str(row.get(key, "")))}</td>' for key in preferred) + '</tr>'
    table += '</tbody></table>'
    sheets = []
    for name in ('chained_4_baseline', 'chained_4_aligned_overlap_encode',
                 'chained_7_aligned_overlap_encode', 'chained_10_aligned',
                 'chained_10_aligned_anchor_every2_overlap',
                 'chained_10_aligned_anchor_every3_overlap',
                 'chained_10_aligned_first_anchor_overlap', 'chained_14_aligned'):
        path = ROOT / 'runs' / name / 'contact_sheet.jpg'
        if path.exists():
            encoded = base64.b64encode(path.read_bytes()).decode('ascii')
            sheets.append(f'<figure><figcaption>{html.escape(name)}</figcaption>'
                          f'<img alt="Boundary contact sheet for {html.escape(name)}" '
                          f'src="data:image/jpeg;base64,{encoded}"></figure>')
    chart = '''<section class="chart" aria-label="Measured one-GPU time per complete video">
<h2>One GPU: time per finished video</h2><p>Shorter bars are faster. The last row changes the generation policy.</p>
<div class="barrow"><span>4 chunks · original</span><div class="track"><div class="bar" style="width:100%"></div></div><strong>306 s</strong></div>
<div class="barrow"><span>7 chunks · aligned</span><div class="track"><div class="bar" style="width:85.1%"></div></div><strong>260 s</strong></div>
<div class="barrow"><span>10 chunks · aligned</span><div class="track"><div class="bar best" style="width:82.1%"></div></div><strong>251 s</strong></div>
<div class="barrow"><span>10 chunks · alternate anchor</span><div class="track"><div class="bar alternate" style="width:78.9%"></div></div><strong>241 s</strong></div>
<p class="chartnote">All values are measured on one RTX PRO 6000. Four and seven chunks have one run each; both ten-chunk rows are medians of two runs.</p>
</section>'''
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>H3 Pika Flex on an RTX PRO 6000 | Configuration study</title>
<style>
:root{{color-scheme:light;font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;line-height:1.65}}
*{{box-sizing:border-box}}body{{margin:0;background:#f5f7fb;color:#172438}}main{{max-width:1000px;margin:auto;padding:28px 24px 80px}}
h1,h2,h3{{line-height:1.2;color:#10233a}}h1{{font-size:clamp(2rem,5vw,3rem);margin:.3em 0}}h2{{font-size:1.45rem;margin-top:2.5rem;border-bottom:1px solid #d8e1eb;padding-bottom:.35rem}}
h3{{font-size:1.12rem}}p,li{{max-width:78ch}}p{{margin:1em 0}}a{{color:#125c9d}}a:hover{{text-decoration-thickness:2px}}
code{{background:#eaf0f6;padding:.13em .3em;border-radius:3px}}pre{{background:#eaf0f6;padding:16px;border-radius:8px;overflow:auto;font-size:.75rem}}
table{{border-collapse:collapse;font-size:.91rem;width:100%}}th,td{{border-bottom:1px solid #d7e0e9;padding:11px 12px;vertical-align:top;text-align:left}}th{{background:#e7eef5;color:#233a50}}
tbody tr:nth-child(even){{background:#f5f8fb}}.scroll{{overflow:auto;max-height:640px;border:1px solid #d7e0e9;border-radius:8px}}
.scroll table{{min-width:1850px}}.scroll th{{position:sticky;top:0}}
.overview{{background:white;border:1px solid #dce5ee;border-radius:14px;padding:28px clamp(18px,4vw,44px);box-shadow:0 4px 25px #18395b0b}}
.overview>h1+p{{font-size:1.1rem;color:#51667c;margin-top:0}}.overview>h2:first-of-type{{margin-top:2rem}}
.chart{{background:#eaf3fc;border:1px solid #cddfed;border-radius:14px;padding:22px 26px;margin:24px 0}}
.chart h2{{margin-top:0;border:0}}.barrow{{display:grid;grid-template-columns:220px minmax(100px,1fr) 56px;gap:12px;align-items:center;margin:12px 0;font-size:.88rem}}
.track{{height:22px;background:#d9e4ef;border-radius:5px;overflow:hidden}}.bar{{height:100%;background:#6485a6;border-radius:5px}}.bar.best{{background:#1671b9}}.bar.alternate{{background:#8b65b3}}
.barrow strong{{font-variant-numeric:tabular-nums}}.chartnote{{font-size:.84rem;color:#4f6174}}
details{{margin:18px 0;padding:0 20px;border:1px solid #d5e0eb;border-radius:10px;background:#fff}}
summary{{cursor:pointer;font-weight:700;padding:17px 0;color:#183d5e}}details[open]{{padding-bottom:20px}}
figure{{margin:22px 0}}figcaption{{font-weight:700;margin-bottom:8px}}img{{display:block;max-width:100%;height:auto;border-radius:6px}}
.eyebrow{{text-transform:uppercase;letter-spacing:.12em;color:#476c8e;font-size:.75rem;font-weight:800;margin:0 0 8px}}
@media(max-width:680px){{main{{padding:14px 12px 50px}}.overview{{padding:18px}}.barrow{{grid-template-columns:1fr 52px;gap:4px 8px}}.barrow span{{grid-column:1/-1}}.chart{{padding:16px}}}}
</style></head><body><main><p class="eyebrow">Measured on one GPU · Eight-GPU numbers are estimates</p>
<section class="overview">{overview_html}</section>
{chart}
<details id="evidence"><summary>Detailed evidence, methods, and limitations</summary>{evidence_html}</details>
<details id="matrix"><summary>Full configuration matrix: timings, memory, utilization, and outcomes</summary><p>Each row comes from a raw result JSON. Failed cases have no valid output.</p><div class="scroll">{table}</div></details>
<details><summary>Boundary-frame contact sheets</summary><p>These show the frames immediately before and after chunk joins, plus final frames, so you can judge continuity yourself.</p>{''.join(sheets)}</details>
<details><summary>Machine-readable CSV</summary><pre>{html.escape(matrix)}</pre></details>
</main></body></html>'''
    (ROOT / 'report.html').write_text(document)
    (ROOT / 'configuration_matrix.csv').write_text(matrix)
    print(ROOT / 'report.html')


if __name__ == '__main__':
    main()

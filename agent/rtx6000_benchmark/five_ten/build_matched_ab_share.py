"""Build a compact, self-contained share package for the matched RTX 6000 A/B."""

import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "matched_ab_share"


def bar(label, value, maximum, color, suffix, note=""):
    width = 100 * value / maximum
    return (f'<div class="barrow"><div class="barlabel">{html.escape(label)}</div>'
            f'<div class="track"><span style="width:{width:.3f}%;background:{color}"></span></div>'
            f'<strong>{value:.2f}{suffix}</strong>{f"<small>{html.escape(note)}</small>" if note else ""}</div>')


def main():
    s = json.loads((ROOT / "matched_ab_summary.json").read_text())
    p = json.loads((ROOT / "matched_ab_profiles.json").read_text())
    v = json.loads((ROOT / "matched_ab_validation.json").read_text())
    a, b = s["simple"], s["parallel"]
    assert not s["threshold_met"] and v["all_final_mp4_bytes_equal"]
    OUT.mkdir(exist_ok=True)
    for name in ("matched_ab_summary.json", "matched_ab_validation.json", "matched_ab_profiles.json",
                 "matched_ab_provenance.json", "matched_ab_surya.txt", "experiment.py", "validate.py"):
        shutil.copy2(ROOT / name, OUT / name)
    for group in ("matched_simple_a2", "matched_simple_a3", "matched_parallel_b2", "matched_parallel_b1"):
        target = OUT / "raw" / group
        target.mkdir(parents=True, exist_ok=True)
        for name in ("result.json", "gpu_trace.json", "workers.json"):
            shutil.copy2(ROOT / "runs" / group / name, target / name)
    # Every measured final MP4 has the same SHA256. One representative preserves visual review.
    shutil.copy2(ROOT / a["outputs"][0], OUT / "representative_50s.mp4")
    assert s["all_output_bytes_equal"]
    target = 0.8 * a["mean_seconds"]
    repeats = "".join(
        f'<tr><th scope="row">{i}</th><td>{a["seconds"][i-1]:.3f}</td><td>{b["seconds"][i-1]:.3f}</td></tr>'
        for i in (1, 2, 3))
    raw_links = "".join(
        f'<li><a href="raw/{name}/result.json">{name} measurements</a> · '
        f'<a href="raw/{name}/gpu_trace.json">GPU trace</a> · '
        f'<a href="raw/{name}/workers.json">worker PIDs</a></li>'
        for name in ("matched_simple_a2", "matched_parallel_b2", "matched_simple_a3", "matched_parallel_b1"))
    sm_a = p["matched_simple"]["metrics"]["SMs Active [Throughput %]"]["mean"]
    sm_b = p["matched_parallel"]["metrics"]["SMs Active [Throughput %]"]["mean"]
    overlap = p["matched_parallel"]["kernel_overlap"]["transformer_vae_kernel_overlap_seconds"]
    latency = (bar("A · simple", a["mean_seconds"], 240, "#4361a5", " s") +
               bar("B · parallel", b["mean_seconds"], 240, "#0a846c", " s") +
               bar("20% target for B", target, 240, "#af6712", " s", "B must be at or below this"))
    memory = (bar("A · simple", a["peak_vram_gib"], 64, "#4361a5", " GiB") +
              bar("B · parallel", b["peak_vram_gib"], 64, "#0a846c", " GiB"))
    activity = (bar("A · simple", sm_a, 100, "#4361a5", "%") +
                bar("B · parallel", sm_b, 100, "#0a846c", "%"))
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RTX PRO 6000 · Pika Flex fast A/B</title>
<style>
:root{{--ink:#182535;--muted:#536273;--line:#dbe3ea;--a:#4361a5;--b:#0a846c;--warm:#af6712;--bg:#f5f8fb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);font:16px/1.55 system-ui,-apple-system,sans-serif;color:var(--ink)}}
main{{max-width:980px;margin:auto;padding:28px 20px 70px}}h1{{font-size:clamp(2rem,4vw,3rem);line-height:1.12;margin:7px 0 14px}}h2{{font-size:1.35rem;margin:0 0 12px}}h3{{font-size:1.08rem;margin:0 0 12px}}
p{{margin:8px 0 14px}}.eyebrow{{color:var(--b);font-weight:750;letter-spacing:.08em;text-transform:uppercase;font-size:.77rem}}.intro{{font-size:1.12rem;color:var(--muted);max-width:720px}}
.hero{{background:#fff;border:1px solid var(--line);border-radius:18px;padding:26px;margin:24px 0;box-shadow:0 8px 24px #20344a0b}}.decision{{font-size:1.2rem;font-weight:720;line-height:1.4;max-width:760px}}
.stats{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:22px 0}}.stat{{background:#eaf3f0;border-radius:12px;padding:16px}}.stat:nth-child(2){{background:#edf2fa}}.stat:nth-child(3){{background:#fff2e5}}.stat b{{display:block;font-size:1.75rem;line-height:1.2}}.stat span{{font-size:.87rem;color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}section.card{{background:#fff;border:1px solid var(--line);border-radius:16px;padding:22px;margin-bottom:18px}}.wide{{grid-column:1/-1}}
.barrow{{display:grid;grid-template-columns:140px minmax(80px,1fr) 80px;align-items:center;gap:10px;margin:14px 0}}.barlabel{{font-size:.92rem}}.barrow strong{{text-align:right;font-variant-numeric:tabular-nums;font-size:.9rem}}.barrow small{{grid-column:2/-1;color:var(--muted);margin-top:-7px}}.track{{height:20px;border-radius:4px;background:#ecf0f4;overflow:hidden}}.track span{{display:block;height:100%;border-radius:4px}}
.caption{{font-size:.86rem;color:var(--muted);margin-top:12px}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{border-bottom:1px solid var(--line);padding:9px;text-align:left}}th{{font-weight:650}}td{{text-align:right}}.tablewrap{{overflow-x:auto}}.tiny{{font-size:.88rem;color:var(--muted)}}
video{{width:100%;max-width:660px;border-radius:10px;background:#101820}}a{{color:#15577d}}details{{margin-top:12px}}summary{{font-weight:680;cursor:pointer}}code{{font-size:.84em}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f7;padding:14px;border-radius:8px;font-size:.83rem}}ul{{padding-left:20px}}li{{margin:5px 0}}
@media(max-width:700px){{.grid,.stats{{grid-template-columns:1fr}}.barrow{{grid-template-columns:100px minmax(55px,1fr) 72px;gap:7px}}main{{padding:20px 14px 50px}}section.card,.hero{{padding:18px}}}}
</style></head><body><main>
<div class="eyebrow">Matched single-GPU benchmark · RTX PRO 6000</div>
<h1>Is the parallel pipeline worth it?</h1>
<p class="intro">Five 10-second chunks of Pika Flex fast mode, measured through the correctly timed final MP4. Three warmed full-video runs per configuration.</p>
<div class="hero"><div class="eyebrow">Decision</div>
<p class="decision">The parallel pipeline saves {s['seconds_saved']:.2f} seconds ({s['time_reduction_pct']:.1f}%), below Surya’s roughly 20% latency reduction bar. For this workload, the extra MPS, VAE worker and pipeline maintenance are not justified.</p>
<div class="stats"><div class="stat"><b>{a['mean_seconds']:.2f} s</b><span>A · simple pipeline</span></div><div class="stat"><b>{b['mean_seconds']:.2f} s</b><span>B · parallel pipeline</span></div><div class="stat"><b>+{b['peak_vram_gib']-a['peak_vram_gib']:.2f} GiB</b><span>B’s extra peak VRAM</span></div></div>
<p class="tiny">B is {s['speedup_factor']:.3f}× faster. The 20% bar means B ≤ {target:.2f} s, equal to 1.25× throughput at this fixed workload.</p></div>
<div class="grid"><section class="card wide"><h2>Generation time</h2>{latency}<p class="caption">Seconds from first chunk dispatch through final 50-second MP4 assembly and lossless timestamp correction. Lower is better. Bars start at zero.</p></section>
<section class="card"><h2>Peak GPU memory</h2>{memory}<p class="caption">Peak device VRAM across measured repeats. B uses two resident VAE workers; A uses one.</p></section>
<section class="card"><h2>GPU activity</h2>{activity}<p class="caption">Mean SM activity across separate Nsight Systems capture windows. Profiler runtimes are excluded from latency results.</p></section></div>
<section class="card"><h2>All six measured runs</h2><div class="tablewrap"><table><thead><tr><th>Repeat</th><th>A · simple (s)</th><th>B · parallel (s)</th></tr></thead><tbody>{repeats}<tr><th>Mean</th><td>{a['mean_seconds']:.3f}</td><td>{b['mean_seconds']:.3f}</td></tr><tr><th>Sample SD</th><td>{a['stdev_seconds']:.3f}</td><td>{b['stdev_seconds']:.3f}</td></tr></tbody></table></div><p class="caption">Order: A warmup + 2 measured, B warmup + 2, A warmup + 1, B warmup + 1. Warmups are excluded.</p></section>
<section class="card"><h2>What changed?</h2><p><b>A · simple:</b> one resident transformer, one resident VAE, no MPS. One job runs five chunks serially, with no cross-chunk stage overlap.</p><p><b>B · parallel:</b> one resident transformer, two resident VAE workers, MPS and cross-chunk transformer/VAE overlap.</p><p>Everything else matched: two-step fast path, prepared conditioning, independent original-image anchors, seeds 9901–9905, dimensions, audio and output encoding. The 49.408-second source audio received the same 0.592 seconds of trailing silence in both.</p></section>
<section class="card"><h2>Why the gain is modest</h2><p>Nsight Systems found <b>{overlap:.2f} s</b> of actual transformer/VAE GPU kernel overlap in B and zero in A. B kept SMs active more often, but overlapping work competed for the same GPU. Its measured end-to-end gain was only {s['seconds_saved']:.2f} s.</p><p class="tiny">B’s host stage calls overlapped for {p['matched_parallel']['host_call_overlap']['transformer_vae_kernel_overlap_seconds']:.2f} s; that is not the same as GPU kernel overlap. Existing Nsight Compute findings were reused; no profiler timing entered the A/B means.</p></section>
<section class="card"><h2>Request and startup time</h2><p>GPU conditioning preparation took <b>{s['preparation_seconds']:.2f} s</b>. Adding this same measured phase gives resident request estimates of <b>{s['request_latency_with_preparation_seconds']['simple']:.2f} s</b> for A and <b>{s['request_latency_with_preparation_seconds']['parallel']:.2f} s</b> for B.</p><p class="tiny">These are phase sums, not separately timed end-to-end requests. Process setup averaged {sum(a['startup_seconds'])/len(a['startup_seconds']):.2f} s (A) and {sum(b['startup_seconds'])/len(b['startup_seconds']):.2f} s (B); lazy model loading happened in excluded full-video warmups, so a pure model-loading subphase was not isolated.</p></section>
<section class="card"><h2>Verified output</h2><p>All six final MP4s were byte-identical. Each has <b>1,200 delivered frames</b> at 24 fps for 50 seconds, with 243 native frames generated per chunk and 240 exported. Frame timestamps, audio start and duration, and full decode were checked; lossless correction left compressed streams unchanged.</p><video controls preload="metadata" src="representative_50s.mp4" aria-label="Representative 50-second output video"></video><p><a href="representative_50s.mp4">Open the 50-second MP4</a>. One file represents all six byte-identical outputs.</p><p class="tiny">Real-time factor: A {s['real_time_factor']['simple']:.3f}×, B {s['real_time_factor']['parallel']:.3f}× (generation seconds / 50). Visual motion and lip sync remain available for human review.</p></section>
<section class="card"><details><summary>Methods, evidence and reproduction</summary><p>Raw metrics, validation and provenance are included with this page. The original Nsight captures remain in the local experiment workspace.</p><ul><li><a href="matched_ab_summary.json">Summary and all repetitions</a></li><li><a href="matched_ab_validation.json">Video/audio validation</a></li><li><a href="matched_ab_profiles.json">Nsight analysis</a></li><li><a href="matched_ab_provenance.json">Hardware and input provenance</a></li><li><a href="experiment.py">Benchmark runner</a> · <a href="validate.py">Validator</a></li>{raw_links}</ul><p>Rerun with the environment and inputs recorded in provenance, first checking GPU availability. Each block includes one warmup; the measured videos follow. Use unused run names.</p><pre>python -m agent.rtx6000_benchmark.five_ten.experiment --mode simple --name rerun_a1 --transformers 1 --vaes 1 --repeats 2
python -m agent.rtx6000_benchmark.five_ten.experiment --mode sequential --name rerun_b1 --transformers 1 --vaes 2 --repeats 2
python -m agent.rtx6000_benchmark.five_ten.experiment --mode simple --name rerun_a2 --transformers 1 --vaes 1 --repeats 1
python -m agent.rtx6000_benchmark.five_ten.experiment --mode sequential --name rerun_b2 --transformers 1 --vaes 2 --repeats 1</pre><p>Historical 20-chunk data were excluded from the matched comparison.</p></details></section>
</main></body></html>'''
    (OUT / "index.html").write_text(page)
    print(OUT)


if __name__ == "__main__":
    main()

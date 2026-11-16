"""Render the measured Nsight investigation with reproducible source links."""
import csv,html,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LABELS={'1t2v_control':'1 transformer + 2 VAEs · control','2t2v':'2 transformers + 2 VAEs','3t1v':'3 transformers + 1 VAE','3t2v_staged':'3 transformers, then 2 VAEs','isolated_1t1v':'1 transformer, then 1 VAE · isolated reference'}
def esc(x):return html.escape(str(x))
def number(x):return f'{x:.1f}'
def friendly(s):
    if 'qk_int_sv_f8' in s:return 'SageAttention 2'
    if 'DefaultGemmWithVisitor' in s:return 'INT8 transformer GEMM' + (' · stream-K' if 'LeanStreamK' in s else ' · tiled')
    if 's16816gemm_relu_f16' in s:return 'FP16 VAE GEMM'
    if 'flash_fwd_kernel' in s:return 'VAE FlashAttention'
    if 'vectorized_layer_norm' in s:return 'VAE layer normalization'
    if 'CUDAFunctor_add' in s:return 'BF16 elementwise addition'
    return s[:110]
def timeline(r):
    chunks=r['chunks'];lanes=sorted(set((x['role'],x['lane']) for x in chunks),key=lambda x:(x[0]!='denoise',x[1]));span=max(x['start_relative_seconds']+x['duration_seconds'] for x in chunks);scale=880/span
    out=[f'<svg viewBox="0 0 1100 {55+len(lanes)*32}" role="img" aria-label="Synchronized sampler and VAE call timeline">']
    for i,(stage,lane) in enumerate(lanes):
        y=22+i*32;out.append(f'<text x="0" y="{y+16}">{"Transformer" if stage=="denoise" else "VAE"} {lane}</text>')
        for c in chunks:
            if (c['role'],c['lane'])!=(stage,lane):continue
            x=140+c['start_relative_seconds']*scale;w=c['duration_seconds']*scale
            out.append(f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="23" rx="3" fill="{"#4e8cf4" if stage=="denoise" else "#33ba9b"}"><title>Chunk {c["index"]}: {c["duration_seconds"]:.2f}s call; {c["kernel_union_seconds"]:.2f}s own GPU kernels</title></rect>')
    for t in range(0,int(span)+1,30):out.append(f'<text x="{140+t*scale}" y="{47+len(lanes)*32}">{t}s</text>')
    out.append('</svg>');return ''.join(out)
def load_ncu():
    out=[]
    for stage in ['denoise','decode']:
        p=ROOT/f'ncu_{stage}.csv'
        if not p.exists():continue
        lines=p.read_text().splitlines();start=next(i for i,l in enumerate(lines) if l.startswith('"ID"'))
        rows=list(csv.DictReader(lines[start:]))
        if 'Metric Name' in rows[0]:
            groups={}
            for r in rows:
                key=(stage,r['ID']);g=groups.setdefault(key,{'stage':stage,'id':r['ID'],'kernel':r['Kernel Name'],'metrics':{}})
                g['metrics'][r['Metric Name']]={'value':r['Metric Value'],'unit':r.get('Metric Unit')}
            out.extend(groups.values())
        else:
            units=next(r for r in rows if not r['ID'])
            for r in rows:
                if not r['ID']:continue
                metrics={k:{'value':v,'unit':units.get(k,'')} for k,v in r.items() if '__' in k and v}
                out.append({'stage':stage,'id':r['ID'],'kernel':r['Kernel Name'],'metrics':metrics})
    return out

def main():
    d=json.loads((ROOT/'analysis.json').read_text());h=d['historical'];ns=d['nsys'];ncu=load_ncu()
    assert len(ns)==4,'All four Nsight captures required'
    for key in ns:
        status=json.loads((ROOT/(key+'_v2_status.json')).read_text())
        assert status['returncode']==0 and status['result_status']=='success' and status['competing_samples']==0
        assert status['output_sha256']==h[key]['output_sha256']
    assert ncu,'NCU evidence required'
    (ROOT/'ncu_summary.json').write_text(json.dumps(ncu,indent=2))
    content=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H3 · Why extra transformers did not help</title><style>
    :root{color-scheme:dark}body{background:#0c1220;color:#dce5f4;font:16px/1.55 system-ui;margin:0}main{max-width:1240px;margin:auto;padding:44px}h1{font-size:38px;line-height:1.15;max-width:900px}h2{margin-top:40px;color:#fff}h3{color:#fff}a{color:#8fbaff}small,.muted{color:#9caec6}table{width:100%;border-collapse:collapse;font-size:14px;margin:20px 0}th,td{text-align:left;padding:12px;border-bottom:1px solid #29364a}th{color:#aabbd0}section,.callout{background:#141e30;border:1px solid #29364a;border-radius:12px;padding:22px;margin:20px 0}.callout{border-left:4px solid #4e8cf4}.tag{font-size:13px;letter-spacing:1px;color:#7edac5}svg{width:100%;font-size:12px;fill:#b5c7e0}code{background:#1e2b40;padding:2px 5px}details{margin:16px 0}summary{cursor:pointer;font-weight:600}li{margin:8px 0}.scroll{overflow:auto}strong{color:white}</style><main>
    <div class="tag">MEASURED PROFILING · RTX PRO 6000 BLACKWELL · NOVEMBER 2026</div>
    <h1>Why extra transformer copies did not make H3 faster</h1>
    <p>Three multi-transformer configurations, one matched control, fresh Nsight Systems captures, and targeted Nsight Compute kernel measurements.</p>
    <div class="callout"><strong>More transformer workers overlapped their requests, but shared the same physical GPU.</strong> Each chunk became slower as more workers competed. Small denoising gains were offset by a longer decoding tail. The staged three-transformer case also lost transformer/VAE overlap.</div>
    <p>Workload: the same 49.4-second podcast, 20 independent original-image anchored chunks, two Pika Flex steps, 24 fps. All compared runs use <strong>default inference mode</strong>; the newer 200.3-second fast result is not a matched comparator. Each capture generated and validated the complete video.</p>
    <h2>1. Original speed measurements</h2><p>These are the original unprofiled runs. Profiler runtimes below are diagnostic and include instrumentation overhead. T = transformer; V = VAE.</p>
    <table><thead><tr><th>Configuration</th><th>End to end</th><th>Denoising call span</th><th>Decode tail after denoising</th><th>Peak VRAM</th></tr></thead><tbody>''']
    for key in LABELS:
        r=h[key];content.append(f'<tr><td>{esc(LABELS[key])}</td><td>{number(r["timed_total_seconds"])} s</td><td>{number(r["stages"]["denoise"]["span_seconds"])} s</td><td>{number(r["decode_tail_after_denoising_seconds"])} s</td><td>{number(r["gpu_trace_summary"]["peak_vram_gib"])} GiB</td></tr>')
    content.append('''</tbody></table><p class="muted">One selected historical run per row; the 2T+2V result is supported by a 225.3-second repeat. Denoising span is first sampler start to last sampler completion; the decode tail ends at the last VAE call, before remaining output work. The isolated 1T reference did not use MPS, so that stage comparison is diagnostic rather than a fully controlled MPS ablation.</p>
    <h2>2. What went wrong in each configuration</h2><ul><li><strong>2T + 2V:</strong> the denoising span shrank by 8.8 s, but the decoding tail grew by 10.9 s. The full video became 2.5 s slower in the selected runs. A repeat gave the same conclusion.</li><li><strong>3T + 1V:</strong> denoising finished 14.4 s earlier, but the decoding tail grew by 39.0 s. Only one chunk had completed VAE decoding before the last denoising call finished, versus six with the control.</li><li><strong>3T, then 2V:</strong> three transformer copies took 120.4 s for their whole phase, versus 121.0 s with the isolated single transformer. Almost no denoising speedup, and sequential stage ordering forfeited the pipeline overlap.</li></ul><h2>The same chunk gets slower as copies increase</h2><table><tr><th>Configuration</th><th>Mean warmed 55/56-frame denoise</th><th>Change versus control</th><th>Chunks decoded before denoising finishes</th></tr>''')
    baseline=h['1t2v_control']['stages']['denoise']['warm_small_mean_seconds']
    for key in list(LABELS)[:4]:
        r=h[key];v=r['stages']['denoise']['warm_small_mean_seconds'];content.append(f'<tr><td>{esc(LABELS[key])}</td><td>{v:.2f}s</td><td>{v/baseline:.2f}× duration</td><td>{r["chunks_decoded_before_last_denoise"]}/20</td></tr>')
    content.append('''</table><p>The first chunk in each worker is excluded from the warmed comparison. All remaining short chunks have the same effective 56-frame model shape. These observed differences also include the changing mix of concurrent decoding, not just transformer count.</p>
    <h2>3. What Nsight Systems measured on the GPU</h2><p>All four fresh runs matched the historical output SHA-256 and completed without competing GPU workload samples. GPU metrics were sampled at 1,000 Hz. SM activity reports whether multiprocessors have active work; it does not mean every arithmetic unit is saturated.</p><table><tr><th>Configuration</th><th>SM active during T window</th><th>Tensor active</th><th>DRAM read / write</th><th>Actual T/T kernel overlap</th><th>Actual T/V kernel overlap</th></tr>''')
    for key,r in ns.items():
        m=r['metric_windows']['denoising_window'];mean=lambda n:m[n]['mean'];o=r['overlap']
        content.append(f'<tr><td>{esc(LABELS[key])}</td><td>{mean("SMs Active [Throughput %]"):.1f}%</td><td>{mean("Tensor Active [Throughput %]"):.1f}%</td><td>{mean("DRAM Read Bandwidth [Throughput %]"):.1f}% / {mean("DRAM Write Bandwidth [Throughput %]"):.1f}%</td><td>{o.get("multiple_transformer_kernel_seconds",0):.2f}s</td><td>{o.get("transformer_vae_kernel_overlap_seconds",0):.2f}s</td></tr>')
    content.append('</table><p>Kernel-overlap durations are the union of time with simultaneous kernels from different relevant processes. They are not sums of Python-call durations. The two-transformer capture has only 1.73 s of T/T kernel overlap in a 129.38 s denoising call span. The three-transformer/one-VAE capture has only 2.95 s of T/T kernel overlap in 121.85 s. Transformer kernel-time totals remain approximately 112–113 s across these captures: more workers mostly divide the same GPU execution capacity and add waiting between kernels for each chunk.</p><p>Repeated full-model offloading is not the explanation in that capture: after the first chunk per transformer lane, only 0.59 GiB travels host-to-device, taking 0.029 s of recorded GPU-copy duration. The historical node telemetry also reports zero transformer model GiB offloaded. Transfer API waiting time is not included in that GPU-copy duration.</p>')
    for key,r in ns.items():
        status=json.loads((ROOT/(key+'_v2_status.json')).read_text());hist=h[key]
        content.append(f'<section><h3>{esc(LABELS[key])}</h3><p>Profiled total: <strong>{status["instrumented_total_seconds"]:.1f}s</strong> versus original {hist["timed_total_seconds"]:.1f}s. Instrumentation changes scheduling; do not rank configurations using these totals.</p>{timeline(r)}<small>Bars are synchronized CPU sampler/decode calls, colored by worker. Hover for the GPU-kernel coverage inside each call. Actual GPU overlap is quantified separately above.</small>')
        content.append('<details><summary>Dominant kernels and transfers</summary><table><tr><th>Stage</th><th>Kernel family</th><th>Calls</th><th>Summed GPU duration</th></tr>')
        for role in ['denoise','decode']:
            for k in r['top_kernels'][role][:4]:content.append(f'<tr><td>{role}</td><td title="{esc(k["name"])}">{esc(friendly(k["name"]))}</td><td>{k["count"]:,}</td><td>{k["sum_seconds"]:.2f}s</td></tr>')
        content.append('</table><p>Overlapping kernel durations cannot be added to infer wall time.</p><table><tr><th>Stage</th><th>H→D including first chunks</th><th>H→D after first chunk per lane</th></tr>')
        for role in ['denoise','decode']:
            a=r['memcpy'].get(role,{}).get('Host-to-Device',{});b=r['memcpy'].get(role+'_excluding_first_chunk_per_lane',{}).get('Host-to-Device',{})
            content.append(f'<tr><td>{role}</td><td>{a.get("bytes",0)/1024**3:.2f}GiB / {a.get("sum_seconds",0):.3f}s</td><td>{b.get("bytes",0)/1024**3:.2f}GiB / {b.get("sum_seconds",0):.3f}s</td></tr>')
        content.append(f'</table></details><p><a href="{key}_v2.nsys-rep">Open raw Nsight report</a> · <a href="{key}_v2_status.json">Capture and ownership status</a> · <a href="../runs/followup_nsys_v2_{key}/output.mp4">Validated output</a></p></section>')
    content.append('''<h2>4. Why the important kernels have limited room to overlap</h2><div class="callout"><strong>Free VRAM is not the same as free execution resources.</strong> The captured transformer GEMMs request 72 KiB of shared memory per block. Each SM has 100 KiB, so two such blocks cannot fit together. Those kernels also use 224–254 registers per thread with 256 threads per block: 57,344–65,024 registers, out of 65,536 registers per SM. A second full transformer copy does not add more SMs, registers, or shared memory.</div><p><a href="kernel_resources.json">Recorded kernel resource limits</a>. The GPU has 188 SMs. The large transformer launches have thousands of blocks, enough to occupy the whole device over many waves. Low warp occupancy therefore does not imply enough free resources for another copy of the same heavy kernel. NCU measured approximately 91–94% SM throughput for the sampled transformer matrix kernels and confirmed a one-block-per-SM occupancy limit from both registers and shared memory. Their roughly 16.7% warp occupancy does not mean 83.3% spare compute capacity: those blocks already exhaust the resources needed to admit another large block. The following NCU measurements quantify selected kernels independently.</p><p>Selected dominant kernels are profiled on a resident, isolated stage using a 73-frame chunk after warming both frame sizes. Kernel replay perturbs execution and is deliberately separate from MPS pipeline timing. One launch per matching kernel launch configuration is sampled; this is not a weighted full-workload average.</p><details><summary>All sampled kernel counters (expand)</summary><div class="scroll"><table><tr><th>Stage</th><th>Kernel</th><th>SM throughput</th><th>DRAM throughput</th><th>Achieved occupancy</th><th>Registers / thread</th><th>Shared memory / block</th><th>Blocks/SM limited by registers</th><th>Blocks/SM limited by shared memory</th></tr>''')
    def metric(r,key):
        v=r['metrics'].get(key);return esc(v['value']+' '+(v['unit'] or '')) if v else 'Unavailable'
    for r in ncu:
        content.append(f'<tr><td>{r["stage"]}</td><td title="{esc(r["kernel"])}">{esc(friendly(r["kernel"]))}</td><td>{metric(r,"sm__throughput.avg.pct_of_peak_sustained_elapsed")}</td><td>{metric(r,"gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")}</td><td>{metric(r,"sm__warps_active.avg.pct_of_peak_sustained_active")}</td><td>{metric(r,"launch__registers_per_thread")}</td><td>{metric(r,"launch__shared_mem_per_block")}</td><td>{metric(r,"launch__occupancy_limit_registers")}</td><td>{metric(r,"launch__occupancy_limit_shared_mem")}</td></tr>')
    content.append('''</table></div></details><p><a href="ncu_denoise.ncu-rep">Transformer NCU report</a> · <a href="ncu_decode.ncu-rep">VAE NCU report</a> · <a href="ncu_summary.json">All exported kernel metrics</a></p>
    <h2>5. What to do with this result</h2><ul><li>Keep the measured 1T+2V layout for this GPU. Extra transformer copies consume memory and stretch per-chunk work without supplying additional hardware.</li><li>Improve the dominant kernels or reduce repeated VAE launch/setup work before adding more copies. Fast-mode attention and VAE optimization target these costs; their separate measured 200.3-second result is consistent with that approach.</li><li>Independent physical GPUs can supply additional execution capacity. These one-GPU profiles do not establish eight-GPU scaling or validate greater-than 8× throughput.</li></ul>
    <h2>Evidence and reproduction</h2><p><a href="analysis.json">Analysis JSON</a> · <a href="historical_chunks.csv">Original per-chunk timings</a> · <a href="analyze_traces.py">Trace analysis script</a> · <a href="run_nsys_matrix.py">NSYS runner</a> · <a href="run_ncu_kernels.py">NCU runner</a> · <a href="ncu_stage.py">NCU stage workload</a></p>
    <p class="muted">Nsight Systems 2026.4.1; Nsight Compute 2026.2.1. CUDA software tracing is used for MPS. CPU sampling/context-switch tracing is disabled because host perf permissions are restricted. No power limits, driver settings, or production deployment changes were applied. NCU used clock-control=none and cache-control=none, with its default stable Tensor Core pipeline boost during kernel profiling. The profiler-only idle baseline was verified separately at 4 MiB, versus 2 MiB without Nsight metrics.</p>
    <p>Profiler interpretation: <a href="https://docs.nvidia.com/nsight-systems/UserGuide/">NVIDIA Nsight Systems guide</a> · <a href="https://docs.nvidia.com/nsight-compute/ProfilingGuide/">NVIDIA Nsight Compute guide</a>. <a href="provenance.json">Profiler and source provenance</a>.</p></main></html>''')
    (ROOT/'index.html').write_text(''.join(content))
    print('Wrote',ROOT/'index.html')
if __name__=='__main__':main()

from pathlib import Path
p=Path('agent/rtx6000_benchmark/stage_pipeline.py')
s=p.read_text()
s=s.replace("'default', 'dynamic', str(prepared), role, str(latents)", "'default', os.environ.get('H3_STAGE_RESIDENCY', 'dynamic'), str(prepared), role, str(latents)")
s=s.replace("parser.add_argument('--reuse-calibration')", "parser.add_argument('--reuse-calibration')\n    parser.add_argument('--chunks', type=int, choices=[5,10,20], default=10)\n    parser.add_argument('--residency', choices=['dynamic','highvram'], default='dynamic')\n    parser.add_argument('--async-encoding', action='store_true')")
s=s.replace("counts = chunk_counts(10, 'aligned')", "os.environ['H3_STAGE_RESIDENCY'] = args.residency\n    os.environ['H3_ASYNC_ENCODING'] = '1' if args.async_encoding else '0'\n    if args.prime_longest and (args.schedule != 'pipeline' or args.transformers != 1 or args.vaes != 1):\n        raise ValueError('Priming requires a one-transformer/one-VAE pipeline')\n    counts = chunk_counts(args.chunks, 'aligned')")
s=s.replace("'chunks': 10, 'chunk_plan'", "'chunks': args.chunks, 'residency': args.residency, 'async_encoding': args.async_encoding, 'chunk_plan'")
s=s.replace("previous['chunk_frames'] != counts:", "previous['chunk_frames'] != counts or previous.get('residency', 'dynamic') != args.residency:")
p.write_text(s)
p=Path('agent/rtx6000_benchmark/parallel_chunk_resident.py')
s=p.read_text()
s=s.replace("    try:\n        if residency == 'highvram':", "    encoder = None\n    try:\n        if residency == 'highvram':",1)
s=s.replace("        # Wall intervals around the actual sampler call", """        async_encoding = os.environ.get('H3_ASYNC_ENCODING') == '1' and stage == 'decode'
        if async_encoding:
            from .deferred_encoding import DeferredEncoding
            encoder = DeferredEncoding(main.nodes.NODE_CLASS_MAPPINGS['H3SaveChunks'], lane_id, result_queue)
        # Wall intervals around the actual sampler call""",1)
s=s.replace("            index, frames, audio, name, anchor = task", "            if encoder:\n                encoder.limit_pending()\n            index, frames, audio, name, anchor = task",1)
s=s.replace("                if stage != 'denoise':\n                    parsed = probe(output)", "                if stage != 'denoise' and not encoder:\n                    parsed = probe(output)",1)
s=s.replace("file_hash(output) if stage != 'denoise' else None", "file_hash(output) if stage != 'denoise' and not encoder else None",1)
s=s.replace("'prepared_dir': prepared_dir, 'stage': stage,", "'prepared_dir': prepared_dir, 'stage': stage, 'residency': residency, 'async_encoding': async_encoding,",1)
s=s.replace("                (case_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\\n')\n                result_queue.put({'lane': lane_id, 'kind': 'result', 'index': index, 'result': result})", """                if encoder:
                    encoder.finish_later(result, case_dir, output, frames, started)
                else:
                    (case_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\\n')
                    result_queue.put({'lane': lane_id, 'kind': 'result', 'index': index, 'result': result})""",1)
s=s.replace("\n\ndef one_pool_run", "\n    finally:\n        if encoder:\n            encoder.close()\n\n\ndef one_pool_run",1)
p.write_text(s)

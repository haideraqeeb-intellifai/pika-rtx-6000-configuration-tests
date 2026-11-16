"""Submit two chained videos together to the existing single-process H3 executor."""

import concurrent.futures
import copy
import json
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from ml_modules.minimax_h3_turbo.infer import infer, load_models

from .run_case import MODEL_ROOT, ROOT, empty_gpu_gate, probe, sha256


def main():
    directory = ROOT / 'runs' / 'two_incoming_jobs_shared'
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    baseline = ROOT / 'runs' / 'chained_10_aligned'
    runtime = directory / 'runtime'
    shutil.copytree(baseline / 'runtime' / 'input', runtime / 'input')
    template = json.loads((baseline / 'workflow.json').read_text())
    gate = empty_gpu_gate()
    started = time.perf_counter()
    models = load_models(MODEL_ROOT, runtime)
    initialized = time.perf_counter() - started
    barrier = threading.Barrier(2)
    deadline = time.monotonic() + 9000

    def submit(index):
        job_dir = directory / f'job_{index}'
        job_dir.mkdir()
        output = job_dir / 'output.mp4'
        graph = copy.deepcopy(template)
        for node in graph.values():
            inputs = node['inputs']
            if node['class_type'] == 'H3StoreChunk':
                inputs['directory'] = str(job_dir)
            elif node['class_type'] == 'H3SaveChunks':
                inputs['output'] = str(output)
            if 'deadline' in inputs:
                inputs['deadline'] = deadline
        (job_dir / 'workflow.json').write_text(json.dumps(graph, indent=2) + '\n')
        barrier.wait()
        entered = time.perf_counter()
        result = infer(models, graph, prompt_id=uuid.uuid4().hex,
                       timeout_seconds=deadline - time.monotonic())
        finished = time.perf_counter()
        return {'job': index, 'entered_seconds': entered - started,
                'finished_seconds': finished - started,
                'including_lock_wait_seconds': finished - entered,
                'telemetry': result['telemetry'], 'output': str(output)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(submit, index) for index in range(2)]
        jobs = [future.result() for future in futures]
    gpu_work_done = time.perf_counter() - started
    for job in jobs:
        output = Path(job['output'])
        job['output_probe'] = probe(output)
        job['output_sha256'] = sha256(output)
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(output),
                        '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'],
                       check=True, timeout=300, capture_output=True)
        job['full_decode_passed'] = True
    details = {'description': 'Two incoming jobs submitted simultaneously; current executor lock applies',
               'source_case': 'chained_10_aligned', 'empty_gpu_gate': gate,
               'initialization_seconds': initialized, 'gpu_work_done_seconds': gpu_work_done,
               'total_seconds': time.perf_counter() - started, 'jobs': jobs}
    (directory / 'result.json').write_text(json.dumps(details, indent=2) + '\n')
    print(json.dumps({'gpu_work_done_seconds': gpu_work_done,
                      'total_seconds': details['total_seconds'],
                      'jobs': [{key: item[key] for key in ('job', 'finished_seconds',
                                   'including_lock_wait_seconds', 'output_sha256')} for item in jobs]}, indent=2))


if __name__ == '__main__':
    main()

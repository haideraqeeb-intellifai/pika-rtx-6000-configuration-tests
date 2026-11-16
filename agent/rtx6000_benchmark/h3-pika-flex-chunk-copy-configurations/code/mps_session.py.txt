"""Private, temporary MPS controller for one benchmark phase only."""
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


def control(command):
    return subprocess.run(['nvidia-cuda-mps-control'], input=command+'\n', text=True,
        capture_output=True, check=True, timeout=30).stdout


def terminate_client(pid):
    servers=control('get_server_list').split()
    for server in servers:
        if server.isdigit(): control(f'terminate_client {server} {pid}')


def run_with_mps(function, directory):
    directory.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='h3-vae-mps-') as pipe:
        changes={'CUDA_MPS_PIPE_DIRECTORY':pipe, 'CUDA_MPS_LOG_DIRECTORY':str(directory/'mps_logs'),
                 'H3_BENCHMARK_MPS':'1'}
        Path(changes['CUDA_MPS_LOG_DIRECTORY']).mkdir()
        old={key:os.environ.get(key) for key in changes}
        os.environ.update(changes)
        started=False
        try:
            subprocess.run(['nvidia-cuda-mps-control','-d'],check=True,capture_output=True,text=True,timeout=30)
            started=True
            for _ in range(50):
                if (Path(pipe)/'control').exists(): break
                time.sleep(.1)
            result=function()
            result['mps_server_list']=control('get_server_list')
            return result
        finally:
            try:
                if started:
                    shutdown=control('quit')
                    (directory/'mps_shutdown.json').write_text(json.dumps({'quit_completed':True,'output':shutdown}))
            finally:
                for key,value in old.items():
                    if value is None: os.environ.pop(key,None)
                    else: os.environ[key]=value

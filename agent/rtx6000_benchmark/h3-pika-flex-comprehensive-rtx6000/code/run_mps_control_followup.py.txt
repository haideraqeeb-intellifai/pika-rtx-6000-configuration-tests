"""Matched one-transformer/one-VAE MPS control for copy-count attribution."""
from .run_chunk_copy_matrix import run

def main():
    for suffix in ('','_repeat2'):
        run('followup_one_mps'+suffix,['--schedule','pipeline','--chunks','20',
            '--transformers','1','--vaes','1','--mps','--reuse-calibration','followup_cal_20'])

if __name__=='__main__':main()

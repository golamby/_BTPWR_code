"""Run native numerical BPQA on declared non-LDP sensitivity settings."""
import argparse
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
from experiments.sweep_cases import cases
from experiments.run_bpqa_native import init, one
from src import data
from src.simulate import world_digest

ROOT=Path(__file__).resolve().parents[1]


def already_complete(case, inputs):
    path=ROOT/'results'/'bpqa_native_sweeps'/(case['name']+'.npz')
    meta_path=path.with_suffix('.json')
    if not path.exists() or not meta_path.exists(): return False
    try:
        meta=json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['case']!=case or meta['error_aggregation']!='sum': return False
        for rel,digest in meta['source_sha256'].items():
            if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest: return False
        if meta['world_sha256']!=world_digest(data.build_world(case,inputs)): return False
        import numpy as np
        with np.load(path,allow_pickle=False) as z:
            if z['predictions'].shape!=(50,100) or z['resources'].shape!=(50,5): return False
        return True
    except (KeyError, ValueError, OSError):
        return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=18)
    parser.add_argument('--default-budget',required=True,help='Per-round amount or original')
    parser.add_argument('--datasets',nargs='+',default=list(data.DATASETS))
    parser.add_argument('--seeds',nargs='+',type=int,default=[9701,9702,9703])
    args=parser.parse_args()
    # Native BPQA has no comparable location-LDP epsilon, so it is not
    # presented as though it varied that privacy mechanism.
    jobs=[c for c in cases(datasets=args.datasets,seeds=args.seeds,default_budget=args.default_budget) if c['eps1'] == 2.1]
    inputs=data.load_inputs()
    total=len(jobs)
    jobs=[c for c in jobs if not already_complete(c,inputs)]
    print('jobs',len(jobs),'reused',total-len(jobs),flush=True)
    with Pool(args.workers,initializer=init,initargs=((),(), 'sum')) as pool:
        list(pool.imap_unordered(one,jobs))


if __name__ == '__main__':
    main()

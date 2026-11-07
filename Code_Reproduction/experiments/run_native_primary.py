"""Re-evaluate available native numerical systems on the 24 primary worlds."""
import json
import os
from pathlib import Path
from multiprocessing import Pool
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import numpy as np
from src import data,native_context
from src.simulate import world_digest,scientific_code_digest,json_default
from experiments.primary_450_cases import cases
ROOT=Path(__file__).resolve().parents[1]
INPUTS=None

def init():
    global INPUTS
    INPUTS=data.load_inputs()

def one(case):
    w=data.build_world(case,INPUTS)
    payload,result=native_context.run_native(w,case)
    out=ROOT/'results'/'native_primary';out.mkdir(parents=True,exist_ok=True)
    native_context.export(payload,result,case,out/(case['name']+'.npz'))
    metadata=dict(case=case,world_sha256=world_digest(w),
        scientific_code_sha256=scientific_code_digest(),**result)
    (out/(case['name']+'.json')).write_text(json.dumps(metadata,default=json_default))
    print(json.dumps({'case':case['name'],'results':result['results']},default=json_default),flush=True)

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    print('jobs',len(cases()),flush=True)
    with Pool(a.workers,initializer=init) as pool:list(pool.imap_unordered(one,cases()))

if __name__=='__main__':main()

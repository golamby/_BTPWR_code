"""Traditional selectors with the common privacy, feedback, and recovery pipeline."""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from experiments.sweep_cases import cases as sweep_cases
from experiments.primary_450_cases import cases as primary_cases
from src import data
from src.admission_experiments import run_case
ROOT=Path(__file__).resolve().parents[1]
INPUTS=None

def init():
    global INPUTS
    INPUTS=data.load_inputs()

def one(item):
    group,case=item
    return run_case(case,data.build_world(case,INPUTS),ROOT/'results'/group,
                    methods=('random','greedy-cover'))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    jobs=[('traditional_primary',c) for c in primary_cases()]
    jobs += [('traditional_sweeps',c) for c in sweep_cases(default_budget='450')]
    print('jobs',len(jobs),flush=True)
    with Pool(args.workers,initializer=init) as pool:
        list(pool.imap_unordered(one,jobs))

if __name__=='__main__':
    main()

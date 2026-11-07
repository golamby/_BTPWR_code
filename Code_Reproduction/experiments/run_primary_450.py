import argparse
import os
from multiprocessing import Pool
from pathlib import Path

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from experiments.primary_450_cases import cases
from src import data
from src.admission_experiments import run_case

ROOT=Path(__file__).resolve().parents[1]
INPUTS=None


def init():
    global INPUTS
    INPUTS=data.load_inputs()


def one(case):
    return run_case(case,data.build_world(case,INPUTS),ROOT/'results'/'primary_450')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=16)
    args=p.parse_args()
    jobs=cases()
    print('jobs',len(jobs),flush=True)
    with Pool(args.workers,initializer=init) as pool:
        list(pool.imap_unordered(one,jobs))


if __name__=='__main__':
    main()

"""Evaluate candidate across the registered 207 sensitivity worlds."""
import argparse,os
from multiprocessing import Pool
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from src import data
from src.admission_experiments import run_case
from experiments.explore_cseed_competition import CoverageTwo
from experiments.sweep_cases import cases
ROOT=Path(__file__).resolve().parents[1]
INPUTS=None
def init():
    global INPUTS
    INPUTS=data.load_inputs()
def one(c):
    return run_case(c,data.build_world(c,INPUTS),ROOT/'results'/'cseed_candidate_sweep',
                    methods=('admission-cover-2',),planner_types={'admission-cover-2':CoverageTwo})
def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=24);a=p.parse_args()
    with Pool(a.workers,initializer=init) as pool:list(pool.imap_unordered(one,cases(default_budget='450')))
if __name__=='__main__':main()

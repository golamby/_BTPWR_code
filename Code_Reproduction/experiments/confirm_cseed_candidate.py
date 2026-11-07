"""Confirm the selected exploratory candidate on 24 held-out seed/world pairs."""
import argparse,os
from pathlib import Path
from multiprocessing import Pool
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from src import data
from src.admission_experiments import run_case
from experiments.explore_cseed_competition import CoverageTwo
from experiments.primary_450_cases import cases
ROOT=Path(__file__).resolve().parents[1]
INPUTS=None
def init():
    global INPUTS
    INPUTS=data.load_inputs()
def one(c):
    return run_case(c,data.build_world(c,INPUTS),ROOT/'results'/'cseed_confirmation',
                    methods=('admission-cover-2',),planner_types={'admission-cover-2':CoverageTwo})
def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    with Pool(a.workers,initializer=init) as pool:list(pool.imap_unordered(one,cases()))
if __name__=='__main__':main()

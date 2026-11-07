"""Explore two predeclared budget penalties on the same nine worlds."""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
from src import data
from src.admission_experiments import run_case
from src.current_coverage_admission import CostAwarePlanner, CostAwareHalf

ROOT=Path(__file__).resolve().parents[1]
INPUTS=None


def init():
    global INPUTS
    INPUTS=data.load_inputs()


def one(job):
    dataset,seed=job
    case=dict(name=f'{dataset}_{seed}_1.00',group='main',kind='main',stage='main',
              dataset=dataset,field='legacy',seed=seed,n=100,H=.7,C=.1,gamma=.6,
              eps1=2.1,eps2=2.7,budget_override=data.BUDGETS[dataset],budget_factor=1.,
              cohort_rule='training-top200-uniform100')
    return run_case(case,data.build_world(case,INPUTS),ROOT/'results'/'cost_aware',
                    methods=('admission-cost-025','admission-cost-050'),
                    planner_types={'admission-cost-025':CostAwarePlanner,
                                   'admission-cost-050':CostAwareHalf})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=9)
    args=parser.parse_args()
    jobs=[(d,s) for d in data.DATASETS for s in (9701,9702,9703)]
    print('jobs',len(jobs),flush=True)
    with Pool(args.workers,initializer=init) as pool:
        list(pool.imap_unordered(one,jobs))


if __name__ == '__main__':
    main()

"""Paired baseline/development planner campaigns on common generated worlds."""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from src import data
from src.admission_experiments import run_case
from src.current_coverage_admission import AdmissionPlanner

ROOT = Path(__file__).resolve().parents[1]
INPUTS = None


def init():
    global INPUTS
    INPUTS = data.load_inputs()


def one(job):
    dataset, seed = job
    case = dict(name=f'{dataset}_{seed}_1.00', group='main', kind='main', stage='main',
                dataset=dataset, field='legacy', seed=seed, n=100, H=.7, C=.1,
                gamma=.6, eps1=2.1, eps2=2.7, budget_override=data.BUDGETS[dataset],
                budget_factor=1., cohort_rule='training-top200-uniform100')
    return run_case(case, data.build_world(case, INPUTS), ROOT / 'results' / 'current_coverage',
                    methods=('admission', 'admission-current-coverage'),
                    planner_types={'admission-current-coverage': AdmissionPlanner})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=list(data.DATASETS))
    p.add_argument('--seeds', nargs='+', type=int, default=[9701, 9702, 9703])
    p.add_argument('--workers', type=int, default=9)
    args = p.parse_args()
    jobs = [(d, s) for d in args.datasets for s in args.seeds]
    print('jobs', len(jobs), flush=True)
    with Pool(args.workers, initializer=init) as pool:
        list(pool.imap_unordered(one, jobs))


if __name__ == '__main__':
    main()

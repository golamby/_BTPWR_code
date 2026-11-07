"""Run the original planner and matched controls on the sensitivity grid."""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
from experiments.sweep_cases import cases
from src import data
from src.admission_experiments import run_case

ROOT = Path(__file__).resolve().parents[1]
INPUTS = None


def init():
    global INPUTS
    INPUTS = data.load_inputs()


def one(case):
    return run_case(case, data.build_world(case, INPUTS), ROOT/'results'/'review_sweeps',
                    methods=('admission','admission-no-promotion','template-first','old','etbp'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--default-budget', required=True, help='Per-round amount or original')
    parser.add_argument('--datasets', nargs='+', default=list(data.DATASETS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[9701,9702,9703])
    args = parser.parse_args()
    jobs = cases(datasets=args.datasets,seeds=args.seeds,default_budget=args.default_budget)
    print('jobs',len(jobs),flush=True)
    with Pool(args.workers, initializer=init) as pool:
        list(pool.imap_unordered(one,jobs))


if __name__ == '__main__':
    main()

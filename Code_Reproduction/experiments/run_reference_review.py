"""Rerun the corrected oracle-reference diagnostic at fixed beta values."""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
from src import data

ROOT = Path(__file__).resolve().parents[1]
INPUTS = None


def init():
    global INPUTS
    INPUTS = data.load_inputs()


def one(job):
    beta, dataset, seed = job
    from src import reference_experiments as mod
    mod.BETA = beta
    mod.METHODS = ('admission',)
    case = dict(name=f'{dataset}_{seed}_1.00', group='main', kind='main', stage='main',
                dataset=dataset, field='legacy', seed=seed, n=100, H=.7, C=.1,
                gamma=.6, eps1=2.1, eps2=2.7, budget_override=data.BUDGETS[dataset],
                budget_factor=1., cohort_rule='training-top200-uniform100')
    out = ROOT / 'results' / 'oracle_reference' / f'beta_{beta:.2f}'
    rows = mod.run_case(case, data.build_world(case, INPUTS), out)
    print(json.dumps({'beta':beta, 'case':case['name'], 'results':rows}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()
    jobs = [(b, d, s) for b in (.1,.25,.5) for d in ('tdrive','brightkite')
            for s in range(9701,9709)]
    print('jobs', len(jobs), flush=True)
    with Pool(args.workers, initializer=init) as pool:
        list(pool.imap_unordered(one, jobs))


if __name__ == '__main__':
    main()

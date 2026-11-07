"""Rerun the three corrected legacy coupling controls on paired worlds."""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from src import data

ROOT = Path(__file__).resolve().parents[1]
INPUTS = None


def init():
    global INPUTS
    INPUTS = data.load_inputs()


def one(job):
    variant, dataset, seed = job
    if variant == 'coupled':
        from src import coupled_experiments as mod
    elif variant == 'gated':
        from src import coupled_gated_experiments as mod
        mod.MIN_ANCHORS = 8
    elif variant == 'oracle':
        from src import coupled_oracle_experiments as mod
    else:
        raise ValueError(variant)
    mod.COUPLED = True
    mod.METHODS = ('admission',)
    case = dict(name=f'{dataset}_{seed}_1.00', group='main', kind='main', stage='main',
                dataset=dataset, field='legacy', seed=seed, n=100, H=.7, C=.1,
                gamma=.6, eps1=2.1, eps2=2.7, budget_override=data.BUDGETS[dataset],
                budget_factor=1., cohort_rule='training-top200-uniform100')
    out = ROOT / 'results' / 'review_coupled_variants' / variant
    rows = mod.run_case(case, data.build_world(case, INPUTS), out)
    print(json.dumps({'variant': variant, 'case': case['name'], 'results': rows}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    jobs = [(v, d, s) for v in ('coupled','gated','oracle')
            for d in ('tdrive','brightkite') for s in range(9701, 9709)]
    print('jobs', len(jobs), flush=True)
    with Pool(args.workers, initializer=init) as pool:
        list(pool.imap_unordered(one, jobs))


if __name__ == '__main__':
    main()

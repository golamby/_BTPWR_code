"""Numerical BPQA/BBOM campaigns with the paper's original cost semantics."""
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src import bpqa_native, data, metrics
from src.simulate import world_digest

INPUTS = None
CALIBRATION = None
ERROR_MODE = 'sum'


def case_for(dataset, seed, factor):
    return dict(name=f'{dataset}_{seed}_{factor:.2f}', group='main', kind='main',
                stage='main', dataset=dataset, field='legacy', seed=int(seed), n=100,
                H=.7, C=.1, gamma=.6, eps1=2.1, eps2=2.7,
                budget_override=data.BUDGETS[dataset] * factor, budget_factor=factor,
                cohort_rule='training-top200-uniform100')


def calibrate(inputs, dataset, factor, template=None):
    """Estimate BPQA's coverage/reliability balance from separate seed 9601.

    Only offers, bids and task incidences from the first ten calibration
    rounds are used. Subset size is the affordable median-bid count; this is
    an explicit adaptation of the paper's independent-sample estimator.
    """
    case = case_for(dataset, 9601, factor) if template is None else dict(template)
    case.update(seed=9601, name=f'calibration_{dataset}_9601_{factor:.2f}')
    world = data.build_world(case, inputs)
    cfg = bpqa_native.Config()
    rng = np.random.default_rng(np.random.SeedSequence([9601, data.DATASETS.index(dataset),
                                                        int(factor * 100), 8124]))
    levels = bpqa_native.reliability_levels(world['initial_reputation'], cfg)
    ratios = []
    for t in range(10):
        tasks = world['requested_task_ids'][t]
        incidence = world['presence'][t][:, tasks]
        bids = world['actual_cost'][t].sum(axis=1)
        pool = np.flatnonzero(world['offer_mask'][t] & (levels > 0) & (bids > 0) &
                              (bids <= world['budget']) & incidence.any(axis=1))
        if not len(pool):
            continue
        k = min(len(pool), max(1, int(world['budget'] / np.median(bids[pool]))))
        for _ in range(100):
            chosen = rng.choice(pool, size=k, replace=False)
            cov = cfg.task_weight * np.log1p(cfg.rho * incidence[chosen].sum(axis=0)).sum()
            rep = (levels[chosen] * incidence[chosen].sum(axis=1)).sum()
            if rep > 0:
                ratios.append(float(cov / rep))
    if not ratios:
        raise ValueError('No valid independent calibration subsets')
    return {'multiplier': float(np.mean(ratios)), 'seed': 9601, 'rounds': 10,
            'subsets': len(ratios), 'dataset': dataset, 'budget_factor': factor,
            'method': 'mean coverage/reputation over independent calibration subsets',
            'world_sha256': world_digest(world)}


def init(datasets, factors, error_mode):
    global INPUTS, CALIBRATION, ERROR_MODE
    ERROR_MODE = error_mode
    INPUTS = data.load_inputs()
    CALIBRATION = {}


def one(job):
    if isinstance(job, dict):
        case = job
        dataset, seed, factor = case['dataset'], case['seed'], case['budget_factor']
    else:
        dataset, seed, factor = job
        case = case_for(dataset, seed, factor)
    key = (dataset, factor, case['n'], case['H'], case['C'])
    if key not in CALIBRATION:
        CALIBRATION[key] = calibrate(INPUTS, dataset, factor, template=case)
    calibration = CALIBRATION[key]
    world = data.build_world(case, INPUTS)
    cfg = bpqa_native.Config(normalize_worker_error=(ERROR_MODE == 'mean'))
    T, n, D = world['presence'].shape
    reputation = world['initial_reputation'].copy()
    histories = [reputation.copy()]
    predictions = np.full((T, D), np.nan)
    selected = np.zeros((T, n), bool)
    resources = []
    start = time.perf_counter()
    for t in range(T):
        tasks = world['requested_task_ids'][t]
        bids = world['actual_cost'][t].sum(axis=1)
        result = bpqa_native.allocate(world['presence'][t][:, tasks], bids, reputation,
                                      world['offer_mask'][t], world['budget'],
                                      calibration['multiplier'], cfg)
        ids = result['selected']
        selected[t, ids] = True
        reports = np.full((n, D), np.nan)
        reports[ids] = world['reports_check'][t, ids]
        reports[:, ~world['task_mask'][t]] = np.nan
        present = np.isfinite(reports)
        passed = present.any(axis=1) & np.all(~present | ((reports >= 10.) & (reports <= 40.)), axis=1)
        reports[~passed] = np.nan
        truth_check, weights = bpqa_native.truth_and_weights(reports, cfg)
        predictions[t] = np.clip(world['native_offset'] + world['native_scale'] *
                                 (truth_check - 10.) / 30., 0., 100.)
        reputation = bpqa_native.update_reputation(reputation, weights,
                                                   np.isfinite(reports).any(axis=1), cfg)
        histories.append(reputation.copy())
        settled = float(result['payments'][passed].sum())
        resources.append([result['total_payment'], float(world['actual_cost'][t, ids].sum()),
                          int(len(ids)), float(bids[ids].sum()), settled])
    met = metrics.requested_metrics(world['truth_native_private'].T, predictions, world['task_mask'])
    folder_name = (('bpqa_native_sweeps' if ERROR_MODE == 'sum' else 'bpqa_native_sweeps_mean') if case['group'] == 'sweep' else
                   ('bpqa_native' if ERROR_MODE == 'sum' else 'bpqa_native_mean'))
    folder = ROOT / 'results' / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / (case['name'] + '.npz'), predictions=predictions,
                        truth=world['truth_native_private'].T, task_mask=world['task_mask'],
                        selected=selected, reputations=np.asarray(histories),
                        resources=np.asarray(resources))
    meta = dict(case=case, world_sha256=world_digest(world), calibration=calibration,
                completion_percent=met['completion_percent'], answered_rate_percent=met['answered_rate_percent'],
                payment=float(np.asarray(resources)[:, 0].sum()),
                physical=float(np.asarray(resources)[:, 1].sum()),
                settled_payment=float(np.asarray(resources)[:, 4].sum()),
                error_aggregation=ERROR_MODE,
                seconds=time.perf_counter()-start,
                implementation='BPQA and BBOM numeric adaptation; no cryptographic deployment',
                source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                               ('src/bpqa_native.py','src/data.py','experiments/run_bpqa_native.py')})
    (folder / (case['name'] + '.json')).write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta), flush=True)
    return case['name']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=list(data.DATASETS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(9701, 9704)))
    parser.add_argument('--factors', nargs='+', type=float, default=[1.])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--error-mode', choices=('sum','mean'), default='sum')
    args = parser.parse_args()
    checked = data.load_inputs()
    missing = [d for d in args.datasets if not checked['mobility'].get(d) or
               'event_worker' not in checked['mobility'][d]]
    if missing:
        raise FileNotFoundError('Missing prepared mobility arrays: ' + ', '.join(missing))
    jobs = [(d, s, f) for d in args.datasets for s in args.seeds for f in args.factors]
    print('jobs', len(jobs), flush=True)
    if args.workers == 1:
        init(args.datasets, args.factors, args.error_mode)
        for job in jobs:
            one(job)
    else:
        with Pool(args.workers, initializer=init, initargs=(args.datasets,args.factors,args.error_mode)) as pool:
            list(pool.imap_unordered(one, jobs))


if __name__ == '__main__':
    main()

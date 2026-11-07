"""Same-input coupled-reputation baselines (BPQA/Deng-style).

Each campaign uses one world, the same offers, budget, reports and initial
reputation. The separated arm updates reputation from actual reports only;
the coupled arms let recovered estimates enter the feedback path.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np  # noqa: E402
from src import data  # noqa: E402
from src.admission_experiments import run_case  # noqa: E402

METHODS = ("admission", "bpqa-coupled", "deng-coupled")
MODES = {"bpqa-coupled": "bpqa", "deng-coupled": "deng"}
_INPUTS = None


def init():
    global _INPUTS
    _INPUTS = data.load_inputs()


def one(job):
    dataset, seed, factor = job
    case = dict(
        name=f"{dataset}_{seed}_{factor:.2f}", group="main", kind="main",
        stage="main", dataset=dataset, field="legacy", seed=int(seed), n=100,
        H=0.7, C=0.1, gamma=0.6, eps1=2.1, eps2=2.7,
        budget_override=data.BUDGETS[dataset] * factor, budget_factor=factor,
        cohort_rule="training-top200-uniform100",
    )
    world = data.build_world(case, _INPUTS)
    out = ROOT / "results" / "coupled_reputation"
    rows = run_case(case, world, out, methods=METHODS, reputation_modes=MODES)
    return case["name"], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(data.DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(9701, 9709)))
    ap.add_argument("--factors", nargs="+", type=float, default=[0.75, 1.0])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    jobs = [(d, s, f) for d in args.datasets for s in args.seeds for f in args.factors]
    print("jobs", len(jobs), flush=True)
    start = time.time()
    if args.workers <= 1:
        init(); out = [one(j) for j in jobs]
    else:
        with Pool(args.workers, initializer=init) as pool:
            out = list(pool.imap_unordered(one, jobs))
    print("done", len(out), "in", round(time.time() - start, 1), "s")


if __name__ == "__main__":
    main()

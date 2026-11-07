"""Worker-population scaling study for the five registered recruitment policies.

Each worker count reuses the same registered field, task schedule and seeds so
that differences across ``n`` reflect the workforce size rather than new data.
Cohorts are drawn from the training-ranked top-200 pool at every size, which
keeps the cohort rule identical for n <= 100 and n > 100.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import time

import numpy as np

from . import data
from .admission_experiments import METHODS, run_case
from .simulate import json_default

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NS = (25, 50, 75, 100, 150, 200)
DEFAULT_SEEDS = (9701, 9702, 9703, 9704)
DEFAULT_DATASETS = ("tdrive", "foursquare_tky", "brightkite")


def make_case(dataset, seed, n):
    return {
        "name": f"ws_{dataset}_n{n}_s{seed}",
        "group": "worker_scale",
        "kind": "worker_scale",
        "stage": "worker_scale",
        "dataset": dataset,
        "field": "legacy",
        "seed": int(seed),
        "n": int(n),
        "H": 0.7,
        "C": 0.1,
        "gamma": 0.6,
        "eps1": 2.1,
        "eps2": 2.7,
        "budget_override": data.BUDGETS[dataset],
        "cohort_rule": "training-top200-uniform100",
    }


def _run(job):
    case, inputs, outdir = job
    started = time.perf_counter()
    rows = run_case(case, data.build_world(case, inputs), Path(outdir))
    record = {
        "case": case,
        "rows": rows,
        "seconds": time.perf_counter() - started,
    }
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", default=",".join(map(str, DEFAULT_NS)))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--out", default=str(ROOT / "results" / "worker_scale"))
    parser.add_argument("--test", action="store_true", help="single quick case")
    args = parser.parse_args()

    ns = [int(x) for x in args.ns.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    datasets = [x for x in args.datasets.split(",") if x]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cases = [make_case(ds, seed, n) for n in sorted(ns, reverse=True) for seed in seeds for ds in datasets]
    if args.test:
        cases = [make_case("brightkite", 9701, ns[0])]
    inputs = data.load_inputs()

    jobs = [(case, inputs, str(out)) for case in cases]
    records = []
    if args.workers <= 1:
        for job in jobs:
            record = _run(job)
            records.append(record)
            print(json.dumps({"case": record["case"]["name"], "seconds": record["seconds"]}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = [pool.submit(_run, job) for job in jobs]
            for future in as_completed(pending):
                record = future.result()
                records.append(record)
                print(json.dumps({"case": record["case"]["name"], "seconds": record["seconds"]}), flush=True)

    records.sort(key=lambda r: r["case"]["name"])
    (out / "worker_scale.json").write_text(
        json.dumps(records, indent=2, default=json_default), encoding="utf-8"
    )
    print("wrote", out / "worker_scale.json", len(records), "cases")


if __name__ == "__main__":
    main()



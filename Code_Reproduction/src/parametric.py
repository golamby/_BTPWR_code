"""Parameter sweeps for privacy budget and payment budget.

Each setting reruns the five registered policies on the same seeds and the
same two mobility sources; only the named parameter changes.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import time

from . import data
from .admission_experiments import run_case
from .simulate import json_default

ROOT = Path(__file__).resolve().parents[1]
EPS = (1.3, 1.5, 1.7, 1.9, 2.1)
EPS2_OFFSET = 0.6
BUDGET_FACTORS = (0.6, 0.8, 1.0, 1.2, 1.4)
SEEDS = (9701, 9702, 9703, 9704)
DATASETS = ("tdrive", "brightkite")


def base_case(dataset, seed, n=100):
    return {
        "group": "parametric",
        "kind": "parametric",
        "stage": "parametric",
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


def privacy_cases():
    cases = []
    for eps in EPS:
        for seed in SEEDS:
            for ds in DATASETS:
                case = base_case(ds, seed)
                case.update(name=f"pv_e{eps}_{ds}_s{seed}", eps1=eps, eps2=2.7)
                cases.append(case)
    return cases


def budget_cases():
    cases = []
    for factor in BUDGET_FACTORS:
        for seed in SEEDS:
            for ds in DATASETS:
                case = base_case(ds, seed)
                case.update(
                    name=f"bg_f{factor}_{ds}_s{seed}",
                    budget_override=data.BUDGETS[ds] * factor,
                )
                cases.append(case)
    return cases


def _run(job):
    case, inputs, outdir = job
    started = time.perf_counter()
    try:
        rows = run_case(case, data.build_world(case, inputs), Path(outdir))
        return {"case": case, "rows": rows, "seconds": time.perf_counter() - started}
    except Exception as exc:  # keep the pool alive and record the failed setting
        return {"case": case, "rows": [], "error": repr(exc), "seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("privacy", "budget", "all"), default="all")
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--out", default=str(ROOT / "results" / "parametric"))
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    cases = []
    if args.variant in ("privacy", "all"):
        cases += privacy_cases()
    if args.variant in ("budget", "all"):
        cases += budget_cases()
    if args.test:
        cases = cases[:1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = data.load_inputs()
    jobs = [(case, inputs, str(out)) for case in cases]
    records = []
    if args.workers <= 1:
        for job in jobs:
            records.append(_run(job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = [pool.submit(_run, j) for j in jobs]
            for future in as_completed(pending):
                records.append(future.result())
    records.sort(key=lambda r: r["case"]["name"])
    (out / "parametric.json").write_text(json.dumps(records, indent=2, default=json_default), encoding="utf-8")
    print("wrote", out / "parametric.json", len(records), "cases")


if __name__ == "__main__":
    main()


"""Run the C-seeded cap grid used for the matched-payment experiment.

Usage (from 03_Code):
  python -m experiments.run_matched_payment --workers 8
Outputs results/matched_grid.json by default.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data, privacy, trust, recovery, metrics  # noqa: E402
from src.footprints import TemplateSupport  # noqa: E402
from src.admission import select_additive  # noqa: E402
from src.simulate import calibration_for  # noqa: E402

CAPS = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25)
_INPUTS = None
_CAL = {}


def init():
    global _INPUTS
    _INPUTS = data.load_inputs()


def make_case(dataset, seed, factor):
    return dict(name=f"main_{dataset}_s{seed}_f{factor}", group="main", kind="main",
                stage="main", dataset=dataset, field="legacy", seed=int(seed), n=100,
                H=0.7, C=0.1, gamma=0.6, eps1=2.1, eps2=2.7,
                budget_override=data.BUDGETS[dataset] * factor,
                budget_factor=factor, cohort_rule="training-top200-uniform100")


def run_one(args):
    dataset, seed, target_factor, cap = args
    case = make_case(dataset, seed, target_factor)
    w = data.build_world(case, _INPUTS)
    cfg = privacy.Config(n=w["presence"].shape[1], m=w["presence"].shape[2], T=50,
                         eps1=case["eps1"], eps2=case["eps2"])
    tc = trust.Config(gamma=case["gamma"])
    key = (dataset, target_factor)
    if key not in _CAL:
        _CAL[key] = calibration_for(w, ROOT / "calibration")
    head = recovery.initialize(w["public_reference"], w["native_offset"], w["native_scale"], _CAL[key])
    model = TemplateSupport(w["presence"].shape[1], w["public_reference"], case["seed"])
    r = w["initial_reputation"].copy()
    budget = data.BUDGETS[dataset] * cap
    pred, pay, phys = [], 0.0, 0.0
    for t in range(50):
        loc = w["presence"][t]; cost = w["actual_cost"][t]; offers = w["offer_mask"][t]
        avg = np.divide(cost.sum(1), loc.sum(1), out=np.full(len(loc), 3.25), where=loc.sum(1) > 0)
        packet = privacy.release(w["virtual_presence"][t], np.where(w["virtual_presence"][t], avg[:, None], 0),
                                 w["RR_coins"][t], cfg.eps1, cfg.eps2, cfg)
        for k in ("bits", "hi", "nonzero"):
            packet[k][~offers] = False
        support, _ = model.predict(packet["bits"], cfg.eps1, offers, trust.classify(r, tc), t)
        ans = select_additive(packet, offers, r, w["requested_task_ids"][t], cfg, budget, tc, support, True)
        ids = ans["members"]
        pay += float(ans["cost_numerator"] / ans["cost_denominator"])
        phys += float(cost[ids].sum())
        raw = np.full((len(loc), w["presence"].shape[2]), np.nan)
        raw[ids] = w["reports_check"][t, ids]
        model.observe(np.asarray(ids, int), np.isfinite(raw[ids]), t)
        raw[trust.classify(r, tc) == 0] = np.nan
        fb = trust.feedback(raw, r, tc)
        actual = recovery.decode_actual(fb["consensus"], head)
        pred.append(recovery.complete(fb["credible_consensus"], actual, head)[0])
        r = fb["next_reputation"]
    m = metrics.requested_metrics(w["truth_native_private"].T, np.asarray(pred), w["task_mask"])
    return dict(dataset=dataset, seed=int(seed), target_factor=target_factor, cap=cap,
                payment=pay, physical=phys, completion=m["completion_percent"],
                rmse=m["all_answered_RMSE"], mae=m["all_answered_MAE"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=str(ROOT / "targets.json"))
    ap.add_argument("--datasets", nargs="+", default=["tdrive", "brightkite"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--factors", nargs="+", type=float, default=[0.75, 1.0])
    ap.add_argument("--caps", nargs="+", type=float, default=list(CAPS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results" / "matched_grid.json"))
    args = ap.parse_args()
    targets = json.loads(Path(args.targets).read_text())
    targets = [t for t in targets if t["dataset"] in args.datasets
               and float(t["factor"]) in args.factors
               and (args.seeds is None or int(t["seed"]) in args.seeds)]
    jobs = [(t["dataset"], int(t["seed"]), float(t["factor"]), c) for t in targets for c in args.caps]
    print("jobs", len(jobs), flush=True)
    t0 = time.time()
    if args.workers <= 1:
        init(); out = [run_one(j) for j in jobs]
    else:
        with Pool(args.workers, initializer=init) as pool:
            out = list(pool.imap_unordered(run_one, jobs))
    out.sort(key=lambda r: (r["dataset"], r["seed"], r["target_factor"], r["cap"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out, len(out), "in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()

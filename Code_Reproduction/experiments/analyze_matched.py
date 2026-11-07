"""Analyze the matched-payment grid.

Usage (from 03_Code):
  python -m experiments.analyze_matched --grid results/matched_grid.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=str(ROOT / "targets.json"))
    ap.add_argument("--grid", default=str(ROOT / "results" / "matched_grid.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "matched_payment_summary.json"))
    args = ap.parse_args()
    targets = json.loads(Path(args.targets).read_text())
    grid = json.loads(Path(args.grid).read_text())
    g = defaultdict(list)
    for r in grid:
        g[(r["dataset"], r["seed"], r["target_factor"])].append(r)
    for k in g:
        g[k].sort(key=lambda r: r["cap"])
    rows = []
    for t in targets:
        k = (t["dataset"], int(t["seed"]), float(t["factor"]))
        if k not in g:
            continue
        h = g[k]
        pays = [r["payment"] for r in h]; comps = [r["completion"] for r in h]; phys = [r["physical"] for r in h]
        if not (min(pays) <= t["payment"] <= max(pays)):
            continue
        c_comp = float(np.interp(t["payment"], pays, comps))
        c_phys = float(np.interp(t["payment"], pays, phys))
        c_pay = float(np.interp(t["completion"], comps, pays)) if min(comps) <= t["completion"] <= max(comps) else float("nan")
        rows.append(dict(dataset=t["dataset"], seed=int(t["seed"]), factor=float(t["factor"]),
                         btp_comp=t["completion"], btp_pay=t["payment"], btp_phys=t["physical"],
                         c_comp=c_comp, c_phys=c_phys, c_pay_at_comp=c_pay))

    def ci(a):
        a = np.asarray(a); tt = stats.t.ppf(.975, len(a) - 1)
        return [float(a.mean()), float(a.mean() - tt * a.std(ddof=1) / np.sqrt(len(a))),
                float(a.mean() + tt * a.std(ddof=1) / np.sqrt(len(a)))]

    out = {}
    for factor in (0.75, 1.0, None):
        sub = [r for r in rows if factor is None or r["factor"] == factor]
        seeds = sorted({r["seed"] for r in sub})
        dcomp = [np.mean([r["btp_comp"] - r["c_comp"] for r in sub if r["seed"] == s]) for s in seeds]
        dphys = [np.mean([r["btp_phys"] - r["c_phys"] for r in sub if r["seed"] == s]) for s in seeds]
        dpay = [np.mean([r["btp_pay"] - r["c_pay_at_comp"] for r in sub if r["seed"] == s]) for s in seeds]
        out[str(factor)] = dict(n=len(sub), seeds=len(seeds), dcomp=ci(dcomp), dphys=ci(dphys), dpay=ci(dpay))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

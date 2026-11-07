"""Reproduce the external-system cumulative-error table.

Usage (from 03_Code):
  python -m experiments.run_native_ce
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "data" / "recruitment_evidence"
NAT = ROOT / "data" / "native_context"


def ce(pred, truth, mask, P=3):
    total = 0.0; tasks = 0; hits = 0
    for t in range(pred.shape[0]):
        req = np.flatnonzero(mask[t])
        p = pred[t, req]; y = truth[t, req]
        ok = np.isfinite(p); err = np.abs(p - y)
        rel = err / np.maximum(np.abs(y), 1e-9)
        hit = ok & ((rel <= .10) | (err < 1e-9))
        total += float(err[hit].sum()) + P * (len(req) - int(hit.sum()))
        tasks += len(req); hits += int(hit.sum())
    return total / tasks, hits / tasks


def main():
    btp = {}
    for f in glob.glob(str(REC / "*.npz")):
        z = np.load(f, allow_pickle=True); c = json.loads(str(z["case_json"]))
        if c.get("stage") != "main" or abs(c.get("budget_factor", 1.0) - 1.0) > 1e-9:
            continue
        i = [str(m) for m in z["methods"]].index("admission")
        btp[(c["dataset"], c["seed"])] = (z["predictions"][i], z["truth"], z["task_mask"])
    rows = {}
    for f in glob.glob(str(NAT / "*.npz")):
        z = np.load(f, allow_pickle=True); c = json.loads(str(z["case_json"]))
        key = (c["dataset"], c["seed"])
        if key not in btp:
            continue
        for i, m in enumerate(str(x) for x in z["methods"]):
            rows.setdefault(m, []).append(ce(z["predictions"][i], z["truth"], z["task_mask"]))
        rows.setdefault("BTPWR", []).append(ce(*btp[key]))
    out = {m: {"n": len(v), "CE3": float(np.mean([x[0] for x in v])), "hit": float(np.mean([x[1] for x in v]))}
           for m, v in rows.items()}
    out_path = ROOT / "results" / "native_ce.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    for m, v in sorted(out.items(), key=lambda kv: kv[1]["CE3"]):
        print(f"{m:12s} n={v['n']:2d} CE3={v['CE3']:.3f} hit={v['hit']:.3f}")
    print("wrote", out_path)


if __name__ == "__main__":
    main()

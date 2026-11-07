"""Analyze the selection x reputation 2x2 factorial."""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results" / "bpqa_factorial"


def ce3(pred, truth, mask):
    total = 0.0; tasks = 0
    for t in range(pred.shape[0]):
        req = np.flatnonzero(mask[t]); p = pred[t, req]; y = truth[t, req]
        ok = np.isfinite(p); err = np.abs(p - y)
        hit = ok & np.where(y == 0, err == 0, err < .1 * np.abs(y))
        total += float(err[hit].sum()) + 3 * (len(req) - int(hit.sum()))
        tasks += len(req)
    return total / tasks


def ci(v):
    x = np.asarray(v, float); n = len(x)
    h = student_t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def main():
    rows = []
    for path in glob.glob(str(FOLDER / "*.json")):
        j = json.loads(Path(path).read_text(encoding="utf-8"))
        if "case" not in j:
            continue
        z = np.load(Path(path).with_suffix(".npz"), allow_pickle=False)
        m = [str(x) for x in z["methods"]]
        r = {x["method"]: x for x in j["results"]}
        d = {name: {"comp": r[name]["completion_percent"],
                    "pay": r[name]["payment"], "phys": r[name]["physical"],
                    "ce3": ce3(z["predictions"][i], z["truth"], z["task_mask"])}
             for i, name in enumerate(m)}
        rows.append(dict(seed=int(j["case"]["seed"]), dataset=j["case"]["dataset"], **d))
    effects = {}
    for key, fn in {
        "select_sep": lambda d: {k: d["bpqa"][k] - d["admission"][k] for k in d["admission"]},
        "couple_btpwr": lambda d: {k: d["admission-coupled"][k] - d["admission"][k] for k in d["admission"]},
        "select_coup": lambda d: {k: d["bpqa-coupled"][k] - d["admission-coupled"][k] for k in d["admission"]},
        "couple_bpqa": lambda d: {k: d["bpqa-coupled"][k] - d["bpqa"][k] for k in d["admission"]},
        "system": lambda d: {k: d["bpqa-coupled"][k] - d["admission"][k] for k in d["admission"]},
    }.items():
        effects[key] = [dict(seed=r["seed"], **fn(r)) for r in rows]
    inter = []
    for r in rows:
        a = {k: r["bpqa-coupled"][k] - r["bpqa"][k] for k in r["admission"]}
        b = {k: r["admission-coupled"][k] - r["admission"][k] for k in r["admission"]}
        inter.append(dict(seed=r["seed"], **{k: a[k] - b[k] for k in a}))
    effects["interaction"] = inter
    out = {"n_cases": len(rows), "effects": {}}
    for name, rs in effects.items():
        seeds = sorted({x["seed"] for x in rs})
        out["effects"][name] = {}
        for k in ("comp", "ce3", "pay", "phys"):
            vals = [np.mean([x[k] for x in rs if x["seed"] == s]) for s in seeds]
            out["effects"][name][k] = {"mean": float(np.mean(vals)), "ci95": ci(vals)}
    (FOLDER / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

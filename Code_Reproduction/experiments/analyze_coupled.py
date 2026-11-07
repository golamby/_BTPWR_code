"""Analyze same-input separated vs coupled reputation baselines and CE(P)."""
import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
PS = (1, 2, 3, 4, 5, 10)


def ce(pred, truth, mask, penalty):
    total = 0.0; tasks = 0; hits = 0
    for t in range(pred.shape[0]):
        req = np.flatnonzero(mask[t])
        p = pred[t, req]; y = truth[t, req]
        ok = np.isfinite(p); err = np.abs(p - y)
        hit = ok & np.where(y == 0, err == 0, err < .1 * np.abs(y))
        total += float(err[hit].sum()) + penalty * (len(req) - int(hit.sum()))
        tasks += len(req); hits += int(hit.sum())
    return total / tasks, hits / tasks


def ci(values):
    x = np.asarray(values, float); n = len(x)
    half = student_t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return [float(x.mean()), float(x.mean() - half), float(x.mean() + half)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=str(ROOT / "results" / "coupled_reputation"))
    ap.add_argument("--out", default=str(ROOT / "results" / "coupled_reputation_summary.json"))
    args = ap.parse_args()
    folder = Path(args.folder)
    files = sorted(glob.glob(str(folder / "*.npz")))
    if not files:
        raise SystemExit("no campaign files in " + str(folder))
    by_seed = defaultdict(lambda: defaultdict(list))
    metrics = defaultdict(lambda: defaultdict(list))
    for path in files:
        match = re.search(r"_(\d{4})_", Path(path).stem)
        if not match:
            raise ValueError("case name lacks seed: " + path)
        seed = int(match.group(1))
        z = np.load(path, allow_pickle=False)
        methods = [str(x) for x in z["methods"]]
        for i, m in enumerate(methods):
            for p in PS:
                value, _ = ce(z["predictions"][i], z["truth"], z["task_mask"], p)
                by_seed[(m, p)][seed].append(value)
        j = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
        for row in j["results"]:
            m = row["method"]
            for k in ("completion_percent", "answered_rate_percent",
                      "all_answered_MAE", "all_answered_RMSE", "payment", "physical"):
                metrics[m][k].append(float(row[k]))
    out = {"n_campaigns": len(files), "P": list(PS), "methods": {}}
    for m in sorted({k[0] for k in by_seed}):
        out["methods"][m] = {}
        for p in PS:
            groups = by_seed[(m, p)]
            out["methods"][m][str(p)] = {
                "mean": float(np.mean([np.mean(v) for v in groups.values()])),
                "seeds": len(groups),
            }
    for m in out["methods"]:
        if m == "admission":
            continue
        out.setdefault("paired_vs_admission", {})[m] = {}
        for p in PS:
            seeds = sorted(set(by_seed[("admission", p)]) & set(by_seed[(m, p)]))
            d = [np.mean(by_seed[(m, p)][s]) - np.mean(by_seed[("admission", p)][s]) for s in seeds]
            out["paired_vs_admission"][m][str(p)] = {"mean": float(np.mean(d)), "ci95": ci(d),
                                                       "seeds": len(seeds)}
    out["outcomes"] = {}
    for m, vals in metrics.items():
        out["outcomes"][m] = {k: float(np.mean(v)) for k, v in vals.items()}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

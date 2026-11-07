"""Aggregate the workforce sweep into paired effects and a LaTeX table."""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.stats import t as student

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("admission", "admission-no-promotion", "template-first", "old", "etbp")
LABEL = {
    "admission": "BTPWR",
    "admission-no-promotion": "BTPWR-NP",
    "template-first": "C-seeded",
    "old": "Mapped",
    "etbp": "ETBP-P",
}
N_TASKS_PER_ROUND = 50


def load(path):
    path = Path(path)
    if path.is_file() and path.name == "worker_scale.json":
        records = json.loads(path.read_text(encoding="utf-8"))
    else:
        folder = path if path.is_dir() else path.parent
        files = sorted(folder.glob("ws_*.json"))
        if not files:
            raise SystemExit(f"no per-case results in {folder}")
        records = []
        for file in files:
            data = json.loads(file.read_text(encoding="utf-8"))
            records.append({"case": data["case"], "rows": data["results"]})
    rows = []
    for record in records:
        case = record["case"]
        for row in record["rows"]:
            rows.append({**row, "n": case["n"], "seed": case["seed"], "dataset": case["dataset"]})
    return rows


def success_per_payment(row):
    return row["completion_percent"] / 100 * (N_TASKS_PER_ROUND * 50) / row["payment"] * 1000


def paired(rows, comparator="template-first"):
    by = {(r["n"], r["seed"], r["dataset"], r["method"]): r for r in rows}
    ns = sorted({r["n"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})
    out = []
    for n in ns:
        seed_means = {k: [] for k in ("dtcr", "dmae", "drmse", "dpay", "dphys", "dspp")}
        for seed in seeds:
            local = {k: [] for k in seed_means}
            for ds in datasets:
                key = (n, seed, ds)
                if (*key, "admission") not in by or (*key, comparator) not in by:
                    continue
                a, b = by[(*key, "admission")], by[(*key, comparator)]
                local["dtcr"].append(a["completion_percent"] - b["completion_percent"])
                local["dmae"].append(100 * (a["all_answered_MAE"] / b["all_answered_MAE"] - 1))
                local["drmse"].append(100 * (a["all_answered_RMSE"] / b["all_answered_RMSE"] - 1))
                local["dpay"].append(100 * (a["payment"] / b["payment"] - 1))
                local["dphys"].append(100 * (a["physical"] / b["physical"] - 1))
                local["dspp"].append(100 * (success_per_payment(a) / success_per_payment(b) - 1))
            for k in seed_means:
                if local[k]:
                    seed_means[k].append(float(np.mean(local[k])))
        entry = {"n": n, "seeds": len(seed_means["dtcr"])}
        if entry["seeds"] < len(seeds):
            continue
        for k, values in seed_means.items():
            v = np.asarray(values, float)
            mean = float(v.mean())
            radius = float(student.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
            entry[k], entry[k + "_lo"], entry[k + "_hi"] = mean, mean - radius, mean + radius
        out.append(entry)
    return out


def timing(rows):
    by = {(r["n"], r["seed"], r["dataset"], r["method"]): r for r in rows}
    out = {}
    for n in sorted({r["n"] for r in rows}):
        vals = [r["seconds"] for r in rows if r["n"] == n and r["method"] == "admission"]
        out[n] = {"campaign_s": float(np.mean(vals)), "cases": len(vals)}
    return out


def main():
    folder = ROOT / "results" / "worker_scale"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else folder / "worker_scale.json"
    if not path.exists():
        path = folder
    rows = load(path)
    table = paired(rows)
    timing_rows = timing(rows)
    dest = folder / "worker_scale_summary.json"
    dest.write_text(json.dumps({"paired_vs_cseeded": table, "timing": timing_rows}, indent=2), encoding="utf-8")
    print("n seeds dTCR[lo,hi] dMAE dRMSE dPay dPhys dSPP campaign_s")
    for e in table:
        t = timing_rows[e["n"]]
        print(
            f'{e["n"]:4d} {e["seeds"]:2d} '
            f'{e["dtcr"]:+7.2f}[{e["dtcr_lo"]:+6.2f},{e["dtcr_hi"]:+6.2f}] '
            f'{e["dmae"]:+6.2f} {e["drmse"]:+6.2f} {e["dpay"]:+6.2f} {e["dphys"]:+6.2f} {e["dspp"]:+6.2f} '
            f'{t["campaign_s"]:8.1f}'
        )
    print("wrote", dest)


if __name__ == "__main__":
    main()

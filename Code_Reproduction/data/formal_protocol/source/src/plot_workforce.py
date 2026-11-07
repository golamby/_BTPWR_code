"""Plot the workforce sweep: service, payment and planning time versus n."""
from pathlib import Path
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter, FixedLocator

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"BTPWR": "#A42B33", "C-seeded": "#536C7A", "BTPWR-NP": "#A5824B"}
MARKERS = {"BTPWR": "o", "C-seeded": "s", "BTPWR-NP": "^"}


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "xtick.direction": "in",
        "ytick.direction": "in",
    })


def load(path):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for record in records:
        case = record["case"]
        for row in record["rows"]:
            rows.append({**row, "n": case["n"], "seed": case["seed"], "dataset": case["dataset"]})
    return rows


def mean_by(rows, method, field):
    out = {}
    for n in sorted({r["n"] for r in rows}):
        vals = [r[field] for r in rows if r["n"] == n and r["method"] == method]
        out[n] = (float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
    return out


def search_per_round(records, folder=None):
    """Mean complete planning-loop seconds per round from campaign rows."""
    out = {}
    for record in records:
        n = record["case"]["n"]
        for row in record["rows"]:
            if row["method"] == "admission":
                out.setdefault(n, []).append(row["seconds"] / 50.0)
    return {n: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))) for n, v in out.items()}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "results" / "worker_scale" / "worker_scale.json"
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = load(path)
    style()

    ns = sorted({r["n"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.05), constrained_layout=True)
    ax = axes[0]
    for method, label in (("admission", "BTPWR"), ("template-first", "C-seeded")):
        series = mean_by(rows, method, "completion_percent")
        x = np.array(ns)
        y = np.array([series[n][0] for n in ns])
        e = np.array([series[n][1] for n in ns])
        ax.errorbar(x, y, yerr=1.96 * e, marker=MARKERS[label], ms=3.2, lw=1.1,
                    color=COLORS[label], capsize=2, label=label)
    ax.set_xlabel("workers $n$")
    ax.set_ylabel("task completion rate (\\%)")
    ax.set_title("(a) service", loc="left")
    ax.legend(loc="best", handlelength=1.4)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())

    ax = axes[1]
    for method, label in (("admission", "BTPWR"), ("template-first", "C-seeded")):
        series = mean_by(rows, method, "payment")
        x = np.array(ns)
        y = np.array([series[n][0] / 1000 for n in ns])
        e = np.array([series[n][1] / 1000 for n in ns])
        ax.errorbar(x, y, yerr=1.96 * e, marker=MARKERS[label], ms=3.2, lw=1.1,
                    color=COLORS[label], capsize=2, label=label)
    ax.set_xlabel("workers $n$")
    ax.set_ylabel("committed payment ($10^3$ units)")
    ax.set_title("(b) payment", loc="left")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())

    ax = axes[2]
    sp = search_per_round(records, Path(path).parent)
    x = np.array(sorted(sp))
    y = np.array([sp[n][0] for n in x])
    e = np.array([sp[n][1] for n in x])
    ax.errorbar(x, y, yerr=1.96 * e, marker="D", ms=3.0, lw=1.1, color="#295D83", capsize=2)
    if len(x) > 1:
        slope = np.polyfit(np.log(x), np.log(y), 1)[0]
        ax.text(0.03, 0.92, f"slope $\\approx$ {slope:.2f}", transform=ax.transAxes, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("workers $n$")
    ax.set_ylabel("planning time per round (s)")
    ax.set_title("(c) planning cost", loc="left")
    ax.yaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4]))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())

    for ext in ("pdf", "svg"):
        for a in axes.flat:
            a.yaxis.grid(True, color="#DEDEDE", linewidth=.5, linestyle=(0, (2, 2)))
            a.set_axisbelow(True)
            t = a.get_title()
            a.set_title(t.replace("(", r"$\mathbf{(}$", 1).replace(")", r"$\mathbf{)}$", 1), loc="left")
        fig.savefig(outdir / f"workforce_scaling.{ext}")
    print("wrote", outdir / "workforce_scaling.pdf", "slope", np.polyfit(np.log(x), np.log(y), 1)[0] if len(x) > 1 else None)


if __name__ == "__main__":
    main()

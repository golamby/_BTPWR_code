"""Summarize and plot the privacy- and budget-parameter sweeps."""
from pathlib import Path
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    ("admission", "BTPWR", "#8C2633", "o", "-"),
    ("admission-no-promotion", "BTPWR-NP", "#B08A4F", "s", (0, (4.0, 2.0))),
    ("template-first", "C-seeded", "#5F7D8C", "^", (0, (1.2, 1.2))),
    ("old", "Mapped", "#2F5D7C", "D", (0, (5.0, 1.5, 1.0, 1.5))),
    ("etbp", "ETBP-P", "#5B7F6A", "v", (0, (6.0, 2.0))),
]


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.6,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
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
        variant = "privacy" if case["name"].startswith("pv_") else "budget"
        if variant == "privacy":
            value = float(case["eps1"])
        else:
            value = float(case["budget_override"]) / 455.0 if case["dataset"] == "brightkite" else float(case["budget_override"]) / 475.0
        for row in record["rows"]:
            rows.append({**row, "variant": variant, "value": value, "seed": case["seed"], "dataset": case["dataset"]})
    return rows


def series(rows, method, field, variant):
    out = {}
    for value in sorted({r["value"] for r in rows if r["variant"] == variant}):
        seed_means = []
        for seed in sorted({r["seed"] for r in rows if r["variant"] == variant}):
            vals = [r[field] for r in rows if r["variant"] == variant and r["value"] == value and r["seed"] == seed and r["method"] == method]
            if vals:
                seed_means.append(float(np.mean(vals)))
        v = np.asarray(seed_means, float)
        out[value] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))))
    return out


def panel(ax, rows, variant, field, ylabel, title):
    from matplotlib.colors import to_rgb
    points = sorted({r["value"] for r in rows if r["variant"] == variant})
    for method, label, color, marker, linestyle in METHODS:
        s = series(rows, method, field, variant)
        x = np.array(points)
        y = np.array([s[p][0] for p in points])
        e = np.array([s[p][1] for p in points])
        light = tuple(1.0 - 0.62 * (1.0 - c) for c in to_rgb(color))
        ax.errorbar(x, y, yerr=e, marker=marker, ms=3.2, lw=1.05,
                    linestyle=linestyle, color=color, ecolor=light,
                    elinewidth=.65, capsize=1.1, capthick=.5,
                    markeredgewidth=.7, label=label, zorder=3)
    ax.set_xlabel("location budget $\\epsilon_1$" if variant == "privacy" else "payment cap / default cap")
    ax.set_ylabel(ylabel)
    ax.set_title(title.replace("(", r"$\mathbf{(}$", 1).replace(")", r"$\mathbf{)}$", 1), loc="center")
    ax.yaxis.grid(True, color="#DEDEDE", linewidth=.5, linestyle=(0, (2, 2)))
    ax.set_axisbelow(True)


def figure(rows, variant, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.1), constrained_layout=True)
    panel(axes[0], rows, variant, "completion_percent", "task completion rate (\\%)", "(a) service")
    panel(axes[1], rows, variant, "all_answered_RMSE", "own-answer RMSE", "(b) error")
    panel(axes[2], rows, variant, "payment", "payment ($10^3$ units)", "(c) payment")
    for ax in axes:
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(ScalarFormatter())
    axes[0].legend(loc="lower right", ncol=1, handlelength=1.3, fontsize=7.5)
    for ext in ("pdf", "svg"):
        fig.savefig(outdir / f"parametric_{variant}.{ext}")
    return fig


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "parametric" / "parametric.json"
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    style()
    rows = load(path)
    summary = {}
    for variant in ("privacy", "budget"):
        figure(rows, variant, outdir)
        summary[variant] = {
            method: {field: series(rows, method, field, variant) for field in ("completion_percent", "all_answered_RMSE", "payment", "physical")}
            for method, *_ in METHODS
        }
    (outdir.parent / "results" / "parametric" / "parametric_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for variant in ("privacy", "budget"):
        print(variant)
        for method, label, *_ in METHODS:
            s = summary[variant][method]["completion_percent"]
            print(" ", label, {k: round(v[0], 2) for k, v in s.items()})


if __name__ == "__main__":
    main()



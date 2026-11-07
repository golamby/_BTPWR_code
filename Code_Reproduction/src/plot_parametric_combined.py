"""Combined two-row parameter-sweep figure for the main text."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter

from plot_parametric import METHODS, load, panel, style


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/parametric/parametric.json")
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures")
    style()
    rows = load(path)
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.55), constrained_layout=True)
    panel(axes[0, 0], rows, "privacy", "completion_percent", "completion (\\%)", "(a) privacy: service")
    panel(axes[0, 1], rows, "privacy", "all_answered_RMSE", "RMSE", "(b) privacy: error")
    panel(axes[0, 2], rows, "privacy", "payment", "payment ($10^3$)", "(c) privacy: payment")
    panel(axes[1, 0], rows, "budget", "completion_percent", "completion (\\%)", "(d) budget: service")
    panel(axes[1, 1], rows, "budget", "all_answered_RMSE", "RMSE", "(e) budget: error")
    panel(axes[1, 2], rows, "budget", "payment", "payment ($10^3$)", "(f) budget: payment")
    for ax in axes.flat:
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(ScalarFormatter())
    handles, labels = axes[0, 1].get_legend_handles_labels()
    axes[0, 1].legend(handles, labels, loc="upper right", fontsize=7.0, handlelength=1.1,
                      handletextpad=.35, borderpad=.25, labelspacing=.3,
                      frameon=True, framealpha=.92, edgecolor="#CCCCCC")
    for ext in ("pdf", "svg"):
        fig.savefig(outdir / f"parametric_combined.{ext}")
    print("wrote", outdir / "parametric_combined.pdf")


if __name__ == "__main__":
    main()

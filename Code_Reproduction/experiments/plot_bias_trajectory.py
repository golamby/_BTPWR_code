import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory containing round_metrics.csv")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    for f in ("C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/timesbd.ttf"):
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 7.0, "axes.labelsize": 7.2,
        "axes.titlesize": 7.4, "xtick.labelsize": 6.4, "ytick.labelsize": 6.4,
        "legend.fontsize": 6.2, "axes.linewidth": 0.8, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.facecolor": "white",
    })

    source = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(source / "round_metrics.csv",
                    usecols=["method", "stress", "group", "round",
                             "false_acceptance", "macro_F1"])
    b = d[(d.method == "BTPWR") & (d.group == "stress") & (d.stress == "small_bias")]
    n = d[(d.method == "BTPWR") & (d.group == "primary")]
    b = b.groupby("round")[["false_acceptance", "macro_F1"]].mean()
    n = n.groupby("round")[["false_acceptance", "macro_F1"]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.25))
    fig.subplots_adjust(left=0.13, right=0.985, top=0.84, bottom=0.29, wspace=0.38)
    for ax, col, letter, title, ylab, ylim, color, marker in [
        (axes[0], "false_acceptance", "a", "False acceptance", "False acceptance (%)", 80, "#A63A2E", "o"),
        (axes[1], "macro_F1", "b", "Identification", "Identification F1 (%)", 100, "#1F4E79", "s"),
    ]:
        ax.plot(b.index, b[col], color=color, lw=1.3, marker=marker, ms=2.6,
                markevery=[0, 4, 9, 19, 29, 49], label="1-unit bias")
        ax.plot(n.index, n[col], color="#777777", lw=0.9, ls="--", label="Nominal")
        ax.set_xlim(1, 50)
        ax.set_ylim(0, ylim)
        ax.set_xticks([1, 10, 20, 30, 40, 50])
        ax.set_xlabel("Round")
        ax.set_ylabel(ylab)
        ax.set_title(f"({letter})  {title}", loc="center", pad=4)
        ax.grid(axis="y", color="#E4E4E4", lw=0.45)
        ax.set_axisbelow(True)
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(True)
            ax.spines[s].set_linewidth(0.8)
            ax.spines[s].set_color("#333333")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.9, columnspacing=1.0, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"bias_trajectory.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", out / "bias_trajectory.pdf")


if __name__ == "__main__":
    main()

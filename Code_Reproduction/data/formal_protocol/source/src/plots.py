"""Rebuild the manuscript's nine data figures and their numerical tables."""
from contextlib import contextmanager
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import CORE

DATASETS = ["tdrive", "foursquare_tky", "brightkite"]
DATASET_LABELS = dict(zip(DATASETS, ["T-Drive", "Foursquare-TKY", "Brightkite"]))
FIELDS = ["legacy", "noaa_east_asia", "noaa_europe", "noaa_north_america"]
FIELD_LABELS = dict(
    zip(FIELDS, ["Controlled reference", "East Asia", "Europe", "North America"])
)
PRIMARY = CORE + ["MMCB", "ETBP-TD"]
STYLES = {
    "BTPWR": dict(
        color="#174C88",
        marker="s",
        linestyle="-",
        linewidth=1.65,
        markersize=3.8,
        zorder=7,
    ),
    "BPP-TW": dict(
        color="#BA3E43",
        marker="o",
        linestyle="--",
        linewidth=1.35,
        markersize=3.5,
        zorder=6,
    ),
    "BPP-T": dict(
        color="#CF842C",
        marker="^",
        linestyle="-.",
        linewidth=1.3,
        markersize=3.7,
        zorder=5,
    ),
    "BPP": dict(
        color="#75518C",
        marker="D",
        linestyle=":",
        linewidth=1.45,
        markersize=3.3,
        zorder=4,
    ),
    "Random": dict(
        color="#626A72",
        marker="x",
        linestyle=(0, (5, 2, 1, 2)),
        linewidth=1.15,
        markersize=3.8,
        zorder=2,
    ),
    "No-Privacy": dict(
        color="#1E8178",
        marker="v",
        linestyle=(0, (6, 2)),
        linewidth=1.3,
        markersize=3.7,
        zorder=3,
    ),
    "MMCB": dict(
        color="#7C8D3B",
        marker="p",
        linestyle=(0, (4, 1.5, 1, 1.5)),
        linewidth=1.25,
        markersize=3.8,
    ),
    "ETBP-TD": dict(
        color="#2B91A9", marker="P", linestyle="-.", linewidth=1.35, markersize=3.7
    ),
}
REFERENCE_NUMBERS = {"BPP": 4, "Random": 5, "No-Privacy": 5, "MMCB": 9, "ETBP-TD": 8}
AXIS_LABELS = {
    "eps1": r"Location privacy $\epsilon_1$",
    "honest_fraction": "Honest-worker fraction",
    "budget": "Budget (payment units)",
    "n": "Registered workers",
    "gamma": r"Score sensitivity $\gamma$",
    "certified_fraction": "Initially certified fraction",
}
AXIS_FILES = {
    "eps1": "v6_epsilon1_service",
    "honest_fraction": "v6_honest_fraction_mechanism",
    "budget": "v6_budget_mechanism",
    "n": "v6_worker_population",
    "gamma": "v6_score_sensitivity",
    "certified_fraction": "v6_certified_fraction",
}


@contextmanager
def publication_style():
    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.labelsize": 8.2,
        "axes.titlesize": 8.4,
        "axes.linewidth": 0.75,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.bottom": True,
        "axes.spines.left": True,
        "axes.edgecolor": "#20252B",
        "axes.labelcolor": "#20252B",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "legend.fontsize": 7.6,
        "legend.frameon": False,
        "legend.handlelength": 2.3,
        "legend.handletextpad": 0.55,
        "legend.columnspacing": 1.15,
        "legend.labelspacing": 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "savefig.dpi": 240,
    }
    with matplotlib.rc_context(rc):
        yield


def journal_axes(ax, xlabel, ylabel, title="", panel=""):
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.set_xlabel(xlabel, labelpad=2.5)
    ax.set_ylabel(ylabel, labelpad=3.0)
    ax.set_title((f"({panel}) " + title).strip() if panel else title, x=0.5, pad=4.0)
    ax.tick_params(top=True, right=True)
    ax.margins(x=0.025)


def method_curve(ax, x, y, method, references, markevery=1):
    opts = dict(STYLES[method], markeredgewidth=0.7)
    if method != "BTPWR":
        opts["markerfacecolor"] = "white"
    label = f"{method} [{references[method]}]" if method in references else method
    return ax.plot(x, y, label=label, markevery=markevery, **opts)[0]


def shared_legend(fig, axes, ncol=3, y=0.995):
    handles, labels = [], []
    for ax in np.asarray(list(np.asarray(axes, dtype=object).flat), dtype=object):
        h, names = ax.get_legend_handles_labels()
        for handle, name in zip(h, names):
            if name not in labels:
                handles.append(handle)
                labels.append(name)
    # Prefer a full-width single row; wrap only when its measured width requires it.
    for columns in range(len(handles), 0, -1):
        legend = fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
            ncol=columns, borderaxespad=0.0, handlelength=1.8,
            handletextpad=0.45, columnspacing=1.0,
        )
        fig.canvas.draw()
        if legend.get_window_extent(fig.canvas.get_renderer()).width <= fig.bbox.width * 0.98:
            break
        legend.remove()


def export(fig, folder, name):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    clipped = []
    unused_ticks = set()
    for ax in fig.axes:
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(limits)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo <= tick.get_loc() <= hi:
                    unused_ticks.update((id(tick.label1), id(tick.label2)))
    for artist in fig.findobj(matplotlib.text.Text):
        if (
            id(artist) not in unused_ticks
            and artist.get_visible()
            and artist.get_text().strip()
        ):
            box = artist.get_window_extent(renderer)
            if (
                box.width
                and box.height
                and (
                    box.x0 < bounds.x0 - 1.5
                    or box.y0 < bounds.y0 - 1.5
                    or box.x1 > bounds.x1 + 1.5
                    or box.y1 > bounds.y1 + 1.5
                )
            ):
                clipped.append(artist.get_text())
    if clipped:
        raise ValueError(f"{name}: labels outside canvas: {clipped}")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(folder / (name + "." + ext))
    plt.close(fig)
    return name


def pooled(rows):
    n = rows.answered_task_times.sum()
    requested = rows.requested_task_times.sum()
    return {
        "TCR": 100 * rows.completed_task_times.sum() / requested
        if requested
        else np.nan,
        "AR": 100 * n / requested if requested else np.nan,
        "N": int(n),
        "MAE": rows.native_absolute_error_sum.sum() / n if n else np.nan,
        "RMSE": np.sqrt(rows.native_squared_error_sum.sum() / n) if n else np.nan,
    }


def default_figure(rounds, folder, references):
    a = rounds[rounds.kind == "confirmation"]
    if a.empty or not set(PRIMARY).issubset(a.method.unique()):
        raise ValueError("The default figure needs all eight comparison methods")
    with publication_style():
        fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35))
        fig.subplots_adjust(left=0.066, right=0.99, bottom=0.23, top=0.73, wspace=0.34)
        for i, dataset in enumerate(DATASETS):
            for method in PRIMARY:
                available = a[(a.dataset == dataset) & (a.method == method)]
                if set(available.seed.unique()) != {602, 603, 604}:
                    plt.close(fig)
                    raise ValueError(
                        f"{dataset}/{method}: three complete default campaigns required"
                    )
                if (
                    not available.groupby("seed")["round"]
                    .apply(lambda x: set(x) == set(range(1, 51)))
                    .all()
                ):
                    plt.close(fig)
                    raise ValueError(f"{dataset}/{method}: missing sensing rounds")
                r = available.groupby("round").running_completion_percent.mean()
                method_curve(
                    axes[i], r.index, r.values, method, references, markevery=8
                )
            journal_axes(
                axes[i],
                "Sensing round",
                "Cumulative completion (%)",
                DATASET_LABELS[dataset],
                "abc"[i],
            )
            axes[i].set_xlim(1, 50)
            axes[i].set_ylim(0, 100)
            axes[i].set_xticks([1, 10, 20, 30, 40, 50])
        shared_legend(fig, axes, ncol=4)
        return export(fig, folder, "v6_default_completion")


def axis_figure(axis, slots, metrics, common, folder, references):
    rows = []
    for name, dataset, value in slots:
        for method in CORE:
            a = metrics[(metrics["case"] == name) & (metrics.method == method)]
            b = common[(common["case"] == name) & (common.method == method)]
            if len(a) != 1 or len(b) != 1:
                raise ValueError(f"{axis}/{name}/{method}: missing generated metrics")
            rows.append(
                dict(
                    case=name,
                    dataset=dataset,
                    method=method,
                    x=value,
                    TCR=a.iloc[0].completion_percent,
                    RMSE_common_all6=b.iloc[0].RMSE,
                    common_N=b.iloc[0].N,
                )
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(folder) / (AXIS_FILES[axis] + ".csv"), index=False)
    two = axis in ("eps1", "honest_fraction")
    with publication_style():
        fig, axes = plt.subplots(
            2 if two else 1, 3, figsize=(7.16, 4.20 if two else 2.35), squeeze=False
        )
        fig.subplots_adjust(
            left=0.067,
            right=0.99,
            bottom=0.14 if two else 0.24,
            top=0.88 if two else 0.82,
            wspace=0.34,
            hspace=0.54,
        )
        for i, dataset in enumerate(DATASETS):
            for method in CORE:
                a = frame[
                    (frame.dataset == dataset) & (frame.method == method)
                ].sort_values("x")
                method_curve(axes[0, i], a.x, a.TCR, method, references)
                if two:
                    method_curve(
                        axes[1, i], a.x, a.RMSE_common_all6, method, references
                    )
            journal_axes(
                axes[0, i],
                AXIS_LABELS[axis],
                "Completion (%)",
                DATASET_LABELS[dataset],
                "abc"[i],
            )
            axes[0, i].set_ylim(0, 100)
            if two:
                journal_axes(
                    axes[1, i],
                    AXIS_LABELS[axis],
                    "RMSE on common tasks",
                    panel="def"[i],
                )
                axes[1, i].set_ylim(bottom=0)
            xs = sorted(frame[frame.dataset == dataset].x.unique())
            for row in axes:
                row[i].set_xticks(xs if len(xs) <= 7 else xs[::2])
        shared_legend(fig, axes, ncol=3)
        return export(fig, folder, AXIS_FILES[axis])


def source_figure(metrics, risk, contamination, folder):
    methods = [
        "C-only blocked KRR",
        "C-only original RBF",
        "C-only linear",
        "All-source recovery service",
        "Direct TD",
    ]
    labels = ["BTPWR", "Fixed KRR", "Linear recovery", "All-source KRR", "BPP-TW"]
    colors = ["#174C88", "#1E8178", "#626A72", "#BA3E43", "#CF842C"]
    unobserved = risk[risk["slice"] == "unobserved_common_recovery4"]
    rows = []
    for method in methods:
        row = {"method": method}
        for dataset in DATASETS:
            a = metrics[(metrics.dataset == dataset) & (metrics.method == method)]
            if set(a.seed.unique()) != {602, 603, 604}:
                raise ValueError(f"{dataset}/{method}: missing source-study campaigns")
            b = unobserved[
                (unobserved.dataset == dataset) & (unobserved.method == method)
            ]
            n = b.N.sum()
            row.update(
                {
                    dataset + "_TCR": a.completion_percent.mean(),
                    dataset + "_unobserved_N": int(n),
                    dataset + "_MAE": b.SAE.sum() / n if n else np.nan,
                    dataset + "_RMSE": np.sqrt(b.SSE.sum() / n) if n else np.nan,
                }
            )
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("method")
    frame.to_csv(Path(folder) / "source_responsibility.csv")
    counts = contamination.groupby("dataset").sum(numeric_only=True)
    with publication_style():
        fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.38))
        fig.subplots_adjust(left=0.055, right=0.995, bottom=0.28, top=0.82, wspace=0.46)
        x = np.arange(3)
        for j, method in enumerate(methods):
            axes[0].bar(
                x + (j - 2) * 0.14,
                [frame.loc[method, d + "_TCR"] for d in DATASETS],
                0.14,
                color=colors[j],
                edgecolor="white",
                lw=0.25,
                label=labels[j],
            )
        for panel, key in ((1, "MAE"), (2, "RMSE")):
            for j, method in enumerate(methods[:4]):
                axes[panel].bar(
                    x + (j - 1.5) * 0.17,
                    [frame.loc[method, d + "_" + key] for d in DATASETS],
                    0.17,
                    color=colors[j],
                    edgecolor="white",
                    lw=0.25,
                )
        bad = [
            100
            * counts.loc[d, "all_anchor_malicious_report_count"]
            / counts.loc[d, "all_anchor_report_count"]
            for d in DATASETS
        ]
        c_bad = [
            100
            * counts.loc[d, "C_anchor_malicious_report_count"]
            / counts.loc[d, "C_anchor_report_count"]
            for d in DATASETS
        ]
        axes[3].bar(x + 0.1, bad, 0.34, color=colors[3], edgecolor="white", lw=0.25)
        axes[3].scatter(
            x - 0.2, c_bad, s=13, color=colors[0], marker="s", zorder=5, clip_on=False
        )
        for xx, value in zip(x, c_bad):
            axes[3].annotate(
                f"{value:g}",
                (xx - 0.2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.2,
                color=colors[0],
            )
        titles = [
            "Task completion",
            "Recovery MAE",
            "Recovery RMSE",
            "Source contamination",
        ]
        ylabs = [
            "Completion (%)",
            "MAE (RH pp)",
            "RMSE (RH pp)",
            "Malicious sources (%)",
        ]
        for i, ax in enumerate(axes):
            journal_axes(ax, "", ylabs[i], titles[i], "abcd"[i])
            ax.set_title(f'({"abcd"[i]}) {titles[i]}', fontsize=7.2, pad=5)
            ax.set_ylabel(ylabs[i], fontsize=7.2, labelpad=3)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [DATASET_LABELS[d] for d in DATASETS],
                rotation=30,
                ha="right",
                fontsize=6.5,
            )
            ax.tick_params(labelsize=6.5, length=2.6, width=0.65)
            ax.set_xlim(-0.58, 2.58)
        axes[0].set_ylim(0, 100)
        axes[1].set_ylim(0, max(5.1, axes[1].get_ylim()[1]))
        axes[2].set_ylim(0, max(7, axes[2].get_ylim()[1]))
        axes[3].set_ylim(0, max(6, axes[3].get_ylim()[1]))
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="upper center",
            bbox_to_anchor=(0.51, 1.005),
            ncol=5,
            frameon=False,
            fontsize=7.1,
            handlelength=1.45,
            handletextpad=0.35,
            columnspacing=1.0,
        )
        return export(fig, folder, "source_responsibility")


def paired_figure(intervals, folder):
    frame = intervals[intervals.comparator == "ETBP-priority+newhead"]
    if frame.empty:
        raise ValueError("No generated resource-matched priority comparisons")
    colors = dict(zip(DATASETS, ["#C84C0F", "#297FBD", "#4F9E55"]))
    markers = dict(zip(DATASETS, ["o", "s", "^"]))
    with publication_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.0))
        fig.subplots_adjust(
            left=0.18, right=0.98, bottom=0.10, top=0.91, wspace=0.68, hspace=0.43
        )
        for row, group in enumerate(("fresh_v3", "cohort_v3")):
            for col, metric in enumerate(("completion_percent", "RMSE")):
                ax = axes[row, col]
                a = frame[(frame.group == group) & (frame.metric == metric)]
                order = [
                    (f, d)
                    for f in FIELDS
                    for d in DATASETS
                    if len(a[(a.field == f) & (a.dataset == d)])
                ]
                expected = [
                    (f, d)
                    for f in (FIELDS if group == "fresh_v3" else FIELDS[1:])
                    for d in DATASETS
                ]
                if order != expected:
                    plt.close(fig)
                    raise ValueError(
                        f"{group}/{metric}: incomplete confirmation intervals"
                    )
                for y, (field, dataset) in enumerate(order):
                    r = a[(a.field == field) & (a.dataset == dataset)].iloc[0]
                    value = r.mean_difference
                    ax.errorbar(
                        value,
                        y,
                        xerr=np.array([[value - r.CI95_low], [r.CI95_high - value]]),
                        fmt=markers[dataset],
                        color=colors[dataset],
                        markersize=3.3,
                        elinewidth=0.8,
                        capsize=2,
                    )
                ax.axvline(0, color=".65", linewidth=0.7)
                ax.set_yticks(range(len(order)))
                ax.set_yticklabels(
                    [
                        FIELD_LABELS[f]
                        + " / "
                        + ("Tokyo" if d == "foursquare_tky" else DATASET_LABELS[d])
                        for f, d in order
                    ],
                    fontsize=6.0,
                )
                ax.invert_yaxis()
                journal_axes(
                    ax,
                    "Completion difference (pp)"
                    if col == 0
                    else "Common-task RMSE difference",
                    "Reference / mobility pattern",
                    "Fixed100: eight RNG replicates"
                    if row == 0
                    else "Top200: four cohort draws",
                    "abcd"[2 * row + col],
                )
        return export(fig, folder, "service_v3_tent_paired_confirmation")


def numerical_tables(metrics, pairs, recognition, folder, axis_slots=None):
    folder = Path(folder)
    defaults = metrics[metrics.kind == "confirmation"]
    rows = [
        {"dataset": ds, "method": method, **pooled(a)}
        for (ds, method), a in defaults.groupby(["dataset", "method"])
    ]
    pd.DataFrame(rows).to_csv(folder / "default_metrics.csv", index=False)
    p = pairs[pairs.kind == "confirmation"]
    paired = []
    for (ds, comparator), a in p.groupby(["dataset", "comparator"]):
        n = a.N.sum()
        row = dict(dataset=ds, comparator=comparator, N=int(n))
        for prefix in ("BTPWR", "comparator"):
            row[prefix + "_MAE"] = a[prefix + "_SAE"].sum() / n if n else np.nan
            row[prefix + "_RMSE"] = (
                np.sqrt(a[prefix + "_SSE"].sum() / n) if n else np.nan
            )
        paired.append(row)
    pd.DataFrame(paired).to_csv(folder / "common_task_precision.csv", index=False)
    methods = CORE[1:] + ["MMCB", "ETBP-TD", "ETBP-priority+newhead"]
    tex = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"&\multicolumn{3}{c}{T-Drive}&\multicolumn{3}{c}{Foursquare-TKY}&\multicolumn{3}{c}{Brightkite}\\",
        r"Comparator&$N$&MAE pair&RMSE pair&$N$&MAE pair&RMSE pair&$N$&MAE pair&RMSE pair\\\midrule",
    ]
    for method in methods:
        cells = [r"ETBP-priority$+R$" if method == "ETBP-priority+newhead" else method]
        for ds in DATASETS:
            found = [
                r for r in paired if r["dataset"] == ds and r["comparator"] == method
            ]
            if not found:
                cells.extend(["--", "--", "--"])
                continue
            row = found[0]
            cells.append(f"{row['N']:,}")
            for key in ("MAE", "RMSE"):
                a, b = row["BTPWR_" + key], row["comparator_" + key]
                aa, bb = f"{a:.2f}", f"{b:.2f}"
                aa = r"\textbf{" + aa + "}" if a <= b else aa
                bb = r"\textbf{" + bb + "}" if b <= a else bb
                cells.append(aa + "/" + bb)
        tex.append("&".join(cells) + r"\\")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (folder / "common_task_precision.tex").write_text(
        "\n".join(tex) + "\n", encoding="utf-8"
    )
    advantages = []
    for ds in DATASETS:
        a = defaults[(defaults.dataset == ds) & (defaults.method == "BTPWR")]
        if a.empty:
            continue
        own = pooled(a)
        for comparator in defaults.method.unique():
            b = defaults[(defaults.dataset == ds) & (defaults.method == comparator)]
            if b.empty or comparator == "BTPWR":
                continue
            other = pooled(b)
            row = dict(
                dataset=ds,
                comparator=comparator,
                completion_difference_pp=own["TCR"] - other["TCR"],
            )
            for pair in paired:
                if pair["dataset"] == ds and pair["comparator"] == comparator:
                    for metric in ("MAE", "RMSE"):
                        denom = pair["comparator_" + metric]
                        row[metric + "_reduction_percent"] = (
                            100 * (denom - pair["BTPWR_" + metric]) / denom
                            if denom
                            else np.nan
                        )
            advantages.append(row)
    pd.DataFrame(advantages).to_csv(folder / "default_comparisons.csv", index=False)
    if not recognition.empty:
        gamma_cases = [r[0] for r in (axis_slots or {}).get("gamma", [])]
        rec = recognition[
            (recognition.seed == 601)
            & (recognition.rounds_completed == 50)
            & recognition.method.isin(["BTPWR", "BPP-T"])
        ]
        if gamma_cases:
            rec = rec[rec["case"].isin(gamma_cases)]
        use = metrics[(metrics.seed == 601) & metrics.method.isin(["BTPWR", "BPP-T"])]
        keys = ["case", "method"]
        rec.merge(
            use[
                keys
                + [
                    "selected_worker_rounds",
                    "selected_malicious_worker_rounds",
                    "malicious_recruited_fraction",
                ]
            ],
            on=keys,
            how="left",
        ).to_csv(folder / "gamma_recognition.csv", index=False)


def auxiliary_tables(result_dir, metrics, folder):
    root, folder = Path(result_dir), Path(folder)
    fresh = metrics[metrics.group.isin(["fresh_v3", "cohort_v3"])]
    rows = []
    for (group, field, method), a in fresh.groupby(["group", "field", "method"]):
        values = [pooled(a[a.dataset == ds]) for ds in DATASETS]
        rows.append(
            dict(
                group=group,
                field=field,
                method=method,
                campaigns=len(a),
                **{
                    key: float(np.mean([r[key] for r in values]))
                    for key in ("TCR", "MAE", "RMSE")
                },
            )
        )
    if rows:
        pd.DataFrame(rows).to_csv(folder / "fresh_reference_metrics.csv", index=False)
    op = root / "operators"
    if (op / "metrics.csv").exists() and (op / "common.csv").exists():
        metrics = pd.read_csv(op / "metrics.csv")
        common = pd.read_csv(op / "common.csv")
        common = common[common["slice"] == "missing_actual_common"]
        rows = []
        for (ds, method), a in metrics.groupby(["dataset", "method"]):
            b = common[(common.dataset == ds) & (common.method == method)]
            n = b.N.sum()
            rows.append(
                dict(
                    dataset=ds,
                    method=method,
                    TCR=a.completion_percent.mean(),
                    missing_N=int(n),
                    missing_MAE=b.SAE.sum() / n if n else np.nan,
                    missing_RMSE=np.sqrt(b.SSE.sum() / n) if n else np.nan,
                )
            )
        pd.DataFrame(rows).to_csv(folder / "operator_comparison.csv", index=False)


def plot_all(result_dir, out_dir, strict=True, references=None, axis_slots=None):
    root, out = Path(result_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    evaluation = root / "evaluation"

    def read(name):
        path = evaluation / (name + ".csv")
        return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()

    metrics, rounds, common = (
        read("combined_metrics"),
        read("combined_rounds"),
        read("all6_common_metrics"),
    )
    references = references or REFERENCE_NUMBERS
    if axis_slots is None:
        config = Path(__file__).parents[1] / "experiments.json"
        axis_slots = (
            json.loads(config.read_text(encoding="utf-8")).get("axes", {})
            if config.exists()
            else {}
        )
    made, missing = [], []
    jobs = [("v6_default_completion", lambda: default_figure(rounds, out, references))]
    for axis in AXIS_FILES:
        jobs.append(
            (
                AXIS_FILES[axis],
                lambda axis=axis: axis_figure(
                    axis, axis_slots[axis], metrics, common, out, references
                ),
            )
        )

    def source():
        sg = root / "source_gate"
        return source_figure(
            pd.read_csv(sg / "metrics.csv"),
            pd.read_csv(sg / "risk_slices.csv"),
            pd.read_csv(sg / "anchor_contamination.csv"),
            out,
        )

    jobs.extend(
        [
            ("source_responsibility", source),
            (
                "service_v3_tent_paired_confirmation",
                lambda: paired_figure(read("paired_intervals"), out),
            ),
        ]
    )
    for name, render in jobs:
        try:
            made.append(render())
        except (FileNotFoundError, KeyError, ValueError, AttributeError) as e:
            missing.append({"figure": name, "reason": str(e)})
    if not metrics.empty:
        numerical_tables(
            metrics,
            read("common_answer_metrics"),
            read("combined_recognition"),
            out,
            axis_slots,
        )
        auxiliary_tables(root, metrics, out)
    receipt = {"figures": made, "missing": missing, "complete": len(made) == 9}
    (out / "figure_index.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    if strict and missing:
        raise ValueError(
            "Incomplete generated experiment inputs: " + json.dumps(missing)
        )
    return receipt

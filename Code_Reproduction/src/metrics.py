"""Requested-task, worker-recognition and campaign-paired evaluation."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

CORE = ["BTPWR", "BPP-TW", "BPP-T", "BPP", "Random", "No-Privacy"]


def _divide(a, b):
    return np.divide(
        a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=np.asarray(b) > 0
    )


def requested_metrics(truth, prediction, request_mask):
    g, p, mask = (
        np.asarray(truth, float),
        np.asarray(prediction, float),
        np.asarray(request_mask, bool),
    )
    if (
        g.ndim != 2
        or p.shape != g.shape
        or mask.shape != g.shape
        or not np.isfinite(g).all()
    ):
        raise ValueError("Finite reference and matched round-by-region arrays required")
    sizes = mask.sum(1)
    if not len(sizes) or np.any(sizes == 0):
        raise ValueError("Each round needs at least one requested task")
    finite = mask & np.isfinite(p)
    error = np.where(finite, p - g, 0.0)
    # Missing answers fail completion but receive no artificial error penalty.
    success = finite & np.where(g == 0, error == 0, abs(error) < 0.1 * abs(g))
    answers, good = finite.sum(1), success.sum(1)
    ae, se = abs(error).sum(1), (error * error).sum(1)
    n, total = int(answers.sum()), int(sizes.sum())
    ca, cs, cae, cse = answers.cumsum(), good.cumsum(), ae.cumsum(), se.cumsum()
    return {
        "evaluation_domain": "native physical requested-task values",
        "success_rule": "strict abs(error)<.1abs(truth); at truth0 exact finite zero only",
        "n_rounds": len(g),
        "n_grid_cells": g.shape[1],
        "requested_tasks_per_round": int(sizes[0])
        if np.all(sizes == sizes[0])
        else None,
        "requested_task_times": total,
        "answered_task_times": n,
        "answered_rate_percent": 100 * n / total,
        "completed_task_times": int(good.sum()),
        "completion_percent": 100 * good.sum() / total,
        "native_absolute_error_sum": float(ae.sum()),
        "native_squared_error_sum": float(se.sum()),
        "all_answered_MAE": float(ae.sum() / n) if n else None,
        "all_answered_RMSE": float(np.sqrt(se.sum() / n)) if n else None,
        "per_round_answered_count": answers,
        "per_round_completed_count": good,
        "per_round_completion_percent": 100 * good / sizes,
        "per_round_answered_rate_percent": 100 * answers / sizes,
        "per_round_native_absolute_error_sum": ae,
        "per_round_native_squared_error_sum": se,
        "per_round_all_answered_MAE": _divide(ae, answers),
        "per_round_all_answered_RMSE": np.sqrt(_divide(se, answers)),
        "running_all_answered_MAE": _divide(cae, ca),
        "running_all_answered_RMSE": np.sqrt(_divide(cse, ca)),
        "running_completion_percent": 100 * cs / sizes.cumsum(),
        "running_answered_rate_percent": 100 * ca / sizes.cumsum(),
    }


def recognition_metrics(reputation, honest, certified=None):
    r, h = np.asarray(reputation, float), np.asarray(honest, bool)
    if r.ndim != 1 or h.shape != r.shape or not np.isfinite(r).all():
        raise ValueError("Finite worker scores and matched evaluator labels required")
    c, m = r >= 0.8, r < 0.3
    u = ~(c | m)

    def rate(v, cohort):
        return float(v[cohort].mean()) if cohort.any() else None

    hr, mr = rate(c, h), rate(m, ~h)
    out = {
        "workers": len(r),
        "C_workers": int(c.sum()),
        "U_workers": int(u.sum()),
        "M_workers": int(m.sum()),
        "recognition_accuracy": float(((h & c) | (~h & m)).mean()),
        "balanced_class_recall": (hr + mr) / 2
        if hr is not None and mr is not None
        else None,
        "honest_C_recall": hr,
        "malicious_M_recall": mr,
        "unknown_fraction": float(u.mean()),
        "honest_false_M_rate": rate(m, h),
        "malicious_false_C_rate": rate(c, ~h),
        "honest_U_fraction": rate(u, h),
        "malicious_U_fraction": rate(u, ~h),
    }
    if certified is not None:
        certified = np.asarray(certified, bool)
        if certified.shape != h.shape:
            raise ValueError("Certified cohort shape mismatch")
        out.update(
            ordinary_honest_C_recall=rate(c, h & ~certified),
            given_certified_workers=int(certified.sum()),
        )
    return out


def recruited_malicious_metrics(selected, honest):
    selected, h = np.asarray(selected, bool), np.asarray(honest, bool)
    if selected.ndim != 2 or selected.shape[1:] != h.shape:
        raise ValueError(
            "Matched round-by-worker selections and cohort labels required"
        )
    total = selected.sum(1)
    bad = (selected & ~h[None, :]).sum(1)
    return {
        "selected_worker_rounds": int(total.sum()),
        "selected_malicious_worker_rounds": int(bad.sum()),
        "malicious_recruited_fraction": float(bad.sum() / total.sum())
        if total.sum()
        else None,
        "per_round_selected_workers": total,
        "per_round_selected_malicious_workers": bad,
        "per_round_malicious_recruited_fraction": _divide(bad, total),
    }


def error_sums(truth, prediction, mask):
    e = np.asarray(prediction)[mask] - np.asarray(truth)[mask]
    n = len(e)
    return {
        "N": n,
        "SAE": float(abs(e).sum()),
        "SSE": float((e * e).sum()),
        "MAE": float(abs(e).mean()) if n else None,
        "RMSE": float(np.sqrt((e * e).mean())) if n else None,
    }


def summarize_case(case, world, payload):
    """Evaluate generated predictions; evaluator labels never enter algorithms."""
    models, selected = payload["models"], payload["selected"]
    reps = payload.get("reputations", {})
    truth = payload.get("truth_native", np.asarray(world["truth_native_private"]).T)
    mask = np.asarray(payload.get("task_mask", world["task_mask"]), bool)
    honest, cert = np.asarray(world["honest_private"], bool), world["certified"]
    meta = {
        k: v
        for k, v in case.items()
        if k not in ("world", "old_prediction", "old_csv_files")
    }
    meta.update(
        case=case["name"],
        honest_fraction=case.get("H"),
        certified_fraction=case.get("C"),
    )
    out = {
        k: []
        for k in (
            "summary",
            "rounds",
            "recognition",
            "tails",
            "pairs",
            "common",
            "interaction",
        )
    }
    for method, pred in models.items():
        pred = np.asarray(pred)
        met = requested_metrics(truth, pred, mask)
        recruit = recruited_malicious_metrics(selected[method], honest)
        scalars = {
            k: v
            for d in (met, recruit)
            for k, v in d.items()
            if not isinstance(v, np.ndarray)
        }
        out["summary"].append({**meta, "method": method, **scalars})
        e = abs(pred[mask & np.isfinite(pred)] - truth[mask & np.isfinite(pred)])
        out["tails"].append(
            {
                **meta,
                "method": method,
                "N": len(e),
                **{
                    f"q{q}": float(np.quantile(e, q / 100)) if len(e) else None
                    for q in (50, 90, 95, 99)
                },
                "CVaR95": float(e[e >= np.quantile(e, 0.95)].mean())
                if len(e)
                else None,
            }
        )
        for t in range(len(truth)):
            out["rounds"].append(
                {
                    **meta,
                    "method": method,
                    "round": t + 1,
                    "completion_percent": met["per_round_completion_percent"][t],
                    "answered_tasks": int(met["per_round_answered_count"][t]),
                    "completed_tasks": int(met["per_round_completed_count"][t]),
                    "running_completion_percent": met["running_completion_percent"][t],
                    "running_answered_rate_percent": met[
                        "running_answered_rate_percent"
                    ][t],
                    "absolute_error_sum": met["per_round_native_absolute_error_sum"][t],
                    "squared_error_sum": met["per_round_native_squared_error_sum"][t],
                    "selected_workers": int(np.asarray(selected[method])[t].sum()),
                }
            )
        for t, r in enumerate(reps.get(method, [])):
            if np.isnan(r).all():
                continue
            out["recognition"].append(
                {
                    **meta,
                    "method": method,
                    "rounds_completed": t,
                    **recognition_metrics(r, honest, cert),
                }
            )
        if method != "BTPWR":
            domain = mask & np.isfinite(pred) & np.isfinite(models["BTPWR"])
            a, b = (
                error_sums(truth, models["BTPWR"], domain),
                error_sums(truth, pred, domain),
            )
            out["pairs"].append(
                {
                    **meta,
                    "comparator": method,
                    "N": a["N"],
                    **{"BTPWR_" + k: v for k, v in a.items() if k != "N"},
                    **{"comparator_" + k: v for k, v in b.items() if k != "N"},
                }
            )
    if all(m in models for m in CORE):
        domain = mask & np.stack([np.isfinite(models[m]) for m in CORE]).all(0)
        for method in CORE:
            out["common"].append(
                {**meta, "method": method, **error_sums(truth, models[method], domain)}
            )
    lookup = {r["method"]: r for r in out["summary"]}
    if all(m in lookup for m in ("BTPWR", "BPP-TW", "Binary+newhead", "BPP-T")):
        delta = (
            lookup["BTPWR"]["completion_percent"]
            - lookup["BPP-TW"]["completion_percent"]
            - lookup["Binary+newhead"]["completion_percent"]
            + lookup["BPP-T"]["completion_percent"]
        )
        out["interaction"].append({**meta, "completion_interaction_pp": delta})
    return out


def paired_intervals(summary, pairs):
    """Student intervals use independent campaign seeds, not task-time samples."""
    if summary.empty or pairs.empty:
        return pd.DataFrame()
    rows = []
    for group in ("fresh_v3", "cohort_v3"):
        source = pairs[pairs.group == group]
        sums = summary[summary.group == group]
        if source.empty:
            continue
        lookup = sums.set_index(["seed", "field", "dataset", "method"])

        def delta_for(frame, metric, comparator):
            if metric in ("MAE", "RMSE"):
                return (
                    frame["BTPWR_" + metric].to_numpy()
                    - frame["comparator_" + metric].to_numpy()
                )
            return np.array(
                [
                    lookup.loc[(r.seed, r.field, r.dataset, "BTPWR"), metric]
                    - lookup.loc[(r.seed, r.field, r.dataset, comparator), metric]
                    for r in frame.itertuples()
                ]
            )

        for (field, dataset, comparator), frame in source.groupby(
            ["field", "dataset", "comparator"]
        ):
            for metric in (
                "completion_percent",
                "answered_rate_percent",
                "MAE",
                "RMSE",
            ):
                delta = delta_for(frame, metric, comparator)
                n, mean = len(delta), float(delta.mean())
                half = (
                    float(student_t.ppf(0.975, n - 1) * delta.std(ddof=1) / np.sqrt(n))
                    if n > 1
                    else np.nan
                )
                rows.append(
                    dict(
                        group=group,
                        field=field,
                        dataset=dataset,
                        comparator=comparator,
                        metric=metric,
                        n_campaigns=n,
                        mean_difference=mean,
                        CI95_low=mean - half,
                        CI95_high=mean + half,
                        unit="Fixed-cohort RNG replicate"
                        if group == "fresh_v3"
                        else "Conditional training-pool cohort draw",
                        comparison="BTPWR minus comparator",
                    )
                )
        for comparator, frame in source.groupby("comparator"):
            for metric in (
                "completion_percent",
                "answered_rate_percent",
                "MAE",
                "RMSE",
            ):
                delta = np.array(
                    [
                        delta_for(f, metric, comparator).mean()
                        for _, f in frame.groupby("seed")
                    ]
                )
                n, mean = len(delta), float(delta.mean())
                half = (
                    float(student_t.ppf(0.975, n - 1) * delta.std(ddof=1) / np.sqrt(n))
                    if n > 1
                    else np.nan
                )
                rows.append(
                    dict(
                        group=group,
                        field="ALL_FIXED_STRATA",
                        dataset="ALL_FIXED_STRATA",
                        comparator=comparator,
                        metric=metric,
                        n_campaigns=n,
                        mean_difference=mean,
                        CI95_low=mean - half,
                        CI95_high=mean + half,
                        unit="Seed-cluster mean over fixed field/pattern strata",
                        comparison="BTPWR minus comparator",
                    )
                )
    return pd.DataFrame(rows)


TABLE_FILES = {
    "summary": "combined_metrics",
    "rounds": "combined_rounds",
    "recognition": "combined_recognition",
    "tails": "error_quantiles",
    "pairs": "common_answer_metrics",
    "common": "all6_common_metrics",
    "interaction": "factorial_interactions",
}


def save_evaluation(tables, result_dir):
    folder = Path(result_dir) / "evaluation"
    folder.mkdir(parents=True, exist_ok=True)
    frames = {}
    for key, stem in TABLE_FILES.items():
        frame = tables.get(key, [])
        frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
        frames[key] = frame
        path = folder / (stem + ".csv")
        if not frame.empty:
            frame.to_csv(path, index=False)
        elif path.exists():
            path.unlink()
    ci = paired_intervals(frames["summary"], frames["pairs"])
    if not ci.empty:
        ci.to_csv(folder / "paired_intervals.csv", index=False)
    elif (folder / "paired_intervals.csv").exists():
        (folder / "paired_intervals.csv").unlink()
    return frames


def collect_evaluation(result_dir):
    """Collect per-case JSON evaluation records generated by the simulator."""
    tables = {k: [] for k in TABLE_FILES}
    files = sorted((Path(result_dir) / "cases").glob("*/evaluation.json"))
    if not files:
        raise FileNotFoundError("No generated cases/*/evaluation.json found")
    for path in files:
        case = json.loads(path.read_text(encoding="utf-8"))
        for key in tables:
            tables[key].extend(case.get(key, []))
    return save_evaluation(tables, result_dir)

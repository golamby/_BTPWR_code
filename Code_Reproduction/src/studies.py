"""Same-acquisition source and recovery-operator controls."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from . import recovery, trust, bgmc
from .metrics import requested_metrics, error_sums

SOURCE_METHODS = [
    "C-only blocked KRR",
    "C-only original RBF",
    "C-only linear",
    "All-source recovery service",
    "Direct TD",
]


def _save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _metadata(case):
    return {k: case[k] for k in ("dataset", "seed")} | {"case": case["name"]}


def source_study(case, world, payload, result_dir, calibration=None):
    """Vary only recovery sources/operators; actual feedback stays frozen."""
    folder = Path(result_dir) / "source_gate"
    meta = _metadata(case)
    h, offset, scale = (
        world["public_reference"],
        float(world["native_offset"]),
        float(world["native_scale"]),
    )
    credible = payload["credible"]["tent-density"]
    actual_check = payload["actual"]["tent-density"]
    actual_native = payload["models"]["BPP-TW"]
    models = [payload["models"]["BTPWR"]]
    fixed = recovery.initialize(h, offset, scale, calibration)
    all_source = recovery.initialize(h, offset, scale, calibration)
    models.append(
        np.asarray(
            [
                recovery.fixed_complete(c, a, fixed)[0]
                for c, a in zip(credible, actual_native)
            ]
        )
    )
    models.append(
        np.asarray(
            [
                recovery.linear_complete(c, a, fixed["linear"])[0]
                for c, a in zip(credible, actual_native)
            ]
        )
    )
    models.append(
        np.asarray(
            [
                recovery.complete(c, a, all_source)[0]
                for c, a in zip(actual_check, actual_native)
            ]
        )
    )
    models.append(actual_native)
    predictions = np.asarray(models)
    _save_npz(
        folder / "predictions" / (case["name"] + ".npz"),
        prediction_native=predictions,
        methods=np.array(SOURCE_METHODS),
        actual_mask=np.isfinite(actual_native),
        credible_mask=np.isfinite(credible),
    )
    truth, mask = world["truth_native_private"].T, np.asarray(world["task_mask"], bool)
    actual = np.isfinite(actual_native)
    common = mask & ~actual & np.isfinite(predictions[:4]).all(0)
    metrics, slices = [], []
    for name, prediction in zip(SOURCE_METHODS, predictions):
        met = requested_metrics(truth, prediction, mask)
        metrics.append(
            {
                **meta,
                "method": name,
                **{k: v for k, v in met.items() if not isinstance(v, np.ndarray)},
            }
        )
        domains = {
            "common_all5": mask & np.isfinite(predictions).all(0),
            "unobserved_common_recovery4": common,
            "default_unanswered_newanswers": mask
            & ~np.isfinite(predictions[0])
            & np.isfinite(prediction),
        }
        for label, domain in domains.items():
            domain = domain & np.isfinite(prediction)
            counts = error_sums(truth, prediction, domain)
            e = abs(prediction[domain] - truth[domain])
            slices.append(
                {
                    **meta,
                    "method": name,
                    "slice": label,
                    **counts,
                    "completed": int((e < 0.1 * abs(truth[domain])).sum()),
                }
            )
    selected = np.asarray(payload["selected"]["BTPWR"], bool)
    pre = np.asarray(payload["reputations"]["BTPWR"])[:-1]
    honest = np.asarray(world["honest_private"], bool)
    counts = []
    cfg = trust.Config(gamma=case["gamma"])
    for t in range(len(actual)):
        raw = np.full_like(world["reports_check"][t], np.nan)
        raw[selected[t]] = world["reports_check"][t, selected[t]]
        raw[trust.classify(pre[t], cfg) == 0] = np.nan
        fb = trust.feedback(raw, pre[t], cfg)
        np.testing.assert_array_equal(fb["credible_consensus"], credible[t])
        np.testing.assert_array_equal(
            fb["next_reputation"], payload["reputations"]["BTPWR"][t + 1]
        )
        c, used = fb["credible_mask"], fb["used_mask"]
        counts.append(
            {
                **meta,
                "round": t + 1,
                "C_only_anchor_cells": int(np.isfinite(credible[t]).sum()),
                "all_source_anchor_cells": int(np.isfinite(fb["consensus"]).sum()),
                "added_U_only_anchor_cells": int(
                    (np.isfinite(fb["consensus"]) & ~np.isfinite(credible[t])).sum()
                ),
                "C_anchor_report_count": int(c.sum()),
                "C_anchor_malicious_report_count": int((c & ~honest[:, None]).sum()),
                "all_anchor_report_count": int(used.sum()),
                "all_anchor_malicious_report_count": int(
                    (used & ~honest[:, None]).sum()
                ),
            }
        )
    record = {"metrics": metrics, "risk_slices": slices, "anchor_contamination": counts}
    target = folder / "cases" / (case["name"] + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record), encoding="utf-8")
    return record


def operator_study(case, world, payload, result_dir, check_cap100=True):
    """Replace only missing-position predictions using the same credible anchors."""
    folder = Path(result_dir) / "operators"
    h = np.asarray(world["public_reference"])
    offset, scale = float(world["native_offset"]), float(world["native_scale"])
    c = payload["credible"]["tent-density"]
    actual = payload["models"]["BPP-TW"]
    default = payload["models"]["BTPWR"]
    fixed, calibrated = default.copy(), default.copy()
    coords = np.c_[np.arange(10).repeat(10) / 9, np.tile(np.arange(10), 10) / 9]
    low = float(h[:, :24].min())
    span = max(float(np.ptp(h[:, :24])), 1e-12)
    train = (h[:, :24] - low) / span
    ranks = [2, 5, 10]
    init = {r: bgmc.factor(train, r) for r in ranks}
    full_init = {r: bgmc.factor(h, r) for r in ranks}
    cache, validation, traces, cap_checks = {}, [], [], []

    def public_rank(a, ids, cap):
        values, records = [], []
        for rank in ranks:
            config = bgmc.Config(
                rank=rank,
                spatial_weight=0.5,
                temporal_weight=0.5,
                als_cap=cap,
                train_columns=24,
            )
            total, n, largest, hits, last = 0.0, 0, 0, 0, []
            for month in range(6):
                target = h[:, 24 + month]
                obs = ((target - low) / span).copy()
                obs[ids] += np.random.default_rng(
                    np.random.SeedSequence([8847, len(ids), month])
                ).normal(0, (0.1 / 30) / span, len(ids))
                estimate, trace = bgmc.bgmc(
                    train, np.where(a, obs, 0), a, coords, config, initial=init[rank]
                )
                e = estimate[~a] - (target[~a] - low) / span
                total += float((e * e).sum())
                n += int((~a).sum())
                largest = max(largest, int(trace[-1]["iteration"]))
                hits += int(trace[-1]["iteration"] >= cap)
                last.append(abs(float(trace[-1]["relative_change"])))
            values.append(total / max(n, 1))
            records.append(
                dict(
                    rank=rank,
                    public_MSE=values[-1],
                    max_ALS_iterations=largest,
                    public_cap_hits=hits,
                    max_final_relative_change=max(last),
                )
            )
        return ranks[int(np.argmin(values))], values, records

    for t in range(len(c)):
        a = np.isfinite(c[t])
        ids = np.flatnonzero(a)
        fill = ~np.isfinite(actual[t]) & np.isfinite(default[t])
        if not fill.any():
            continue
        config = bgmc.Config(
            rank=5,
            spatial_weight=0.5,
            temporal_weight=0.5,
            als_cap=40,
            train_columns=30,
        )
        estimate, trace = bgmc.bgmc(
            h, np.nan_to_num((c[t] - 10) / 30), a, coords, config, initial=full_init[5]
        )
        fixed[t, fill] = np.clip(offset + scale * estimate[fill], 0, 100)
        traces.append(
            dict(
                case=case["name"],
                round=t + 1,
                method="BGMC rank5",
                rank=5,
                ALS_iterations=trace[-1]["iteration"],
                relative_change=trace[-1]["relative_change"],
                cap_hit=int(trace[-1]["iteration"] >= 40),
            )
        )
        key = tuple(ids.tolist())
        if key not in cache:
            rank, mse, stats = public_rank(a, ids, 40)
            cache[key] = rank
            digest = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
            validation.extend(
                dict(
                    case=case["name"],
                    mask_sha256=digest,
                    anchor_count=len(ids),
                    selected_rank=rank,
                    **s
                )
                for s in stats
            )
            if check_cap100 and any(s["public_cap_hits"] for s in stats):
                rank100, mse100, stats100 = public_rank(a, ids, 100)
                cap_checks.append(
                    dict(
                        case=case["name"],
                        mask_sha256=digest,
                        rank40=rank,
                        rank100=rank100,
                        selected_rank_unchanged=rank == rank100,
                        max_iterations=max(s["max_ALS_iterations"] for s in stats100),
                        cap_hits=sum(s["public_cap_hits"] for s in stats100),
                        MSE40=mse,
                        MSE100=mse100,
                    )
                )
        rank = cache[key]
        config = bgmc.Config(
            rank=rank,
            spatial_weight=0.5,
            temporal_weight=0.5,
            als_cap=40,
            train_columns=30,
        )
        estimate, trace = bgmc.bgmc(
            h,
            np.nan_to_num((c[t] - 10) / 30),
            a,
            coords,
            config,
            initial=full_init[rank],
        )
        calibrated[t, fill] = np.clip(offset + scale * estimate[fill], 0, 100)
        traces.append(
            dict(
                case=case["name"],
                round=t + 1,
                method="BGMC public-rank",
                rank=rank,
                ALS_iterations=trace[-1]["iteration"],
                relative_change=trace[-1]["relative_change"],
                cap_hit=int(trace[-1]["iteration"] >= 40),
            )
        )
    names = ["BTPWR", "BGMC rank5", "BGMC public-rank"]
    predictions = np.asarray([default, fixed, calibrated])
    for prediction in predictions:
        np.testing.assert_array_equal(np.isfinite(prediction), np.isfinite(default))
        np.testing.assert_array_equal(
            prediction[np.isfinite(actual)], default[np.isfinite(actual)]
        )
    _save_npz(
        folder / "predictions" / (case["name"] + ".npz"),
        methods=np.array(names),
        prediction_native=predictions,
    )
    truth, mask = world["truth_native_private"].T, world["task_mask"]
    metrics, common = [], []
    for name, prediction in zip(names, predictions):
        met = requested_metrics(truth, prediction, mask)
        metrics.append(
            {
                **_metadata(case),
                "method": name,
                **{k: v for k, v in met.items() if not isinstance(v, np.ndarray)},
            }
        )
        for label, domain in {
            "all_common": mask & np.isfinite(default),
            "missing_actual_common": mask & ~np.isfinite(actual) & np.isfinite(default),
        }.items():
            common.append(
                {
                    **_metadata(case),
                    "method": name,
                    "slice": label,
                    **error_sums(truth, prediction, domain),
                }
            )
    record = dict(
        metrics=metrics,
        common=common,
        validation=validation,
        final_trace=traces,
        cap100=cap_checks,
    )
    target = folder / "cases" / (case["name"] + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record), encoding="utf-8")
    return record


def collect_studies(result_dir):
    for name in ("source_gate", "operators"):
        folder = Path(result_dir) / name
        tables = {}
        for path in sorted((folder / "cases").glob("*.json")):
            for key, rows in json.loads(path.read_text(encoding="utf-8")).items():
                tables.setdefault(key, []).extend(rows)
        for key, rows in tables.items():
            if rows:
                pd.DataFrame(rows).to_csv(folder / (key + ".csv"), index=False)
    return True

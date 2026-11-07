"""Public-output diagnostics and exact recruitment implementation checks."""
import csv
import json
import math
import gc
import statistics
import time
import tracemalloc
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .data import build_world


def save(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def balanced(y, prediction):
    return 0.5 * (np.mean(prediction[y]) + np.mean(~prediction[~y]))


def auc(y, score):
    positive, negative = score[y], score[~y]
    return float(
        np.mean(
            (positive[:, None] > negative[None, :])
            + 0.5 * (positive[:, None] == negative[None, :])
        )
    )


def public_output_audit(results, inputs):
    root = Path(results)
    records, exposure, folds = [], [], []
    for seed in [602, 603, 604]:
        folder = root / "cases" / f"confirmation_tdrive_seed{seed}"
        case = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        world = build_world(case, inputs)
        with np.load(folder / "predictions.npz", allow_pickle=False) as z:
            selected = z["selected_masks"][0]
            reputation = z["reputation_history"][0]
        visits, offers = world["presence"], world["offer_mask"]
        offered = offers.sum(0)
        rate = np.divide(
            selected.sum(0), offered, out=np.zeros(len(offered)), where=offered > 0
        )
        malicious = reputation[-1] < 0.3
        for i in range(len(offered)):
            records.append(
                {
                    "seed": seed,
                    "worker": i,
                    "selection_rate_given_offer": float(rate[i]),
                    "final_M": int(malicious[i]),
                }
            )
        admitted = selected[:, :, None] & visits
        exposure.append(
            {
                "seed": seed,
                "offered_worker_rounds": int(offers.sum()),
                "selected_worker_rounds": int(selected.sum()),
                "selected_among_offers_pct": 100 * selected.sum() / offers.sum(),
                "positive_visit_bits": int(visits.sum()),
                "disclosed_admitted_positive_visit_bits": int(admitted.sum()),
                "positive_visit_recall_pct": 100 * admitted.sum() / visits.sum(),
                "positive_visit_precision_pct": 100.0,
                "cell_bits_fixed_zero_by_nonoffer_pct": 100 * np.mean(~offers),
            }
        )
    for heldout in [602, 603, 604]:
        train = [r for r in records if r["seed"] != heldout]
        test = [r for r in records if r["seed"] == heldout]
        x = np.array([r["selection_rate_given_offer"] for r in train])
        y = np.array([r["final_M"] for r in train], bool)
        unique = np.unique(x)
        thresholds = np.r_[-1, (unique[:-1] + unique[1:]) / 2, 2]
        scores = np.array([balanced(y, x < threshold) for threshold in thresholds])
        threshold = thresholds[np.argmax(scores)]
        xt = np.array([r["selection_rate_given_offer"] for r in test])
        yt = np.array([r["final_M"] for r in test], bool)
        folds.append(
            {
                "heldout_seed": heldout,
                "train_worker_count": len(train),
                "test_worker_count": len(test),
                "test_M_count": int(yt.sum()),
                "threshold_selected_per_offer": float(threshold),
                "training_balanced_accuracy": float(scores.max()),
                "test_balanced_accuracy": float(balanced(yt, xt < threshold)),
                "test_AUC": auc(yt, -xt),
                "test_accuracy": float(np.mean((xt < threshold) == yt)),
            }
        )
    save(root / "privacy/worker_records.csv", records)
    save(root / "privacy/exposure.csv", exposure)
    save(root / "privacy/metrics.csv", folds)
    return folds


def privacy_accounting(results):
    e1, e2 = 2.1, 2.7
    p = 1 / (1 + math.exp(-e2))
    joint = max(e2, e1 + math.log(2 * p))
    rows = []
    for cells, rounds in [(1, 1), (1, 50), (2, 1), (100, 1), (100, 50)]:
        rows.append(
            {
                "changed_virtual_cells": cells,
                "offering_rounds": rounds,
                "recruitment_epsilon_upper": cells * rounds * joint,
            }
        )
    save(Path(results) / "privacy/accounting.csv", rows)
    rows = []
    for total in [1.0, 4.0, 10.0]:
        cell = total / 5000
        c1 = cell - math.log(2 / (1 + math.exp(-cell)))
        p1 = 1 / (1 + math.exp(-c1))
        rows.append(
            {
                "whole_packet_50round_budget": total,
                "epsilon1": c1,
                "epsilon2": cell,
                "location_retention": p1,
                "cost_retention": 1 / (1 + math.exp(-cell)),
                "location_estimator_variance": p1 * (1 - p1) / (2 * p1 - 1) ** 2,
            }
        )
    save(Path(results) / "privacy/budget_allocation.csv", rows)


def packet(world, case, t):
    from . import privacy

    loc, cost = world["presence"][t], world["actual_cost"][t]
    offers = world["offer_mask"][t]
    average = np.divide(
        cost.sum(1), loc.sum(1), out=np.full(len(loc), 3.25), where=loc.sum(1) > 0
    )
    virtual = world["virtual_presence"][t]
    cfg = privacy.Config(n=len(loc), m=100, T=50, eps1=case["eps1"], eps2=case["eps2"])
    released = privacy.release(
        virtual,
        np.where(virtual, average[:, None], 0.0),
        world["RR_coins"][t],
        cfg.eps1,
        cfg.eps2,
        cfg,
    )
    for value in released.values():
        value[~offers] = False
    budget = case.get("budget_override")
    return (
        released,
        offers,
        world["virtual_task_cells"][t],
        cfg,
        float(world["budget"] if budget is None else budget),
    )


def equal_trace(a, b):
    if a.keys() != b.keys():
        raise AssertionError("Selector fields differ")
    for key in a:
        if isinstance(a[key], np.ndarray):
            np.testing.assert_array_equal(a[key], b[key])
        elif a[key] != b[key]:
            raise AssertionError("Selector mismatch: " + key)


def selector_benchmark(results, inputs, repetitions=7, limit=None):
    from . import recruitment, trust

    rows, checks = [], []
    folders = sorted((Path(results) / "cases").glob("*"))
    for folder in folders:
        if not (folder / "metadata.json").exists():
            continue
        case = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if case.get("kind") == "confirmation":
            rounds = [0, 1, 10, 20, 30, 40, 49]
        elif (
            case.get("group") == "fresh_v3"
            and case["field"] != "legacy"
            and case["seed"] in [730, 737]
        ):
            rounds = [0, 10, 20, 30, 40, 49]
        else:
            continue
        world = build_world(case, inputs)
        with np.load(folder / "predictions.npz", allow_pickle=False) as z:
            pre, selected = z["reputation_pre"][0], z["selected_masks"][0]
        tc = trust.Config(gamma=case["gamma"])
        for t in rounds:
            released, offers, tasks, cfg, budget = packet(world, case, t)
            args = released, offers, pre[t], tasks, cfg, budget, tc
            whole, cached = (
                recruitment.reference_greedy(*args),
                recruitment.greedy(*args),
            )
            equal_trace(whole, cached)
            np.testing.assert_array_equal(
                cached["members"], np.flatnonzero(selected[t])
            )
            equal_trace(
                recruitment.reference_greedy(*args, binary=True),
                recruitment.greedy(*args, binary=True),
            )
            with patch.object(
                recruitment, "mapweights", lambda r, c: trust.mapweights(r, "BTPWR", c)
            ):
                sine_reference = recruitment.reference_greedy(*args)
            equal_trace(sine_reference, recruitment.greedy_sine(*args))
            checks.append(
                {
                    "case": case["name"],
                    "round": t,
                    "all_three_profiles_equal": True,
                    "matches_selected_workers": True,
                }
            )
            times = {"whole": [], "cached": []}
            for repeat in range(repetitions):
                order = [
                    ("whole", recruitment.reference_greedy),
                    ("cached", recruitment.greedy),
                ]
                if repeat % 2:
                    order.reverse()
                for label, fn in order:
                    start = time.perf_counter_ns()
                    fn(*args)
                    times[label].append((time.perf_counter_ns() - start) / 1e6)
            memory = {}
            for label, fn in [
                ("whole", recruitment.reference_greedy),
                ("cached", recruitment.greedy),
            ]:
                gc.collect()
                tracemalloc.start()
                fn(*args)
                memory[label] = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
            old, new = (
                statistics.median(times["whole"]),
                statistics.median(times["cached"]),
            )
            rows.append(
                {
                    "case": case["name"],
                    "round": t,
                    "workers": len(offers),
                    "selected": len(cached["members"]),
                    "whole_median_ms": old,
                    "cached_median_ms": new,
                    "speedup": old / new,
                    "whole_peak_traced_bytes": memory["whole"],
                    "cached_peak_traced_bytes": memory["cached"],
                }
            )
            if limit is not None and len(rows) >= limit:
                break
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise FileNotFoundError("Run confirmation cases before the selector audit.")
    save(Path(results) / "selection/metrics.csv", rows)
    save(Path(results) / "selection/equivalence.csv", checks)
    return rows


def worker_evidence_audit(results, cases):
    from . import trust

    rows = []
    for case in cases:
        folder = Path(results) / "cases" / case["name"]
        with np.load(folder / "world.npz", allow_pickle=False) as z:
            reports, offers = z["reports_check"], z["offer_mask"]
        with np.load(folder / "predictions.npz", allow_pickle=False) as z:
            selected, pre, post = (
                z["acquisition_selected_masks"],
                z["reputation_pre"],
                z["reputation_post"],
            )
            credible, actual, scored = (
                z["credible_consensus_check"],
                z["actual_consensus_check"],
                z["scored_count"],
            )
        cfg = trust.Config(gamma=case["gamma"])
        for arm in range(len(pre)):
            for t in range(pre.shape[1]):
                if np.any(selected[arm, t] & ~offers[t]):
                    raise AssertionError("Recruitment of a nonparticipant")
                raw = np.full_like(reports[t], np.nan)
                raw[selected[arm, t]] = reports[t, selected[arm, t]]
                raw[trust.classify(pre[arm, t], cfg) == 0] = np.nan
                replay = trust.feedback(raw, pre[arm, t], cfg)
                for key, expected in [
                    ("next_reputation", post[arm, t]),
                    ("credible_consensus", credible[arm, t]),
                    ("consensus", actual[arm, t]),
                    ("scored_count", scored[arm, t]),
                ]:
                    np.testing.assert_array_equal(replay[key], expected)
                unchanged = ~selected[arm, t] | (trust.classify(pre[arm, t], cfg) == 0)
                np.testing.assert_array_equal(
                    pre[arm, t, unchanged], post[arm, t, unchanged]
                )
        rows.append(
            {
                "case": case["name"],
                "feedback_frames": int(pre.shape[0] * pre.shape[1]),
                "worker_rounds": int(pre.size),
                "selected_reports_and_feedback_equal": True,
            }
        )
    save(Path(results) / "verification/worker_evidence.csv", rows)
    print(
        json.dumps(
            {
                "cases": len(rows),
                "feedback_frames": sum(r["feedback_frames"] for r in rows),
                "worker_rounds": sum(r["worker_rounds"] for r in rows),
                "status": "PASS",
            }
        )
    )
    return rows

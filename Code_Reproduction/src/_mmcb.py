"""MMCB allocation and payments (DOI 10.1109/TDSC.2024.3368655)."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    diminishing_return: float = 0.8
    importance_mix: float = 0.5


def task_weights(location_importance, task_value, cfg=Config()):
    (l, v) = (np.asarray(location_importance, float), np.asarray(task_value, float))
    if (
        l.ndim != 1
        or v.shape != l.shape
        or (not np.isfinite(l).all())
        or (not np.isfinite(v).all())
    ):
        raise ValueError("Finite equal-length task vectors required")
    if min(l.min(initial=0), v.min(initial=0)) < 0 or not 0 <= cfg.importance_mix <= 1:
        raise ValueError("Nonnegative importance/value and mixture in [0,1] required")
    return cfg.importance_mix * l + (1 - cfg.importance_mix) * v


class Coverage:
    def __init__(self, incidence, weights, cfg=Config()):
        self.A = np.asarray(incidence, bool)
        self.h = np.asarray(weights, float)
        self.lam = cfg.diminishing_return
        if self.A.ndim != 2 or self.h.shape != (self.A.shape[1],):
            raise ValueError("Worker-by-task incidence and task weights required")
        if (
            not np.isfinite(self.h).all()
            or np.any(self.h < 0)
            or (not np.isfinite(self.lam))
            or (self.lam <= 0)
        ):
            raise ValueError("Nonnegative finite weights and positive lambda required")

    def value(self, counts):
        return float(np.log1p(self.lam * counts) @ self.h)

    def marginal(self, counts, workers):
        ids = np.asarray(workers, int)
        return (
            np.log1p(self.lam * (counts[None, :] + self.A[ids]))
            - np.log1p(self.lam * counts)[None, :]
        ) @ self.h


def _greedy(oracle, bids, eligible, budget):
    counts = np.zeros(oracle.A.shape[1])
    remaining = list(sorted(eligible))
    (selected, trace) = ([], [])
    value = 0.0
    while remaining:
        gains = oracle.marginal(counts, remaining)
        k = int(np.argmax(gains / bids[remaining]))
        (worker, gain) = (int(remaining[k]), float(gains[k]))
        if gain <= 0:
            break
        threshold = budget * gain / (2 * (value + gain))
        admit = bool(bids[worker] <= threshold)
        trace.append(
            {
                "worker": worker,
                "marginal": gain,
                "bid": float(bids[worker]),
                "proportional_threshold": threshold,
                "admitted": admit,
            }
        )
        if not admit:
            break
        selected.append(worker)
        counts += oracle.A[worker]
        value += gain
        remaining.pop(k)
    return (np.asarray(selected, int), trace)


def critical_payment(oracle, bids, eligible, budget, winner):
    counts = np.zeros(oracle.A.shape[1])
    remaining = [int(i) for i in sorted(eligible) if i != winner]
    (payment, trace) = (0.0, [])
    value = 0.0
    while True:
        own_gain = float(oracle.marginal(counts, [winner])[0])
        if own_gain <= 0:
            break
        share = budget * own_gain / (2 * (value + own_gain))
        if not remaining:
            payment = max(payment, share)
            trace.append(
                {
                    "prefix_worker_count": int(len(eligible) - 1 - len(remaining)),
                    "competitor": None,
                    "share_cap": share,
                    "density_cap": float("inf"),
                    "candidate_payment": share,
                    "terminal": "exhausted",
                }
            )
            break
        gains = oracle.marginal(counts, remaining)
        k = int(np.argmax(gains / bids[remaining]))
        (competitor, gain) = (int(remaining[k]), float(gains[k]))
        density_cap = own_gain * bids[competitor] / gain if gain > 0 else float("inf")
        candidate = min(share, density_cap)
        payment = max(payment, candidate)
        other_share = budget * gain / (2 * (value + gain)) if gain > 0 else 0.0
        other_admitted = bool(gain > 0 and bids[competitor] <= other_share)
        trace.append(
            {
                "competitor": competitor,
                "share_cap": share,
                "density_cap": density_cap,
                "candidate_payment": candidate,
                "competitor_admitted_in_replay": other_admitted,
            }
        )
        if not other_admitted:
            break
        counts += oracle.A[competitor]
        value += gain
        remaining.pop(k)
    return (float(payment), trace)


def allocate(incidence, bids, budget, weights, rng, cfg=Config(), branch_draw=None):
    oracle = Coverage(incidence, weights, cfg)
    bids = np.asarray(bids, float)
    n = oracle.A.shape[0]
    if bids.shape != (n,) or not np.isfinite(bids).all() or np.any(bids <= 0):
        raise ValueError(
            "Paper domain requires strictly positive finite bids; empty/zero-cost taskless bidders must be omitted by a declared adapter"
        )
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("Nonnegative finite monetary budget required")
    draw = int(rng.integers(1, 6)) if branch_draw is None else int(branch_draw)
    if draw not in [1, 2, 3, 4, 5]:
        raise ValueError(
            "Native random branch must be one of five equiprobable outcomes"
        )
    eligible = np.flatnonzero(bids <= budget)
    payments = np.zeros(n)
    (trace, payment_trace) = ([], {})
    if not len(eligible):
        selected = np.array([], int)
    elif draw in [1, 2]:
        values = oracle.marginal(np.zeros(oracle.A.shape[1]), eligible)
        winner = int(eligible[int(np.argmax(values))])
        selected = np.array([winner], int)
        payments[winner] = budget
    else:
        (selected, trace) = _greedy(oracle, bids, eligible, budget)
        for winner in selected:
            (payments[winner], payment_trace[str(winner)]) = critical_payment(
                oracle, bids, eligible, budget, int(winner)
            )
    total = float(payments.sum())
    if total > budget + 1e-08 * max(1.0, budget):
        raise AssertionError(
            "Native critical-payment total exceeds B; do not silently rescale"
        )
    if len(selected) and np.any(payments[selected] + 1e-09 < bids[selected]):
        raise AssertionError("A selected positive-bid worker is underpaid")
    counts = oracle.A[selected].sum(0) if len(selected) else np.zeros(oracle.A.shape[1])
    return {
        "selected": selected,
        "payments": payments,
        "total_payment": total,
        "selected_bid_sum": float(bids[selected].sum()),
        "coverage_objective": oracle.value(counts),
        "branch_draw": draw,
        "branch": "singleton" if draw in [1, 2] else "proportional_share",
        "allocation_trace": trace,
        "critical_payment_trace": payment_trace,
        "implementation_scope": "Paper-based numeric allocation/critical payments; not a cryptographic or Fabric deployment.",
    }


def mean_service_head(reports, selected):
    data = np.asarray(reports, float)[np.asarray(selected, int)]
    count = np.isfinite(data).sum(0)
    return np.divide(
        np.nansum(data, axis=0),
        count,
        out=np.full(data.shape[1], np.nan),
        where=count > 0,
    )

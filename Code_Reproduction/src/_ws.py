"""Exact positive-cost maximum-coverage WS without privacy."""
import itertools
import numpy as np


def utility(counts, size, q, d):
    size = np.asarray(size)
    lower = q * size
    scaled = (counts - lower[..., None]) / d
    partial = ((scaled > 0) & (scaled < 1)).sum(-1)
    full = (scaled >= 1).sum(-1)
    coefficient = (np.ceil(lower) - lower) / d
    return full + partial * coefficient


def candidate_utilities(counts, newsize, features, q, d):
    lower = q * newsize
    f0 = (counts - lower[:, None]) / d
    f1 = (counts + 1 - lower[:, None]) / d
    full0 = f0 >= 1
    full1 = f1 >= 1
    p0 = (f0 > 0) & (f0 < 1)
    p1 = (f1 > 0) & (f1 < 1)
    nfull = (
        full0.sum(1)[:, None] + (full1.astype(float) - full0.astype(float)) @ features.T
    )
    npart = p0.sum(1)[:, None] + (p1.astype(float) - p0.astype(float)) @ features.T
    coefficient = (np.ceil(lower) - lower) / d
    return nfull + npart * coefficient[:, None]


def exact_ws(features, costs, budget, q=0.0, denominator=1.0, batch_size=256):
    x = np.asarray(features, float)
    c = np.asarray(costs, float)
    (n, m) = x.shape
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("Exact count-bin WS requires binary worker incidence features")
    if not np.isfinite(c).all() or np.any(c < 0):
        raise ValueError(
            "WS profile requires nonnegative additive cost; do not silently clip baseline PCP"
        )
    if np.any((c == 0) & (x.sum(1) > 0)):
        raise ValueError("Positive utility zero-cost WS tie needs an explicit policy")
    safe = np.where(c > 0, c, np.inf)
    seedsets = [()]
    for k in range(1, min(3, n) + 1):
        seedsets.extend(itertools.combinations(range(n), k))
    sizes = np.array([len(s) for s in seedsets])
    membership = np.zeros((len(seedsets), n), bool)
    for (i, s) in enumerate(seedsets):
        membership[i, list(s)] = True
    counts = membership.astype(float) @ x
    charge = membership @ c
    values = utility(counts, sizes, q, denominator)
    feasible = charge <= budget + 1e-09
    values = np.where(feasible, values, -np.inf)
    best = int(np.argmax(values))
    best_value = float(values[best])
    best_members = membership[best].copy()
    triples = np.flatnonzero((sizes == 3) & feasible)
    evaluated = 0
    extensions = 0
    upper = float(np.any(x > 0, axis=0).sum()) if q == 0 else float(m)
    if best_value >= upper - 1e-12:
        return (
            np.flatnonzero(best_members),
            {
                "utility": best_value,
                "cost": float(c[best_members].sum()),
                "all_initial_sets": len(seedsets),
                "feasible_triplets": len(triples),
                "extensions": 0,
                "globally_attained_upper_bound": True,
            },
        )
    for start in range(0, len(triples), batch_size):
        ids = triples[start : start + batch_size]
        sel = membership[ids].copy()
        cnt = counts[ids].copy()
        cost = charge[ids].copy()
        sz = sizes[ids].copy()
        value = values[ids].copy()
        active = np.ones(len(ids), bool)
        while active.any():
            live = np.flatnonzero(active)
            candidate = candidate_utilities(cnt[live], sz[live] + 1, x, q, denominator)
            gain = candidate - value[live, None]
            allowed = ~sel[live] & (cost[live, None] + c[None, :] <= budget + 1e-09)
            ratios = np.where(allowed, gain / safe[None, :], -np.inf)
            choice = np.argmax(ratios, axis=1)
            benefit = ratios[np.arange(len(live)), choice]
            stop = ~(benefit > 0)
            active[live[stop]] = False
            take = live[~stop]
            worker = choice[~stop]
            if len(take):
                sel[take, worker] = True
                cnt[take] += x[worker]
                cost[take] += c[worker]
                sz[take] += 1
                value[take] = candidate[np.flatnonzero(~stop), worker]
                extensions += len(take)
            evaluated += int(len(live) * n)
        local = int(np.argmax(value))
        if value[local] > best_value:
            best_value = float(value[local])
            best_members = sel[local].copy()
        if best_value >= upper - 1e-12:
            break
    assert c[best_members].sum() <= budget + 1e-08
    return (
        np.flatnonzero(best_members),
        {
            "utility": best_value,
            "cost": float(c[best_members].sum()),
            "all_initial_sets": len(seedsets),
            "feasible_triplets": len(triples),
            "extensions": extensions,
            "marginal_evaluations": evaluated,
            "globally_attained_upper_bound": bool(best_value >= upper - 1e-12),
            "WS_exact_no_template_sampling": True,
        },
    )

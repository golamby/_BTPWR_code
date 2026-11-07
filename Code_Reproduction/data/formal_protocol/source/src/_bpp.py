"""BPP: literal TDSC Algorithms 5/6 (DOI 10.1109/TDSC.2022.3186023)."""
import itertools
import numpy as np


def task_column_weights(task_cells, k=10):
    t = np.asarray(task_cells)
    if t.ndim != 1 or not np.issubdtype(t.dtype, np.integer):
        raise ValueError("Integer zero-based true task cell IDs required")
    if k < 1 or np.any((t < 0) | (t >= k * k)) or len(np.unique(t)) != len(t):
        raise ValueError("Native binary task matrix requires unique in-grid cells")
    return np.tile(np.bincount(t % k, minlength=k), k).astype(float)


def published_pcp_costs(release, eps2, cmin=1.5, cmax=5.0):
    bits = np.asarray(release["bits"])
    hi = np.asarray(release["hi"])
    if bits.ndim != 2 or hi.shape != bits.shape:
        raise ValueError("Full-grid release arrays required")
    p = 1 / (1 + np.exp(-eps2))
    N = bits.sum(1).astype(float)
    n_hi = hi.sum(1).astype(float)
    return cmin * N + (cmax - cmin) * (n_hi - (1 - p) * N) / (2 * p - 1)


def native_utility(counts, size, eps1, task_cells, k=10):
    N = np.asarray(counts, float)
    w = task_column_weights(task_cells, k)
    if N.shape[-1] != k * k:
        raise ValueError("Counts must retain D grid cells")
    p = 1 / (1 + np.exp(-eps1))
    d = 2 * p - 1
    if not np.isfinite(d) or d <= 0:
        raise ValueError("Strictly positive finite eps1 required")
    f = (N - (1 - p) * np.asarray(size)[..., None]) / d
    return np.clip(f, 0.0, 1.0) @ w


def _bin_utility(counts, size, q, d, weights):
    lower = q * np.asarray(size)
    f = (counts - lower[..., None]) / d
    partial = (f > 0) & (f < 1)
    full = f >= 1
    coefficient = (np.ceil(lower) - lower) / d
    return full @ weights + partial @ weights * coefficient


def _candidate_utilities(counts, newsize, features, q, d, weights):
    lower = q * newsize
    f0 = (counts - lower[:, None]) / d
    f1 = (counts + 1 - lower[:, None]) / d
    full0 = f0 >= 1
    full1 = f1 >= 1
    part0 = (f0 > 0) & (f0 < 1)
    part1 = (f1 > 0) & (f1 < 1)
    full = (full0 @ weights)[:, None] + (
        full1.astype(float) - full0
    ) * weights @ features.T
    part = (part0 @ weights)[:, None] + (
        part1.astype(float) - part0
    ) * weights @ features.T
    coefficient = (np.ceil(lower) - lower) / d
    return full + part * coefficient[:, None]


def exact_ws(bits, costs, budget, task_cells, eps1, k=10, batch_size=256):
    x = np.asarray(bits, float)
    c = np.asarray(costs, float)
    weights = task_column_weights(task_cells, k)
    if x.ndim != 2 or x.shape[1] != k * k or c.shape != (len(x),):
        raise ValueError("Expected bits[n,D], costs[n], D=k*k")
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("Original WLC outputs must be binary")
    if not np.isfinite(c).all() or np.any(c < 0):
        raise ValueError(
            "Native PCP costs must be nonnegative; do not silently clip them"
        )
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("Nonnegative finite budget required")
    if np.any((c == 0) & (x.sum(1) > 0)):
        raise ValueError(
            "Positive-feature zero-cost worker needs a separately specified WS rule"
        )
    p = 1 / (1 + np.exp(-eps1))
    q = 1 - p
    d = 2 * p - 1
    if not np.isfinite(d) or d <= 0:
        raise ValueError("Strictly positive finite eps1 required")
    (n, D) = x.shape
    safe = np.where(c > 0, c, np.inf)
    seedsets = [()]
    for size in range(1, min(3, n) + 1):
        seedsets.extend(itertools.combinations(range(n), size))
    sizes = np.fromiter((len(s) for s in seedsets), int, count=len(seedsets))
    membership = np.zeros((len(seedsets), n), bool)
    for (i, s) in enumerate(seedsets):
        membership[i, list(s)] = True
    counts = membership.astype(float) @ x
    charge = membership @ c
    values = _bin_utility(counts, sizes, q, d, weights)
    values = np.where(charge <= budget + 1e-09, values, -np.inf)
    best = int(np.argmax(values))
    best_value = float(values[best])
    best_members = membership[best].copy()
    triples = np.flatnonzero((sizes == 3) & (charge <= budget + 1e-09))
    evaluated = extensions = paths = 0
    upper = float(weights.sum())
    if best_value < upper - 1e-12:
        for start in range(0, len(triples), batch_size):
            ids = triples[start : start + batch_size]
            sel = membership[ids].copy()
            cnt = counts[ids].copy()
            cost = charge[ids].copy()
            sz = sizes[ids].copy()
            value = values[ids].copy()
            active = np.ones(len(ids), bool)
            paths += len(ids)
            while active.any():
                live = np.flatnonzero(active)
                candidate = _candidate_utilities(
                    cnt[live], sz[live] + 1, x, q, d, weights
                )
                gain = candidate - value[live, None]
                allowed = ~sel[live] & (cost[live, None] + c[None, :] <= budget + 1e-09)
                ratio = np.where(allowed, gain / safe[None, :], -np.inf)
                choice = np.argmax(ratio, axis=1)
                benefit = ratio[np.arange(len(live)), choice]
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
    selected = np.flatnonzero(best_members)
    if c[selected].sum() > budget + 1e-08:
        raise AssertionError("Native WS budget violation")
    return (
        selected,
        {
            "utility": best_value,
            "cost": float(c[selected].sum()),
            "all_initial_sets": len(seedsets),
            "feasible_triplets": len(triples),
            "triplet_paths_evaluated": paths,
            "extensions": extensions,
            "marginal_evaluations": evaluated,
            "native_utility_upper_bound": upper,
            "task_count": len(task_cells),
            "grid_regions": D,
            "column_task_counts": np.bincount(
                np.asarray(task_cells) % k, minlength=k
            ).tolist(),
            "WTM_literal_column_alias": True,
            "WS_exact_no_template_sampling": True,
            "globally_attained_proxy_upper_bound": bool(best_value >= upper - 1e-12),
            "scope": "Original dense-RMM Algorithm5 proxy and complete Algorithm6 WS, not corrected exact-cell matching or physical task completion.",
        },
    )

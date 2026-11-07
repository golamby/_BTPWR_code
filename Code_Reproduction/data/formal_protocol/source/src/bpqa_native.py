"""Numerical BPQA/BBOM adaptation on shared mobility and sensing worlds.

BPQA equations (3), (5)-(9), (14)-(24); Zhou et al. algorithms 1-2.
This numerical implementation does not claim Fabric or cryptographic execution.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    threshold: float = .5
    levels: int = 5
    rho: float = .9
    task_weight: float = 10.
    delta: float = .5
    alpha: float = 1. / 3.
    mu: float = .1
    epsilon: float = .5
    numerical_epsilon: float = 1e-10
    td_tolerance: float = 1e-3
    td_iterations: int = 200
    normalize_worker_error: bool = False


def reliability_levels(reputation, cfg=Config()):
    r = np.asarray(reputation, dtype=float)
    if not np.isfinite(r).all() or np.any((r < 0) | (r > 1)):
        raise ValueError('Reputations must be finite and in [0, 1]')
    width = (1. - cfg.threshold) / cfg.levels
    ratio = (r - cfg.threshold) / width
    nearest = np.rint(ratio)
    ratio = np.where(np.abs(ratio - nearest) <= 1e-12, nearest, ratio)
    level = np.floor(ratio).astype(int) + 1
    return np.where(r >= cfg.threshold, np.clip(level, 1, cfg.levels), 0)


def _objective(counts, reputation_sum, multiplier, cfg):
    coverage = float(cfg.task_weight * np.log1p(cfg.rho * counts).sum())
    reputation = float(multiplier * reputation_sum)
    return coverage, reputation


def allocate(incidence, bids, reputation, eligible, budget, multiplier, cfg=Config()):
    """Budget BBOM: binary search, greedy submodular cover and critical pay.

    When a winner has no substitute capable of meeting a target, its critical
    payment is capped at the full budget. Ties prefer the lowest worker index.
    These necessary edge conventions are not stated in Zhou's pseudocode.
    """
    A = np.asarray(incidence, dtype=bool)
    bids = np.asarray(bids, dtype=float)
    r = np.asarray(reputation, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    n, m = A.shape
    if bids.shape != (n,) or r.shape != (n,) or eligible.shape != (n,):
        raise ValueError('Mismatched worker inputs')
    if not np.isfinite(bids).all() or np.any(bids < 0) or not (budget >= 0 and multiplier > 0):
        raise ValueError('Finite nonnegative bids and valid budget/multiplier required')
    levels = reliability_levels(r, cfg).astype(float)
    # Zhou's W includes all qualified bidders; high bids may still constrain
    # another bidder's critical payment and must not be silently removed.
    pool = np.flatnonzero(eligible & (levels > 0) & (bids > 0) & A.any(axis=1))
    if not len(pool) or not m or not budget:
        return {'selected': np.empty(0, dtype=int), 'payments': np.zeros(n), 'total_payment': 0.}
    worker_rep = levels * A.sum(axis=1)

    def greedy(target, exclude=None, collect_payment_for=None):
        remain = pool[pool != exclude]
        counts = np.zeros(m, dtype=int)
        rep = 0.
        selected = []
        payment = 0.
        while len(remain):
            cov, rv = _objective(counts, rep, multiplier, cfg)
            current = .5 * (min(cov, target) + min(rv, target))
            if current >= target - 1e-10:
                return selected, payment, True
            cov_gain = cfg.task_weight * (np.log1p(cfg.rho * (counts[None, :] + A[remain])) -
                        np.log1p(cfg.rho * counts)[None, :]).sum(axis=1)
            dcov = np.minimum(cov + cov_gain, target) - min(cov, target)
            drep = np.minimum(rv + multiplier * worker_rep[remain], target) - min(rv, target)
            gain = .5 * (dcov + drep)
            q = int(np.argmax(gain / bids[remain]))
            winner, best = int(remain[q]), float(gain[q])
            if best <= 1e-12:
                break
            if collect_payment_for is not None:
                own = int(collect_payment_for)
                own_cov = float(cfg.task_weight * (np.log1p(cfg.rho * (counts + A[own])) - np.log1p(cfg.rho * counts)).sum())
                own_gain = .5 * ((min(cov + own_cov, target) - min(cov, target)) +
                                 (min(rv + multiplier * worker_rep[own], target) - min(rv, target)))
                payment = max(payment, own_gain * bids[winner] / best)
            selected.append(winner)
            counts += A[winner]
            rep += worker_rep[winner]
            remain = np.delete(remain, q)
        cov, rv = _objective(counts, rep, multiplier, cfg)
        complete = .5 * (min(cov, target) + min(rv, target)) >= target - 1e-10
        return selected, payment, complete

    def payments_for(selected, target):
        p = np.zeros(n)
        for worker in selected:
            _, payment, feasible = greedy(target, exclude=worker, collect_payment_for=worker)
            p[worker] = min(float(budget), payment) if feasible else float(budget)
        return p

    full_counts = A[pool].sum(axis=0)
    full_rep = worker_rep[pool].sum()
    lo, hi = 0., min(_objective(full_counts, full_rep, multiplier, cfg))
    selected = []
    while hi - lo > cfg.delta:
        target = (lo + hi) / 2.
        trial, _, feasible = greedy(target)
        p = payments_for(trial, target) if feasible else None
        if feasible and p.sum() <= budget + 1e-9 and all(p[i] + 1e-9 >= bids[i] for i in trial):
            lo, selected = target, trial
        else:
            hi = target
    p = payments_for(selected, lo) if selected else np.zeros(n)
    if p.sum() > budget + 1e-8 or any(p[i] + 1e-8 < bids[i] for i in selected):
        raise ArithmeticError('Computed critical payments violate budget or bid')
    return {'selected': np.asarray(selected, int), 'payments': p,
            'total_payment': float(p.sum()), 'target': lo}


def truth_and_weights(reports, cfg=Config()):
    """BPQA Eq. 16 weights with explicit finite, normalized truth update."""
    y = np.asarray(reports, float)
    if y.ndim != 2 or np.isinf(y).any():
        raise ValueError('A worker-by-task matrix of finite values or NaNs is required')
    valid = np.isfinite(y)
    count = valid.sum(axis=0)
    z = np.divide(np.nansum(y, axis=0), count, out=np.full(y.shape[1], np.nan), where=count > 0)
    active = valid.any(axis=1)
    weights = np.zeros(y.shape[0])
    if not active.any():
        return z, weights
    for _ in range(cfg.td_iterations):
        error = np.where(valid, y - z[None, :], 0.)
        d = (error * error).sum(axis=1)
        if cfg.normalize_worker_error:
            d = np.divide(d, valid.sum(axis=1), out=np.zeros_like(d), where=active)
        total = float(d[active].sum())
        if total <= cfg.numerical_epsilon:
            weights[active] = 1.
        else:
            weights[active] = np.log((total + cfg.numerical_epsilon) /
                                     (d[active] + cfg.numerical_epsilon))
        weighted = np.where(valid, weights[:, None], 0.)
        den = weighted.sum(axis=0)
        fallback = valid & (den[None, :] <= cfg.numerical_epsilon)
        weighted = np.where(fallback, 1., weighted)
        den = weighted.sum(axis=0)
        next_z = np.divide(np.nansum(weighted * y, axis=0), den,
                           out=np.full(y.shape[1], np.nan), where=den > 0)
        delta = np.max(np.abs(next_z[count > 0] - z[count > 0]))
        z = next_z
        if delta <= cfg.td_tolerance:
            break
    return z, weights


def update_reputation(reputation, weights, scored, cfg=Config()):
    """BPQA equations 17-24; only workers with accepted reports update."""
    r = np.asarray(reputation, float)
    weights = np.asarray(weights, float)
    scored = np.asarray(scored, bool)
    level = reliability_levels(r, cfg)
    mapped = cfg.threshold + (1. - cfg.threshold) / cfg.levels * (level - 1)
    out = r.copy()
    active = scored & (level > 0)
    if not active.any():
        return out
    low = active & (weights < cfg.epsilon)
    p_low = low.sum() / active.sum()
    beta = cfg.mu * (cfg.alpha / p_low - cfg.alpha) if p_low > 0 else 0.
    mapped = np.clip(mapped, 1e-12, 1 - 1e-12)
    z = np.log(mapped / (1 - mapped)) + cfg.alpha * (weights - cfg.epsilon)
    updated = 1. / (1. + np.exp(-np.clip(z, -700, 700)))
    change = np.where(low, beta * (weights - cfg.epsilon) * mapped,
                      (updated - mapped) * (1. - mapped))
    out[active] = np.clip(r[active] + change[active], 0., 1.)
    return out

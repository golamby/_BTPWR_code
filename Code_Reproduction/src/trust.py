"""Credible-first residual consensus and OTHER-C reputation feedback."""
from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class Config:
    credible_threshold: float = 0.8
    malicious_threshold: float = 0.3
    gamma: float = 0.6
    score_tolerance: float = 0.1
    score_width: float = 30.0
    score_shape: float = 1.0
    td_delta: float = 0.1
    td_tolerance: float = 0.001
    td_cap: int = 200
    reputation_scale: int = 10000
    map_amplitude: float = 2.0
    identified_low: float = 0.3
    identified_high: float = 0.85


def validate_reputations(r):
    r = np.asarray(r, dtype=float)
    if r.ndim != 1 or not np.isfinite(r).all() or np.any((r < 0) | (r > 1)):
        raise ValueError("Estimated reputations must be a finite vector in[0,1]")
    return r


def classify(estimated_reputation, cfg=Config()):
    r = validate_reputations(estimated_reputation)
    return np.where(
        r < cfg.malicious_threshold, 0, np.where(r >= cfg.credible_threshold, 2, 1)
    ).astype(np.int8)


def mapweights(estimated_reputation, kind="BTPWR", cfg=Config()):
    r = validate_reputations(estimated_reputation)
    classes = classify(r, cfg)
    eta = float(np.mean(classes != 1)) if len(r) else 0.0
    if kind == "BPP-T":
        return ((classes != 0).astype(float), classes, eta)
    if kind not in {"BTPWR", "BPP-TW"}:
        return (np.ones(len(r)), classes, eta)
    a = cfg.map_amplitude
    lo = cfg.identified_low
    hi = cfg.identified_high
    fc = a if eta <= lo else a - (a - 1) * (eta - lo) / (hi - lo) if eta <= hi else 1.0
    fu = 1.0 if eta <= lo else 1.0 + (a - 1) * np.sin(np.pi * (eta - lo) / (1 - lo))
    return (np.where(classes == 0, 0.0, np.where(classes == 2, fc, fu)), classes, eta)


def residual_consensus(values, initial_reputations, cfg=Config()):
    y = np.asarray(values, float)
    r = np.asarray(initial_reputations, float)
    if y.ndim != 1 or y.shape != r.shape or (not np.isfinite(y).all()):
        raise ValueError("Finite actual report and reputation vectors required")
    if len(y) == 0:
        return (np.nan, 0)
    if np.any(r <= 0) or not np.isfinite(r).all():
        raise ValueError("Only positive eligible estimated reputations initialize TD")
    z = float(np.dot(r, y) / np.sum(r))
    if len(y) == 1:
        return (float(y[0]), 0)
    for iteration in range(1, cfg.td_cap + 1):
        error = np.abs(y - z)
        a = np.log((np.sum(error) + cfg.td_delta * len(y)) / (error + cfg.td_delta))
        if not np.isfinite(a).all() or not np.sum(a) > 0:
            raise ArithmeticError("Invalid regularized CRH weight sum")
        nxt = float(np.dot(a, y) / np.sum(a))
        if abs(nxt - z) <= cfg.td_tolerance:
            return (nxt, iteration)
        z = nxt
    return (z, cfg.td_cap)


def task_consensus(reports, estimated_reputation, cfg=Config()):
    y = np.asarray(reports, float)
    r = validate_reputations(estimated_reputation)
    if y.ndim != 2 or y.shape[0] != len(r):
        raise ValueError("Worker/task shape mismatch")
    if np.isinf(y).any():
        raise ValueError("Reports may be finite or missing, not infinite")
    classes = classify(r, cfg)
    available = np.isfinite(y)
    eligible = available & (classes[:, None] != 0)
    C = available & (classes[:, None] == 2)
    has_C = C.any(axis=0)
    used = np.where(has_C[None, :], C, eligible)
    m = y.shape[1]
    z = np.full(m, np.nan)
    trusted_z = np.full(m, np.nan)
    iterations = np.zeros(m, int)
    for j in range(m):
        ids = np.flatnonzero(used[:, j])
        (z[j], iterations[j]) = residual_consensus(y[ids, j], r[ids], cfg)
        if has_C[j]:
            trusted_z[j] = z[j]
    return {
        "consensus": z,
        "credible_consensus": trusted_z,
        "has_credible": has_C,
        "credible_mask": C,
        "used_mask": used,
        "credible_count": C.sum(0),
        "eligible_count": eligible.sum(0),
        "iterations": iterations,
        "classes": classes,
    }


def panel_references(values, reputations, credible, reporters, limit):
    """REAPP-style capped weighted reference, always excluding the scored report."""
    ranked = sorted(credible, key=lambda i: (-reputations[i], int(i)))[: limit + 1]
    panel = ranked[:limit]
    backup = ranked[limit] if len(ranked) > limit else None
    numerator = sum(reputations[i] * values[i] for i in panel)
    denominator = sum(reputations[i] for i in panel)
    member = set(panel)
    refs = {}
    for i in reporters:
        total, weight = numerator, denominator
        if i in member:
            total -= reputations[i] * values[i]
            weight -= reputations[i]
            if backup is not None:
                total += reputations[backup] * values[backup]
                weight += reputations[backup]
        if weight > 1e-12:
            refs[i] = total / weight
    return refs


def feedback(reports, estimated_reputation, cfg=Config(), reference_limit=None,
             reference_iteration_cap=None):
    y = np.asarray(reports, float)
    r = validate_reputations(estimated_reputation)
    d = task_consensus(y, r, cfg)
    classes = d["classes"]
    (n, m) = y.shape
    if reference_limit is not None and (
        type(reference_limit) is not int or reference_limit < 1
    ):
        raise ValueError("The reference panel limit must be a positive integer")
    refs = np.full((n, m), np.nan)
    scores = np.full((n, m), np.nan)
    ref_cfg = cfg if reference_iteration_cap is None else replace(cfg, td_cap=reference_iteration_cap)
    for j in range(m):
        credible = np.flatnonzero(d["credible_mask"][:, j])
        if len(credible) == 0:
            continue
        reporters = np.flatnonzero(np.isfinite(y[:, j]) & (classes != 0))
        panel = None if reference_limit is None else panel_references(
            y[:, j], r, credible, reporters, reference_limit
        )
        shared = d["credible_consensus"][j] if ref_cfg is cfg else residual_consensus(
            y[credible, j], r[credible], ref_cfg
        )[0]
        for i in reporters:
            peers = credible[credible != i]
            if len(peers) == 0:
                continue
            if panel is not None:
                reference = panel[i]
            elif classes[i] != 2:
                reference = shared
            else:
                (reference, _) = residual_consensus(y[peers, j], r[peers], ref_cfg)
            refs[i, j] = reference
            error = abs(y[i, j] - reference) / cfg.score_width
            A = (np.exp(-r[i] / cfg.score_shape) - np.exp(-1 / cfg.score_shape)) / (
                1 - np.exp(-1 / cfg.score_shape)
            )
            if error <= cfg.score_tolerance:
                scores[i, j] = cfg.gamma * A * (1 - error / cfg.score_tolerance)
            else:
                scores[i, j] = -cfg.gamma * (1 - A) * (1 - cfg.score_tolerance / error)
    valid = np.isfinite(scores)
    counts = valid.sum(1)
    change = np.divide(
        np.nansum(scores, axis=1), counts, out=np.zeros(n), where=counts > 0
    )
    quantized = (
        np.rint(np.clip(r + change, 0, 1) * cfg.reputation_scale) / cfg.reputation_scale
    )
    next_r = np.where(counts > 0, quantized, r)
    assert np.array_equal(
        next_r[classes == 0], r[classes == 0]
    ), "Excluded M state must be preserved"
    d.update(
        {
            "reference_by_worker": refs,
            "scores": scores,
            "scored_count": counts,
            "next_reputation": next_r,
            "classes_next": classify(next_r, cfg),
        }
    )
    return d

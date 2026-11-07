"""Public affine covariance transfer from actual credible anchors."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Config:
    relative_ridge: float = 0.0001
    absolute_ridge: float = 1e-10
    measurement_floor: float = 1e-12
    measurement_fallback: float = (0.1 / 30.0) ** 2
    update_history: bool = True


def psd(matrix):
    a = (matrix + matrix.T) / 2
    (w, U) = np.linalg.eigh(a)
    return U * np.maximum(w, 0.0) @ U.T


def initialize(public, cfg=Config(), training_prior=None):
    H = np.asarray(public, float)
    if H.ndim != 2 or H.shape[1] < 3 or (not np.isfinite(H).all()):
        raise ValueError("Finite public region-time prefix required")
    if training_prior is None:
        E = H.copy()
        mode = "fixed-public-mean"
    else:
        P = np.asarray(training_prior, float)
        if P.shape != H.shape:
            raise ValueError("Training prior shape mismatch")
        E = H - P
        mode = "external-causal-prior"
    valid = np.isfinite(E)
    if np.any(valid.sum(1) < 3):
        raise ValueError("At least3 public residual observations per region")
    mu = np.nanmean(np.where(valid, E, np.nan), axis=1)
    centered = np.where(valid, E - mu[:, None], 0.0)
    count = valid.astype(float) @ valid.astype(float).T
    cross = centered @ centered.T
    raw = np.divide(cross, count, out=np.zeros_like(cross), where=count > 0)
    K = psd(raw)
    return {
        "mean": mu,
        "kernel": K,
        "pair_count": count,
        "pair_cross": cross,
        "mode": mode,
        "public_columns": H.shape[1],
        "rounds": 0,
        "actual_C_entries_used_for_history": 0,
        "cfg": asdict(cfg),
        "interpretation": "Empirical PSD working kernel/measurement scales; no calibrated Gaussian posterior claim.",
    }


def recover(
    anchor_regions,
    anchor_values,
    measurement_variance,
    state,
    cfg=Config(),
    prior=None,
    physical_bounds=None,
):
    a = np.asarray(anchor_regions, int)
    z = np.asarray(anchor_values, float)
    if a.ndim != 1 or z.shape != a.shape or (not np.isfinite(z).all()):
        raise ValueError("Finite C region/value vectors required")
    D = len(state["mean"])
    if len(set(a.tolist())) != len(a) or np.any(a < 0) or np.any(a >= D):
        raise ValueError("Unique in-domain C regions required")
    if state["mode"] == "external-causal-prior":
        if prior is None:
            raise ValueError("External-prior kernel requires the causal forecast")
        exogenous = np.asarray(prior, float)
        if exogenous.shape != (D,) or not np.isfinite(exogenous).all():
            raise ValueError("Finite causal prior required")
    elif prior is not None:
        raise ValueError(
            "Initialize with matching training_prior before passing an external prior"
        )
    else:
        exogenous = np.zeros(D)
    m = exogenous + state["mean"]
    K = state["kernel"]
    b = np.setdiff1d(np.arange(D), a)
    if measurement_variance is None:
        R = np.full(len(a), cfg.measurement_fallback)
    else:
        R = np.asarray(measurement_variance, float)
        if R.ndim == 0:
            R = np.full(len(a), float(R))
        if R.shape != a.shape or not np.isfinite(R).all() or np.any(R < 0):
            raise ValueError("Nonnegative working C measurement variances required")
        R = np.maximum(R, cfg.measurement_floor)
    ridge = max(cfg.absolute_ridge, cfg.relative_ridge * float(np.trace(K)) / max(D, 1))
    service = m.copy()
    transfer = np.zeros((len(b), len(a)))
    if len(a):
        A = K[np.ix_(a, a)] + np.diag(R + ridge)
        transfer = np.linalg.solve(A, K[np.ix_(a, b)]).T
        service[b] = m[b] + transfer @ (z - m[a])
        service[a] = z
        condition = float(np.linalg.cond(A))
    else:
        condition = 1.0
    if physical_bounds is not None:
        (lo, hi) = physical_bounds
        if np.any(z < lo) or np.any(z > hi):
            raise ValueError("Trusted consensus is outside declared physical bounds")
        service = np.clip(service, lo, hi)
    assert np.isfinite(service).all()
    nxt = {**state, "rounds": state["rounds"] + 1}
    if cfg.update_history and len(a):
        count = state["pair_count"].copy()
        cross = state["pair_cross"].copy()
        residual = z - exogenous[a] - state["mean"][a]
        cross[np.ix_(a, a)] += np.outer(residual, residual)
        count[np.ix_(a, a)] += 1.0
        raw = np.divide(cross, count, out=np.zeros_like(cross), where=count > 0)
        nxt.update(
            {
                "pair_cross": cross,
                "pair_count": count,
                "kernel": psd(raw),
                "actual_C_entries_used_for_history": state[
                    "actual_C_entries_used_for_history"
                ]
                + len(a),
            }
        )
    diag = {
        "C_regions": a.copy(),
        "inferred_regions": b.copy(),
        "prior": m.copy(),
        "innovation": z - m[a],
        "transfer": transfer,
        "regularization": ridge,
        "measurement_scale": R.copy(),
        "anchor_system_condition": condition,
        "anchor_preservation_max_error": float(np.max(abs(service[a] - z)))
        if len(a)
        else 0.0,
        "history_entries_added": len(a) if cfg.update_history else 0,
        "inferred_entries_added_to_history": 0,
        "kernel_used_before_current_C_update": True,
        "source_scope": "actual C consensus only; no q/U/target truth",
    }
    return (service, nxt, diag)

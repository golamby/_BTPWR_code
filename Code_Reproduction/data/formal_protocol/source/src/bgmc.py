"""Spatial-temporal masked alternating matrix completion."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    rank: int = 5
    spatial_weight: float = 0.5
    temporal_weight: float = 0.5
    spatial_mix: float = 0.5
    sigma_geo: float = 0.25
    sigma_value: float = 0.25
    als_cap: int = 40
    als_tolerance: float = 1e-05
    cg_cap: int = 150
    cg_tolerance: float = 1e-08
    numerical_ridge: float = 1e-08
    observation_fraction: float = 0.35
    noise_fraction: float = 0.01
    contamination: float = 0.0
    train_columns: int = 40
    dev_columns: int = 20


def constraints(history, coords, cfg):
    d = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    geo = np.exp(-d / max(cfg.sigma_geo ** 2, 1e-12))
    np.fill_diagonal(geo, 0)
    means = history.mean(1)
    dif = np.abs(means[:, None] - means[None, :])
    val = np.exp(-dif / max(cfg.sigma_value ** 2, 1e-12))
    np.fill_diagonal(val, 0)

    def lap(w):
        return w / np.maximum(w.sum(1, keepdims=True), 1e-12) - np.eye(len(w))

    S = (1 - cfg.spatial_mix) * lap(geo) + cfg.spatial_mix * lap(val)
    return S


def temporal(T):
    D = np.zeros((T, max(0, T - 1)))
    for j in range(T - 1):
        D[j, j] = 1
        D[j + 1, j] = -1
    return D @ D.T


def pcg(op, b, diag, x=None, tol=1e-08, maxiter=150):
    x = np.zeros_like(b) if x is None else x.copy()
    r = b - op(x)
    z = r / np.maximum(diag, 1e-12)
    p = z.copy()
    rz = float(np.sum(r * z))
    bn = max(float(np.linalg.norm(b)), 1e-12)
    for it in range(maxiter):
        Ap = op(p)
        den = float(np.sum(p * Ap))
        if den <= 1e-30:
            break
        a = rz / den
        x += a * p
        r -= a * Ap
        if np.linalg.norm(r) <= tol * bn:
            return (x, it + 1, float(np.linalg.norm(r) / bn))
        z = r / np.maximum(diag, 1e-12)
        nrz = float(np.sum(r * z))
        p = z + nrz / max(rz, 1e-30) * p
        rz = nrz
    return (x, it + 1, float(np.linalg.norm(r) / bn))


def factor(history, rank):
    (u, s, vt) = np.linalg.svd(history, full_matrices=False)
    r = min(rank, len(s))
    root = np.sqrt(np.maximum(s[:r], 1e-12))
    return (u[:, :r] * root, vt[:r].T * root)


def objective(phi, omega, Y, W, S, TT, cfg):
    X = phi @ omega.T
    return float(
        np.sum(W * (X - Y) ** 2)
        + cfg.spatial_weight * np.sum((S @ X) ** 2)
        + cfg.temporal_weight * np.sum(X @ TT * X)
    )


def bgmc(
    history, current, mask, coords, cfg, history_mask=None, initial=None, trace=False
):
    """Alternating global least-squares solves of the published objective.

    The observed history/current data, not evaluation truth, are Y. Numerical
    ridge stabilizes singular normal equations; its fixed size is exported.
    """
    if initial is None:
        (phi, oldomega) = factor(history, cfg.rank)
    else:
        (phi, oldomega) = initial
    phio = phi[mask]
    om = (
        np.linalg.lstsq(phio, current[mask], rcond=1e-10)[0]
        if mask.any()
        else oldomega[-1].copy()
    )
    omega = np.vstack([oldomega, om])
    Y = np.column_stack([history, np.where(mask, current, 0.0)])
    W = np.ones_like(Y)
    W[:, -1] = mask
    if history_mask is not None:
        W[:, :-1] = history_mask
    S = constraints(history[:, : cfg.train_columns], coords, cfg)
    SS = S.T @ S
    TT = temporal(Y.shape[1])
    ridge = cfg.numerical_ridge
    records = []
    prev = objective(phi, omega, Y, W, S, TT, cfg)
    records.append(
        {
            "iteration": 0,
            "objective": prev,
            "relative_change": 0.0,
            "phi_cg_residual": 0.0,
            "omega_cg_residual": 0.0,
        }
    )
    for it in range(cfg.als_cap):
        og = omega.T @ omega
        ot = omega.T @ TT @ omega
        grams = np.einsum("it,ta,tb->iab", W, omega, omega, optimize=True)
        rhs = W * Y @ omega

        def op_phi(x):
            return (
                np.einsum("iab,ib->ia", grams, x)
                + cfg.spatial_weight * (SS @ x) @ og
                + cfg.temporal_weight * x @ ot
                + ridge * x
            )

        diag = (
            np.diagonal(grams, axis1=1, axis2=2)
            + cfg.spatial_weight * np.diag(SS)[:, None] * np.diag(og)[None, :]
            + cfg.temporal_weight * np.diag(ot)[None, :]
            + ridge
        )
        if cfg.spatial_weight == 0 and cfg.temporal_weight == 0:
            phi = np.linalg.solve(
                grams + ridge * np.eye(grams.shape[-1])[None, :, :], rhs[:, :, None]
            )[:, :, 0]
            pr = float(
                np.linalg.norm(op_phi(phi) - rhs) / max(np.linalg.norm(rhs), 1e-12)
            )
        else:
            (phi, _, pr) = pcg(op_phi, rhs, diag, phi, cfg.cg_tolerance, cfg.cg_cap)
        pg = phi.T @ phi
        ps = phi.T @ SS @ phi
        grams = np.einsum("it,ia,ib->tab", W, phi, phi, optimize=True)
        rhs = (W * Y).T @ phi

        def op_omega(x):
            return (
                np.einsum("tab,tb->ta", grams, x)
                + cfg.spatial_weight * x @ ps
                + cfg.temporal_weight * (TT @ x) @ pg
                + ridge * x
            )

        diag = (
            np.diagonal(grams, axis1=1, axis2=2)
            + cfg.spatial_weight * np.diag(ps)[None, :]
            + cfg.temporal_weight * np.diag(TT)[:, None] * np.diag(pg)[None, :]
            + ridge
        )
        if cfg.spatial_weight == 0 and cfg.temporal_weight == 0:
            omega = np.linalg.solve(
                grams + ridge * np.eye(grams.shape[-1])[None, :, :], rhs[:, :, None]
            )[:, :, 0]
            orr = float(
                np.linalg.norm(op_omega(omega) - rhs) / max(np.linalg.norm(rhs), 1e-12)
            )
        else:
            (omega, _, orr) = pcg(
                op_omega, rhs, diag, omega, cfg.cg_tolerance, cfg.cg_cap
            )
        val = objective(phi, omega, Y, W, S, TT, cfg)
        change = (prev - val) / max(abs(prev), 1e-12)
        records.append(
            {
                "iteration": it + 1,
                "objective": val,
                "relative_change": change,
                "phi_cg_residual": pr,
                "omega_cg_residual": orr,
            }
        )
        if val > prev + max(1e-07, abs(prev) * 1e-06):
            raise ArithmeticError(f"ALS objective increased {prev}->{val}")
        if abs(change) < cfg.als_tolerance:
            break
        prev = val
    prediction = phi @ omega[-1]
    return (prediction, records)

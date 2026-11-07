"""Credible-only public-blocked KRR and fixed affine/RBF controls."""
import hashlib
import numpy as np
from ._projection import (
    initialize as linear_initialize,
    complete as linear_complete,
    decode_actual,
)
from . import _covariance as recovery

_primary = recovery.Config(update_history=False)
RIDGE = 0.01
KRR_GRID = [
    dict(bandwidth_factor=1.0, ridge=0.01),
    dict(bandwidth_factor=0.25, ridge=0.01),
    dict(bandwidth_factor=1.0, ridge=0.1),
]


def initialize(public_reference, native_offset, native_scale, calibration=None):
    H = np.asarray(public_reference, float)
    cal = (
        public_calibration(H, native_offset, native_scale)
        if calibration is None
        else calibration
    )
    if hashlib.sha256(H.tobytes()).hexdigest() != cal["public_sha256"]:
        raise ValueError("Calibration must match the public field")
    state = {
        "public": H,
        "linear": linear_initialize(H, native_offset, native_scale),
        "kappas": {"public-RBF": float(cal["kappa"])},
        "cache": {},
        "blocked_fill_cache": {},
    }
    state.update(state["linear"])
    return state


def fixed_complete(credible_check, actual_native, state):
    base, diag = linear_complete(credible_check, actual_native, state["linear"])
    C = diag["credible_mask"]
    a = np.flatnonzero(C)
    out = base.copy()
    if not len(a):
        return out, {**diag, "chosen_RBF": np.zeros(len(C), bool)}
    key = tuple(a.tolist())
    if key not in state["cache"]:
        state["cache"][key] = conditional(
            state["public"], a, state["linear"]["recovery"]
        )
    fitted = state["cache"][key]
    z = (np.asarray(credible_check)[a] - 10) / 30
    mean = np.clip(
        state["offset"] + state["scale"] * nonlinear_predict(fitted, z), 0.0, 100.0
    )
    mean[C] = actual_native[C]
    supported = diag["recovery_mask"]
    actual = np.isfinite(actual_native)
    recovered = np.where(supported, mean, np.nan)
    out = np.where(actual, actual_native, recovered)
    eligible = actual & supported & ~C
    radius = state["kappas"]["public-RBF"] * diag["working_sigma_native"]
    lower = np.maximum(0.0, mean - radius)
    upper = np.minimum(100.0, mean + radius)
    out[eligible] = np.clip(actual_native[eligible], lower[eligible], upper[eligible])
    np.testing.assert_array_equal(out[C], actual_native[C])
    np.testing.assert_array_equal(np.isfinite(out), actual | supported)
    return (
        out,
        {
            **diag,
            "recovery_native": recovered,
            "chosen_RBF": np.ones(len(C), bool),
            "band_lower_native": lower,
            "band_upper_native": upper,
        },
    )


def complete(credible_consensus_check, actual_consensus_native, state):
    baseline, diag = fixed_complete(
        credible_consensus_check, actual_consensus_native, state
    )
    credible = np.asarray(credible_consensus_check, float)
    actual = np.asarray(actual_consensus_native, float)
    a = np.flatnonzero(np.isfinite(credible))
    fill = ~np.isfinite(actual) & np.isfinite(baseline)
    prediction = baseline.copy()
    choice = np.full(len(credible), -1, int)
    new_mean = np.full(len(credible), np.nan)
    if fill.any():
        if not len(a):
            raise AssertionError("Actual credible observations are required")
        key = tuple(a.tolist())
        if key not in state["blocked_fill_cache"]:
            state["blocked_fill_cache"][key] = blocked_conditional(state["public"], a)
        fitted = state["blocked_fill_cache"][key]
        values = np.asarray(
            [krr_predict(fit, (credible[a] - 10) / 30) for fit in fitted["fits"]]
        )
        choice = fitted["choice"]
        mean = values[choice, np.arange(len(credible))]
        new_mean = np.clip(state["offset"] + state["scale"] * mean, 0, 100)
        prediction[fill] = new_mean[fill]
    np.testing.assert_array_equal(np.isfinite(prediction), np.isfinite(baseline))
    np.testing.assert_array_equal(
        prediction[np.isfinite(actual)], baseline[np.isfinite(actual)]
    )
    return (
        prediction,
        {
            **diag,
            "blocked_fill_mask": fill,
            "blocked_fill_config_choice": choice,
            "blocked_fill_mean_native": new_mean,
            "actual_consensus_service_bitwise_preserved": True,
            "public_validation_used_collection_truth": False,
            "inferred_or_U_values_used_for_fit": False,
        },
    )


def fit_kernel(K, Y, ridge):
    n = len(K)
    A = np.linalg.solve(K + ridge * np.eye(n), np.eye(n))
    a1 = A.sum(1)
    den = a1.sum()
    Q = A - np.outer(a1, a1) / den
    alpha = Q @ Y
    intercept = a1 @ Y / den
    diagonal = np.diag(Q)
    assert diagonal.min() > 0
    risk = np.mean((alpha / diagonal[:, None]) ** 2, axis=0)
    return {"alpha": alpha, "intercept": intercept, "risk": risk}


def conditional(public, anchors, linear_state):
    H = np.asarray(public, float)
    a = np.asarray(anchors, int)
    N = H.shape[1]
    X = H[a].T
    Y = H.T
    mean = X.mean(0)
    Xc = X - mean
    eta = N * (
        _primary.measurement_fallback
        + max(
            _primary.absolute_ridge,
            _primary.relative_ridge * np.trace(linear_state["kernel"]) / len(H),
        )
    )
    lin = fit_kernel(Xc @ Xc.T, Y, eta)
    square = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    positive = square[np.triu_indices(N, 1)]
    positive = positive[positive > 1e-14]
    bandwidth = max(float(np.median(positive)) if len(positive) else 1.0, 1e-12)
    K = np.exp(-square / (2 * bandwidth))
    rbf = fit_kernel(K, Y, RIDGE)
    choose = rbf["risk"] < lin["risk"]
    return {
        "X": X,
        "X_centered": Xc,
        "X_mean": mean,
        "bandwidth_squared": bandwidth,
        "linear": lin,
        "RBF": rbf,
        "choose_RBF": choose,
    }


def nonlinear_predict(model, z):
    distances = np.sum((model["X"] - z[None, :]) ** 2, axis=1)
    k = np.exp(-distances / (2 * model["bandwidth_squared"]))
    return model["RBF"]["intercept"] + k @ model["RBF"]["alpha"]


def krr_fit(H, a, config):
    X = H[a].T
    Y = H.T
    n = len(X)
    square = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    positive = square[np.triu_indices(n, 1)]
    positive = positive[positive > 1e-14]
    s2 = (
        max(float(np.median(positive)) if len(positive) else 1.0, 1e-12)
        * config["bandwidth_factor"]
    )
    K = np.exp(-square / (2 * s2))
    fit = fit_kernel(K, Y, config["ridge"])
    return {"X": X, "s2": s2, **fit}


def krr_predict(fit, z):
    k = np.exp(-np.sum((fit["X"] - z[None, :]) ** 2, axis=1) / (2 * fit["s2"]))
    return fit["intercept"] + k @ fit["alpha"]


def blocked_conditional(H, a):
    errors = []
    full = []
    for config in KRR_GRID:
        fit = krr_fit(H[:, :24], a, config)
        validation = []
        for month in range(6):
            z = H[a, 24 + month] + np.random.default_rng(
                np.random.SeedSequence([8847, len(a), month])
            ).normal(0, 0.1 / 30, len(a))
            validation.append(krr_predict(fit, z) - H[:, 24 + month])
        errors.append(np.mean(np.asarray(validation) ** 2, axis=0))
        full.append(krr_fit(H, a, config))
    errors = np.asarray(errors)
    choice = np.argmin(errors, axis=0)
    return {"fits": full, "choice": choice, "public_validation_MSE": errors}


def public_calibration(public_reference, native_offset, native_scale):
    H = np.asarray(public_reference, float)
    offset = float(native_offset)
    scale = float(native_scale)
    assert H.shape == (100, 30) and np.isfinite(H).all() and (scale > 0)
    ls = linear_initialize(H[:, :24], offset, scale)
    ratios = []
    for k in [4, 8, 12, 16, 20]:
        for month in range(6):
            target = H[:, 24 + month]
            for rep in range(20):
                a = np.sort(
                    np.random.default_rng(
                        np.random.SeedSequence([7701, k, month, rep])
                    ).choice(100, k, replace=False)
                )
                z = target[a] + np.random.default_rng(
                    np.random.SeedSequence([7702, k, month, rep])
                ).normal(0, 0.1 / 30, k)
                cred = np.full(100, np.nan)
                cred[a] = 10 + 30 * z
                actual = decode_actual(cred, ls)
                (_, diag) = linear_complete(cred, actual, ls)
                use = diag["recovery_mask"] & ~diag["credible_mask"]
                fitted = conditional(H[:, :24], a, ls["recovery"])
                center = np.clip(offset + scale * nonlinear_predict(fitted, z), 0, 100)
                ratios.extend(
                    (
                        np.abs(offset + scale * target[use] - center[use])
                        / diag["working_sigma_native"][use]
                    ).tolist()
                )
    order = np.sort(ratios)
    rank = min(len(order), int(np.ceil(0.95 * (len(order) + 1))))
    if not len(order):
        raise ValueError("No public supported non-anchor residuals for calibration")
    return {
        "kappa": float(order[rank - 1]),
        "residual_count": len(order),
        "rank": rank,
        "public_sha256": hashlib.sha256(H.tobytes()).hexdigest(),
        "native_offset": offset,
        "native_scale": scale,
        "interpretation": "Public empirical95%residual scaling of oldSchurSD, not coverage guarantee.",
    }

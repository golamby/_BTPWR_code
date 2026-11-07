"""Actual-consensus preserving affine service and public intervals."""
from dataclasses import dataclass
import numpy as np
from . import _covariance as recovery, _support as qualification


@dataclass(frozen=True)
class Config:
    kappa: float = 7.470407589647988
    physical_lower: float = 0.0
    physical_upper: float = 100.0
    schur_floor: float = 1e-14


def initialize(public_reference, native_offset, native_scale, cfg=Config()):
    if not cfg.kappa > 0 or not native_scale > 0:
        raise ValueError("Positive public parameters required")
    return {
        "recovery": recovery.initialize(
            public_reference, recovery.Config(update_history=False)
        ),
        "offset": float(native_offset),
        "scale": float(native_scale),
        "cfg": cfg,
    }


def decode_actual(consensus_check, state):
    return np.clip(
        state["offset"]
        + state["scale"] * ((np.asarray(consensus_check, float) - 10) / 30),
        state["cfg"].physical_lower,
        state["cfg"].physical_upper,
    )


def complete(credible_consensus_check, actual_consensus_native, state):
    credible = np.asarray(credible_consensus_check, float)
    actual = np.asarray(actual_consensus_native, float)
    if (
        credible.ndim != 1
        or actual.shape != credible.shape
        or np.isinf(credible).any()
        or np.isinf(actual).any()
    ):
        raise ValueError("Matched finite/NaN vectors required")
    cfg = state["cfg"]
    pre = state["recovery"]
    a = np.flatnonzero(np.isfinite(credible))
    D = len(credible)
    (service, unused, diag) = recovery.recover(
        a, (credible[a] - 10) / 30, None, pre, recovery.Config(update_history=False)
    )
    (supported, gain) = qualification.qualify(a, pre, diag)
    base = np.clip(
        state["offset"] + state["scale"] * service,
        cfg.physical_lower,
        cfg.physical_upper,
    )
    recovered = np.where(supported, base, np.nan)
    sd = np.full(D, np.nan)
    if len(a):
        b = diag["inferred_regions"]
        K = pre["kernel"]
        reduction = np.einsum("ij,ij->i", diag["transfer"], K[np.ix_(b, a)])
        variance = np.diag(K)[b] - reduction
        if len(variance) and variance.min() < -1e-08:
            raise ArithmeticError("Indefinite working Schur scale")
        sd[b] = state["scale"] * np.sqrt(np.maximum(variance, cfg.schur_floor))
        sd[a] = 0.0
    C = np.isfinite(credible)
    observed = np.isfinite(actual)
    if not observed[C].all():
        raise ValueError("Credible consensus must also be an actual consensus")
    np.testing.assert_array_equal(recovered[C], actual[C])
    prediction = np.where(observed, actual, recovered)
    eligible = observed & supported & ~C
    lower = np.maximum(cfg.physical_lower, base - cfg.kappa * sd)
    upper = np.minimum(cfg.physical_upper, base + cfg.kappa * sd)
    prediction[eligible] = np.clip(actual[eligible], lower[eligible], upper[eligible])
    np.testing.assert_array_equal(prediction[C], actual[C])
    np.testing.assert_array_equal(np.isfinite(prediction), observed | supported)
    return (
        prediction,
        {
            "credible_mask": C,
            "recovery_mask": supported,
            "working_gain": gain,
            "recovery_native": recovered,
            "working_sigma_native": sd,
            "projection_eligible": eligible,
            "projected_mask": eligible & (prediction != actual),
            "actual_mask": observed,
            "band_lower_native": lower,
            "band_upper_native": upper,
            "kernel_or_history_updated": False,
            "inferred_or_U_entries_used_as_recovery_observations": 0,
        },
    )

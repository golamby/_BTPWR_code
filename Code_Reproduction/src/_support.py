"""Explained-variance support qualification."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    minimum_explained_fraction: float = 0.5
    nonzero_variance_floor: float = 1e-14


def qualify(anchor_regions, state, recovery_diagnostics, cfg=Config()):
    kernel = np.asarray(state["kernel"], float)
    anchors = np.asarray(anchor_regions, int)
    D = len(kernel)
    supported = np.zeros(D, bool)
    gain = np.zeros(D, float)
    if not len(anchors):
        return (supported, gain)
    target = np.asarray(recovery_diagnostics["inferred_regions"], int)
    transfer = np.asarray(recovery_diagnostics["transfer"], float)
    diagonal = np.diag(kernel)
    numerator = np.einsum("ij,ij->i", transfer, kernel[np.ix_(target, anchors)])
    valid = diagonal[target] > cfg.nonzero_variance_floor
    fraction = np.divide(
        numerator, diagonal[target], out=np.zeros(len(target)), where=valid
    )
    if np.any(fraction < -1e-07) or np.any(fraction > 1 + 1e-07):
        raise ArithmeticError(
            "PSD working-information fraction outside mathematical bounds"
        )
    gain[target] = np.clip(fraction, 0, 1)
    gain[anchors] = 1.0
    supported[anchors] = True
    supported[target] = valid & (fraction >= cfg.minimum_explained_fraction)
    return (supported, gain)

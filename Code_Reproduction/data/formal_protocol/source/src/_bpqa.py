"""BPQA-style coverage-reliability reverse auction on the shared inputs."""
import numpy as np
from . import baselines, recruitment


def auction_order(incidence, charge, offers, reliability, rho=1.0):
    """Priority order: marginal coverage x reliability / exact charge."""
    n, m = incidence.shape
    covered = np.zeros(m, bool)
    available = np.asarray(offers, bool).copy()
    order = []
    value = np.clip(np.asarray(reliability, float), 1e-6, 1.0) ** rho
    while available.any():
        gain = (incidence & ~covered[None, :]).sum(1) * value
        score = np.where(available, gain / np.maximum(charge, 1e-12), -np.inf)
        i = int(np.argmax(score))
        if not np.isfinite(score[i]) or score[i] <= 0:
            break
        order.append(i)
        covered |= incidence[i]
        available[i] = False
    return np.asarray(order, int)


def select(packet, offers, reputation, virtual_cells, cfg, budget, rho=1.0):
    incidence = np.asarray(packet["bits"])[:, np.asarray(virtual_cells, int)].astype(bool)
    charge = baselines.bpp.published_pcp_costs(packet, cfg.eps2)
    order = auction_order(incidence, charge, offers, reputation, rho)
    return recruitment.prefix_budget(order, packet, cfg, budget)

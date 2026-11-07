"""Location and cost randomized response on virtual locations."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    n: int = 100
    m: int = 100
    T: int = 50
    eps1: float = 2.1
    eps2: float = 2.7
    cmin: float = 1.5
    cmax: float = 5.0


def release(bits, cost, coin, eps1, eps2, cfg):
    p1 = 1 / (1 + np.exp(-eps1))
    p2 = 1 / (1 + np.exp(-eps2))
    obs = np.where(coin[0] < p1, bits, ~bits)
    src = np.where(bits, cost, (cfg.cmin + cfg.cmax) / 2)
    endpoint = coin[1] < (src - cfg.cmin) / (cfg.cmax - cfg.cmin)
    rr = np.where(coin[2] < p2, endpoint, ~endpoint)
    return {"bits": obs, "hi": obs & rr, "nonzero": obs}


def projected_cost_matrix(rel, cfg):
    p2 = 1 / (1 + np.exp(-cfg.eps2))
    nc = rel["nonzero"].astype(float)
    nh = np.clip((rel["hi"] - (1 - p2) * nc) / (2 * p2 - 1), 0, nc)
    return cfg.cmin * nc + (cfg.cmax - cfg.cmin) * nh


def candidatecost(M, rel, cfg):
    p2 = 1 / (1 + np.exp(-cfg.eps2))
    nc = M @ rel["nonzero"].astype(np.float32)
    zh = M @ rel["hi"].astype(np.float32)
    nh = np.clip((zh - (1 - p2) * nc) / (2 * p2 - 1), 0, nc)
    return (cfg.cmin * nc + (cfg.cmax - cfg.cmin) * nh).sum(1)

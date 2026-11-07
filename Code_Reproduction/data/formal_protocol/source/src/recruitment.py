"""Exact rational monetary quotes and equivalent cached density selection."""
from dataclasses import dataclass
from fractions import Fraction
import hashlib, json
import numpy as np
from . import trust as td

PRICE_Q = 1_000_000
I64 = np.iinfo(np.int64).max


@dataclass(frozen=True)
class PriceProfile:
    probability_numerator: int
    probability_denominator: int
    twice_minimum: int
    twice_maximum: int

    @classmethod
    def from_config(cls, cfg):
        P = int(np.rint(PRICE_Q / (1 + np.exp(-float(cfg.eps2)))))
        lo = int(np.rint(2 * cfg.cmin))
        hi = int(np.rint(2 * cfg.cmax))
        if lo != 2 * cfg.cmin or hi != 2 * cfg.cmax:
            raise ValueError("Registered costs must be half-integers")
        if not PRICE_Q / 2 < P < PRICE_Q:
            raise ValueError("Nondegenerate registered price probability required")
        return cls(P, PRICE_Q, lo, hi)

    @property
    def difference(self):
        return 2 * self.probability_numerator - self.probability_denominator

    @property
    def cost_denominator(self):
        return 2 * self.difference

    @property
    def monotone(self):
        return (
            self.probability_numerator * (self.twice_minimum + self.twice_maximum)
            >= self.twice_maximum * self.probability_denominator
        )


def quote_numerators(membership, released, profile):
    M = np.asarray(membership, dtype=np.int64)
    N = M @ np.asarray(released["nonzero"], dtype=np.int64)
    H = M @ np.asarray(released["hi"], dtype=np.int64)
    d = profile.difference
    Qc = profile.probability_denominator
    P = profile.probability_numerator
    recovered = np.minimum(np.maximum(Qc * H - (Qc - P) * N, 0), d * N)
    return (
        profile.twice_minimum * d * N
        + (profile.twice_maximum - profile.twice_minimum) * recovered
    ).sum(1)


def prefix_budget(order, released, cfg, budget):
    order = np.asarray(order, int)
    M = np.zeros((len(order) + 1, len(released["bits"])), bool)
    for (j, i) in enumerate(order):
        M[j + 1] = M[j]
        M[j + 1, i] = True
    profile = PriceProfile.from_config(cfg)
    costs = quote_numerators(M, released, profile)
    allowed = np.flatnonzero(feasible(costs, profile, budget))
    k = int(allowed[-1]) if len(allowed) else 0
    return (order[:k], int(costs[k]), profile.cost_denominator)


def mapweights(estimated_reputation, cfg=td.Config()):
    (w, classes, d) = td.mapweights(estimated_reputation, "BTPWR", cfg)
    x = (d - cfg.identified_low) / (1 - cfg.identified_low)
    fu = (
        1.0
        if d <= cfg.identified_low
        else 1.0 + (cfg.map_amplitude - 1) * max(0.0, 1.0 - abs(2 * x - 1.0))
    )
    return (np.where(classes == 1, fu, w), classes, d)


def reference_greedy(
    released,
    offers,
    estimated_reputation,
    virtual_task_cells,
    cfg,
    budget,
    td_config=td.Config(),
    binary=False,
):
    bits = np.asarray(released["bits"], bool)
    (n, D) = bits.shape
    r = td.validate_reputations(estimated_reputation)
    offer = np.asarray(offers, bool)
    if r.shape != (n,) or offer.shape != (n,):
        raise ValueError("Matchedregistryvectorsrequired")
    profile = PriceProfile.from_config(cfg)
    if not profile.monotone:
        raise ValueError("Registeredmonotonequote required")
    (w, classes, d) = (
        td.mapweights(r, "BPP-T", td_config) if binary else mapweights(r, td_config)
    )
    weights = np.rint(td_config.reputation_scale * w).astype(np.int64)
    utility = weights * bits[:, np.asarray(virtual_task_cells, int)].sum(
        1, dtype=np.int64
    )
    eligible = offer & (classes != 0)
    chosen = np.zeros(n, bool)
    trace = []
    while True:
        pool = np.flatnonzero(eligible & ~chosen)
        if not len(pool):
            break
        trial = np.repeat(chosen[None], len(pool) + 1, axis=0)
        trial[np.arange(1, len(pool) + 1), pool] = True
        costs = quote_numerators(trial, released, profile)
        increment = costs[1:] - costs[0]
        if np.any(increment < 0):
            raise ArithmeticError("Quotemonotonicityviolated")
        gain = utility[pool]
        positions = np.flatnonzero(feasible(costs[1:], profile, budget) & (gain > 0))
        if not len(positions):
            break
        zero = positions[increment[positions] == 0]
        if len(zero):
            best = int(zero[np.argmax(gain[zero])])
        else:
            best = int(positions[0])
            for j in positions[1:]:
                if int(gain[j]) * int(increment[best]) > int(gain[best]) * int(
                    increment[j]
                ):
                    best = int(j)
        i = int(pool[best])
        chosen[i] = True
        trace.append(
            {
                "slot": i,
                "utility_gain": int(gain[best]),
                "price_increment_numerator": int(increment[best]),
                "price_after_numerator": int(costs[1 + best]),
            }
        )
    C = int(quote_numerators(chosen[None], released, profile)[0])
    assert bool(feasible(np.array([C]), profile, budget)[0])
    return {
        "members": np.flatnonzero(chosen),
        "cost_numerator": C,
        "cost_denominator": profile.cost_denominator,
        "cost": C / profile.cost_denominator,
        "utility": int(utility[chosen].sum()),
        "trace": trace,
        "price_profile": profile,
        "mapped_weights_integer": weights,
        "classes": classes,
        "identified_fraction": d,
    }


def budget_limit(profile, budget):
    B = Fraction(str(float(budget)))
    if B < 0:
        raise ValueError("Nonnegative payment budget required")
    return B.numerator * int(profile.cost_denominator) // B.denominator


def quote_counts(N, H, profile):
    N = np.asarray(N, dtype=np.int64)
    H = np.asarray(H, dtype=np.int64)
    d = profile.difference
    Qc = profile.probability_denominator
    P = profile.probability_numerator
    recovered = np.minimum(np.maximum(Qc * H - (Qc - P) * N, 0), d * N)
    return (
        profile.twice_minimum * d * N
        + (profile.twice_maximum - profile.twice_minimum) * recovered
    ).sum(axis=-1)


def _greedy(
    released,
    offers,
    estimated_reputation,
    virtual_task_cells,
    cfg,
    budget,
    td_config,
    binary,
    map_kind,
):
    raw_bits = np.asarray(released["bits"])
    if raw_bits.ndim != 2 or not np.all((raw_bits == 0) | (raw_bits == 1)):
        raise ValueError("Binary worker by virtual-cell matrix required")
    bits = raw_bits.astype(bool, copy=False)
    (n, D) = bits.shape
    r = td.validate_reputations(estimated_reputation)
    raw_offer = np.asarray(offers)
    if not np.all((raw_offer == 0) | (raw_offer == 1)):
        raise ValueError("Binary offer indicators required")
    offer = raw_offer.astype(bool, copy=False)
    if not n or not D:
        raise ValueError(
            "Nonempty registered roster and virtual-cell universe required"
        )
    if r.shape != (n,) or offer.shape != (n,):
        raise ValueError("Matched registry vectors required")
    endpoints = []
    for key in ["nonzero", "hi"]:
        x = np.asarray(released[key])
        if x.shape != (n, D) or not np.all((x == 0) | (x == 1)):
            raise ValueError("Binary endpoint indicator required: " + key)
        endpoints.append(x.astype(np.int64, copy=False))
    (nz, hi) = endpoints
    if np.any(hi > nz):
        raise ValueError("High endpoint must be contained in nonzero")
    profile = PriceProfile.from_config(cfg)
    if not profile.monotone:
        raise ValueError("Registered monotone clipped quote required")
    if profile.twice_minimum < 0 or profile.twice_maximum < profile.twice_minimum:
        raise ValueError("Ordered nonnegative cost endpoints required")
    bound = max(
        profile.probability_denominator * n,
        profile.difference * n,
        profile.twice_maximum * profile.difference * n * D,
    )
    if bound > I64:
        raise OverflowError("Quote profile exceeds proven int64 count bound")
    limit = budget_limit(profile, budget)
    raw_tasks = np.asarray(virtual_task_cells)
    tasks = np.asarray(virtual_task_cells, int)
    if (
        tasks.ndim != 1
        or np.any(raw_tasks != tasks)
        or np.any(tasks < 0)
        or np.any(tasks >= D)
    ):
        raise ValueError("Valid integer logical virtual-task tokens required")
    if binary:
        (w, classes, d) = td.mapweights(r, "BPP-T", td_config)
    elif map_kind == "tent":
        (w, classes, d) = mapweights(r, td_config)
    elif map_kind == "sine":
        (w, classes, d) = td.mapweights(r, "BTPWR", td_config)
    else:
        raise ValueError("Registered map name required")
    scaled = td_config.reputation_scale * w
    if (
        not np.isfinite(scaled).all()
        or max([abs(float(v)) for v in scaled] + [0]) * max(len(tasks), 1) * max(n, 1)
        > I64
    ):
        raise OverflowError("Utility profile exceeds int64 bound")
    weights = np.rint(scaled).astype(np.int64)
    utility = weights * bits[:, tasks].sum(1, dtype=np.int64)
    eligible = offer & (classes != 0)
    chosen = np.zeros(n, bool)
    trace = []
    N = np.zeros(D, np.int64)
    H = np.zeros(D, np.int64)
    current = 0
    while True:
        pool = np.flatnonzero(eligible & ~chosen)
        if not len(pool):
            break
        trialN = N[None, :] + nz[pool]
        trialH = H[None, :] + hi[pool]
        costs = quote_counts(trialN, trialH, profile)
        increment = costs - current
        if np.any(increment < 0):
            raise ArithmeticError("Quote monotonicity violated")
        gain = utility[pool]
        affordable = np.ones(len(costs), bool) if limit > I64 else costs <= limit
        positions = np.flatnonzero(affordable & (gain > 0))
        if not len(positions):
            break
        zero = positions[increment[positions] == 0]
        if len(zero):
            best = int(zero[np.argmax(gain[zero])])
        else:
            best = int(positions[0])
            for j in positions[1:]:
                if int(gain[j]) * int(increment[best]) > int(gain[best]) * int(
                    increment[j]
                ):
                    best = int(j)
        i = int(pool[best])
        chosen[i] = True
        N += nz[i]
        H += hi[i]
        current = int(costs[best])
        trace.append(
            {
                "slot": i,
                "utility_gain": int(gain[best]),
                "price_increment_numerator": int(increment[best]),
                "price_after_numerator": current,
            }
        )
    assert current <= limit
    return {
        "members": np.flatnonzero(chosen),
        "cost_numerator": current,
        "cost_denominator": profile.cost_denominator,
        "cost": current / profile.cost_denominator,
        "utility": int(utility[chosen].sum()),
        "trace": trace,
        "price_profile": profile,
        "mapped_weights_integer": weights,
        "classes": classes,
        "identified_fraction": d,
    }


def greedy(
    released,
    offers,
    estimated_reputation,
    virtual_task_cells,
    cfg,
    budget,
    td_config=td.Config(),
    binary=False,
):
    return _greedy(
        released,
        offers,
        estimated_reputation,
        virtual_task_cells,
        cfg,
        budget,
        td_config,
        binary,
        "tent",
    )


def greedy_sine(
    released,
    offers,
    estimated_reputation,
    virtual_task_cells,
    cfg,
    budget,
    td_config=td.Config(),
    binary=False,
):
    return _greedy(
        released,
        offers,
        estimated_reputation,
        virtual_task_cells,
        cfg,
        budget,
        td_config,
        binary,
        "sine",
    )


def feasible(cost_numerator, profile, budget):
    limit = budget_limit(profile, budget)
    return np.asarray(
        [int(c) <= limit for c in np.asarray(cost_numerator).ravel()]
    ).reshape(np.asarray(cost_numerator).shape)


def public_templates(n, seed, t, budget=None):
    if n < 1:
        raise ValueError("Positive registered roster size required")
    count = int(np.ceil(3200 / n))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 9601, t]))
    base = np.argsort(rng.random((count, n)), axis=1, kind="stable").astype("<u2")
    p = np.repeat(base, n, axis=0)
    k = np.tile(np.arange(1, n + 1, dtype="<u2"), count)
    (a, b) = np.triu_indices(n, 1)
    header = {
        "policy": "BTPWR/v6/all-public-positive-prefixes/1",
        "n": n,
        "seed": seed,
        "round": t,
        "permutation_count": count,
        "targets": "1..n, permutation-major then target-ascending",
        "budget_independent": True,
    }
    digest = hashlib.sha256(
        json.dumps(header, sort_keys=True).encode() + p.tobytes() + k.tobytes()
    ).hexdigest()
    return {
        "permutations": p,
        "targets": k,
        "left": a.astype("<u2"),
        "right": b.astype("<u2"),
        "digest": digest,
    }


def compact(public, positive):
    e = np.asarray(positive, bool)
    perm = public["permutations"]
    k = public["targets"]
    ordered = e[perm]
    prefix = np.cumsum(ordered, axis=1, dtype=np.int16)
    picked = ordered & (prefix <= k[:, None])
    M = np.zeros(perm.shape, np.float32)
    np.put_along_axis(M, perm, picked.astype(np.float32), axis=1)
    available = e.sum() >= k
    assert np.all(M.sum(1) == np.minimum(k, int(e.sum())))
    assert np.all(np.sum(M[:, ~e], axis=1) == 0)
    assert np.all(M[available].sum(1) == k[available])
    assert np.array_equal(np.take_along_axis(M, perm, axis=1).astype(bool), picked)
    assert np.all(prefix[:, -1] == e.sum())
    return (M, available)


def pair_costs(public, released, cmin, cmax, eps2):
    (a, b) = (public["left"], public["right"])
    nc = released["bits"][a].astype(float) + released["bits"][b]
    hi = released["hi"][a].astype(float) + released["hi"][b]
    p = 1 / (1 + np.exp(-eps2))
    nh = np.clip((hi - (1 - p) * nc) / (2 * p - 1), 0, nc)
    return (cmin * nc + (cmax - cmin) * nh).sum(1)


def choose(
    public,
    positive,
    weights,
    individual_utility,
    compact_cost,
    pair_cost,
    budget,
    cache,
):
    e = np.asarray(positive, bool)
    key = np.packbits(e).tobytes()
    if key not in cache:
        (M, available) = compact(public, e)
        cache[key] = (M, available, compact_cost(M))
    (M, available, cc) = cache[key]
    (a, b) = (public["left"], public["right"])
    pair_available = e[a] & e[b]
    cost = np.concatenate((cc, pair_cost))
    utility = np.concatenate(
        (
            M @ (weights * individual_utility),
            (weights * individual_utility)[a] + (weights * individual_utility)[b],
        )
    )
    availability = np.concatenate((available, pair_available))
    feasible = availability & (cost <= budget)
    j = (
        int(np.flatnonzero(feasible)[np.argmax(utility[feasible])])
        if feasible.any()
        else -1
    )
    if j < 0:
        members = np.empty(0, int)
        target = 0
        kind = "abort"
    elif j < len(M):
        members = np.flatnonzero(M[j])
        target = int(public["targets"][j])
        kind = "compacted"
    else:
        q = j - len(M)
        members = np.array([a[q], b[q]], int)
        target = 2
        kind = "pair"
    assert j < 0 or (
        len(members) == target and np.all(e[members]) and (cost[j] <= budget + 1e-08)
    )
    assert not np.any(pair_available & (pair_cost <= budget)) or feasible.any()
    return {
        "index": j,
        "members": members,
        "costs": cost,
        "utilities": utility,
        "availability": availability,
        "feasible": feasible,
        "compacted_available": int(available.sum()),
        "pair_available": int(pair_available.sum()),
        "pair_feasible": int(np.sum(pair_available & (pair_cost <= budget))),
        "selected_target": target,
        "selected_kind": kind,
        "n_compacted": len(M),
        "all_candidate_count": len(cost),
        "available_cardinality_checks": int(available.sum() + pair_available.sum()),
        "all_private_membership_checks": True,
    }

"""Native baseline interfaces and their declared resource constraints."""
from dataclasses import replace
from collections import OrderedDict
import hashlib, time
import numpy as np
from . import _bpp as bpp, _ws as ws, _etbp as etbp, _mmcb as mmcb

DATASETS = ["tdrive", "foursquare_tky", "brightkite"]
ETBP_CONFIG = etbp.Config(
    alpha0=1.0,
    alpha1=0.2,
    sigma=0.1,
    history_window=5,
    initial_stage_periods=5,
    period_weight_rule="linear_recency",
    within_period_weight_rule="linear_recency",
)
_SELECTION_CACHE = OrderedDict()


def etbp_initial_state(world):
    """Shared certified identities AND sampled initial scores for native ETBP."""
    initial = np.asarray(world["initial_reputation"], float)
    state = etbp.initialize(len(initial), initial >= .8, ETBP_CONFIG)
    state.trust = initial.copy()
    state.history = [initial.copy()]
    return state


def etbp_labels(reputation, cfg=ETBP_CONFIG):
    """Native decisions: 0 excluded, 1 unresolved, 2 high-trust.

    Eq.27 uses a strict high boundary and Eq.28 a strict low boundary.
    Equality remains unresolved; Eq.32's separately printed >= reference
    condition is retained inside etbp.feedback rather than changing this rule.
    """
    r = np.asarray(reputation, float)
    if not np.isfinite(r).all() or np.any((r < 0) | (r > 1)):
        raise ValueError("Finite unit-interval native reputation required")
    return np.where(r > cfg.high_threshold, 2,
                    np.where(r < cfg.low_threshold, 0, 1))


def mean_reports(raw):
    count = np.isfinite(raw).sum(0)
    return np.divide(
        np.nansum(raw, axis=0),
        count,
        out=np.full(raw.shape[1], np.nan),
        where=count > 0,
    )


def plain_selections(
    plain, private, presence, costs, offers, tasks, budget, seed, t, eps1, eps2
):
    key = hashlib.sha256(
        np.asarray([budget, seed, t, eps1, eps2], dtype="<f8").tobytes()
    )
    for value in (
        plain["bits"],
        plain["hi"],
        private["bits"],
        private["hi"],
        presence,
        costs,
        offers,
        tasks,
    ):
        x = np.ascontiguousarray(value)
        key.update(str(x.dtype).encode())
        key.update(str(x.shape).encode())
        key.update(x.tobytes())
    digest = key.digest()
    if digest in _SELECTION_CACHE:
        _SELECTION_CACHE.move_to_end(digest)
        selections, payments, info = _SELECTION_CACHE[digest]
        return (
            {k: v.copy() for k, v in selections.items()},
            dict(payments),
            {
                k: {**v, "selection_cache_hit": True, "selector_seconds": None}
                for k, v in info.items()
            },
        )
    start = time.perf_counter()
    active = np.flatnonzero(offers)
    charges = bpp.published_pcp_costs(plain, eps2)
    local, binfo = bpp.exact_ws(
        plain["bits"][active], charges[active], budget, tasks, eps1
    )
    selections = {"BPP": active[local]}
    payments = {"BPP": float(charges[active[local]].sum())}
    binfo.update(
        selector_seconds=time.perf_counter() - start, selection_cache_hit=False
    )
    start = time.perf_counter()
    quotes = bpp.published_pcp_costs(private, eps2)
    permutation = np.random.default_rng(
        np.random.SeedSequence([seed, 43501, t])
    ).permutation(len(offers))
    accepted = []
    bill = 0.0
    for worker in permutation[offers[permutation]]:
        if bill + quotes[worker] > budget + 1e-9:
            break
        accepted.append(int(worker))
        bill += float(quotes[worker])
    selections["Random"] = np.array(accepted, int)
    payments["Random"] = bill
    rinfo = {
        "selector_seconds": time.perf_counter() - start,
        "selection_cache_hit": False,
    }
    start = time.perf_counter()
    bids = costs.sum(1)
    local, ninfo = ws.exact_ws(presence[active][:, tasks], bids[active], budget)
    selections["No-Privacy"] = active[local]
    payments["No-Privacy"] = float(bids[active[local]].sum())
    ninfo.update(
        selector_seconds=time.perf_counter() - start, selection_cache_hit=False
    )
    info = {"BPP": binfo, "Random": rinfo, "No-Privacy": ninfo}
    _SELECTION_CACHE[digest] = (
        {k: v.copy() for k, v in selections.items()},
        dict(payments),
        info,
    )
    if len(_SELECTION_CACHE) > 512:
        _SELECTION_CACHE.popitem(last=False)
    return selections, payments, info


def priority_state(r):
    cfg = replace(
        ETBP_CONFIG,
        high_threshold=float(np.nextafter(0.8, -np.inf)),
        low_threshold=float(np.nextafter(0.3, -np.inf)),
    )
    state = etbp.initialize(len(r), r >= 0.8, cfg)
    state.trust = r.copy()
    return state, cfg


def native_round(world, t, state, seed, dataset):
    tasks = world["requested_task_ids"][t]
    loc = world["presence"][t]
    offers = world["offer_mask"][t]
    active = np.flatnonzero(offers)
    bids = world["actual_cost"][t].sum(1)
    A = loc[:, tasks]
    di = DATASETS.index(dataset)
    auction = mmcb.allocate(
        A[active],
        bids[active],
        float(world["budget"]),
        np.ones(len(tasks)),
        np.random.default_rng(np.random.SeedSequence([seed, 78101, di, t])),
    )
    selected = active[auction["selected"]]
    raw = np.full((len(offers), len(tasks)), np.nan)
    raw[selected] = world["reports_check"][t, selected][:, tasks]
    means = mmcb.mean_service_head(raw, selected)
    recruitment = etbp.recruit_offers(
        A,
        min(int(0.4 * len(offers)), len(active)),
        state,
        np.random.default_rng(np.random.SeedSequence([seed, 78102, di, t])),
        ETBP_CONFIG,
        offers,
    )
    chosen = recruitment["selected"]
    raw = np.full_like(raw, np.nan)
    raw[chosen] = world["reports_check"][t, chosen][:, tasks]
    fb = etbp.feedback(
        raw,
        recruitment,
        state,
        float(np.ptp(10 + 30 * world["public_reference"])),
        ETBP_CONFIG,
    )
    values = {"MMCB": means, "ETBP-TD": fb["truth"]}
    selected = {"MMCB": selected, "ETBP-TD": chosen}
    records = {
        "MMCB": {
            "constraint": "native_monetary_payment_budget",
            "commitment": auction["total_payment"],
            "native_bid_sum": auction["selected_bid_sum"],
            "branch": auction["branch"],
        },
        "ETBP-TD": {
            "constraint": "native_fixed_cardinality",
            "commitment": None,
            "target_K": int(0.4 * len(offers)),
            "capacity_shortfall": recruitment["capacity_shortfall"],
        },
    }
    return values, selected, fb["next_state"], records


def vlr_campaign(world, case, rounds=50):
    from . import privacy, bgmc

    T, n, D = world["presence"].shape
    T = min(T, int(rounds))
    seed = case["seed"]
    cfg = privacy.Config(n=n, m=D, T=T, eps1=case["eps1"], eps2=case["eps2"])
    bcfg = bgmc.Config(
        rank=5, spatial_weight=0.5, temporal_weight=0.5, train_columns=30, als_cap=40
    )
    coords = np.c_[np.arange(10).repeat(10) / 9, np.tile(np.arange(10), 10) / 9]
    history = world["public_reference"].copy()
    history_mask = np.ones_like(history, bool)
    prediction = np.full((T, D), np.nan)
    chosen = np.zeros((T, n), bool)
    records = []
    for t in range(T):
        loc = world["presence"][t]
        virtual = world["virtual_presence"][t]
        cost = world["actual_cost"][t]
        offers = world["offer_mask"][t]
        avg = np.divide(
            cost.sum(1), loc.sum(1), out=np.full(n, 3.25), where=loc.sum(1) > 0
        )
        coins = world["RR_coins"][t]
        packet = privacy.release(
            virtual,
            np.where(virtual, avg[:, None], 0.0),
            coins,
            cfg.eps1,
            cfg.eps2,
            cfg,
        )
        for key in ("bits", "hi", "nonzero"):
            packet[key][~offers] = False
        charges = bpp.published_pcp_costs(packet, cfg.eps2)
        active = np.flatnonzero(offers)
        ids, diag = bpp.exact_ws(
            packet["bits"][active],
            charges[active],
            float(world["budget"]),
            np.unique(world["virtual_task_cells"][t]),
            cfg.eps1,
        )
        ids = active[ids]
        chosen[t, ids] = True
        raw = np.full((n, D), np.nan)
        raw[ids] = world["reports_check"][t, ids]
        current = (mean_reports(raw) - 10) / 30
        observed = np.isfinite(current)
        estimate, trace = bgmc.bgmc(
            history,
            np.nan_to_num(current),
            observed,
            coords,
            bcfg,
            history_mask=history_mask,
        )
        estimate = np.where(observed, current, estimate)
        history = np.column_stack([history, estimate])
        history_mask = np.column_stack([history_mask, observed])
        prediction[t] = np.clip(
            float(world["native_offset"]) + float(world["native_scale"]) * estimate,
            0,
            100,
        )
        records.append(
            {
                "case": case["name"],
                "dataset": case["dataset"],
                "seed": seed,
                "round": t + 1,
                "method": "VLR-BPP numerical reproduction",
                "constraint": "native_monetary_budget",
                "selected_workers": len(ids),
                "commitment": float(charges[ids].sum()),
                "budget": float(world["budget"]),
                "physical_cost": float(cost[ids].sum()),
                "ALS_iterations": len(trace) - 1,
                "last_objective": trace[-1]["objective"],
            }
        )
    return prediction, chosen, records

"""Generate complete recurrent method trajectories from one shared world."""
from pathlib import Path
from dataclasses import asdict
import csv, hashlib, json, time, os, uuid
import numpy as np
from .data import build_world
from . import privacy, trust, recruitment, recovery, baselines
from .metrics import summarize_case

ACQUISITIONS = [
    "tent-density",
    "binary-density",
    "ETBP-priority controlled",
    "sine-density",
    "original-prefix",
]


def scientific_code_digest():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def world_digest(world):
    h = hashlib.sha256()
    for name, value in sorted(world.items()):
        if isinstance(value, np.ndarray) or np.isscalar(value):
            x = np.ascontiguousarray(value)
            h.update(name.encode())
            h.update(str(x.dtype).encode())
            h.update(str(x.shape).encode())
            h.update(x.tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + uuid.uuid4().hex[:12] + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=json_default), encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def save_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
        )
        w.writeheader()
        w.writerows(rows)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def calibration_for(world, results):
    H = world["public_reference"]
    digest = hashlib.sha256(H.tobytes()).hexdigest()
    folder = Path(results) / "calibration"
    folder.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        H.tobytes()
        + np.asarray(
            [world["native_offset"], world["native_scale"]], dtype="<f8"
        ).tobytes()
    ).hexdigest()
    path = folder / (key + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    cal = recovery.public_calibration(
        H, float(world["native_offset"]), float(world["native_scale"])
    )
    atomic_json(path, cal)
    return cal


def run_case(case, inputs, outputdir, rounds=None, all_baselines=None, overwrite=False):
    result = Path(outputdir)
    folder = result / "cases" / case["name"]
    folder.mkdir(parents=True, exist_ok=True)
    intended_rounds = 50 if rounds is None else min(50, int(rounds))
    intended_native = (
        (case.get("group") == "registered")
        if all_baselines is None
        else bool(all_baselines)
    )
    code_digest = scientific_code_digest()
    w = build_world(case, inputs)
    input_digest = world_digest(w)
    if (folder / "evaluation.json").exists() and not overwrite:
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if (
            all(meta.get(k) == v for k, v in case.items())
            and meta.get("actual_rounds") == intended_rounds
            and meta.get("native_baselines_executed") == intended_native
            and meta.get("scientific_code_sha256") == code_digest
            and meta.get("world_input_sha256") == input_digest
        ):
            return json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
        raise ValueError(
            "Existing case has different parameters, round count or method scope. Use overwrite=True."
        )
    started = time.perf_counter()
    T, n, D = w["presence"].shape
    T = T if rounds is None else min(T, int(rounds))
    cfg = privacy.Config(n=n, m=D, T=T, eps1=case["eps1"], eps2=case["eps2"])
    tcfg = trust.Config(gamma=case["gamma"])
    B = float(w["budget"])
    seed = case["seed"]
    H = w["public_reference"]
    offset = float(w["native_offset"])
    scale = float(w["native_scale"])
    cal = calibration_for(w, result)
    registered = case.get("group") == "registered"
    use_native = registered if all_baselines is None else bool(all_baselines)
    acqs = ACQUISITIONS if registered else ACQUISITIONS[:4]
    A = len(acqs)
    heads = [recovery.initialize(H, offset, scale, cal) for _ in acqs]
    reps = np.repeat(w["initial_reputation"][None], A, axis=0)
    priority, pcfg = baselines.priority_state(reps[2])
    native = baselines.etbp.initialize(
        n, w["initial_reputation"] >= 0.8, baselines.ETBP_CONFIG
    )
    direct = np.full((A, T, D), np.nan)
    service = direct.copy()
    rbf = direct.copy()
    static = direct.copy()
    credible = direct.copy()
    actual = direct.copy()
    pre = np.empty((A, T, n))
    post = pre.copy()
    selected = np.zeros((A, T, n), bool)
    scored = np.zeros((A, T, n), np.int16)
    Ccount = np.zeros((A, T, D), np.int16)
    runtime = []
    selector_traces = []
    native_names = (
        ["BPP", "Random", "No-Privacy", "MMCB", "ETBP-TD"] if use_native else []
    )
    native_preds = {m: np.full((T, D), np.nan) for m in native_names}
    native_selected = {m: np.zeros((T, n), bool) for m in native_names}
    native_reps = [native.trust.copy()]
    for t in range(T):
        loc = w["presence"][t]
        cost = w["actual_cost"][t]
        offers = w["offer_mask"][t]
        v = w["virtual_presence"][t]
        coins = (
            w["RR_coins"][t]
            if "RR_coins" in w
            else np.random.default_rng(np.random.SeedSequence([seed, 53102, t])).random(
                (3, n, D)
            )
        )
        avg = np.divide(
            cost.sum(1), loc.sum(1), out=np.full(n, 3.25), where=loc.sum(1) > 0
        )
        packet = privacy.release(
            v, np.where(v, avg[:, None], 0.0), coins, cfg.eps1, cfg.eps2, cfg
        )
        for key in ("bits", "hi", "nonzero"):
            packet[key][~offers] = False
        tasks = w["virtual_task_cells"][t]
        for mi, acq in enumerate(acqs):
            tick = time.perf_counter()
            pre[mi, t] = reps[mi]
            if mi in (0, 1, 3):
                fn = recruitment.greedy_sine if mi == 3 else recruitment.greedy
                ans = fn(packet, offers, reps[mi], tasks, cfg, B, tcfg, binary=mi == 1)
                ids = ans["members"]
                num = ans["cost_numerator"]
                den = ans["cost_denominator"]
                selector_traces.append(
                    {"round": t + 1, "acquisition": acq, "trace": ans["trace"]}
                )
            elif mi == 2:
                order = baselines.etbp.recruit_offers(
                    packet["bits"][:, tasks],
                    min(40, int(offers.sum())),
                    priority,
                    np.random.default_rng(
                        np.random.SeedSequence(
                            [seed, 78102, baselines.DATASETS.index(case["dataset"]), t]
                        )
                    ),
                    pcfg,
                    offers,
                )["selected"]
                ids, num, den = recruitment.prefix_budget(order, packet, cfg, B)
            else:
                public = recruitment.public_templates(n, seed, t, B)
                pairs = recruitment.pair_costs(
                    public, packet, cfg.cmin, cfg.cmax, cfg.eps2
                )
                weights, _, _ = trust.mapweights(reps[mi], "BTPWR", tcfg)
                p = 1 / (1 + np.exp(-cfg.eps1))
                utility = np.clip(
                    (packet["bits"][:, tasks].astype(float) - (1 - p)) / (2 * p - 1),
                    0,
                    1,
                ).sum(1)
                ans = recruitment.choose(
                    public,
                    offers & (weights > 0),
                    weights,
                    utility,
                    lambda M: privacy.candidatecost(M, packet, cfg),
                    pairs,
                    B,
                    {},
                )
                ids = ans["members"]
                num = float(ans["costs"][ans["index"]]) if ans["index"] >= 0 else 0.0
                den = 1
            raw = np.full((n, D), np.nan)
            raw[ids] = w["reports_check"][t, ids]
            raw[trust.classify(reps[mi], tcfg) == 0] = np.nan
            fb = trust.feedback(raw, reps[mi], tcfg)
            act = recovery.decode_actual(fb["consensus"], heads[mi])
            direct[mi, t] = act
            credible[mi, t] = fb["credible_consensus"]
            actual[mi, t] = fb["consensus"]
            service[mi, t] = recovery.complete(
                fb["credible_consensus"], act, heads[mi]
            )[0]
            rbf[mi, t] = recovery.fixed_complete(
                fb["credible_consensus"], act, heads[mi]
            )[0]
            static[mi, t] = recovery.linear_complete(
                fb["credible_consensus"], act, heads[mi]["linear"]
            )[0]
            reps[mi] = fb["next_reputation"]
            post[mi, t] = reps[mi]
            selected[mi, t, ids] = True
            scored[mi, t] = fb["scored_count"]
            Ccount[mi, t] = fb["credible_count"]
            if mi == 2:
                priority.trust = reps[mi].copy()
                priority.evaluation_count += fb["scored_count"]
                priority.recruited_count[ids] += 1
                priority.history.append(reps[mi].copy())
                priority.period += 1
            assert np.all(offers[ids]) and num / den <= B + 1e-7
            runtime.append(
                {
                    "case": case["name"],
                    "dataset": case["dataset"],
                    "field": case["field"],
                    "seed": seed,
                    "method": acq,
                    "round": t + 1,
                    "constraint": "shared_rational_monetary_budget"
                    if mi != 4
                    else "original_float_monetary_budget",
                    "selected_workers": len(ids),
                    "commitment": num / den,
                    "cost_numerator": num,
                    "cost_denominator": den,
                    "budget": B,
                    "physical_cost": float(cost[ids].sum()),
                    "actual_C_anchors": int(fb["has_credible"].sum()),
                    "OTHER_C_scored_reports": int(fb["scored_count"].sum()),
                    "seconds": time.perf_counter() - tick,
                }
            )
        if use_native:
            plain = privacy.release(loc, cost, coins, cfg.eps1, cfg.eps2, cfg)
            for key in ("bits", "hi", "nonzero"):
                plain[key][~offers] = False
            tick = time.perf_counter()
            picks, pays, selection_info = baselines.plain_selections(
                plain,
                packet,
                loc,
                cost,
                offers,
                w["requested_task_ids"][t],
                B,
                seed,
                t,
                cfg.eps1,
                cfg.eps2,
            )
            for name, ids in picks.items():
                raw = np.full((n, D), np.nan)
                raw[ids] = w["reports_check"][t, ids]
                native_preds[name][t] = recovery.decode_actual(
                    baselines.mean_reports(raw), heads[0]
                )
                native_selected[name][t, ids] = True
                runtime.append(
                    {
                        "case": case["name"],
                        "dataset": case["dataset"],
                        "field": case["field"],
                        "seed": seed,
                        "method": name,
                        "round": t + 1,
                        "constraint": "native_monetary_budget",
                        "selected_workers": len(ids),
                        "commitment": pays[name],
                        "budget": B,
                        "physical_cost": float(cost[ids].sum()),
                        "selector_seconds": selection_info[name]["selector_seconds"],
                        "selection_cache_hit": selection_info[name][
                            "selection_cache_hit"
                        ],
                    }
                )
            values, picks, native, records = baselines.native_round(
                w, t, native, seed, case["dataset"]
            )
            native_reps.append(native.trust.copy())
            for name, ids in picks.items():
                native_preds[name][t, w["requested_task_ids"][t]] = np.clip(
                    offset + scale * (values[name] - 10) / 30, 0.0, 100.0
                )
                native_selected[name][t, ids] = True
                runtime.append(
                    {
                        "case": case["name"],
                        "dataset": case["dataset"],
                        "field": case["field"],
                        "seed": seed,
                        "method": name,
                        "round": t + 1,
                        "selected_workers": len(ids),
                        "budget": B,
                        "physical_cost": float(cost[ids].sum()),
                        **records[name],
                    }
                )
        if (t + 1) % 10 == 0:
            print(
                json.dumps(
                    {
                        "case": case["name"],
                        "round": t + 1,
                        "seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    models = {
        "BTPWR": service[0],
        "BPP-TW": direct[0],
        "BPP-T": direct[1],
        "Binary+newhead": service[1],
        "Mapped+RBF": rbf[0],
        "Mapped+static": static[0],
        "ETBP-priority+newhead": service[2],
        "ETBP-priority+RBF": rbf[2],
        "Sin+newhead": service[3],
        **native_preds,
    }
    index = {
        "BTPWR": 0,
        "BPP-TW": 0,
        "BPP-T": 1,
        "Binary+newhead": 1,
        "Mapped+RBF": 0,
        "Mapped+static": 0,
        "ETBP-priority+newhead": 2,
        "ETBP-priority+RBF": 2,
        "Sin+newhead": 3,
    }
    if registered:
        for suffix, arr in [("newhead", service), ("RBF", rbf), ("static", static)]:
            name = "Original-prefix+" + suffix
            models[name] = arr[4]
            index[name] = 4
    picks = {name: selected[i] for name, i in index.items()}
    picks.update(native_selected)
    histories = {
        name: np.vstack([pre[i, 0][None], post[i]]) for name, i in index.items()
    }
    if use_native:
        histories["ETBP-TD"] = np.asarray(native_reps)
    payload = {
        "models": models,
        "selected": picks,
        "reputations": histories,
        "task_mask": w["task_mask"][:T],
        "truth_native": w["truth_native_private"][:, :T].T,
        "credible": {name: credible[i] for i, name in enumerate(acqs)},
        "actual": {name: actual[i] for i, name in enumerate(acqs)},
        "runtime": runtime,
    }
    if case.get("kind") == "confirmation" and use_native:
        vp, vs, vr = baselines.vlr_campaign(w, case, T)
        models["VLR-BPP numerical reproduction"] = vp
        picks["VLR-BPP numerical reproduction"] = vs
        runtime.extend(vr)
    names = list(models)
    empty = np.full((T + 1, n), np.nan)
    np.savez_compressed(
        folder / "predictions.npz",
        methods=np.array(names),
        prediction_native=np.array([models[m] for m in names]),
        selected_masks=np.array([picks[m] for m in names]),
        reputation_history=np.array([histories.get(m, empty) for m in names]),
        acquisitions=np.array(acqs),
        credible_consensus_check=credible,
        actual_consensus_check=actual,
        prediction_service_v3=service,
        prediction_direct=direct,
        prediction_RBF=rbf,
        prediction_static=static,
        acquisition_selected_masks=selected,
        reputation_pre=pre,
        reputation_post=post,
        credible_count=Ccount,
        scored_count=scored,
        task_mask=w["task_mask"][:T],
        requested_task_ids=w["requested_task_ids"][:T],
        truth_native_private=w["truth_native_private"][:, :T],
        honest_private=w["honest_private"],
        certified=w["certified"],
        public_reference=H,
        native_offset=offset,
        native_scale=scale,
    )
    np.savez_compressed(
        folder / "world.npz",
        **{
            k: v
            for k, v in w.items()
            if k != "RR_coins" and (isinstance(v, np.ndarray) or np.isscalar(v))
        }
    )
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                **case,
                "actual_rounds": T,
                "native_baselines_executed": use_native,
                "public_calibration": cal,
                "elapsed_seconds": time.perf_counter() - started,
                "trust_config": asdict(tcfg),
                "scientific_code_sha256": code_digest,
                "world_input_sha256": input_digest,
            },
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )
    (folder / "selector_traces.json").write_text(
        json.dumps(selector_traces), encoding="utf-8"
    )
    save_rows(folder / "runtime.csv", runtime)
    tables = summarize_case(case, w, payload)
    (folder / "evaluation.json").write_text(
        json.dumps(tables, default=json_default), encoding="utf-8"
    )
    return tables

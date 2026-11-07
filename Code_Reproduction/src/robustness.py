"""Reference scarcity, compromised certification, and heterogeneous report noise."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np

from .data import build_world, rng
from . import privacy, recruitment, recovery, trust
from .simulate import calibration_for, save_rows


CASES = [
    {"name": "baseline_C10", "C": 10},
    {"name": "scarce_C0", "C": 0},
    {"name": "scarce_C1", "C": 1},
    {"name": "wrong_C1", "C": 1, "wrong": 1, "bias": 9.0},
    {"name": "wrong2_C10", "C": 10, "wrong": 2, "bias": 9.0},
    {"name": "onoff2_C10", "C": 10, "onoff": 2, "bias": 9.0},
    {"name": "hetero14_sigma45", "C": 10, "noisy_honest": 14, "sigma": 4.5},
    {"name": "Uonly14_sigma45", "C": 10},
]
PROFILE = "final_Tent_U_exact_density_public_blocked_missing_fill"


def generate(world, seed, case):
    n = len(world["initial_reputation"])
    order = rng(seed, 43001).permutation(n)
    initial = np.full(n, 0.5)
    cert = order[: case["C"]]
    initial[cert] = world["initial_reputation"][cert]
    reports = world["reports_check"].copy()
    attackers = np.array(cert[: case.get("wrong", case.get("onoff", 0))], int)
    honest = world["honest_private"].copy()
    honest[attackers] = False
    truth = 10 + 30 * world["truth_normalized_private"].T
    active = np.zeros((50, n), bool)
    if len(attackers):
        active[:, attackers] = True
        if "onoff" in case:
            active[:] = False
            active[20:25, attackers] = True
            active[30:35, attackers] = True
        reports = np.where(
            active[:, :, None] & world["presence"],
            np.round(truth[:, None, :] + case["bias"], 2),
            reports,
        )
    noisy = np.empty(0, int)
    if "noisy_honest" in case:
        noisy = order[world["honest_private"][order]][: case["noisy_honest"]]
    elif case["name"] == "Uonly14_sigma45":
        pool = np.flatnonzero(world["honest_private"] & ~np.isin(np.arange(n), cert))
        noisy = rng(seed, 1151103).choice(pool, 14, replace=False)
    if len(noisy):
        noise = rng(seed, 1151102).normal(
            0, case.get("sigma", 4.5), (50, len(noisy), 100)
        )
        reports[:, noisy] = np.where(
            world["presence"][:, noisy], np.round(truth[:, None, :] + noise, 2), np.nan
        )
    np.testing.assert_array_equal(np.isfinite(reports), world["presence"])
    return {
        "reports": reports,
        "initial": initial,
        "cert": np.isin(np.arange(n), cert),
        "attack_ids": attackers,
        "clean_honest": honest,
        "active_attack": active,
        "ordinary_malicious": ~world["honest_private"],
        "noisy": noisy,
    }


def acquire(world, generated, calibration):
    cfg = privacy.Config(n=100, m=100, T=50, eps1=2.1, eps2=2.7)
    tcfg = trust.Config(gamma=0.6)
    model = recovery.initialize(
        world["public_reference"],
        world["native_offset"],
        world["native_scale"],
        calibration,
    )
    r = generated["initial"].copy()
    states, prediction, selected, scored, counts, credible, actual = (
        [r.copy()],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    numerator, denominator = [], []
    start = time.perf_counter()
    for t in range(50):
        loc, cost, virtual, offer = (
            world[k][t]
            for k in ("presence", "actual_cost", "virtual_presence", "offer_mask")
        )
        average = np.divide(
            cost.sum(1), loc.sum(1), out=np.full(100, 3.25), where=loc.sum(1) > 0
        )
        packet = privacy.release(
            virtual,
            np.where(virtual, average[:, None], 0.0),
            world["RR_coins"][t],
            2.1,
            2.7,
            cfg,
        )
        for k in ("bits", "hi", "nonzero"):
            packet[k][~offer] = False
        ans = recruitment.greedy(
            packet,
            offer,
            r,
            world["virtual_task_cells"][t],
            cfg,
            float(world["budget"]),
            tcfg,
        )
        ids = ans["members"]
        chosen = np.zeros(100, bool)
        chosen[ids] = True
        classes = trust.classify(r, tcfg)
        raw = np.full((100, 100), np.nan)
        raw[ids] = generated["reports"][t, ids]
        raw[classes == 0] = np.nan
        fb = trust.feedback(raw, r, tcfg)
        act = recovery.decode_actual(fb["consensus"], model)
        p, _ = recovery.complete(fb["credible_consensus"], act, model)
        nxt = fb["next_reputation"]
        assert (
            np.all(offer[ids])
            and ans["cost_numerator"] / ans["cost_denominator"]
            <= float(world["budget"]) + 1e-9
        )
        np.testing.assert_array_equal(nxt[~chosen], r[~chosen])
        np.testing.assert_array_equal(nxt[classes == 0], r[classes == 0])
        np.testing.assert_array_equal(
            p[np.isfinite(fb["credible_consensus"])],
            act[np.isfinite(fb["credible_consensus"])],
        )
        prediction.append(p)
        selected.append(chosen)
        scored.append(fb["scored_count"])
        counts.append(fb["credible_count"])
        credible.append(fb["credible_consensus"])
        actual.append(act)
        numerator.append(ans["cost_numerator"])
        denominator.append(ans["cost_denominator"])
        states.append(nxt.copy())
        r = nxt
    return {
        "prediction_native": np.asarray(prediction),
        "reputation_states": np.asarray(states),
        "selected_masks": np.asarray(selected),
        "scored_count": np.asarray(scored),
        "credible_count": np.asarray(counts),
        "credible_consensus_check": np.asarray(credible),
        "actual_consensus_native": np.asarray(actual),
        "cost_numerator": np.asarray(numerator, np.int64),
        "cost_denominator": np.asarray(denominator, np.int64),
        "clean_honest": generated["clean_honest"],
        "ordinary_malicious": generated["ordinary_malicious"],
        "initial_certified": generated["cert"],
        "compromised_ids": generated["attack_ids"],
        "active_attack": generated["active_attack"],
        "noisy_ids": generated["noisy"],
        "elapsed_s": np.array(time.perf_counter() - start),
    }


def evaluate_arrays(truth, prediction, mask):
    finite = np.isfinite(prediction) & mask
    error = (prediction - truth)[finite]
    success = finite & (np.abs(prediction - truth) < 0.1 * np.abs(truth))
    success |= finite & (truth == 0) & (prediction == 0)
    return {
        "TCR": 100 * success.sum() / mask.sum(),
        "AR": 100 * finite.sum() / mask.sum(),
        "MAE": float(np.abs(error).mean()) if len(error) else None,
        "RMSE": float(np.sqrt(np.mean(error * error))) if len(error) else None,
        "answered": int(finite.sum()),
        "completed": int(success.sum()),
    }


def evaluate(world, payload, case, seed):
    p, r = payload["prediction_native"], payload["reputation_states"]
    selected, scored, counts = (
        payload[k] for k in ("selected_masks", "scored_count", "credible_count")
    )
    honest, mal, cert, attack = (
        payload[k]
        for k in (
            "clean_honest",
            "ordinary_malicious",
            "initial_certified",
            "compromised_ids",
        )
    )
    classes = trust.classify(r.reshape(-1)).reshape(r.shape)
    false = honest & np.any(classes[1:] == 0, axis=0)
    promoted = mal & ~cert & np.any(classes[1:] == 2, axis=0)
    truth, mask = world["truth_native_private"].T, world["task_mask"]
    row = {
        "profile": PROFILE,
        "case": case["name"],
        "seed": seed,
        **evaluate_arrays(truth, p, mask),
        "honest_false_excluded": int(false.sum()),
        "honest_denominator": int(honest.sum()),
        "ordinary_malicious_promoted_ever": int(promoted.sum()),
        "ordinary_malicious_denominator": int(mal.sum()),
        "ordinary_malicious_final_M": int((mal & (classes[-1] == 0)).sum()),
        "ordinary_malicious_final_C": int((mal & (classes[-1] == 2)).sum()),
        "initial_attacker_final_C": int((classes[-1, attack] == 2).sum()),
        "initial_attacker_final_M": int((classes[-1, attack] == 0).sum()),
        "initial_attacker_count": len(attack),
        "scored_honest_worker_rounds": int((scored[:, honest] > 0).sum()),
        "selected_mean": float(selected.sum(1).mean()),
        "singleton_anchor_fraction": float(
            (counts == 1).sum() / max((counts > 0).sum(), 1)
        ),
        "elapsed_s": float(payload["elapsed_s"]),
        "false_exclusion_recovery_rounds": "infinite_without_external_reset_if_any"
        if false.any()
        else "not_applicable",
    }
    row.update(
        {
            "tail_" + k: v
            for k, v in evaluate_arrays(truth[40:], p[40:], mask[40:]).items()
        }
    )
    rounds = []
    for t in range(50):
        num, den = (
            int(payload["cost_numerator"][t]),
            int(payload["cost_denominator"][t]),
        )
        rounds.append(
            {
                "case": case["name"],
                "seed": seed,
                "round": t + 1,
                "selected": int(selected[t].sum()),
                "cost_numerator": num,
                "cost_denominator": den,
                "commitment": num / den,
                "budget": float(world["budget"]),
                "C_anchors": int((counts[t] > 0).sum()),
                "singleton_C_anchors": int((counts[t] == 1).sum()),
                "OTHER_C_scored_reports": int(scored[t].sum()),
            }
        )
    return row, rounds


def run_stress_suite(inputs, result_dir, overwrite=False, case_filter=None):
    from .simulate import scientific_code_digest, world_digest

    root = Path(result_dir)
    out = root / "stress"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    filters = (
        []
        if case_filter is None
        else ([case_filter] if isinstance(case_filter, str) else list(case_filter))
    )
    if filters and not any(
        f in c["name"] + "_seed" + str(s)
        for f in filters
        for c in CASES
        for s in (602, 603, 604)
    ):
        raise ValueError("No reference-stress campaigns matched the requested filter")
    code = scientific_code_digest()
    contexts = {}

    def fingerprint(case, seed):
        world, calibration, digest = contexts[seed]
        record = {
            "code": code,
            "world": digest,
            "case": case,
            "seed": seed,
            "calibration": calibration,
        }
        return hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    for seed in (602, 603, 604):
        world = build_world(
            {
                "dataset": "tdrive",
                "seed": seed,
                "field": "legacy",
                "n": 100,
                "H": 0.7,
                "C": 0.1,
                "budget_override": None,
            },
            inputs,
        )
        calibration = calibration_for(world, root)
        contexts[seed] = (world, calibration, world_digest(world))
        for case in CASES:
            name = case["name"] + "_seed" + str(seed)
            if filters and not any(f in name for f in filters):
                continue
            path = out / "predictions" / (name + ".npz")
            if path.exists() and not overwrite:
                with np.load(path, allow_pickle=False) as z:
                    payload = {k: z[k].copy() for k in z.files}
                if str(payload.get("input_fingerprint", "")) != fingerprint(case, seed):
                    raise ValueError(
                        "Stress source or input changed; rerun with --overwrite: "
                        + name
                    )
            else:
                payload = acquire(world, generate(world, seed, case), calibration)
                payload["input_fingerprint"] = np.array(fingerprint(case, seed))
                np.savez_compressed(path, **payload)
            row, _ = evaluate(world, payload, case, seed)
            print(
                json.dumps({"stress": name, "TCR": row["TCR"], "MAE": row["MAE"]}),
                flush=True,
            )
    if scientific_code_digest() != code:
        raise ValueError(
            "Scientific source changed during stress execution; rerun with --overwrite"
        )
    rows, rounds = [], []
    for seed in (602, 603, 604):
        world = contexts[seed][0]
        for case in CASES:
            path = out / "predictions" / (case["name"] + "_seed" + str(seed) + ".npz")
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as z:
                payload = {k: z[k].copy() for k in z.files}
            if str(payload.get("input_fingerprint", "")) != fingerprint(case, seed):
                continue
            row, detail = evaluate(world, payload, case, seed)
            rows.append(row)
            rounds.extend(detail)
    if not rows:
        raise ValueError("No reference-stress campaigns matched the requested filter")
    save_rows(out / "per_campaign.csv", rows)
    save_rows(out / "per_round.csv", rounds)
    return rows

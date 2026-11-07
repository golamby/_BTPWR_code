"""Replay the unchanged BTPWR and export authenticated-test inputs for EthereumJS."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import secrets
import time

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from src import data, privacy, recruitment, trust
from src.admission_experiments import audit_worker_feedback, run_case
from src.simulate import json_default
from .run_native_ce import ce

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "btpwr-original-replay-v1"
PAYMENT_SUBUNITS = 10**9


def digest(*items, salt=b""):
    state = hashlib.sha256(salt)
    for item in items:
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            for metadata in (array.dtype.str, list(array.shape)):
                state.update(json.dumps(metadata, separators=(",", ":")).encode())
            value = array.tobytes()
        else:
            value = json.dumps(item, sort_keys=True, separators=(",", ":"),
                               default=json_default).encode()
        state.update(len(value).to_bytes(8, "big"))
        state.update(value)
    return "0x" + state.hexdigest()


def allocate(amount, weights, slots):
    """Largest-remainder conversion of the original proportional payout."""
    if not amount:
        return [0] * len(weights)
    if not sum(weights):
        raise ValueError("Positive charge requires positive singleton quote mass")
    total = sum(weights)
    qr = [divmod(amount * weight, total) for weight in weights]
    payments = [base for base, _ in qr]
    rank = sorted(range(len(slots)), key=lambda j: (-qr[j][1], slots[j]))
    for j in rank[: amount - sum(payments)]:
        payments[j] += 1
    assert sum(payments) == amount
    return payments


def pack_input(case, path):
    occupancy = data.presence(case, data.load_inputs())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, name=np.array(case["name"]),
                        shape=np.array(occupancy.shape), bits=np.packbits(occupancy.reshape(-1)))


def world_for(case, occupancy):
    if occupancy:
        with np.load(occupancy, allow_pickle=False) as packed:
            shape = tuple(packed["shape"])
            if str(packed["name"]) != case["name"] or shape != (50, case["n"], 100):
                raise ValueError("Occupancy is not registered for this case")
            cells = np.unpackbits(packed["bits"], count=int(np.prod(shape))).reshape(shape)
        return data.build_world(case, data.load_inputs(fields_only=True), cells.astype(bool))
    return data.build_world(case, data.load_inputs())


def replay(case, folder, occupancy, capture_rounds, expected):
    folder.mkdir(parents=True, exist_ok=True)
    world = world_for(case, occupancy)
    tmax, n, d = world["presence"].shape
    cfg = privacy.Config(n=n, m=d, T=tmax, eps1=case["eps1"], eps2=case["eps2"])
    price = recruitment.PriceProfile.from_config(cfg)
    cap = recruitment.budget_limit(price, world["budget"]) * PAYMENT_SUBUNITS
    policy_sources = ("admission.py", "admission_experiments.py", "trust.py", "recovery.py",
                      "recruitment.py", "progress.py", "footprints.py", "privacy.py")
    versions = {name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
                for name in policy_sources}
    policy = digest(PROFILE, case, versions, PAYMENT_SUBUNITS)
    previous = genesis = digest("enrollment", world["initial_reputation"], salt=secrets.token_bytes(32))
    rounds = []

    def observe(index, selected, numerator, denominator, packet, offers, pre, reports,
                feedback, predictions, events, frames, selection_stats, recovery_seconds, **_):
        nonlocal previous
        assert denominator == price.cost_denominator
        slots = list(map(int, selected))
        if slots:
            membership = np.eye(n, dtype=bool)[slots]
            weights = list(map(int, recruitment.quote_numerators(membership, packet, price)))
        else:
            weights = []
        charge = int(numerator) * PAYMENT_SUBUNITS
        amounts = allocate(charge, weights, slots)
        offer_root = digest("offers", index, offers, packet["bits"], packet["hi"],
                            world["requested_task_ids"][index], world["virtual_task_cells"][index])
        reports_root = digest("reports", index, reports, salt=secrets.token_bytes(32))
        next_root = digest("state", index, previous, reports_root, feedback["next_reputation"],
                           predictions, [asdict(event) for event in events if event.round == index],
                           salt=secrets.token_bytes(32))
        rounds.append(dict(round=index + 1, selected=slots, budget=str(cap), charge=str(charge),
                           payouts=list(map(str, amounts)), inputs=offer_root, reports=reports_root,
                           previous=previous, next=next_root, estimates=int(np.isfinite(predictions).sum()),
                           reference_planner_seconds=selection_stats[0],
                           reference_recovery_seconds=recovery_seconds))
        previous = next_root
        if index + 1 in capture_rounds:
            np.savez_compressed(folder / f"private-feedback-{index+1:02}.npz",
                reputation=pre[slots], reports=reports[slots], cells=np.arange(d),
                next_reputation=feedback["next_reputation"][slots],
                scored_count=feedback["scored_count"][slots],
                consensus=feedback["consensus"], gamma=np.array(case["gamma"]))

    start = time.perf_counter()
    results = run_case(case, world, folder, methods=("admission",), round_observer=observe)
    raw_path = folder / (case["name"] + ".npz")
    audit = audit_worker_feedback(world, raw_path, case)
    archive_match = None
    if expected:
        with np.load(raw_path, allow_pickle=False) as observed, np.load(expected, allow_pickle=False) as frozen:
            index = frozen["methods"].tolist().index("admission")
            np.testing.assert_array_equal(observed["task_mask"], frozen["task_mask"])
            np.testing.assert_array_equal(observed["truth"], frozen["truth"])
            np.testing.assert_array_equal(observed["predictions"][0], frozen["predictions"][index])
            np.testing.assert_array_equal(observed["admission__stats"][:, :6], frozen["resources"][index])
            archive_match = True
    manifest = dict(profile=PROFILE, case=case["name"], registeredWorkers=n,
                    scope="Original numerical BTPWR, locally generated roots, synthetic signing keys",
                    policy=policy, genesis=genesis, rounds=rounds)
    (folder / "rounds.local.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    receipt = dict(case=case["name"], registeredWorkers=n, rounds=len(rounds), archive_match=archive_match,
                   worker_audit=audit, reference_results=results,
                   elapsed_seconds=time.perf_counter() - start, policy_hash=policy)
    (folder / "reference_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def diagnose(case, folder, occupancy):
    """Compare single-round updates on identical reports and frozen prestates."""
    world = world_for(case, occupancy)
    cfg = trust.Config(gamma=case["gamma"])
    with np.load(folder / (case["name"] + ".npz"), allow_pickle=False) as archive:
        policy = archive["methods"].tolist().index("admission")
        previous = archive["admission__pre"]
        picked = archive["admission__selected"]
        expected = archive["reputations"][policy, 1:]
    difference, categories, total = 0, 0, 0
    largest = 0
    rounds = []
    for t, r in enumerate(previous):
        reports = np.where(picked[t, :, None], world["reports_check"][t], np.nan)
        alternate = trust.feedback(reports, r, cfg, reference_iteration_cap=0)
        candidate = alternate["next_reputation"]
        codes = np.rint(cfg.reputation_scale * abs(candidate - expected[t])).astype(int)
        classes = trust.classify(candidate, cfg) != trust.classify(expected[t], cfg)
        a = int(np.count_nonzero(codes))
        b = int(np.count_nonzero(classes))
        difference += a
        categories += b
        total += int(np.count_nonzero(alternate["scored_count"]))
        largest = max(largest, int(codes.max()))
        rounds.append(dict(round=t + 1, changed_codes=a, class_disagreements=b))
    result = dict(case=case["name"], rounds=len(previous), scored_worker_rounds=total,
                  changed_codes=difference, class_disagreements=categories, maximum_code_error=largest,
                  scope="Zero-iteration diagnostic on identical paid reports and frozen original prestates",
                  per_round=rounds)
    (folder / "zero_iteration_diagnostic.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_round"}))


def ablate(case, folder, occupancy):
    """Replay a separate no-iteration feedback arm, preserving sensing consensus."""
    world = world_for(case, occupancy)
    target = folder / "zero-iteration-reference"
    rows = run_case(case, world, target, methods=("admission",), reference_iteration_cap=0)
    with np.load(target / (case["name"] + ".npz"), allow_pickle=False) as candidate:
        with np.load(folder / (case["name"] + ".npz"), allow_pickle=False) as original:
            p, q = candidate["predictions"][0], original["predictions"][0]
            truth, requested = original["truth"], original["task_mask"]
            np.testing.assert_array_equal(candidate["truth"], truth)
            np.testing.assert_array_equal(candidate["task_mask"], requested)
            joint = requested & np.isfinite(p) & np.isfinite(q)
            error, baseline = abs(p[joint] - truth[joint]), abs(q[joint] - truth[joint])
            if not joint.any():
                raise ValueError("No common answers for the MAE/RMSE comparison")
            receipt = dict(case=case["name"], candidate=rows[0],
                changed_roster_rounds=int(np.count_nonzero(np.any(
                    candidate["admission__selected"] != original["admission__selected"], axis=1))),
                changed_reputation_codes=int(np.count_nonzero(
                    candidate["reputations"] != original["reputations"])),
                common_tasks=int(joint.sum()),
                common_mae_change=float(error.mean() - baseline.mean()),
                common_rmse_change=float(np.sqrt((error**2).mean()) - np.sqrt((baseline**2).mean())),
                cumulative_error_change={str(k): ce(p, truth, requested, k)[0] -
                    ce(q, truth, requested, k)[0] for k in (3, 5, 10)},
                scope="Only OTHER-C scoring references omit iteration; registered task consensus and recovery are unchanged")
    (folder / "zero_iteration_replay.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("prepare", "replay", "diagnose-iterations", "ablate-iterations"))
    cli.add_argument("--case", default="tdrive_9701_B475.0_C0.1")
    cli.add_argument("--occupancy", type=Path)
    cli.add_argument("--output", type=Path, default=ROOT / "results" / "original-committee")
    cli.add_argument("--capture-rounds", type=int, nargs="*", default=())
    cli.add_argument("--allow-new-world", action="store_true")
    args = cli.parse_args()
    manifest = json.loads((ROOT / "recruitment_manifest.json").read_text(encoding="utf-8-sig"))
    case = next((item for item in manifest["cases"] if item["name"] == args.case), None)
    if case is None or case["stage"] != "main":
        cli.error("Use a registered main campaign from recruitment_manifest.json")
    if args.action == "prepare":
        if not args.occupancy:
            cli.error("prepare requires --occupancy")
        pack_input(case, args.occupancy)
    elif args.action == "diagnose-iterations":
        diagnose(case, args.output, args.occupancy)
    elif args.action == "ablate-iterations":
        ablate(case, args.output, args.occupancy)
    else:
        expected = ROOT / "data" / "recruitment_evidence" / (case["name"] + ".npz")
        if not expected.exists() and not args.allow_new_world:
            cli.error("Frozen evidence missing. Use --allow-new-world for a separately labeled replay.")
        replay(case, args.output, args.occupancy, args.capture_rounds,
               None if args.allow_new_world else expected)


if __name__ == "__main__":
    main()

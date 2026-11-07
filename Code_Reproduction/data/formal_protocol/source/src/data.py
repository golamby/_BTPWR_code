"""Real mobility preprocessing and paired crowdsensing experiment generation."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import csv
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np


DATASETS = ("tdrive", "foursquare_tky", "brightkite")
BOUNDS = {
    "tdrive": (39.6, 40.2, 116.1, 116.7),
    "foursquare_tky": (35.4, 35.9, 139.3, 140.0),
    "brightkite": (39.4, 40.1, -105.3, -104.6),
}
BUDGETS = {"tdrive": 475.0, "foursquare_tky": 475.0, "brightkite": 455.0}
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data"
MONTHS = {
    s: i + 1
    for i, s in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())
}
FREQUENCIES = np.array([0.05, 0.6, 1.6, 2.75, 2.7, 1.6, 0.6, 0.1])
FREQUENCIES /= FREQUENCIES.sum()


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rng(seed, family, t=None):
    return np.random.default_rng(
        np.random.SeedSequence([seed, family] if t is None else [seed, family, t])
    )


def load_inputs(data_dir=DEFAULT_DATA, fields_only=False):
    root = Path(data_dir)
    source = json.loads((root / "sources.json").read_text(encoding="utf-8"))
    inputs = {"fields": {}, "mobility": {}}
    with np.load(root / "reference_fields.npz", allow_pickle=False) as z:
        for key, expected in source["NOAA"]["array_sha256"].items():
            if key not in z.files or array_hash(z[key]) != expected:
                raise ValueError(
                    "Different public reference input; run python src/data.py reference: "
                    + key
                )
        for name in ("noaa_east_asia", "noaa_europe", "noaa_north_america"):
            inputs["fields"][name] = {
                k: z[name + "__" + k].copy() for k in ("rh", "dates")
            }
    legacy = root / "controlled_reference.npz"
    if not legacy.exists():
        legacy = root / "local" / "reference.npz"
    if legacy.exists():
        with np.load(legacy, allow_pickle=False) as z:
            for key, expected in source["legacy_controlled_field"][
                "array_sha256"
            ].items():
                if key not in z.files or array_hash(z[key]) != expected:
                    raise ValueError(
                        "Different provided controlled reference input: " + key
                    )
            inputs["fields"]["legacy"] = {
                "rh": z["rh"].astype(float),
                "dates": z["dates"].copy(),
            }
    if fields_only:
        return inputs
    path = root / "local" / "mobility.npz"
    if not path.exists():
        raise FileNotFoundError(
            "Run python src/data.py prepare --raw-dir YOUR_SOURCE_FOLDER first. Mobility derivatives stay local."
        )
    with np.load(path, allow_pickle=False) as z:
        for ds in DATASETS:
            prefix = ds + "__"
            inputs["mobility"][ds] = {
                k[len(prefix) :]: z[k].copy() for k in z.files if k.startswith(prefix)
            }
            if not inputs["mobility"][ds]:
                continue
            for key, expected in source["mobility"][ds]["normalized_arrays"].items():
                if (
                    key not in inputs["mobility"][ds]
                    or array_hash(inputs["mobility"][ds][key]) != expected
                ):
                    raise ValueError(
                        "Different local mobility input; run python src/data.py prepare: "
                        + ds
                        + "/"
                        + key
                    )
    return inputs


def presence(case, inputs):
    p = inputs["mobility"][case["dataset"]]
    seed, n = int(case["seed"]), int(case.get("n", 100))
    cohort = case.get("cohort_rule") == "training-top200-uniform100"
    pool = 200 if cohort else 100
    chosen = np.sort(rng(seed, 101).choice(pool, n, replace=False))
    remap = np.full(200, -1, dtype=np.int64)
    remap[chosen] = np.arange(n)
    worker = remap[p["event_worker"]]
    times, cells = p["event_time"], p["event_cell"]
    cutoff, end = int(p["cutoff"]), int(p["source_end"])
    days = p["source_days"]
    starts, stops = np.maximum(days, cutoff), np.minimum(days + 86400, end)
    keep = starts < stops
    starts, stops = starts[keep], stops[keep]
    prefix = np.r_[np.int64(0), np.cumsum(stops - starts)]
    use = (worker >= 0) & (times >= cutoff) & (times < end)
    t = times[use]
    interval = np.searchsorted(starts, t, side="right") - 1
    if not (np.all(interval >= 0) and np.all(t < stops[interval])):
        raise ValueError(
            "Mobility observations fall outside the source-supported clock."
        )
    assignment = (prefix[interval] + t - starts[interval]) * 50 // int(prefix[-1])
    loc = np.zeros((50, n, 100), bool)
    loc[assignment, worker[use], cells[use]] = True
    return loc


def population(n, H, C, seed):
    order, u = rng(seed, 43001).permutation(n), rng(seed, 43002).random(n)
    honest, cert = np.zeros(n, bool), np.zeros(n, bool)
    honest[order[: int(np.floor(n * H))]] = True
    cert[order[: int(np.floor(n * C))]] = True
    if not np.all(honest[cert]):
        raise ValueError("Certified workers must belong to the honest population.")
    q = np.where(honest, 0.8 + 0.2 * u, 0.3 * u)
    reputation = np.full(n, 0.5)
    # The certified subset is known honest at initialization; its initial
    # reputation is a separately sampled grade, not the latent quality q.
    certified_grade = rng(seed, 43003).uniform(0.95, 1.0, n)
    reputation[cert] = np.rint(certified_grade[cert] * 10000) / 10000
    return q, honest, cert, reputation


def route_cost(loc, random):
    out = np.zeros(loc.shape, float)
    for i in range(len(loc)):
        interval = random.choice(np.arange(1, 9), p=FREQUENCIES)
        total = random.uniform(10 * interval, 10 * (interval + 1))
        cells = np.flatnonzero(loc[i])
        if len(cells):
            out[i, cells] = np.clip(
                random.normal(total // len(cells), 1.5, len(cells)), 1.5, 5.0
            )
    return out


def neighbours(reference):
    corr = np.nan_to_num(np.corrcoef(reference))
    np.fill_diagonal(corr, 0)
    out = []
    for j in range(len(corr)):
        ids = np.argsort(-np.abs(corr[j]), kind="stable")[:2]
        out.append(ids[np.abs(corr[j, ids]) >= 0.93])
    return tuple(out)


def replace_locations(loc, nb, random):
    result = np.zeros_like(loc)
    for i, j in zip(*np.where(loc)):
        result[i, int(random.choice(nb[j])) if len(nb[j]) else j] = True
    return result


def reports(truth, q, loc, random):
    noise = random.normal(0, 0.5 * (1 - q[:, None]), loc.shape)
    corrupt = random.uniform(10.0, 40.0, loc.shape)
    value = np.where(q[:, None] >= 0.8, truth[None, :] + noise, corrupt)
    return np.where(loc, np.round(value, 2), np.nan)


def build_world(case, inputs, occupancy=None):
    seed, ds = int(case["seed"]), case["dataset"]
    field_name = case.get("field", "legacy")
    if field_name not in inputs["fields"]:
        raise FileNotFoundError(
            "Missing controlled_reference.npz for the original-paper cases; restore the supplied input or use --legacy."
        )
    field = inputs["fields"][field_name]
    g = field["rh"].astype(float)
    offset, scale = float(g[:, :30].min()), float(np.ptp(g[:, :30]))
    history, truth = (g[:, :30] - offset) / scale, (g[:, 30:80] - offset) / scale
    if occupancy is None:
        loc = presence(case, inputs)
    else:
        loc = np.asarray(occupancy)
        if loc.dtype != bool or loc.shape != (50, int(case.get("n", 100)), 100):
            raise ValueError("Expected Boolean 50-round grid occupancy")
    T, n, D = loc.shape
    q, honest, cert, r0 = population(
        n, float(case.get("H", 0.7)), float(case.get("C", 0.1)), seed
    )
    nb = neighbours(history)
    costs, coins, virtual, tasks, vtasks, values = [], [], [], [], [], []
    for t in range(T):
        costs.append(route_cost(loc[t], rng(seed, 53101, t)))
        coins.append(rng(seed, 53102, t).random((3, n, D)))
        virtual.append(replace_locations(loc[t], nb, rng(seed, 53201, t)))
        task = rng(seed, 53501, t).choice(D, 50, replace=False)
        tasks.append(task)
        random = rng(seed, 53301, t)
        mapping = np.array(
            [int(random.choice(v)) if len(v) else j for j, v in enumerate(nb)], int
        )
        vtasks.append(mapping[task])
        values.append(reports(10 + 30 * truth[:, t], q, loc[t], rng(seed, 53401, t)))
    tasks = np.asarray(tasks, np.int16)
    mask = np.zeros((T, D), bool)
    np.put_along_axis(mask, tasks, True, axis=1)
    adjacency = np.zeros((D, D), bool)
    for j, ids in enumerate(nb):
        adjacency[j, ids] = True
    cap = case.get("budget_override")
    return {
        "presence": loc,
        "offer_mask": loc.any(2),
        "actual_cost": np.asarray(costs),
        "RR_coins": np.asarray(coins),
        "virtual_presence": np.asarray(virtual),
        "requested_task_ids": tasks,
        "virtual_task_cells": np.asarray(vtasks, np.int16),
        "task_mask": mask,
        "reports_check": np.asarray(values),
        "public_reference": history,
        "public_priority": np.ones(D),
        "source_support_adjacency": adjacency,
        "initial_reputation": r0,
        "q_private": q,
        "honest_private": honest,
        "certified": cert,
        "truth_native_private": g[:, 30:80],
        "truth_normalized_private": truth,
        "native_offset": offset,
        "native_scale": scale,
        "budget": BUDGETS[ds] if cap is None else float(cap),
        "column_indices": np.arange(30, 80),
        "dates": field["dates"][30:80].copy(),
    }


def raw_lines(dataset, raw_dir):
    root = Path(raw_dir)
    if dataset == "brightkite":
        with gzip.open(root / "loc-brightkite_totalCheckins.txt.gz", "rb") as stream:
            yield from stream
    else:
        names = (
            ["dataset_tsmc2014.zip"]
            if dataset == "foursquare_tky"
            else ["0" + str(i) + ".zip" for i in range(6, 15)]
        )
        seen = set()
        for name in names:
            with zipfile.ZipFile(root / name) as archive:
                members = (
                    ["dataset_tsmc2014/dataset_TSMC2014_TKY.txt"]
                    if dataset == "foursquare_tky"
                    else [
                        p
                        for p in sorted(archive.namelist())
                        if p.lower().endswith(".txt") and Path(p).stem.isdigit()
                    ]
                )
                for member in members:
                    key = Path(member).name
                    if key in seen:
                        raise ValueError("Duplicate source file: " + member)
                    seen.add(key)
                    with archive.open(member) as stream:
                        yield from stream


def raw_events(dataset, raw_dir):
    la, lb, lo, lp = BOUNDS[dataset]
    for line in raw_lines(dataset, raw_dir):
        try:
            if dataset == "brightkite":
                a = line.split()
                user, lat, lon = a[0].decode("ascii"), float(a[2]), float(a[3])
                moment = datetime.fromisoformat(
                    a[1].decode("ascii").replace("Z", "+00:00")
                )
            elif dataset == "foursquare_tky":
                a = line.rstrip(b"\r\n").split(b"\t")
                user, lat, lon = a[0].decode("ascii"), float(a[4]), float(a[5])
                d = a[7].decode("ascii").split()
                hh, mm, ss = map(int, d[3].split(":"))
                if d[4] != "+0000":
                    continue
                moment = datetime(
                    int(d[5]), MONTHS[d[1]], int(d[2]), hh, mm, ss, tzinfo=timezone.utc
                )
            else:
                a = line.rstrip(b"\r\n").split(b",")
                user, lat, lon = a[0].decode("ascii"), float(a[3]), float(a[2])
                moment = datetime.fromisoformat(a[1].decode("ascii")).replace(
                    tzinfo=timezone(timedelta(hours=8))
                )
            if la <= lat <= lb and lo <= lon <= lp and 2000 <= moment.year <= 2020:
                yield user, int(moment.timestamp()), lat, lon
        except (ValueError, IndexError, UnicodeDecodeError, OverflowError):
            continue


def normalize(dataset, raw_dir):
    # Fit the cohort and spatial bounds using the first 30% of the temporal horizon.
    days, first, last = Counter(), None, None
    for _, sec, _, _ in raw_events(dataset, raw_dir):
        days[sec // 86400] += 1
        first, last = (
            sec if first is None else min(first, sec),
            sec if last is None else max(last, sec),
        )
    if first is None:
        raise ValueError("No valid source observations for " + dataset)
    start, end = min(days) * 86400, (max(days) + 1) * 86400
    cutoff = start + int(0.3 * (end - start))
    counts, box = Counter(), [90.0, -90.0, 180.0, -180.0]
    for user, sec, lat, lon in raw_events(dataset, raw_dir):
        if sec < cutoff:
            counts[user] += 1
            box = [
                min(box[0], lat),
                max(box[1], lat),
                min(box[2], lon),
                max(box[3], lon),
            ]
    users = sorted(counts, key=lambda u: (-counts[u], int(u)))[:200]
    lookup = {u: i for i, u in enumerate(users)}
    rows = []
    la, lb, lo, lp = box
    for user, sec, lat, lon in raw_events(dataset, raw_dir):
        if user in lookup and la <= lat <= lb and lo <= lon <= lp:
            cell = min(9, int((lat - la) / (lb - la) * 10)) * 10 + min(
                9, int((lon - lo) / (lp - lo) * 10)
            )
            rows.append((lookup[user], sec, cell))
    x = np.asarray(rows, np.int64)
    return {
        "user_ids": np.array(users),
        "training_counts": np.array([counts[u] for u in users], np.int32),
        "event_worker": x[:, 0].astype(np.int16),
        "event_time": x[:, 1],
        "event_cell": x[:, 2].astype(np.int16),
        "cutoff": np.array(cutoff, np.int32),
        "bbox": np.asarray(box),
        "source_days": np.array(sorted(days), np.int64) * 86400,
        "source_first": np.array(first, np.int32),
        "source_end": np.array(last + 1, np.int32),
    }


def prepare(raw_dir, data_dir=DEFAULT_DATA, legacy=None, datasets=DATASETS):
    root = Path(data_dir)
    local = root / "local"
    local.mkdir(parents=True, exist_ok=True)
    source = json.loads((root / "sources.json").read_text(encoding="utf-8"))
    target = local / "mobility.npz"
    payload = {}
    if target.exists():
        with np.load(target, allow_pickle=False) as z:
            payload = {k: z[k].copy() for k in z.files}
    for ds in datasets:
        for item in source["mobility"][ds]["archives"]:
            path = Path(raw_dir) / item["file"]
            if not path.exists() or file_hash(path) != item["sha256"]:
                raise ValueError(
                    "Missing or different official source archive: " + str(path)
                )
        p = normalize(ds, raw_dir)
        for key, value in p.items():
            expected = source["mobility"][ds]["normalized_arrays"].get(key)
            if expected is not None and array_hash(value) != expected:
                raise ValueError(
                    "Preprocessing output differs from the study input: "
                    + ds
                    + "/"
                    + key
                )
            payload[ds + "__" + key] = value
        print("Prepared real source events:", ds, flush=True)
    np.savez_compressed(target, **payload)
    if legacy is not None:
        with np.load(legacy, allow_pickle=False) as z:
            fields = {
                "rh": z["rh"].copy(),
                "dates": z["timestamps" if "timestamps" in z.files else "dates"].copy(),
            }
        for key, value in fields.items():
            if (
                array_hash(value)
                != source["legacy_controlled_field"]["array_sha256"][key]
            ):
                raise ValueError(
                    "The supplied controlled reference field differs: " + key
                )
        np.savez_compressed(local / "reference.npz", **fields)


def download(dataset, raw_dir, data_dir=DEFAULT_DATA):
    source = json.loads((Path(data_dir) / "sources.json").read_text(encoding="utf-8"))
    root = Path(raw_dir)
    root.mkdir(parents=True, exist_ok=True)
    for item in source["mobility"][dataset]["archives"]:
        target = root / item["file"]
        if not target.exists():
            urllib.request.urlretrieve(item["url"], target)
        if file_hash(target) != item["sha256"]:
            raise ValueError("Download differs from the study source: " + item["file"])


def reference(raw_dir, data_dir=DEFAULT_DATA):
    root = Path(data_dir)
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    source = json.loads((root / "sources.json").read_text(encoding="utf-8"))["NOAA"]
    payload = {}
    for name, url in source["queries"].items():
        path = raw / (name + ".csv")
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))[1:]
        dates = sorted({r["time"] for r in rows})
        lat = sorted({float(r["latitude"]) for r in rows}, reverse=True)
        lon = sorted({float(r["longitude"]) for r in rows})
        g = np.full((100, 84), np.nan)
        for r in rows:
            g[
                10 * lat.index(float(r["latitude"])) + lon.index(float(r["longitude"])),
                dates.index(r["time"]),
            ] = float(r["rhum"])
        values = {
            "rh": g,
            "dates": np.array(dates),
            "coords": np.array([(a, b) for a in lat for b in lon]),
        }
        for k, value in values.items():
            key = name + "__" + k
            if array_hash(value) != source["array_sha256"][key]:
                raise ValueError(
                    "Public reference data differ from the study source: " + key
                )
            payload[key] = value
    np.savez_compressed(root / "reference_fields.npz", **payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "download", "reference"))
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_DATA / "raw")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--legacy", type=Path)
    parser.add_argument(
        "--dataset", choices=DATASETS, nargs="+", default=list(DATASETS)
    )
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.raw_dir, args.data_dir, args.legacy, args.dataset)
    elif args.action == "reference":
        reference(args.raw_dir, args.data_dir)
    else:
        for name in args.dataset:
            download(name, args.raw_dir, args.data_dir)

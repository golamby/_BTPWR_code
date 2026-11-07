"""Beijing hourly field replay using real taxi visits near monitoring sites."""
from pathlib import Path
import sys, io, json, zipfile, hashlib, argparse
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from functools import lru_cache

CODE = Path(__file__).resolve().parents[1]
from . import data, recovery, privacy, trust, recruitment, baselines
from .metrics import requested_metrics

URL = "https://archive.ics.uci.edu/static/public/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip"
ARCHIVE_SHA256 = "b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4"


def download():
    from urllib.request import urlretrieve

    path = CODE / "data/raw/beijing_hourly.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(URL, str(path))
    if data.file_hash(path) != ARCHIVE_SHA256:
        raise ValueError("The UCI archive differs from the registered public source")
    return path


STATIONS = {
    "Aotizhongxin": (116.397, 39.982),
    "Changping": (116.230, 40.217),
    "Dingling": (116.220, 40.292),
    "Dongsi": (116.417, 39.929),
    "Guanyuan": (116.339, 39.929),
    "Gucheng": (116.184, 39.914),
    "Huairou": (116.628, 40.328),
    "Nongzhanguan": (116.461, 39.937),
    "Shunyi": (116.655, 40.127),
    "Tiantan": (116.407, 39.886),
    "Wanliu": (116.287, 39.987),
    "Wanshouxigong": (116.352, 39.878),
}
DATES = ("2014-01-01", "2014-04-01", "2014-07-01", "2014-10-01")


@lru_cache(maxsize=1)
def fields():
    archive = download()
    with zipfile.ZipFile(archive) as z:
        nested = zipfile.ZipFile(
            io.BytesIO(z.read("PRSA2017_Data_20130301-20170228.zip"))
        )
        series = {}
        for name in nested.namelist():
            if not name.endswith(".csv"):
                continue
            frame = pd.read_csv(nested.open(name))
            station = str(frame.station.iloc[0])
            if station in STATIONS:
                stamp = pd.to_datetime(frame[["year", "month", "day", "hour"]])
                series[station] = pd.Series(frame["PM2.5"].values, index=stamp)
    return pd.DataFrame(series)[list(STATIONS)]


def local_visits(raw_dir):
    cache = CODE / "data/local/hourly_taxi_visits.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = CODE / "data/raw" if raw_dir is None else Path(raw_dir)
    source = dict(np.load(CODE / "data/local/mobility.npz", allow_pickle=False))
    ids = source["tdrive__user_ids"]
    wanted = {str(x): i for i, x in enumerate(ids)}
    start = int(np.ceil(float(source["tdrive__cutoff"]) / 3600) * 3600)
    radius, hours = 2.0, 50
    xy = np.array(list(STATIONS.values()))
    archives = json.loads((CODE / "data/sources.json").read_text(encoding="utf-8"))["mobility"]["tdrive"]["archives"]
    specification = dict(
        version="hourly-nearest-station-v1",
        worker_ids=[str(x) for x in ids],
        cutoff_unix=float(source["tdrive__cutoff"]),
        start_unix=start,
        end_unix_exclusive=start + hours * 3600,
        hour_seconds=3600,
        radius_km=radius,
        stations=[(name, list(position)) for name, position in STATIONS.items()],
        archives={item["file"]: item["sha256"] for item in archives},
    )
    encoded = json.dumps(specification, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def content_hash(values):
        digest = hashlib.sha256()
        for name in ("presence", "point_count", "station_points", "worker_ids", "start_unix"):
            value = np.ascontiguousarray(values[name])
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()

    complete_sources = True
    for item in archives:
        path = raw_dir / item["file"]
        if not path.exists():
            complete_sources = False
        elif data.file_hash(path) != item["sha256"]:
            raise ValueError("T-Drive source hash mismatch: " + str(path))
    reason = "absent"
    if cache.exists():
        saved = dict(np.load(cache, allow_pickle=False))
        if str(saved.get("cache_key", "")) == key:
            if str(saved.get("content_sha256", "")) != content_hash(saved):
                raise ValueError("Hourly visit cache content checksum mismatch; remove it and rebuild from --raw-dir")
            return saved
        reason = "missing provenance or a changed worker/window/station/source configuration"
    if not complete_sources:
        raise FileNotFoundError(
            "Hourly visit cache is " + reason + ". Rebuild with --raw-dir pointing to the nine registered T-Drive archives."
        )
    loc = np.zeros((hours, len(ids), len(xy)), bool)
    point_count = np.zeros((hours, len(ids)), int)
    nearest_count = np.zeros(len(xy), int)
    for line in data.raw_lines("tdrive", raw_dir):
        try:
            a = line.decode().strip().split(",")
            i = wanted.get(a[0])
            if i is None:
                continue
            second = int(
                datetime.fromisoformat(a[1])
                .replace(tzinfo=timezone(timedelta(hours=8)))
                .timestamp()
            )
            t = (second - start) // 3600
            if not 0 <= t < hours:
                continue
            lon, lat = float(a[2]), float(a[3])
            distance = 111.2 * np.sqrt(
                ((xy[:, 0] - lon) * np.cos(np.deg2rad(lat))) ** 2
                + (xy[:, 1] - lat) ** 2
            )
            j = int(np.argmin(distance))
            point_count[t, i] += 1
            if distance[j] <= radius:
                loc[t, i, j] = True
                nearest_count[j] += 1
        except (ValueError, IndexError, UnicodeDecodeError):
            continue
    values = dict(
        presence=loc,
        point_count=point_count,
        station_points=nearest_count,
        worker_ids=ids,
        start_unix=np.array(start),
    )
    values.update(cache_key=np.array(key), cache_specification=np.array(encoded),
                  content_sha256=np.array(content_hash(values)))
    temporary = cache.with_suffix(".pending.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(cache)
    return dict(np.load(cache, allow_pickle=False))


def build(date, seed, budget=90.0, raw_dir=None):
    frame = fields()
    first = pd.Timestamp(date)
    block = frame.loc[first : first + pd.Timedelta(hours=79)].to_numpy().T
    assert block.shape == (12, 80)
    history = pd.DataFrame(block[:, :30].T).ffill().bfill().to_numpy().T
    if not np.isfinite(history).all():
        raise ValueError("A station has no historical reference in the fixed window")
    valid = np.isfinite(block[:, 30:]) & (block[:, 30:] >= 0) & (block[:, 30:] <= 1000)
    g = np.where(valid, block[:, 30:] / 10.0, 0.0)
    H = history / 1000.0
    v = local_visits(raw_dir)
    chosen = np.sort(
        data.rng(seed, 101).choice(len(v["worker_ids"]), 100, replace=False)
    )
    loc = v["presence"][:, chosen].copy() & valid.T[:, None, :]
    n, D = loc.shape[1:]
    q, honest, cert, r0 = data.population(n, 0.7, 0.1, seed)
    nb = data.neighbours(H)
    costs = []
    virtual = []
    tasks = []
    vt = []
    reports = []
    coins = []
    for t in range(50):
        costs.append(data.route_cost(loc[t], data.rng(seed, 53101, t)))
        coins.append(data.rng(seed, 53102, t).random((3, n, D)))
        virtual.append(data.replace_locations(loc[t], nb, data.rng(seed, 53201, t)))
        available = np.flatnonzero(valid[:, t])
        if len(available) < 6:
            raise ValueError(
                "Fewer than six observed target stations in the predefined window"
            )
        task = data.rng(seed, 53501, t).choice(available, 6, replace=False)
        tasks.append(task)
        random = data.rng(seed, 53301, t)
        mapping = np.array(
            [int(random.choice(a)) if len(a) else j for j, a in enumerate(nb)]
        )
        vt.append(mapping[task])
        reports.append(
            data.reports(10 + 30 * g[:, t] / 100, q, loc[t], data.rng(seed, 53401, t))
        )
    mask = np.zeros((50, D), bool)
    np.put_along_axis(mask, np.array(tasks), True, axis=1)
    w = dict(
        presence=loc,
        offer_mask=loc.any(2),
        actual_cost=np.asarray(costs),
        RR_coins=np.asarray(coins),
        virtual_presence=np.asarray(virtual),
        requested_task_ids=np.asarray(tasks),
        virtual_task_cells=np.asarray(vt),
        task_mask=mask,
        reports_check=np.asarray(reports),
        public_reference=H,
        initial_reputation=r0,
        q_private=q,
        honest_private=honest,
        certified=cert,
        truth_native_private=g,
        truth_valid_private=valid,
        native_offset=np.array(0.0),
        native_scale=np.array(100.0),
        budget=np.array(budget),
        dates=frame.loc[first : first + pd.Timedelta(hours=79)]
        .index.astype(str)
        .values.astype(str),
        field_unit=np.array("10 microgram/m3"),
        worker_ids_private=v["worker_ids"][chosen],
    )
    return w


def calibrate(H, offset, scale):
    D = H.shape[0]
    ls = recovery.linear_initialize(H[:, :24], offset, scale)
    ratios = []
    for k in sorted(set(min(k, D - 1) for k in [4, 8, 12, 16, 20])):
        for t in range(6):
            for rep in range(20):
                a = np.sort(
                    data.rng(7701, k * 100 + t, rep).choice(D, k, replace=False)
                )
                z = H[a, 24 + t] + data.rng(7702, k * 100 + t, rep).normal(
                    0, 0.1 / 30, k
                )
                cred = np.full(D, np.nan)
                cred[a] = 10 + 30 * z
                act = recovery.decode_actual(cred, ls)
                _, diag = recovery.linear_complete(cred, act, ls)
                use = diag["recovery_mask"] & ~diag["credible_mask"]
                fit = recovery.conditional(H[:, :24], a, ls["recovery"])
                center = np.clip(
                    offset + scale * recovery.nonlinear_predict(fit, z), 0, 100
                )
                ratios.extend(
                    (
                        abs(offset + scale * H[use, 24 + t] - center[use])
                        / diag["working_sigma_native"][use]
                    ).tolist()
                )
    order = np.sort(ratios)
    if not len(order):
        raise ValueError(
            "No held-out supported targets; cannot calibrate a recovery band"
        )
    rank = min(len(order), int(np.ceil(0.95 * (len(order) + 1))))
    return dict(
        kappa=float(order[rank - 1]),
        public_sha256=hashlib.sha256(H.tobytes()).hexdigest(),
        residual_count=len(order),
        rank=rank,
    )


def run(w, seed, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    D = w["presence"].shape[-1]
    n = w["presence"].shape[1]
    cfg = privacy.Config(n=n, m=D, T=50)
    tc = trust.Config()
    cal = calibrate(w["public_reference"], 0, 100)
    names = ["Frozen BTPWR", "Shared ETBP priority", "Binary mapping"]
    preds = {name: np.full((50, D), np.nan) for name in names}
    reps = {name: w["initial_reputation"].copy() for name in names}
    heads = {
        name: recovery.initialize(w["public_reference"], 0, 100, cal) for name in names
    }
    priority, pcfg = baselines.priority_state(reps[names[1]])
    rows = []
    for t in range(50):
        loc = w["presence"][t]
        cost = w["actual_cost"][t]
        offers = w["offer_mask"][t]
        avg = np.divide(
            cost.sum(1), loc.sum(1), out=np.full(n, 3.25), where=loc.sum(1) > 0
        )
        packet = privacy.release(
            w["virtual_presence"][t],
            np.where(w["virtual_presence"][t], avg[:, None], 0),
            w["RR_coins"][t],
            cfg.eps1,
            cfg.eps2,
            cfg,
        )
        for key in packet:
            packet[key][~offers] = False
        for name in names:
            r = reps[name]
            if name == "Shared ETBP priority":
                priority.trust = r.copy()
                order = baselines.etbp.recruit_offers(
                    packet["bits"][:, w["virtual_task_cells"][t]],
                    min(40, int(offers.sum())),
                    priority,
                    data.rng(seed, 78102, t),
                    pcfg,
                    offers,
                )["selected"]
                ids, num, den = recruitment.prefix_budget(
                    order, packet, cfg, float(w["budget"])
                )
            else:
                ans = recruitment.greedy(
                    packet,
                    offers,
                    r,
                    w["virtual_task_cells"][t],
                    cfg,
                    float(w["budget"]),
                    tc,
                    binary=name == "Binary mapping",
                )
                ids = ans["members"]
                num, den = ans["cost_numerator"], ans["cost_denominator"]
            raw = np.full((n, D), np.nan)
            raw[ids] = w["reports_check"][t, ids]
            raw[trust.classify(r, tc) == 0] = np.nan
            fb = trust.feedback(raw, r, tc)
            actual = recovery.decode_actual(fb["consensus"], heads[name])
            preds[name][t] = recovery.complete(
                fb["credible_consensus"], actual, heads[name]
            )[0]
            reps[name] = fb["next_reputation"]
            if name == "Shared ETBP priority":
                priority.evaluation_count += fb["scored_count"]
                priority.recruited_count[ids] += 1
                priority.history.append(reps[name].copy())
                priority.period += 1
            rows.append(
                dict(
                    method=name,
                    round=t,
                    payment=num / den,
                    route_cost=float(cost[ids].sum()),
                    C_anchors=int(fb["has_credible"].sum()),
                    reference_reports=int(fb["scored_count"].sum()),
                    selected=len(ids),
                )
            )
    history = w["public_reference"] * 100
    preds["History mean"] = np.repeat(history.mean(1)[None, :], 50, axis=0)
    preds["Last historical reading"] = np.repeat(history[:, -1][None, :], 50, axis=0)
    preds["Historical 24h repeat"] = np.array(
        [history[:, 6 + t % 24] for t in range(50)]
    )
    X = history[:, :-1]
    Y = history[:, 1:]
    slope = np.sum((X - X.mean(1)[:, None]) * (Y - Y.mean(1)[:, None]), 1) / (
        np.sum((X - X.mean(1)[:, None]) ** 2, 1) + 1e-8
    )
    slope = np.clip(slope, -1, 1)
    intercept = Y.mean(1) - slope * X.mean(1)
    last = history[:, -1].copy()
    ar = []
    for t in range(50):
        last = np.clip(intercept + slope * last, 0, 100)
        ar.append(last.copy())
    preds["Historical AR(1)"] = np.array(ar)
    summary = []
    for name, pred in preds.items():
        m = requested_metrics(
            w["truth_native_private"].T * 10, pred * 10, w["task_mask"]
        )
        summary.append(
            dict(
                method=name,
                seed=seed,
                **{k: v for k, v in m.items() if np.isscalar(v) or v is None}
            )
        )
    np.savez_compressed(
        outdir / "predictions.npz",
        methods=np.array(list(preds)),
        prediction=np.array(list(preds.values())) * 10,
        truth=w["truth_native_private"].T * 10,
        mask=w["task_mask"],
    )
    pd.DataFrame(summary).to_csv(outdir / "summary.csv", index=False)
    pd.DataFrame(rows).to_csv(outdir / "resources.csv", index=False)
    return summary

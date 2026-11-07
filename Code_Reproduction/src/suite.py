"""Registered experiment configurations and complete numerical execution."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from .simulate import run_case

ROOT = Path(__file__).resolve().parents[1]


def load_cases(suite="all", config=None):
    config = ROOT / "experiments.json" if config is None else Path(config)
    cases = json.loads(config.read_text(encoding="utf-8"))["cases"]
    if suite == "all":
        return cases
    if suite == "main":
        return [c for c in cases if c["kind"] == "confirmation"]
    if suite == "default":
        return [c for c in cases if c["kind"] == "default"]
    if suite == "registered":
        return [c for c in cases if c["group"] == "registered"]
    if suite == "sensitivity":
        return [c for c in cases if c["kind"] in ("default", "axis", "epsilon1")]
    if suite == "smoke":
        return [
            next(c for c in cases if c["name"] == "confirmation_brightkite_seed602")
        ]
    if suite == "confirmation":
        return [c for c in cases if c["group"] in ("fresh_v3", "cohort_v3")]
    if suite == "fresh":
        return [c for c in cases if c["group"] == "fresh_v3"]
    if suite == "cohort":
        return [c for c in cases if c["group"] == "cohort_v3"]
    if suite == "historical":
        return [c for c in cases if c["group"] == "fresh"]
    if suite == "privacy":
        return [
            c for c in cases if c["kind"] == "confirmation" and c["dataset"] == "tdrive"
        ]
    if suite == "noaa":
        return [
            c
            for c in cases
            if c["group"] in ("fresh_v3", "cohort_v3") and c["field"] != "legacy"
        ]
    match = [c for c in cases if c["name"] == suite]
    if match:
        return match
    raise ValueError("Unknown suite or case name: " + suite)


def select_cases(name, case_filter=None):
    cases = load_cases(name)
    if case_filter:
        filters = [case_filter] if isinstance(case_filter, str) else list(case_filter)
        cases = [c for c in cases if any(f in c["name"] for f in filters)]
    if not cases:
        raise ValueError("No cases matched the requested filter")
    return cases


def _execute(job):
    case, inputs, outputdir, kwargs = job
    tables = run_case(case, inputs, outputdir, **kwargs)
    return {"case": case["name"], "methods": len(tables["summary"]), "completed": True}


def run_suite(cases, inputs, outputdir, workers=2, **kwargs):
    cases = list(cases)
    jobs = [(c, inputs, str(outputdir), kwargs) for c in cases]
    records = []
    if workers == 1:
        for job in jobs:
            rec = _execute(job)
            records.append(rec)
            print(json.dumps(rec), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            pending = [pool.submit(_execute, j) for j in jobs]
            for future in as_completed(pending):
                rec = future.result()
                records.append(rec)
                print(json.dumps(rec), flush=True)
    return records

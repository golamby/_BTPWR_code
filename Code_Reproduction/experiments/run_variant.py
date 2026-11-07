"""Run one mechanism variant on selected main worlds.

Usage (from 03_Code):
  python -m experiments.run_variant --variant reference --beta 0.25
  python -m experiments.run_variant --variant hitreward --hit-weight 1.0
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data  # noqa: E402
from src import admission_experiments as ae  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["coupled", "gated", "oracle", "reference", "hitreward", "fallback"])
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--hit-weight", type=float, default=1.0)
    ap.add_argument("--uth", type=float, default=0.35)
    ap.add_argument("--datasets", nargs="+", default=["tdrive", "brightkite"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[9701])
    ap.add_argument("--factors", nargs="+", type=float, default=[1.0])
    ap.add_argument("--out", default=str(ROOT / "results" / "variant_runs"))
    args = ap.parse_args()

    if args.variant == "coupled":
        from src import coupled_experiments as mod
        mod.COUPLED = True
    elif args.variant == "gated":
        from src import coupled_gated_experiments as mod
        mod.COUPLED = True
        mod.MIN_ANCHORS = 8
    elif args.variant == "oracle":
        from src import coupled_oracle_experiments as mod
        mod.COUPLED = True
    elif args.variant == "reference":
        from src import reference_experiments as mod
        mod.BETA = args.beta
    elif args.variant == "hitreward":
        from src import hitreward_admission
        hitreward_admission.HIT_WEIGHT = args.hit_weight
        ae.AdmissionPlanner = hitreward_admission.AdmissionPlanner
        mod = ae
    else:
        from src import fallback_experiments as mod
        mod.UTH = args.uth
    mod.METHODS = ("admission",)
    inputs = data.load_inputs()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        for seed in args.seeds:
            for factor in args.factors:
                case = dict(name=f"{args.variant}_{ds}_s{seed}_f{factor}", group="main", kind="main",
                            stage="main", dataset=ds, field="legacy", seed=int(seed), n=100,
                            H=0.7, C=0.1, gamma=0.6, eps1=2.1, eps2=2.7,
                            budget_override=data.BUDGETS[ds] * factor, budget_factor=factor,
                            cohort_rule="training-top200-uniform100")
                mod.run_case(case, data.build_world(case, inputs), out)
                print("done", case["name"], flush=True)


if __name__ == "__main__":
    main()

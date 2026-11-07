"""Bounded evidence-count updates with direct and credible-recovered references."""
import os
from pathlib import Path
from multiprocessing import Pool
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from src import data
from src.fast_qualification import DynamicBalanced
from src.admission_experiments import run_case
from experiments.sweep_cases import cases
ROOT=Path(__file__).resolve().parents[1];INPUTS=None
def init():
    global INPUTS
    INPUTS=data.load_inputs()
def one(c):
    methods=('admission-count-direct','admission-count-recovery','admission-count-recovery-full')
    return run_case(c,data.build_world(c,INPUTS),ROOT/'results'/'count_updates',methods=methods,
        planner_types={m:DynamicBalanced for m in methods},reputation_modes={
            'admission-count-direct':'count-direct','admission-count-recovery':'count-recovery',
            'admission-count-recovery-full':'count-recovery-full'})
def main():
    jobs=[c for c in cases(default_budget='450') if c['n']==100 and c['H']==.7 and c['C']==.1 and c['budget_override']==450 and c['eps1']==2.1]
    assert len(jobs)==9
    with Pool(9,initializer=init) as pool:list(pool.imap_unordered(one,jobs))
if __name__=='__main__':main()

"""Eight held-out seeds for credible-anchor secondary references."""
import os
from pathlib import Path
from multiprocessing import Pool
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from src import data
from src.fast_qualification import DynamicBalanced
from src.admission_experiments import run_case
from experiments.primary_450_cases import cases
ROOT=Path(__file__).resolve().parents[1];INPUTS=None
def init():
    global INPUTS
    INPUTS=data.load_inputs()
def one(c):
    methods=('admission-separate','admission-recovered','admission-recovered-025')
    return run_case(c,data.build_world(c,INPUTS),ROOT/'results'/'recovery_confirm',methods=methods,
        planner_types={m:DynamicBalanced for m in methods},
        reputation_modes={'admission-recovered':'recovery-ref','admission-recovered-025':'recovery-ref-025'})
def main():
    with Pool(16,initializer=init) as pool:list(pool.imap_unordered(one,cases()))
if __name__=='__main__':main()

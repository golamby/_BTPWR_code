"""Compare two simple opportunity selectors on held-out seed/world pairs."""
import os
from multiprocessing import Pool
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from src import data
from src.fast_qualification import FastQualification,WeightedOpportunity,DynamicBalanced
from src.admission_experiments import run_case
from experiments.primary_450_cases import cases
ROOT=Path(__file__).resolve().parents[1]
INPUTS=None
def init():
    global INPUTS
    INPUTS=data.load_inputs()
def one(c):
    return run_case(c,data.build_world(c,INPUTS),ROOT/'results'/'fast_confirm',
                    methods=('admission-fast','admission-weighted-opportunity','admission-dynamic'),
                    planner_types={'admission-fast':FastQualification,'admission-weighted-opportunity':WeightedOpportunity,
                                   'admission-dynamic':DynamicBalanced})
def main():
    with Pool(16,initializer=init) as pool:list(pool.imap_unordered(one,cases()))
if __name__=='__main__':main()

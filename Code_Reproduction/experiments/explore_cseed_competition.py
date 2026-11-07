"""Explore two coverage weights on the registered nine development worlds."""
import argparse,os
from multiprocessing import Pool
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from src import data
from src.admission import AdmissionPlanner as Original
from src.current_coverage_admission import direct_coverage
from src.admission_experiments import run_case
from experiments.sweep_cases import cases
ROOT=Path(__file__).resolve().parents[1]
INPUTS=None
class CoverageOne(Original):
    weight=1.
    def value(self,members,backend='numba'):
        ans=dict(super().value(members,backend=backend))
        ans['value']+=self.weight*direct_coverage(self.state.support,self.state.reputation,self.state.requested_tasks,members)
        return ans
class CoverageTwo(CoverageOne):
    weight=2.
def init():
    global INPUTS
    INPUTS=data.load_inputs()
def one(case):
    return run_case(case,data.build_world(case,INPUTS),ROOT/'results'/'cseed_exploration',
        methods=('admission-cover-1','admission-cover-2'),
        planner_types={'admission-cover-1':CoverageOne,'admission-cover-2':CoverageTwo})
def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=9);a=p.parse_args()
    jobs=[c for c in cases(default_budget='450') if c['n']==100 and c['H']==.7 and c['C']==.1 and c['budget_override']==450 and c['eps1']==2.1]
    assert len(jobs)==9
    with Pool(a.workers,initializer=init) as pool:list(pool.imap_unordered(one,jobs))
if __name__=='__main__':main()

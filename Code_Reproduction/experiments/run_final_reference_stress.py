"""Final-planner reference stress with fixed paired worlds and explicit deviations."""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path
import numpy as np
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
from src import data
from src.admission_experiments import run_case
from experiments.sweep_cases import cases
ROOT=Path(__file__).resolve().parents[1]
INPUTS=None

def init():
    global INPUTS
    INPUTS=data.load_inputs()

def one(job):
    scenario,base=job
    world=data.build_world(base,INPUTS)
    cert=np.flatnonzero(world['certified'])
    if len(cert)!=10: raise ValueError('Expected ten initially certified workers')
    if scenario in ('sole-correct','sole-biased'):
        world['initial_reputation'][cert[1:]]=.5
        world['certified'][cert[1:]]=False
    if scenario in ('sole-biased','two-biased','onoff-biased'):
        chosen=cert[:1] if scenario=='sole-biased' else cert[:2]
        rounds=np.r_[20:25,30:35] if scenario=='onoff-biased' else np.arange(50)
        for t in rounds:
            reports=world['reports_check'][t,chosen,:]
            world['reports_check'][t,chosen,:]=np.clip(reports+9.,10.,40.)
    case=base.copy();case['name']=f"stress_{scenario}_{base['dataset']}_{base['seed']}";case['stress']=scenario
    return run_case(case,world,ROOT/'results'/'final_reference_stress',methods=('admission',))

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=12);args=p.parse_args()
    defaults=[c for c in cases(default_budget='450') if c['n']==100 and c['H']==.7
              and c['C']==.1 and c['budget_override']==450 and c['eps1']==2.1]
    if len(defaults)!=9:raise AssertionError(f'Expected 9 default worlds, got {len(defaults)}')
    jobs=[(s,c) for s in ('sole-correct','sole-biased','two-biased','onoff-biased') for c in defaults]
    print('jobs',len(jobs),flush=True)
    with Pool(args.workers,initializer=init) as pool:list(pool.imap_unordered(one,jobs))

if __name__=='__main__':main()

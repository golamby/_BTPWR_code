"""Audit secondary-reference accuracy against held-out simulation truth."""
import os,json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
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
    w=data.build_world(c,INPUTS);audit=[]
    def observer(**event):
        fb=event['feedback'];mask=fb.get('recovery_reference_mask')
        if mask is None:return
        t=event['index'];truth_check=10+30*(w['truth_native_private'][:,t]-w['native_offset'])/w['native_scale']
        err=np.abs(fb['reference_by_worker']-truth_check[None,:])[mask]
        audit.append(dict(method=event['method'],round=t+1,count=int(len(err)),
                          abs_error_sum=float(err.sum()),sq_error_sum=float((err*err).sum()),
                          within_3=int((err<=3).sum())))
    methods=('admission-recovered','admission-recovered-025')
    out=ROOT/'results'/'secondary_reference_audit'
    run_case(c,w,out,methods=methods,planner_types={m:DynamicBalanced for m in methods},
        reputation_modes={'admission-recovered':'recovery-ref','admission-recovered-025':'recovery-ref-025'},round_observer=observer)
    (out/(c['name']+'.audit.json')).write_text(json.dumps(audit,indent=2))
def main():
    jobs=[c for c in cases(default_budget='450') if c['n']==100 and c['H']==.7 and c['C']==.1 and c['budget_override']==450 and c['eps1']==2.1]
    assert len(jobs)==9
    with Pool(9,initializer=init) as pool:list(pool.imap_unordered(one,jobs))
if __name__=='__main__':main()

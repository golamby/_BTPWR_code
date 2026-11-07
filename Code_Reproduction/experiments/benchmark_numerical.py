"""Serial numerical wall-time measurement on paired default worlds."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
from src import data
from src.admission_experiments import run_case
from experiments.sweep_cases import cases
from experiments import run_bpqa_native
ROOT=Path(__file__).resolve().parents[1]

def main():
    inputs=data.load_inputs()
    run_bpqa_native.init((),(),'sum')
    defaults=[c for c in cases(default_budget='450') if c['n']==100 and c['H']==.7
              and c['C']==.1 and c['budget_override']==450 and c['eps1']==2.1]
    if len(defaults)!=9:raise AssertionError('Expected nine default paired worlds')
    for case in defaults:
        run_case(case,data.build_world(case,inputs),ROOT/'results'/'serial_timing',
                 methods=('admission','admission-no-promotion','template-first','old','etbp','random','greedy-cover'))
        run_bpqa_native.one(case)
    print('SERIAL_BENCHMARK_COMPLETE',flush=True)

if __name__=='__main__':main()

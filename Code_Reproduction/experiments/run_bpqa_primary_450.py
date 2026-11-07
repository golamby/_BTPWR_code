import argparse
import os
from multiprocessing import Pool

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from experiments.primary_450_cases import cases
from experiments.run_bpqa_native import init, one


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=12)
    args=p.parse_args()
    jobs=cases()
    print('jobs',len(jobs),flush=True)
    with Pool(args.workers,initializer=init,initargs=((),(), 'sum')) as pool:
        list(pool.imap_unordered(one,jobs))


if __name__=='__main__':
    main()

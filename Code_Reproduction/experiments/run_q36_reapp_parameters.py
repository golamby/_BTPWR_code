"""Prespecified REAPP recruitment-convention sensitivity; no primary changes."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
from dataclasses import replace,asdict
from multiprocessing import Pool
import numpy as np
SOURCE=Path(os.environ.get('BTPWR_Q36_SOURCE',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(SOURCE))
from experiments import run_formal_q36 as formal
from experiments.formal_q36_cases import base,finish
from src.reapp_native import Config,campaign
from src.simulate import atomic_json,world_digest

OUT=None;PRIMARY=None
def init(out,primary):
    global OUT,PRIMARY
    OUT=Path(out);PRIMARY=Path(primary);formal.init(out)

def hashes():
    return dict(formal.hashes(),parameter_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

def one(job):
    c,phi,quota=job;method='REAPP-phi%d-U%d'%(phi,quota);dest=OUT/'cases'/c['name']/method
    dest.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    try:
        w=formal.get_world(c);wid=world_digest(w);source=hashes()
        if (dest/'result.json').exists():
            m=json.loads((dest/'result.json').read_text())
            if m['source_sha256']!=source or m['world_sha256']!=wid or m['output_sha256']!=hashlib.sha256((dest/'result.npz').read_bytes()).hexdigest():raise AssertionError('Invalid resume')
            return dict(case=c['name'],method=method,status='verified-resume')
        atomic_json(dest/'progress.json',dict(status='running',pid=os.getpid()))
        cfg=replace(Config(),phi=phi,unknown_per_region=quota)
        reused=phi==3 and quota==3
        if reused:
            name=c['name'].replace('q36_primary_','q36_sensitivity_')
            source_folder=PRIMARY/'cases'/name/'REAPP';meta=json.loads((source_folder/'result.json').read_text())
            assert meta['world_sha256']==wid and meta['source_sha256']==formal.hashes()
            assert meta['config']['phi']==3 and meta['config']['unknown_per_region']==3
            f=source_folder/'result.npz';assert hashlib.sha256(f.read_bytes()).hexdigest()==meta['output_sha256']
            with np.load(f) as z:arrays={k:z[k].copy() for k in z.files}
            records=meta['records'];thresholds=meta['class_thresholds']
        else:
            result=campaign(w,c,50,cfg)
            arrays={k:result[k] for k in ('predictions','selected','reputations','resources')}
            arrays.update(truth=w['truth_native_private'].T,task_mask=w['task_mask'],honest=w['honest_private'],certified=w['certified'])
            records=result['records'];thresholds=result['class_thresholds']
        if arrays['predictions'].shape!=(50,100) or arrays['reputations'].shape!=(51,80):raise AssertionError('Incomplete result')
        np.savez_compressed(dest/'result.npz',**arrays)
        atomic_json(dest/'result.json',dict(case=c,method=method,status='completed',rounds=50,source_sha256=source,
            world_sha256=wid,output_sha256=hashlib.sha256((dest/'result.npz').read_bytes()).hexdigest(),
            config=asdict(cfg),records=records,class_thresholds=thresholds,resource_columns=formal.RESOURCE_COLUMNS,
            reused_existing_primary_convention=reused,seconds=time.perf_counter()-start,
            interpretation='REAPP source-unspecified recruitment conventions; native unit-worker cost, no monetary cap; all four settings retained',
            original_equations='Eq10/11 symbolic phi; no numeric phi or unknown recruitment quota given; SGD and feedback defaults unchanged'))
        atomic_json(dest/'progress.json',dict(status='completed'))
        return dict(case=c['name'],method=method,status='completed',reused=reused,seconds=time.perf_counter()-start)
    except Exception:
        atomic_json(dest/'progress.json',dict(status='failed',traceback=traceback.format_exc()));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--primary',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    jobs=[(finish(base(d,s)),phi,quota) for d in formal.data.DATASETS for s in range(9701,9704) for phi in [1,3] for quota in [1,3]]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'jobs_reapp_parameters.json',dict(jobs=[dict(case=c['name'],method='REAPP-phi%d-U%d'%(ph,u)) for c,ph,u in jobs],source_sha256=hashes(),cases=[c for c,ph,u in jobs if ph==1 and u==1]))
    print(json.dumps(dict(configurations=36,new_campaigns=27,reused=9,workers=a.workers)),flush=True)
    with Pool(a.workers,initializer=init,initargs=(a.output,a.primary)) as pool:
        for i,r in enumerate(pool.imap_unordered(one,jobs)):print(json.dumps(dict(done=i+1,total=36,**r)),flush=True)
    atomic_json(out/'done_reapp_parameters.json',dict(status='completed',configurations=36,new_campaigns=27,reused=9))

if __name__=='__main__':
    import signal
    out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
    if hasattr(signal,'SIGALRM'):
        def timeout(*args):raise TimeoutError('REAPP sensitivity hard timeout2700seconds')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(2700)
    try:
        main();atomic_json(out/'exit.json',dict(status='completed',exit_code=0))
    except BaseException:
        atomic_json(out/'exit.json',dict(status='failed',exit_code=1,traceback=traceback.format_exc()));raise

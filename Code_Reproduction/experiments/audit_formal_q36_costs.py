"""Rebuild frozen worlds and audit selected-route costs without policy reruns.

Run next to the frozen source with the source data available. The completed
Q36 audit used the same operations inline before idle protocol measurements.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import data
from src.simulate import world_digest
from experiments.run_formal_q36 import perturb


def audit(result_root,output):
    started=time.time();inputs=data.load_inputs();campaigns=worlds=0;maxphysical=maxbids=0.
    for folder in sorted((Path(result_root)/'cases').iterdir()):
        paths=sorted(folder.glob('*/result.json'))
        if not paths:continue
        case=json.loads(paths[0].read_text(encoding='utf-8'))['case']
        world=perturb(data.build_world(case,inputs),case);worldhash=world_digest(world);worlds+=1
        for p in paths:
            meta=json.loads(p.read_text(encoding='utf-8'))
            if meta['world_sha256']!=worldhash:raise ValueError('World hash mismatch '+case['name'])
            npz=p.with_name('result.npz')
            if hashlib.sha256(npz.read_bytes()).hexdigest()!=meta['output_sha256']:raise ValueError('Output hash mismatch')
            with np.load(npz,allow_pickle=False) as z:
                selected=z['selected'];resources=z['resources']
                if np.any(selected&~world['offer_mask']):raise ValueError('Unavailable worker selected')
                physical=np.array([world['actual_cost'][t,selected[t]].sum() for t in range(50)])
                bids=np.sum(world['actual_cost'].sum(2)*selected,axis=1)
                d=float(np.max(np.abs(physical-resources[:,1])))
                e=float(np.max(np.abs(bids-resources[:,3])))
                if d>=1e-8 or e>=1e-8:raise ValueError((case['name'],meta['method'],d,e))
                maxphysical=max(maxphysical,d);maxbids=max(maxbids,e);campaigns+=1
    receipt=dict(status='passed',worlds=worlds,campaigns=campaigns,round_resource_rows=campaigns*50,
                 all_frozen_world_hashes_match=True,all_output_hashes_match=True,all_selections_available=True,
                 max_physical_cost_abs_difference=maxphysical,max_actual_bid_sum_abs_difference=maxbids,
                 tolerance=1e-8,seconds=time.time()-started,
                 scope='Independent rebuild of frozen worlds and selected-route costs; no policy execution or private input export')
    Path(output).write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    audit(a.root,a.output)

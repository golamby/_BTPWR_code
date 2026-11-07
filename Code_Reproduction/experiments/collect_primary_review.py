"""Verify shared worlds and compute seed-paired primary results."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from experiments.primary_450_cases import cases
from experiments.collect_review_sweeps import quality, LABELS
from src import data

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();base=Path(a.results);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for case in cases():
        name=case['name']; path=base/'primary_450'/(name+'.npz'); bp=base/'bpqa_native'/(name+'.npz')
        meta=json.loads(path.with_suffix('.json').read_text()); bm=json.loads(bp.with_suffix('.json').read_text())
        if meta['case']!=case or bm['case']!=case or meta['world_sha256']!=bm['world_sha256']:
            raise ValueError('World mismatch '+name)
        if bm['error_aggregation']!='sum': raise ValueError('Unexpected BPQA aggregation')
        with np.load(path) as z, np.load(bp) as b:
            truth=z['truth'];mask=z['task_mask'].astype(bool);ours=z['predictions'][0]
            np.testing.assert_array_equal(truth,b['truth']);np.testing.assert_array_equal(mask,b['task_mask'])
            q,honest,_,_=data.population(case['n'],case['H'],case['C'],case['seed'])
            candidates=[(m,z['predictions'][i],z[m+'__stats'],z['reputations'][i],None)
                        for i,m in enumerate(z['methods'].tolist())]
            candidates.append(('bpqa',b['predictions'],b['resources'],b['reputations'],b['resources'][:,4]))
            trad=base/'traditional_primary'/(name+'.npz')
            if trad.exists():
                tm=json.loads(trad.with_suffix('.json').read_text())
                if tm['world_sha256']!=meta['world_sha256']:raise ValueError('Traditional world mismatch')
                with np.load(trad) as tr:
                    np.testing.assert_array_equal(tr['truth'],truth);np.testing.assert_array_equal(tr['task_mask'],mask)
                    candidates += [(m,tr['predictions'][i].copy(),tr[m+'__stats'].copy(),tr['reputations'][i].copy(),None)
                                   for i,m in enumerate(tr['methods'].tolist())]
            for method,pred,stats,rep,settled in candidates:
                common=mask & np.isfinite(ours) & np.isfinite(pred)
                value=quality(truth,pred,mask,common);ref=quality(truth,ours,mask,common)
                value['mae_delta_vs_btpwr']=value['mae_common']-ref['mae_common']
                value['rmse_delta_vs_btpwr']=value['rmse_common']-ref['rmse_common']
                value.update(payment=float(stats[:,0].sum()),physical=float(stats[:,1].sum()),
                             settled_payment=float(settled.sum()) if settled is not None else float(stats[:,0].sum()),
                             honest_recognition=float((rep[-1,honest]>=.8).mean()),
                             malicious_recognition=float((rep[-1,~honest]<.3).mean()),
                             malicious_accepted=float((rep[-1,~honest]>=.8).mean()))
                rows.append(dict(dataset=case['dataset'],seed=case['seed'],method=LABELS.get(method,method),**value))
    df=pd.DataFrame(rows);df.to_csv(out/'primary_cases.csv',index=False)
    summary=[]
    columns=['completion','answered','ce3','mae_common','rmse_common','mae_delta_vs_btpwr',
             'rmse_delta_vs_btpwr','payment','settled_payment','physical','honest_recognition',
             'malicious_recognition','malicious_accepted']
    for method,group in df.groupby('method'):
        for metric in columns:
            v=group.groupby('seed')[metric].mean().dropna().to_numpy()
            err=t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)) if len(v)>1 else float('nan')
            summary.append(dict(method=method,metric=metric,mean=float(v.mean()),low=float(v.mean()-err),high=float(v.mean()+err),seeds=len(v)))
    pd.DataFrame(summary).to_csv(out/'primary_summary.csv',index=False)
    contrasts=[]
    ours=df[df.method=='BTPWR'].set_index(['dataset','seed'])
    for method in sorted(set(df.method)-{'BTPWR'}):
        other=df[df.method==method].set_index(['dataset','seed'])
        for metric in ('completion','answered','ce3','payment','physical','honest_recognition','malicious_recognition','malicious_accepted'):
            delta=ours[metric]-other[metric]
            if metric in ('payment','physical'): delta=100*delta/other[metric]
            v=delta.groupby(level='seed').mean().to_numpy()
            e=t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
            contrasts.append(dict(method=method,metric=metric,mean=float(v.mean()),low=float(v.mean()-e),high=float(v.mean()+e)))
    pd.DataFrame(contrasts).to_csv(out/'primary_paired.csv',index=False)
    print(json.dumps({'worlds':len(cases()),'rows':len(rows),'methods':sorted(df.method.unique().tolist())}))

if __name__=='__main__': main()

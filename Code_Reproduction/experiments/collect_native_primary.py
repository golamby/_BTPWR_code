"""Check native numerical system outputs against the same primary worlds."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.primary_450_cases import cases
from experiments.collect_review_sweeps import quality
ROOT=Path(__file__).resolve().parents[2]/'final_results'

def main():
    rows=[]
    for case in cases():
        name=case['name'];path=ROOT/'native_primary'/(name+'.npz');p=ROOT/'primary_450'/(name+'.npz')
        a=json.loads(path.with_suffix('.json').read_text());b=json.loads(p.with_suffix('.json').read_text())
        if a['world_sha256']!=b['world_sha256'] or a['case']!=case:raise ValueError('World mismatch '+name)
        with np.load(path) as z,np.load(p) as other:
            if json.loads(str(z['case_json']))!=case:raise ValueError('Case mismatch '+name)
            np.testing.assert_array_equal(z['truth'],other['truth']);np.testing.assert_array_equal(z['task_mask'],other['task_mask'])
            truth=z['truth'];mask=z['task_mask'].astype(bool);ours=other['predictions'][0]
            for i,method in enumerate(z['methods'].tolist()):
                pred=z['predictions'][i];common=mask & np.isfinite(ours) & np.isfinite(pred)
                q=quality(truth,pred,mask,common);ref=quality(truth,ours,mask,common)
                q['rmse_delta_vs_btpwr']=q['rmse_common']-ref['rmse_common']
                res=z['resources'][i]
                q['payment']=float(res[:,0].sum()) if np.isfinite(res[:,0]).all() else np.nan
                q['physical']=float(res[:,1].sum())
                rows.append(dict(dataset=case['dataset'],seed=case['seed'],method=method,**q))
    df=pd.DataFrame(rows);df.to_csv(ROOT/'tables'/'native_primary_cases.csv',index=False)
    print(df.groupby('method')[['completion','answered','ce3','payment','physical','rmse_delta_vs_btpwr']].mean().round(3).to_string())

if __name__=='__main__':main()

"""Compare final-planner reference stresses to the paired default worlds."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from experiments.collect_review_sweeps import quality
from experiments.sweep_cases import cases
ROOT=Path(__file__).resolve().parents[2]/'final_results'

def main():
    base=pd.read_csv(ROOT/'tables'/'sweep_cases.csv')
    base=base[(base.method=='BTPWR')&(base.axis=='budget')&(base.x==450)].set_index(['dataset','seed'])
    defaults={(c['dataset'],c['seed']):c for c in cases(default_budget='450') if c['n']==100
              and c['H']==.7 and c['C']==.1 and c['budget_override']==450 and c['eps1']==2.1}
    rows=[]
    for path in sorted((ROOT/'final_reference_stress').glob('stress*.json')):
        j=json.loads(path.read_text());case=j['case'];r=j['results'][0]
        with np.load(path.with_suffix('.npz')) as z:
            pred=z['predictions'][0];truth=z['truth'];mask=z['task_mask'].astype(bool)
            reference=ROOT/'review_sweeps'/(defaults[(case['dataset'],case['seed'])]['name']+'.npz')
            with np.load(reference) as original:
                np.testing.assert_array_equal(truth,original['truth'])
                np.testing.assert_array_equal(mask,original['task_mask'])
            q=quality(truth,pred,mask,mask & np.isfinite(pred))
        default=base.loc[(case['dataset'],case['seed'])]
        rows.append(dict(scenario=case['stress'],dataset=case['dataset'],seed=case['seed'],
            completion=q['completion'],answered=q['answered'],ce3=q['ce3'],
            payment=r['payment'],physical=r['physical'],
            delta_completion=q['completion']-default['completion'],delta_ce3=q['ce3']-default['ce3']))
    df=pd.DataFrame(rows)
    if len(df)!=36:raise ValueError('Expected 36 stress worlds')
    df.to_csv(ROOT/'tables'/'reference_stress_cases.csv',index=False)
    results=[]
    for scenario,g in df.groupby('scenario'):
        row={'scenario':scenario}
        for col in ['completion','answered','ce3','payment','physical','delta_completion','delta_ce3']:
            x=g.groupby('seed')[col].mean().to_numpy();e=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
            row[col]=x.mean();row[col+'_low']=x.mean()-e;row[col+'_high']=x.mean()+e
        results.append(row)
    out=pd.DataFrame(results);out.to_csv(ROOT/'tables'/'reference_stress_summary.csv',index=False)
    print(out[['scenario','completion','delta_completion','delta_completion_low','delta_completion_high','ce3']].round(3).to_string(index=False))

if __name__=='__main__':main()

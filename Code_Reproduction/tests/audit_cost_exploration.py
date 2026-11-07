"""Compare four planners on identical development worlds."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import t

base, cost = Path(sys.argv[1]), Path(sys.argv[2])
records=[]
for dataset in ('tdrive','foursquare_tky','brightkite'):
    for seed in (9701,9702,9703):
        stem=f'{dataset}_{seed}_1.00'
        j1=json.loads((base/(stem+'.json')).read_text(encoding='utf8'))
        j2=json.loads((cost/(stem+'.json')).read_text(encoding='utf8'))
        assert j1['world_sha256']==j2['world_sha256']
        with np.load(base/(stem+'.npz'),allow_pickle=False) as a, np.load(cost/(stem+'.npz'),allow_pickle=False) as b:
            np.testing.assert_array_equal(a['truth'],b['truth'])
            np.testing.assert_array_equal(a['task_mask'],b['task_mask'])
            entries={}
            for z in (a,b):
                for i,name in enumerate(z['methods'].tolist()):
                    y=z['truth']; p=z['predictions'][i]; m=z['task_mask'].astype(bool)
                    finite=m & np.isfinite(p); e=np.abs(p-y)
                    hit=finite & np.where(y==0,e==0,e<.1*np.abs(y))
                    stats=z[name+'__stats']
                    entries[name]={'completion':float(100*hit.sum()/m.sum()),
                                   'answered':float(100*finite.sum()/m.sum()),
                                   'ce3':float((e[hit].sum()+3*(m.sum()-hit.sum()))/m.sum()),
                                   'payment':float(stats[:,0].sum()),'physical':float(stats[:,1].sum()),
                                   'honest_C_final':int(z['worker_counts'][i,-1,0]),
                                   'malicious_C_final':int(z['worker_counts'][i,-1,2])}
            records.append({'dataset':dataset,'seed':seed,'methods':entries})
effects={}
for method in ('admission-current-coverage','admission-cost-025','admission-cost-050'):
    effects[method]={}
    for metric in ('completion','answered','ce3','payment','physical','honest_C_final','malicious_C_final'):
        x=np.asarray([np.mean([r['methods'][method][metric]-r['methods']['admission'][metric]
                              for r in records if r['seed']==s]) for s in (9701,9702,9703)])
        h=t.ppf(.975,2)*x.std(ddof=1)/np.sqrt(3)
        effects[method][metric]={'mean':float(x.mean()),'ci95':[float(x.mean()-h),float(x.mean()+h)]}
print(json.dumps({'scope':'exploratory 3 seeds x 3 datasets', 'effects':effects,'records':records},indent=2))

"""Check and summarize nine paired exploration cases (not confirmation)."""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t

root = Path(sys.argv[1])
ds_names = ('tdrive','foursquare_tky','brightkite')
records = []
for ds in ds_names:
    for seed in (9701,9702,9703):
        stem = f'{ds}_{seed}_1.00'
        main_json = json.loads((root/'current_coverage'/(stem+'.json')).read_text(encoding='utf-8'))
        bpqa_json = json.loads((root/'bpqa_native'/(stem+'.json')).read_text(encoding='utf-8'))
        assert main_json['world_sha256'] == bpqa_json['world_sha256']
        with np.load(root/'current_coverage'/(stem+'.npz'),allow_pickle=False) as z, \
             np.load(root/'bpqa_native'/(stem+'.npz'),allow_pickle=False) as b:
            np.testing.assert_array_equal(z['truth'],b['truth'])
            np.testing.assert_array_equal(z['task_mask'],b['task_mask'])
            assert list(z['methods']) == ['admission','admission-current-coverage']
            mask = z['task_mask'].astype(bool); truth = z['truth']
            preds = [z['predictions'][0],z['predictions'][1],b['predictions']]
            def perf(pred):
                finite = mask & np.isfinite(pred); err = np.abs(pred-truth)
                success = finite & np.where(truth == 0, err == 0, err < .1*np.abs(truth))
                return {'completion':float(100*success.sum()/mask.sum()),
                        'answered':float(100*finite.sum()/mask.sum()),
                        'ce3':float((err[success].sum()+3*(mask.sum()-success.sum()))/mask.sum())}
            common = mask.copy()
            for pred in preds: common &= np.isfinite(pred)
            methods = []
            for i,pred in enumerate(preds):
                p = perf(pred)
                p['rmse_common_all'] = float(np.sqrt(np.mean((pred[common]-truth[common])**2)))
                if i < 2:
                    stats = z[list(z['methods'])[i]+'__stats']
                    p['payment'] = float(stats[:,0].sum()); p['physical'] = float(stats[:,1].sum())
                    p['honest_C_final'] = int(z['worker_counts'][i,-1,0])
                    p['malicious_C_final'] = int(z['worker_counts'][i,-1,2])
                else:
                    p['payment'] = float(b['resources'][:,0].sum()); p['physical'] = float(b['resources'][:,1].sum())
                    p['settled_payment'] = float(b['resources'][:,4].sum())
                methods.append(p)
            assert abs(methods[2]['completion']-bpqa_json['completion_percent'])<1e-8
            records.append({'dataset':ds,'seed':seed,'common_tasks':int(common.sum()),
                            'original':methods[0], 'candidate':methods[1], 'bpqa_numeric':methods[2]})
metrics = ('completion','answered','ce3','rmse_common_all','payment','physical','honest_C_final','malicious_C_final')
effects = {}
for metric in metrics:
    values = []
    for seed in (9701,9702,9703):
        pairs = [r for r in records if r['seed']==seed]
        values.append(np.mean([r['candidate'][metric]-r['original'][metric] for r in pairs]))
    arr = np.asarray(values); h = t.ppf(.975,2)*arr.std(ddof=1)/np.sqrt(3)
    effects[metric] = {'mean_candidate_minus_original':float(arr.mean()),
                       'ci95':[float(arr.mean()-h),float(arr.mean()+h)],
                       'seed_means':arr.tolist()}
print(json.dumps({'scope':'exploration; three seed means averaged over three fixed datasets',
                  'effects':effects,'records':records},indent=2))

"""Pair BPQA numeric output with the registered BTPWR evidence archives."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import t

root = Path(__file__).resolve().parents[1]
bpqa = Path(sys.argv[1])
archive = root / 'data' / 'recruitment_evidence'
records = []
for dataset in ('tdrive', 'foursquare_tky', 'brightkite'):
    budget = 455. if dataset == 'brightkite' else 475.
    for seed in (9701, 9702, 9703):
        name = f'{dataset}_{seed}_1.00'
        meta = json.loads((bpqa / (name + '.json')).read_text(encoding='utf-8'))
        path = archive / f'{dataset}_{seed}_B{budget}_C0.1.npz'
        with np.load(path, allow_pickle=False) as base, np.load(bpqa / (name + '.npz'), allow_pickle=False) as other:
            np.testing.assert_array_equal(base['truth'], other['truth'])
            np.testing.assert_array_equal(base['task_mask'], other['task_mask'])
            assert base['methods'][0] == 'admission'
            truth, mask = base['truth'], base['task_mask'].astype(bool)
            bp, op = other['predictions'], base['predictions'][0]
            res, bres = other['resources'], base['resources'][0]
            assert np.all(res[:, 0] <= budget + 1e-7)
            assert np.all(res[:, 0] + 1e-7 >= res[:, 3])
            assert np.array_equal(other['selected'].sum(axis=1), res[:, 2])
            assert abs(meta['payment'] - res[:, 0].sum()) < 1e-6
            def calculate(pred):
                finite = mask & np.isfinite(pred)
                err = np.abs(pred - truth)
                hit = finite & np.where(truth == 0, err == 0, err < .1 * np.abs(truth))
                return float(hit.sum() / mask.sum() * 100), float(finite.sum() / mask.sum() * 100)
            otcr, oar = calculate(op)
            btcr, bar = calculate(bp)
            assert abs(meta['completion_percent'] - btcr) < 1e-8
            common = mask & np.isfinite(bp) & np.isfinite(op)
            def rmse(pred):
                return float(np.sqrt(np.mean((pred[common]-truth[common])**2))) if common.any() else None
            records.append({'dataset':dataset, 'seed':seed, 'common_tasks':int(common.sum()),
                            'ours_completion':otcr, 'bpqa_completion':btcr,
                            'ours_answered':oar, 'bpqa_answered':bar,
                            'ours_rmse_common':rmse(op), 'bpqa_rmse_common':rmse(bp),
                            'ours_payment':float(bres[:,0].sum()), 'bpqa_payment':float(res[:,0].sum()),
                            'ours_physical':float(bres[:,1].sum()), 'bpqa_physical':float(res[:,1].sum())})
print(json.dumps({'records':records, 'cases':len(records),
                  'note':'Descriptive 3-seed comparison; native BPQA lacks the BTPWR recovery service.'}, indent=2))

"""Independently recalculate seed-paired effects from archived predictions."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

folder = Path(sys.argv[1])
rows = defaultdict(lambda: defaultdict(list))
expected = {(ds, seed, cap) for ds in ('tdrive', 'brightkite')
            for seed in range(9701, 9709) for cap in (.75, 1.)}
seen = set()
for path in sorted(folder.glob('*.json')):
    obj = json.loads(path.read_text(encoding='utf-8'))
    if 'case' not in obj:
        continue
    case = obj['case']
    key = (case['dataset'], case['seed'], case['budget_factor'])
    if key not in expected or key in seen:
        raise ValueError('Unexpected or duplicate campaign: ' + repr(key))
    seen.add(key)
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
        truth, mask = z['truth'], z['task_mask'].astype(bool)
        records = {r['method']: r for r in obj['results']}
        val = {}
        for name, pred in zip(z['methods'].tolist(), z['predictions']):
            err = np.abs(pred - truth)
            hit = mask & np.isfinite(pred) & np.where(truth == 0, err == 0, err < .1 * np.abs(truth))
            completion = 100. * hit.sum() / mask.sum()
            if abs(completion - records[name]['completion_percent']) > 1e-8:
                raise ValueError('Completion mismatch: ' + repr((key, name)))
            val[name] = {'comp': completion,
                         'ce3': (err[hit].sum() + 3 * (mask.sum()-hit.sum())) / mask.sum(),
                         'pay': records[name]['payment'], 'phys': records[name]['physical']}
        pairs = {'select_sep': ('bpqa', 'admission'),
                 'couple_btpwr': ('admission-coupled', 'admission'),
                 'select_coup': ('bpqa-coupled', 'admission-coupled'),
                 'couple_bpqa': ('bpqa-coupled', 'bpqa'),
                 'system': ('bpqa-coupled', 'admission')}
        for effect, (a, b) in pairs.items():
            for metric in ('comp', 'ce3', 'pay', 'phys'):
                rows[(effect, metric)][case['seed']].append(val[a][metric]-val[b][metric])
        for metric in ('comp', 'ce3', 'pay', 'phys'):
            diff = ((val['bpqa-coupled'][metric]-val['bpqa'][metric]) -
                    (val['admission-coupled'][metric]-val['admission'][metric]))
            rows[('interaction', metric)][case['seed']].append(diff)
if seen != expected:
    raise ValueError('Missing cases: ' + repr(sorted(expected-seen)))
out = {'n_cases': len(seen), 'effects': {}}
for (effect, metric), seeds in sorted(rows.items()):
    if set(seeds) != set(range(9701, 9709)) or any(len(v) != 4 for v in seeds.values()):
        raise ValueError('Unbalanced seed strata')
    x = np.array([np.mean(seeds[s]) for s in sorted(seeds)])
    h = t.ppf(.975, 7) * x.std(ddof=1) / np.sqrt(8)
    out['effects'].setdefault(effect, {})[metric] = [float(x.mean()), float(x.mean()-h), float(x.mean()+h)]
print(json.dumps(out, indent=2))

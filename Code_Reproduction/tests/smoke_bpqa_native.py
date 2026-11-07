"""Check the full BPQA numeric path on one actual experiment round."""
import time
import numpy as np
from experiments.run_bpqa_native import calibrate, case_for
from src import bpqa_native, data

inputs = data.load_inputs()
case = case_for('tdrive', 9701, 1.)
world = data.build_world(case, inputs)
calibration = calibrate(inputs, 'tdrive', 1.)
tasks = world['requested_task_ids'][0]
bids = world['actual_cost'][0].sum(axis=1)
started = time.perf_counter()
out = bpqa_native.allocate(world['presence'][0][:, tasks], bids, world['initial_reputation'],
                           world['offer_mask'][0], world['budget'], calibration['multiplier'])
ids = out['selected']
assert np.all(world['offer_mask'][0, ids])
assert out['total_payment'] <= world['budget'] + 1e-8
assert np.all(out['payments'][ids] + 1e-8 >= bids[ids])
reports = np.full(world['reports_check'][0].shape, np.nan)
reports[ids] = world['reports_check'][0, ids]
reports[:, ~world['task_mask'][0]] = np.nan
reports[(reports < 10.) | (reports > 40.)] = np.nan
pred, weights = bpqa_native.truth_and_weights(reports)
new_r = bpqa_native.update_reputation(world['initial_reputation'], weights, np.isfinite(reports).any(axis=1))
assert not np.isinf(pred).any() and np.isfinite(new_r).all()
print({'selected': len(ids), 'payment': out['total_payment'], 'physical': float(bids[ids].sum()),
       'seconds': time.perf_counter()-started, 'calibration': calibration})

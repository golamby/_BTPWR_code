"""Integration contracts for the approved Q36 runner, without campaign data."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from experiments import run_formal_q36 as runner
from experiments.formal_q36_cases import cases


class FormalQ36Contract(unittest.TestCase):
    def test_grade_float_bytes_equal_signed_integer_codes(self):
        state = np.arange(10001, dtype=int)
        expected = (np.minimum(state//1000, 9)*1000+500)/10000.
        actual = runner.grade(state/10000.)
        self.assertEqual(actual.astype('<f8').tobytes(), expected.astype('<f8').tobytes())

    def test_approved_world_counts_and_deduplicated_factor_ranges(self):
        primary = cases('primary')
        self.assertEqual(len(primary), 24)
        self.assertTrue(all(c['n']==80 and c['budget_override']==450 for c in primary))
        sweep = cases('sensitivity')
        self.assertEqual(len(sweep), 207)
        self.assertEqual(len({c['name'] for c in sweep}), 207)
        for dataset in ('tdrive','foursquare_tky','brightkite'):
            for seed in range(9701,9704):
                local = [c for c in sweep if c['dataset']==dataset and c['seed']==seed]
                self.assertEqual(len(local), 23)
                self.assertEqual(sorted({c['n'] for c in local}), [50,60,70,80,90,100])
                self.assertEqual(sorted({c['budget_override'] for c in local}), [250,300,350,400,450,500,550,600])
                self.assertEqual(sum(c['n']==80 and c['H']==.7 and c['C']==.1 and c['eps1']==2.1 and c['budget_override']==450 for c in local), 1)

    def test_native_quota_keeps_undefined_payment_and_full_route_cost(self):
        # No truth, behavior labels or future private reports in this fixture:
        # the wrapper needs only native results and actual selected route cost.
        initial=np.array([.98,.5,.5]);state=SimpleNamespace(trust=initial.copy())
        world=dict(presence=np.ones((2,3,100),bool),initial_reputation=initial,
                   requested_task_ids=np.array([[0],[1]]),native_offset=0.,native_scale=100.,
                   actual_cost=np.ones((2,3,100))*2)
        def native(*args):
            return ({'ETBP-TD':np.array([25.])}, {'ETBP-TD':np.array([1])}, state,
                    {'ETBP-TD':dict(commitment=None,constraint='native_fixed_cardinality')})
        with patch.object(runner.baselines,'native_round',side_effect=native):
            result=runner.run_native_other(world,dict(seed=9801,dataset='tdrive'),'ETBP-TD')
        self.assertTrue(np.isnan(result['resources'][:,[0,4]]).all())
        np.testing.assert_array_equal(result['resources'][:,1], [200.,200.])
        self.assertEqual(result['class_thresholds'],dict(low=.4,high=.9,credible_strict=True))
        self.assertEqual(result['reputations'][0].tolist(),initial.tolist())


if __name__=='__main__':
    unittest.main()

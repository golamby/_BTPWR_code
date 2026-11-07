import unittest
import numpy as np
from experiments import run_q36_supplement as s
from src import privacy


class SupplementTests(unittest.TestCase):
    def test_wrapper_matches_canonical_current_support_policy(self):
        p=dict(bits=np.array([[1,0,0,1],[0,1,1,0],[1,1,0,0]],bool),
               hi=np.zeros((3,4),bool),nonzero=np.ones((3,4),bool))
        r=np.array([.95,.55,.55]);active=np.ones(3,bool);tasks=np.array([0,2])
        cfg=privacy.Config(n=3,m=4)
        ids,charge=s.select_control('PaidFootprint-Privileged',p,active,r,s.current_support(p,2.1),tasks,cfg,30.,50)
        expected=s.formal.policy.select_round(p,active,r,tasks,30.,50)
        np.testing.assert_array_equal(np.sort(ids),np.sort(expected['members']))
        self.assertEqual(charge,round(expected['cost_numerator']/expected['cost_denominator']*1e6)/1e6)

    def test_reference_audit_counts_worker_checks_and_uses_truth_only_for_errors(self):
        raw=np.array([[20.,21.,np.nan],[np.nan,25.,26.],[20.,21.,22.]])
        r=np.array([.5,.5,.95]);direct=dict(has_credible=np.array([1,0,0],bool),reference_by_worker=np.array([[20.,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan]]))
        secondary=np.array([np.nan,15.,18.]);w=dict(native_offset=0.,native_scale=30.,truth_normalized_private=np.array([[.3],[.4],[.5]]))
        a=s.reference_audit(raw,r,direct,secondary,w,0)
        self.assertEqual(a['secondary_worker_region_checks'],3)
        self.assertEqual(a['secondary_distinct_regions'],2)
        self.assertEqual(a['secondary_sum_absolute_error'],9.)
        self.assertEqual(a['secondary_sum_squared_error'],27.)
        self.assertEqual(a['secondary_within3'],3)
        before=secondary.copy();w['truth_normalized_private']*=2
        b=s.reference_audit(raw,r,direct,secondary,w,0)
        np.testing.assert_array_equal(before,secondary)
        self.assertEqual(a['secondary_worker_region_checks'],b['secondary_worker_region_checks'])
        self.assertNotEqual(a['secondary_sum_squared_error'],b['secondary_sum_squared_error'])


if __name__=='__main__':unittest.main()

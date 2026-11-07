import unittest
import numpy as np
from src import trust
from src.count_feedback import feedback

class CountFeedbackTests(unittest.TestCase):
    def test_three_successes_and_failures_have_stronger_effect(self):
        r=np.array([.95,.95,.5]);cfg=trust.Config()
        def run(value,count):
            raw=np.array([[25.]*count,[25.]*count,[value]*count])
            return feedback(raw,r,None,cfg,0.,100.)['next_reputation'][2]
        self.assertGreater(run(25.,3),run(25.,1))
        self.assertLess(run(40.,3),run(40.,1))
    def test_region_order_does_not_change_update(self):
        r=np.array([.95,.95,.5]);raw=np.array([[25.,25.,25.],[25.,25.,25.],[25.,26.,40.]])
        a=feedback(raw,r,None,trust.Config(),0.,100.)
        b=feedback(raw[:,[2,0,1]],r,None,trust.Config(),0.,100.)
        np.testing.assert_array_equal(a['next_reputation'],b['next_reputation'])
        np.testing.assert_array_equal(a['effective_evidence'],b['effective_evidence'])

    def test_recovery_units_and_sole_credible_self_reference(self):
        r=np.array([.95,.5]);raw=np.array([[25.,np.nan],[25.,25.]])
        # The second region has a recovered physical value encoding checking value 25.
        a=feedback(raw,r,np.array([np.nan,50.]),trust.Config(),0.,100.)
        b=feedback(raw,r,np.array([np.nan,45.]),trust.Config(),10.,70.)
        np.testing.assert_array_equal(a['next_reputation'],b['next_reputation'])
        self.assertEqual(a['next_reputation'][0],r[0])
        self.assertEqual(a['effective_evidence'][1],1.5)

    def test_evidence_cap_and_secondary_discount(self):
        r=np.array([.95,.95,.5]);cfg=trust.Config()
        four=feedback(np.full((3,4),25.),r,None,cfg,0.,100.)
        eight=feedback(np.full((3,8),25.),r,None,cfg,0.,100.)
        np.testing.assert_array_equal(four['next_reputation'],eight['next_reputation'])
        raw=np.array([[np.nan],[np.nan],[25.]])
        half=feedback(raw,r,np.array([50.]),cfg,0.,100.,.5)
        full=feedback(raw,r,np.array([50.]),cfg,0.,100.,1.)
        self.assertLess(half['next_reputation'][2],full['next_reputation'][2])

if __name__=='__main__':unittest.main()

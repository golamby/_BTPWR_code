import unittest
import numpy as np
from src import bpqa_native as bpqa


class NativeBPQAChecks(unittest.TestCase):
    def test_levels_respect_threshold_and_upper_endpoint(self):
        np.testing.assert_array_equal(bpqa.reliability_levels([.49, .5, .65, .99, 1.]), [0, 1, 2, 5, 5])
        np.testing.assert_array_equal(bpqa.reliability_levels([.5, .6, .7, .8, .9, 1.]), [1, 2, 3, 4, 5, 5])
        np.testing.assert_array_equal(bpqa.reliability_levels([.6-1e-8, .6+1e-8]), [1, 2])

    def test_budget_critical_payments_and_membership(self):
        incidence = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=bool)
        bids = np.array([2., 3., 2., 4.])
        out = bpqa.allocate(incidence, bids, np.array([.8, .8, .8, .8]),
                            np.ones(4, dtype=bool), 10., .4,
                            bpqa.Config(delta=.001))
        self.assertGreater(len(out['selected']), 0)
        self.assertLessEqual(out['total_payment'], 10. + 1e-8)
        self.assertTrue(np.all(out['payments'][out['selected']] >= bids[out['selected']] - 1e-8))
        self.assertEqual(out['total_payment'], out['payments'].sum())

    def test_truth_uses_only_submitted_values_and_handles_perfect_agreement(self):
        reports = np.array([[25., np.nan], [25., np.nan], [np.nan, 17.]])
        z, weights = bpqa.truth_and_weights(reports)
        np.testing.assert_allclose(z, [25., 17.])
        self.assertTrue(np.isfinite(weights).all())
        self.assertGreaterEqual(weights.min(), 0.)

    def test_unchosen_or_unscored_worker_never_updates(self):
        r = np.array([.9, .5, .5])
        out = bpqa.update_reputation(r, np.array([.9, .1, .9]), [True, True, False])
        self.assertEqual(out[2], r[2])
        self.assertGreater(out[0], r[0])
        self.assertLess(out[1], r[1])


if __name__ == '__main__':
    unittest.main()

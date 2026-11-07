"""Independent matrix and state checks for the original numerical baselines."""
import unittest
import numpy as np
from src import _bpp as bpp, baselines


class NativeBaselineContract(unittest.TestCase):
    def test_literal_dense_rmm_matches_explicit_matrix_products(self):
        k = 3
        R = np.array([[1., 2., .7], [.5, 1.3, 2.1], [2.2, .8, 1.7]])
        tasks = np.array([0, 4, 7])
        counts = np.array([1, 0, 1, 0, 2, 1, 2, 1, 1])
        size, eps = 3, 1.7
        p = 1 / (1 + np.exp(-eps))
        calibrated = np.clip((counts-(1-p)*size)/(2*p-1), 0, 1)
        expected = 0.
        for cell, value in enumerate(calibrated):
            worker = np.zeros((k, k)); worker.flat[cell] = value
            for task in tasks:
                target = np.zeros((k, k)); target.flat[task] = 1
                product = (R@target).T @ (R@worker)
                if np.any(np.abs(product)>1e-12) and np.allclose(product, product.T):
                    expected += value
        self.assertAlmostEqual(float(bpp.native_utility(counts, size, eps, tasks, k)), expected)

    def test_shared_initial_draws_are_not_replaced_by_one(self):
        initial = np.array([.9512, .9876, .5, .5])
        state = baselines.etbp_initial_state(dict(initial_reputation=initial))
        np.testing.assert_array_equal(state.trust, initial)
        np.testing.assert_array_equal(state.history[0], initial)
        np.testing.assert_array_equal(state.initial_trusted, [True, True, False, False])

    def test_native_class_boundaries_do_not_use_btpwr_thresholds(self):
        np.testing.assert_array_equal(baselines.etbp_labels([.3,.4,.5,.8,.9,.95]), [0,1,1,1,1,2])


if __name__ == '__main__':
    unittest.main()

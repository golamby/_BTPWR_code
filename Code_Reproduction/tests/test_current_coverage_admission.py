import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from src.current_coverage_admission import direct_coverage, CostAwarePlanner, OriginalPlanner


class CurrentCoverageChecks(unittest.TestCase):
    def test_zero_duplicate_and_reputation_weighting(self):
        p = np.array([[1., 0.], [1., 1.]])
        r = np.array([1., .5])
        self.assertEqual(direct_coverage(p, r, [0, 1], []), 0.)
        self.assertAlmostEqual(direct_coverage(p, r, [0, 1], [0]), .5)
        self.assertAlmostEqual(direct_coverage(p, r, [0, 1], [1]), .5)
        self.assertAlmostEqual(direct_coverage(p, r, [0, 1], [0, 1]), .75)

    def test_cost_penalty_uses_commitment_and_does_not_modify_base_cache(self):
        planner = object.__new__(CostAwarePlanner)
        planner.state = SimpleNamespace(support=np.array([[1.]]), reputation=np.array([1.]),
                                        requested_tasks=np.array([0]), budget=100.)
        planner.price = SimpleNamespace(cost_denominator=1.)
        cached = {'value': 2., 'current_charge': 80}
        with patch.object(OriginalPlanner, 'value', return_value=cached):
            first = planner.value([0]); second = planner.value([0])
        self.assertAlmostEqual(first['value'], 2. + .5 - .25 * .8)
        self.assertEqual(first, second)
        self.assertEqual(cached['value'], 2.)


if __name__ == '__main__':
    unittest.main()

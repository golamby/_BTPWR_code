import unittest
import numpy as np
from src import data


class PopulationInitializationChecks(unittest.TestCase):
    def test_true_type_and_observable_initial_grade_are_distinct(self):
        q, honest, certified, estimate = data.population(100, .7, .1, 9701)
        self.assertEqual(certified.sum(), 10)
        self.assertTrue(np.all(honest[certified]))
        self.assertTrue(np.all(q[honest] >= .8))
        self.assertTrue(np.all(q[~honest] < .3))
        self.assertTrue(np.all((estimate[certified] >= .95) & (estimate[certified] <= 1.)))
        self.assertTrue(np.all(estimate[~certified] == .5))
        self.assertTrue(np.allclose(estimate * 10000, np.rint(estimate * 10000)))
        again = data.population(100, .7, .1, 9701)
        for a, b in zip((q, honest, certified, estimate), again):
            np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()

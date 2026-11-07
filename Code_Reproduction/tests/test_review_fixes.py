import unittest

import numpy as np

from experiments.analyze_bpqa_factorial import ce3
from experiments.analyze_coupled import ce
from src import metrics, trust
from src.admission_experiments import scored_events
from src.reference_experiments import reference_feedback


class ReviewFixes(unittest.TestCase):
    def test_physical_units_do_not_change_check_feedback(self):
        r = np.array([.9, .9, .5])
        outcomes = []
        for offset, scale in [(0., 100.), (12., 87.), (-20., 25.)]:
            physical = offset + scale * .5
            check = 10. + 30. * (physical - offset) / scale
            reports = np.array([[25.], [check], [25.]])
            outcomes.append(trust.feedback(reports, r)['next_reputation'])
        for outcome in outcomes[1:]:
            np.testing.assert_array_equal(outcomes[0], outcome)

    def test_history_records_actual_applied_feedback(self):
        cfg = trust.Config()
        r = np.array([.9, .9, .5, .5])
        reports = np.array([[25.], [25.], [25.], [25.]])
        feedback = trust.feedback(reports, r, cfg)
        events = scored_events(reports, feedback, r, 0, cfg)
        self.assertEqual([event.worker for event in events], [0, 1, 2, 3])
        for event in events:
            self.assertEqual(event.next_r, feedback['next_reputation'][event.worker])
            self.assertEqual(len(event.residuals), feedback['scored_count'][event.worker])

    def test_success_boundary_matches_primary_metric(self):
        truth = np.array([[100., 100., 0., 0.]])
        prediction = np.array([[110., 109., 0., .0000000001]])
        mask = np.ones_like(truth, dtype=bool)
        primary = metrics.requested_metrics(truth, prediction, mask)
        self.assertEqual(primary['completed_task_times'], 2)
        expected = (9. + 2 * 3.) / 4
        self.assertEqual(ce3(prediction, truth, mask), expected)
        self.assertEqual(ce(prediction, truth, mask, 3)[0], expected)

    def test_oracle_reference_uses_check_scale_and_actual_transition(self):
        raw = np.array([[25., np.nan], [np.nan, 25.]])
        r = np.array([.9, .5])
        results = []
        for offset, scale in ((0., 100.), (12., 87.)):
            pred = np.array([offset + .5 * scale, offset + .5 * scale])
            next_r, n, refs, factors, counts = reference_feedback(raw, r, pred, pred, trust.Config(), .25, offset, scale)
            self.assertEqual(n, 1)
            self.assertAlmostEqual(refs[1, 1], 25.)
            self.assertEqual(factors[1, 1], .25)
            self.assertEqual(counts[1], 1)
            self.assertGreater(next_r[1], r[1])
            results.append(next_r)
        np.testing.assert_array_equal(results[0], results[1])


if __name__ == '__main__':
    unittest.main()

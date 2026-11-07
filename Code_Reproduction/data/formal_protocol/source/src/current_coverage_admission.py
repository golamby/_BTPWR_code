"""Development candidate: current report coverage plus future qualification."""
import numpy as np
from .admission import AdmissionPlanner as OriginalPlanner


DIRECT_WEIGHT = .5


def direct_coverage(support, reputation, tasks, members):
    """Bounded score using only released-data support and current reputation.

    Reputation is a ranking proxy here, not a calibrated probability of truth.
    """
    p = np.asarray(support, float)
    r = np.asarray(reputation, float)
    task_ids = np.asarray(tasks, int)
    ids = np.asarray(members, int)
    if p.ndim != 2 or r.shape != (p.shape[0],) or not len(task_ids):
        raise ValueError('Matched supports, reputation and nonempty tasks required')
    if not len(ids):
        return 0.
    q = np.clip(p[ids][:, task_ids] * r[ids, None], 0., 1.)
    return float(np.mean(1. - np.prod(1. - q, axis=0)))


class AdmissionPlanner(OriginalPlanner):
    def value(self, members, backend='numba'):
        # The parent's cached result must remain unchanged: select() asks for
        # the same candidate repeatedly while comparing marginal values.
        result = dict(super().value(members, backend=backend))
        direct = direct_coverage(self.state.support, self.state.reputation,
                                 self.state.requested_tasks, members)
        result['value'] += DIRECT_WEIGHT * direct
        result['current_report_coverage_score'] = direct
        return result


class CostAwarePlanner(AdmissionPlanner):
    cost_weight = .25

    def value(self, members, backend='numba'):
        result = dict(super().value(members, backend=backend))
        budget = float(self.state.budget)
        fraction = (float(result['current_charge']) / self.price.cost_denominator / budget
                    if budget > 0 else 0.)
        result['value'] -= self.cost_weight * fraction
        result['current_budget_fraction'] = fraction
        return result


class CostAwareHalf(CostAwarePlanner):
    cost_weight = .5

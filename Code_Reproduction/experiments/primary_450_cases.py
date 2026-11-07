"""Independent-seed primary comparison under the shared 450 per-round cap."""
from src import data


def cases():
    return [dict(name=f'main_{d}_{s}_b450', group='confirmation', kind='main',
                 stage='confirmation', dataset=d, field='legacy', seed=s, n=100,
                 H=.7, C=.1, gamma=.6, eps1=2.1, eps2=2.7, budget_override=450.,
                 budget_factor=450./data.BUDGETS[d], cohort_rule='training-top200-uniform100')
            for d in data.DATASETS for s in range(9801,9809)]

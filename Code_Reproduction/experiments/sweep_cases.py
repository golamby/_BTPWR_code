"""Declared one-factor sensitivity grid; identical defaults are shared."""
from src import data


AXES = {
    'workers': [50, 60, 70, 80, 90, 100],
    'honest_fraction': [.5, .6, .7, .8, .9],
    'certified_fraction': [.1, .2, .3],
    'budget': [250, 300, 350, 400, 450, 500, 550, 600],
    'location_epsilon': [1.3, 1.5, 1.7, 1.9, 2.1],
}


def cases(datasets=data.DATASETS, seeds=(9701,9702,9703), default_budget=None):
    if default_budget is None:
        raise ValueError('Specify the default per-round budget explicitly')
    result = []
    for dataset in datasets:
        for seed in seeds:
            seen = {}
            for axis, values in AXES.items():
                for value in values:
                    n, H, C, eps = 100, .7, .1, 2.1
                    budget = data.BUDGETS[dataset] if default_budget == 'original' else float(default_budget)
                    if axis == 'workers': n = value
                    elif axis == 'honest_fraction': H = value
                    elif axis == 'certified_fraction': C = value
                    elif axis == 'budget': budget = float(value)
                    elif axis == 'location_epsilon': eps = value
                    factor = float(budget) / data.BUDGETS[dataset]
                    key = (int(n),float(H),float(C),float(budget),float(eps))
                    if key not in seen:
                        name = f'sweep_{dataset}_{seed}_n{n}_h{H:.2f}_c{C:.2f}_b{budget:.2f}_e{eps:.2f}'
                        seen[key] = dict(name=name, group='sweep', kind='sweep', stage='sweep',
                            dataset=dataset, field='legacy', seed=int(seed), n=int(n), H=H, C=C,
                            gamma=.6, eps1=eps, eps2=2.7, budget_override=budget,
                            budget_factor=factor, cohort_rule='training-top200-uniform100', sweep_axes=[])
                    seen[key]['sweep_axes'].append({'axis':axis, 'value':value})
            result.extend(seen.values())
    return result

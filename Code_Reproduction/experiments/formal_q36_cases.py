"""Frozen Q36 worlds: default80, approved sweeps and targeted review checks."""
from src import data

def base(dataset, seed):
    return dict(dataset=dataset, seed=int(seed), n=80, H=.7, C=.1, gamma=.6,
                eps1=2.1, eps2=2.7, budget_override=450., field='legacy',
                cohort_rule='training-top200-uniform100', kind='main', group='primary',
                stage='formal-q36', factor='default', factor_value=0.)

def finish(c):
    c=dict(c)
    c['budget_factor']=c['budget_override']/data.BUDGETS[c['dataset']]
    c['name']='q36_{group}_{dataset}_{seed}_n{n}_h{H:.3f}_c{C:.4f}_b{budget_override:g}_e{eps1:.2f}_{field}'.format(**c)
    if c.get('stress'):c['name']+='_'+c['stress']
    return c

def cases(stage='primary'):
    result=[]
    if stage=='primary':
        return [finish(base(d,s)) for d in data.DATASETS for s in range(9801,9809)]
    if stage=='sensitivity':
        values=[('workers','n',[50,60,70,80,90,100]),('budget','budget_override',[250,300,350,400,450,500,550,600]),
                ('honest','H',[.5,.6,.7,.8,.9]),('certified','C',[.1,.2,.3]),('privacy','eps1',[1.3,1.5,1.7,1.9,2.1])]
        for d in data.DATASETS:
            for s in range(9701,9704):
                default=base(d,s);seen=set()
                for factor,key,points in values:
                    for value in points:
                        c=dict(default,group='sensitivity',kind='sweep',factor=factor,factor_value=value)
                        c[key]=value;c=finish(c)
                        if c['name'] not in seen:result.append(c);seen.add(c['name'])
        assert len(result)==207
        return result
    if stage=='traceable':
        fields=dict(tdrive='noaa_east_asia',foursquare_tky='noaa_east_asia',brightkite='noaa_north_america')
        return [finish(dict(base(d,s),group='traceable',field=fields[d])) for d in data.DATASETS for s in range(9701,9704)]
    if stage=='stress':
        for d in data.DATASETS:
            for s in range(9701,9704):
                for attack in ['small_bias','large_bias','on_off','reference_noise','sole_bad_certificate','two_bad_certificates']:
                    c=dict(base(d,s),group='stress',stress=attack)
                    if attack=='sole_bad_certificate':c['C']=1/c['n']
                    result.append(finish(c))
        return result
    raise ValueError(stage)

def methods_for(c, include_ablations=False):
    names=['BTPWR','BPP','VLR-BPP','ETBP-TD','REAPP','BPQA','MMCB','Random','Greedy-C/R','C-seeded','ETBP-P']
    # These original systems have no such native parameter. Their default/native
    # resource points remain available; no fictitious flat sensitivity is run.
    if c['group']=='sensitivity' and c['factor'] in ('budget','privacy'):
        if (c['factor']=='budget' and c['budget_override']!=450) or (c['factor']=='privacy' and c['eps1']!=2.1):
            names=[m for m in names if m not in ('ETBP-TD','REAPP')]
        if c['factor']=='privacy' and c['eps1']!=2.1:names=[m for m in names if m!='MMCB']
    if c['group']=='stress':names=['BTPWR','ETBP-TD','REAPP','C-seeded']
    if c['group']=='traceable':names=['BTPWR','BPP','VLR-BPP','ETBP-TD','REAPP','C-seeded']
    if include_ablations and c['group']=='primary':
        names += ['Recruit-NoBalance','Recruit-NoOpportunity','Recruit-Neither',
                  'Feedback-NoCount','Feedback-NoSecondary','Feedback-Neither',
                  'Secondary-FullWeight','Evidence-Cap1','Evidence-Cap9']
    return names

"""Secondary class-only forecast; true feedback still uses exact reputation."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t as student
from .admission import AdmissionPlanner as OriginalPlanner
from .progress import make_profile
from . import metrics


def class_of(r, cfg):
    r = np.asarray(r, float)
    if not np.isfinite(r).all():
        raise ValueError('Finite authenticated state required')
    return np.where(r < cfg.malicious_threshold, 0, np.where(r >= cfg.credible_threshold, 2, 1))

def aggregate_class_counts(events, current_round, cfg):
    counts = np.zeros((3, 3), np.int64)
    seen = set()
    for e in events:
        key = (int(e.round), int(e.worker))
        if e.round < 0 or e.round >= current_round or key in seen:
            raise ValueError('One strictly past authenticated transition per worker-round required')
        seen.add(key)
        (a, b) = (int(class_of(e.pre_r, cfg)), int(class_of(e.next_r, cfg)))
        if a == 0:
            raise ValueError('Ordinary M cannot supply a scored transition')
        counts[a, b] += 1
    return counts

def pooled_kernel(counts, profile):
    counts = np.asarray(counts)
    if counts.shape != (3, 3) or np.any(counts < 0) or np.any(counts[0]):
        raise ValueError('Class-only scored counts required')
    alpha5 = 0.5 * profile.allowed[[1, 2, 3]].sum(0)
    alpha = np.array([alpha5[0], alpha5[1:4].sum(), alpha5[4]])
    p = np.zeros((3, 3))
    p[0, 0] = 1
    p[1] = (counts[1] + alpha) / (counts[1].sum() + alpha.sum())
    numerator = counts[2].astype(float)
    numerator[2] += 1
    p[2] = numerator / numerator.sum()
    return (p, alpha)

class ClassOnlyPlanner(OriginalPlanner):

    def __init__(self, state, remaining_contract_rounds, scenarios=4, allow_admission=True):
        if not allow_admission:
            raise ValueError('Do not combine two distinct secondary controls')
        cfg = state.trust_config
        current = class_of(state.reputation, cfg)
        counts = aggregate_class_counts(state.scored_rounds, state.round, cfg)
        profile = make_profile(cfg)
        (probability, alpha) = pooled_kernel(counts, profile)
        representative = np.array([0.0, (cfg.malicious_threshold + cfg.credible_threshold) / 2, 1.0])[current]
        representative = np.rint(representative * cfg.reputation_scale) / cfg.reputation_scale
        representative.setflags(write=False)
        redacted = replace(state, reputation=representative, scored_rounds=())
        super().__init__(redacted, remaining_contract_rounds, scenarios, True)
        destinations = np.array([0, 1, 4])
        self.z0 = destinations[current].astype(np.int8)
        self.kernel = np.eye(5)
        self.kernel[np.ix_(destinations, destinations)] = probability
        self.kernel[destinations[:, None], np.array([2, 3])] = 0.0
        self.kernel.setflags(write=False)
        self.counts = counts.copy()
        self.counts.setflags(write=False)
        self.class_prior = alpha.copy()
        self.class_prior.setflags(write=False)
        assert not self.state.scored_rounds

def cases(manifest):
    return [c for c in manifest['cases'] if c['stage']=='main' and c['budget_factor']==1.]


def run_case(case,world,output,calibration=None):
    from . import admission_experiments as runner
    original_methods,original_planner=runner.METHODS,runner.AdmissionPlanner
    runner.METHODS=('admission-pooled',);runner.AdmissionPlanner=ClassOnlyPlanner
    try:return runner.run_case(case,world,output,calibration)
    finally:runner.METHODS,runner.AdmissionPlanner=original_methods,original_planner


def check(folder,primary,manifest,output,partial=False):
    folder=Path(folder);primary=Path(primary);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];missing=[]
    for case in cases(manifest):
        path=folder/(case['name']+'.npz');main=primary/path.name
        if not path.exists() or not main.exists():missing.append(case['name']);continue
        with np.load(path,allow_pickle=False) as z,np.load(main,allow_pickle=False) as p:
            if p['methods'].tolist()!=manifest['methods'] or p['methods'][0]!='admission':raise ValueError('Primary method identity differs')
            if z['methods'].tolist()!=['admission-pooled'] or json.loads(str(z['case_json']))!=case:raise ValueError('Class-only evidence differs')
            if z['predictions'].shape!=(1,50,100) or z['resources'].shape!=(1,50,6):raise ValueError('Class-only shape differs')
            np.testing.assert_array_equal(z['truth'],p['truth']);np.testing.assert_array_equal(z['task_mask'],p['task_mask'])
            truth=z['truth'];mask=z['task_mask'];a=p['predictions'][0];b=z['predictions'][0]
            ar=p['resources'][0];br=z['resources'][0]
            if not np.isfinite(br).all() or np.any(br[:,0]<0) or np.any(br[:,0]>case['budget_override']+1e-7):raise ValueError('Invalid class-only payment')
            ma=metrics.requested_metrics(truth,a,mask);mb=metrics.requested_metrics(truth,b,mask)
            common=mask&np.isfinite(a)&np.isfinite(b);ea=(a-truth)[common];eb=(b-truth)[common]
            row=dict(case=case['name'],dataset=case['dataset'],seed=case['seed'],ours_TCR=ma['completion_percent'],control_TCR=mb['completion_percent'],
                TCR_delta_pp=ma['completion_percent']-mb['completion_percent'],common_tasks=int(common.sum()),
                MAE_change_percent=100*(np.abs(ea).mean()/np.abs(eb).mean()-1),
                RMSE_change_percent=100*(np.sqrt((ea*ea).mean())/np.sqrt((eb*eb).mean())-1))
            for name,col in [('payment',0),('physical',1)]:row[name+'_change_percent']=100*(ar[:,col].sum()/br[:,col].sum()-1)
            row['successes_per_payment_change_percent']=100*((ma['completion_percent']/ar[:,0].sum())/(mb['completion_percent']/br[:,0].sum())-1)
            rows.append(row)
    if missing and not partial:raise ValueError('Missing secondary class-only cases: '+str(missing))
    if not rows:raise ValueError('No class-only evidence')
    paired=pd.DataFrame(rows);paired.to_csv(output/'class_only_paired_cases.csv',index=False)
    columns=['TCR_delta_pp','MAE_change_percent','RMSE_change_percent','payment_change_percent','physical_change_percent','successes_per_payment_change_percent']
    result=[]
    if not missing:
        for dataset,group in [('all',paired)]+[(d,paired[paired.dataset==d]) for d in ('tdrive','foursquare_tky','brightkite')]:
            expected=3 if dataset=='all' else 1
            if not np.all(group.groupby('seed').size().to_numpy()==expected):raise ValueError('Unbalanced class-only strata')
            means=group.groupby('seed')[columns].agg(lambda x:np.mean(np.asarray(x,float)))
            if len(means)!=8:raise ValueError('Eight independent seeds required')
            for key in columns:
                v=means[key].to_numpy()
                if not np.isfinite(v).all():raise ValueError('Undefined secondary metric; retain all strata and resolve its interpretation before inference: '+key)
                mean=float(np.mean(v));radius=float(student.ppf(.975,7)*np.std(v,ddof=1)/np.sqrt(8))
                result.append(dict(dataset=dataset,metric=key,mean=mean,lower=mean-radius,upper=mean+radius,seeds=8))
    intervals=pd.DataFrame(result);intervals.to_csv(output/'class_only_intervals.csv',index=False)
    receipt=dict(complete=not missing,checked_cases=len(rows),missing=missing,
        role='Post-registration secondary state-granularity attribution; not part of the primary five-policy registration.',
        inference='Three fixed dataset strata within eachseed, then df7 paired Student intervals; no dropped undefined values.')
    (output/'class_only_check.json').write_text(json.dumps(receipt,indent=2))
    return paired,intervals,receipt


def plot(folder,primary,manifest,output,partial=False):
    paired,intervals,receipt=check(folder,primary,manifest,output,partial)
    if intervals.empty:raise ValueError('All24 class-only cases are needed for the interval figure')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':9,
        'axes.linewidth':.75,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
    from .recruitment_evidence import figure_style,finish_axes
    figure_style()
    fig,axes=plt.subplots(1,3,figsize=(7.15,1.9),layout='constrained')
    ds=['all','tdrive','foursquare_tky','brightkite'];labels=['All strata','T-Drive','Foursquare','Brightkite'];y=np.arange(4)
    for ax,key,title,xlabel in zip(axes,['TCR_delta_pp','MAE_change_percent','payment_change_percent'],
            ['(a) Task success','(b) Common-task precision','(c) Actual payment'],['Difference (pp)','MAE change (%)','Payment change (%)']):
        selected=intervals[intervals.metric==key].set_index('dataset').loc[ds]
        ax.errorbar(selected['mean'],y,xerr=np.vstack([selected['mean']-selected.lower,selected.upper-selected['mean']]),fmt='o',
            markersize=4,color='#A42B33',ecolor='#314F68',capsize=2,linewidth=1)
        ax.axvline(0,color='#666666',linewidth=.7,linestyle='--');ax.set_yticks(y,labels);ax.invert_yaxis()
        ax.set_title(title,pad=4);ax.set_xlabel(xlabel);finish_axes(ax)
    for suffix in ('pdf','svg'):fig.savefig(Path(output)/('class_only_attribution.'+suffix),bbox_inches='tight',pad_inches=.02)
    plt.close(fig)
    return receipt

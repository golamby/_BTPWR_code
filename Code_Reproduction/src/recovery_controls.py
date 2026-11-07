"""Same-collection recovery controls; acquisition and real feedback are frozen."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from . import trust,recovery,metrics
from .simulate import calibration_for

METHODS=('BTPWR','Direct TD','All-source KRR')


def cases(manifest):
    return [c for c in manifest['cases'] if c['stage']=='main' and c['budget_factor']==1. and c['seed']<=9703]


def run_case(world,case,primary_source,output,calibration=None):
    w=world;output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cal=calibration_for(w,output) if calibration is None else calibration
    heads=[recovery.initialize(w['public_reference'],w['native_offset'],w['native_scale'],cal) for _ in range(2)]
    tc=trust.Config(gamma=case['gamma']);predictions=[];observed=[];credible=[];diagnostics=[]
    with np.load(primary_source,allow_pickle=False) as z:
        i=z['methods'].tolist().index('admission');selected=z['admission__selected'];pre=z['admission__pre']
        expected=z['predictions'][i];history=z['reputations'][i]
        np.testing.assert_array_equal(pre,history[:-1]);np.testing.assert_array_equal(history[0],w['initial_reputation'])
        for t,pick in enumerate(selected):
            raw=np.full(w['reports_check'][t].shape,np.nan);raw[pick]=w['reports_check'][t,pick]
            raw[trust.classify(pre[t],tc)==0]=np.nan
            fb=trust.feedback(raw,pre[t],tc)
            np.testing.assert_array_equal(fb['next_reputation'],history[t+1])
            actual=recovery.decode_actual(fb['consensus'],heads[0])
            ours,cd=recovery.complete(fb['credible_consensus'],actual,heads[0])
            all_source,ad=recovery.complete(fb['consensus'],actual,heads[1])
            np.testing.assert_allclose(ours,expected[t],rtol=0,atol=1e-10,equal_nan=True)
            seen=np.isfinite(actual);C=np.isfinite(fb['credible_consensus']);task=w['task_mask'][t]
            predictions.append([ours,actual,all_source]);observed.append(seen);credible.append(C)
            diagnostics.append([int(C.sum()),int(seen.sum()),int(cd['recovery_mask'].sum()),int(ad['recovery_mask'].sum()),
                int((seen&np.isfinite(ours)&(ours!=actual)).sum()),int((seen&np.isfinite(all_source)&(all_source!=actual)).sum()),
                int((task&~seen&np.isfinite(ours)&np.isfinite(all_source)).sum())])
    payload=dict(methods=np.asarray(METHODS),predictions=np.asarray(predictions).transpose(1,0,2),
        truth=w['truth_native_private'].T,task_mask=w['task_mask'],diagnostics=np.asarray(diagnostics),case_json=np.array(json.dumps(case)))
    np.savez_compressed(output/(case['name']+'.npz'),**payload)
    return payload


def check(folder,manifest,output,partial=False):
    folder=Path(folder);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];paired=[];missing=[]
    for case in cases(manifest):
        path=folder/(case['name']+'.npz')
        if not path.exists():missing.append(case['name']);continue
        with np.load(path,allow_pickle=False) as z:
            if z['methods'].tolist()!=list(METHODS) or json.loads(str(z['case_json']))!=case:raise ValueError('Recovery-control identity differs')
            truth=z['truth'];task=z['task_mask'];pred=z['predictions'];actual=np.isfinite(pred[1]);diag=z['diagnostics']
            if pred.shape!=(3,50,100) or diag.shape!=(50,7):raise ValueError('Incomplete recovery-control evidence')
            for i,method in enumerate(METHODS):
                m=metrics.requested_metrics(truth,pred[i],task)
                rows.append(dict(case=case['name'],dataset=case['dataset'],seed=case['seed'],method=method,
                    TCR=m['completion_percent'],AR=m['answered_rate_percent'],own_MAE=m['all_answered_MAE'],own_RMSE=m['all_answered_RMSE'],
                    credible_anchor_cells=int(diag[:,0].sum()),eligible_consensus_cells=int(diag[:,1].sum()),
                    credible_projection_cells=int(diag[:,4].sum()),all_source_projection_cells=int(diag[:,5].sum())))
            for i,method in enumerate(METHODS[1:],1):
                common=task&np.isfinite(pred[0])&np.isfinite(pred[i])
                for label,domain in [('common answered',common),('common originally missing',common&~actual)]:
                    a=(pred[0]-truth)[domain];b=(pred[i]-truth)[domain]
                    paired.append(dict(case=case['name'],dataset=case['dataset'],seed=case['seed'],comparison=method,domain=label,
                        tasks=int(domain.sum()),ours_MAE=float(np.abs(a).mean()) if len(a) else np.nan,
                        control_MAE=float(np.abs(b).mean()) if len(b) else np.nan,
                        ours_RMSE=float(np.sqrt((a*a).mean())) if len(a) else np.nan,control_RMSE=float(np.sqrt((b*b).mean())) if len(b) else np.nan,
                        ours_only_answers=int((task&np.isfinite(pred[0])&~np.isfinite(pred[i])).sum()),
                        control_only_answers=int((task&~np.isfinite(pred[0])&np.isfinite(pred[i])).sum())))
    if missing and not partial:raise ValueError('Missing recovery controls: '+str(missing))
    if not rows:raise ValueError('No recovery-control evidence')
    table=pd.DataFrame(rows);pairs=pd.DataFrame(paired)
    table.to_csv(output/'recovery_control_cases.csv',index=False);pairs.to_csv(output/'recovery_control_common_errors.csv',index=False)
    receipt=dict(complete=not missing,cases=len(rows)//3,missing=missing,
        role='Same paid reports, selections and true feedback. Direct TD disables both recovery and projection. All-source KRR admits every non-M local consensus as a query anchor.',
        interpretation='Anchor dimensions, answer support and projection branches can change; this is not a fixed-dimension pure poisoning-rate intervention.')
    (output/'recovery_control_check.json').write_text(json.dumps(receipt,indent=2))
    return table,pairs,receipt


def plot(folder,manifest,output,partial=False):
    table,pairs,receipt=check(folder,manifest,output,partial)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    from .recruitment_evidence import figure_style,finish_axes
    figure_style()
    datasets=['tdrive','foursquare_tky','brightkite'];labels=['T-Drive','Foursquare','Brightkite']
    colors=['#8C2633','#6E6E6E','#B08A4F']
    fig,axes=plt.subplots(1,3,figsize=(7.15,2.0),layout='constrained');x=np.arange(3);width=.24
    for k,(metric,title,ylabel) in enumerate([('TCR',r'$\mathbf{(a)}$ Task completion','Successful tasks (%)'),('AR',r'$\mathbf{(b)}$ Answer support','Answered tasks (%)')]):
        for i,method in enumerate(METHODS):
            values=[table[(table.dataset==d)&(table.method==method)][metric].mean() for d in datasets]
            axes[k].bar(x+(i-1)*width,values,width,color=colors[i],edgecolor='#2A2A2A',linewidth=.3,label=method)
        axes[k].set_xticks(x,labels,fontsize=7.2);axes[k].set_title(title,pad=4);axes[k].set_ylabel(ylabel);axes[k].set_xlabel('Mobility source',labelpad=2.5)
        axes[k].set_ylim(0,100 if metric=='TCR' else 110)
    for i,(column,label) in enumerate([('ours_RMSE','Credible-source KRR'),('control_RMSE','All-source KRR')]):
        values=[]
        for d in datasets:
            subset=pairs[(pairs.dataset==d)&(pairs.comparison=='All-source KRR')&(pairs.domain=='common originally missing')]
            if np.any(subset.tasks==0):raise ValueError('Missing common imputation support; do not silently omit a case')
            values.append(subset[column].mean())
        axes[2].plot(x,values,marker=['o','s'][i],markersize=3.6,linewidth=1.15,color=[colors[0],colors[2]][i],markerfacecolor=colors[0] if i==0 else 'white',markeredgewidth=.9,label=label)
    axes[2].set_xticks(x,labels,fontsize=7.2);axes[2].set_title(r'$\mathbf{(c)}$ Missing-task precision',pad=4);axes[2].set_ylabel('RMSE (RH pp)');axes[2].set_xlabel('Mobility source',labelpad=2.5)
    for ax in axes:finish_axes(ax)
    handles,labels_=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels_,loc='outside upper center',ncol=3,fontsize=7.4,columnspacing=.8,handlelength=1.0,frameon=False)
    for suffix in ('pdf','svg'):fig.savefig(Path(output)/('recovery_controls.'+suffix),bbox_inches='tight',pad_inches=.02)
    plt.close(fig)
    return receipt

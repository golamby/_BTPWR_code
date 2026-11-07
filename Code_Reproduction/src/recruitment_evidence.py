"""Independent numeric checks and compact figures from anonymous evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t as student
from . import metrics

METRICS=('TCR_delta_pp','payment_change_percent','physical_change_percent',
         'successes_per_1000_payment_change_percent','MAE_change_percent','RMSE_change_percent')


def cited(label):
    path=Path(__file__).resolve().parents[1]/'plot_citations.json'
    mapping=json.loads(path.read_text()) if path.exists() else {}
    number=mapping.get(label)
    if number is None:return label
    if not isinstance(number,int) or number<1:raise ValueError('Citation must be a positive bibliography number')
    return f'{label} [{number}]'


def check(folder,manifest,output,partial=False):
    folder=Path(folder);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];pairs=[];missing=[]
    for case in manifest['cases']:
        path=folder/(case['name']+'.npz')
        if not path.exists():missing.append(case['name']);continue
        with np.load(path,allow_pickle=False) as z:
            if z['methods'].tolist()!=manifest['methods']:raise ValueError('Methods differ: '+str(path))
            if json.loads(str(z['case_json']))!=case:raise ValueError('Case differs: '+str(path))
            truth=z['truth'];pred=z['predictions'];mask=z['task_mask'];res=z['resources']
            D=12 if case['stage']=='hourly' else 100
            if truth.shape!=(50,D) or pred.shape!=(5,50,D) or mask.shape!=(50,D) or res.shape!=(5,50,6):raise ValueError('Shape mismatch: '+str(path))
            if not np.isfinite(truth[mask]).all() or not np.isfinite(res).all():raise ValueError('Nonfinite evidence: '+str(path))
            workers=z['worker_counts']
            if workers.shape!=(5,50,9) or np.any(workers<0) or not np.all(workers[:,:,:6].sum(2)==case['n']):raise ValueError('Invalid aggregate worker audit: '+str(path))
            if np.any(res[:,:,0]<0) or np.any(res[:,:,0]>case['budget_override']+1e-7):raise ValueError('Payment exceeds cap: '+str(path))
            local=[]
            for i,method in enumerate(manifest['methods']):
                m=metrics.requested_metrics(truth,pred[i],mask)
                row={k:case.get(k,'') for k in ('name','stage','dataset','seed','budget_factor','requested_start')}
                row.update(method=method,TCR=m['completion_percent'],answered_rate=m['answered_rate_percent'],
                    own_MAE=m['all_answered_MAE'],own_RMSE=m['all_answered_RMSE'],payment=float(res[i,:,0].sum()),
                    physical=float(res[i,:,1].sum()),anchors=float(res[i,:,2].sum()))
                row['successes_per_1000_payment']=row['TCR']/100*mask.sum()/row['payment']*1000
                honest=int(np.floor(case['n']*case['H']));malicious=case['n']-honest
                row['final_balanced_recognition_percent']=50*(workers[i,-1,0]/honest+workers[i,-1,1]/malicious)
                row['new_C_paid_worker_rounds']=int(workers[i,:,8].sum())
                rows.append(row);local.append(row)
            for i,method in enumerate(manifest['methods'][1:],1):
                common=mask&np.isfinite(pred[0])&np.isfinite(pred[i]);x=(pred[0]-truth)[common];y=(pred[i]-truth)[common]
                row={k:local[0][k] for k in ('name','stage','dataset','seed','budget_factor','requested_start')}
                row.update(comparison=method,common_tasks=int(common.sum()),TCR_delta_pp=local[0]['TCR']-local[i]['TCR'],
                    MAE_change_percent=100*(np.abs(x).mean()/np.abs(y).mean()-1),
                    RMSE_change_percent=100*(np.sqrt((x*x).mean())/np.sqrt((y*y).mean())-1))
                for key in ('payment','physical','successes_per_1000_payment'):
                    row[key+'_change_percent']=100*(local[0][key]/local[i][key]-1)
                pairs.append(row)
    if missing and not partial:raise ValueError(f'Missing {len(missing)} registered cases; first: {missing[0]}')
    if not rows:raise ValueError('No matching evidence files found')
    allrows=pd.DataFrame(rows);paired=pd.DataFrame(pairs);summary=[]
    allrows.to_csv(output/'verified_cases.csv',index=False);paired.to_csv(output/'verified_pairs.csv',index=False)
    for stage,strata,nseeds in (('main',6,8),('hourly',4,3)):
        for comparator in manifest['methods'][1:]:
            f=paired[(paired.stage==stage)&(paired.comparison==comparator)]
            counts=f.groupby('seed').size();good=counts[counts==strata].index
            if len(good)!=nseeds:continue
            values=f[f.seed.isin(good)].groupby('seed')[list(METRICS)].agg(lambda x:np.mean(np.asarray(x,float)))
            if len(values)!=nseeds:continue
            for metric in METRICS:
                v=values[metric].to_numpy();mean=float(v.mean());valid=np.isfinite(v).all()
                radius=float(student.ppf(.975,nseeds-1)*v.std(ddof=1)/np.sqrt(nseeds)) if valid else np.nan
                summary.append(dict(stage=stage,comparison=comparator,metric=metric,mean=mean,lower=mean-radius,upper=mean+radius,seeds=nseeds))
    intervals=pd.DataFrame(summary);intervals.to_csv(output/'verified_intervals.csv',index=False)
    receipt=dict(checked_cases=len(rows)//5,registered_cases=len(manifest['cases']),complete=not missing,missing=missing,
                 privacy='Evidence contains no worker identifiers, routes, reports, or per-worker reputations.',
                 inference='paired seed means over all registered strata; df7 main and df2 hourly; no NaN omission')
    (output/'check.json').write_text(json.dumps(receipt,indent=2))
    return allrows,paired,intervals,receipt


def figure_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':8.0,'axes.titlesize':8.6,'axes.labelsize':8.0,
        'axes.linewidth':.7,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'savefig.facecolor':'white','figure.facecolor':'white',
        'axes.facecolor':'white','axes.axisbelow':True,'legend.frameon':False})


def figure_labels():
    return ['BTPWR','BTPWR-NP','C-seeded','Mapped',cited('ETBP-priority').replace('ETBP-priority','ETBP-P')]


COLORS=['#8C2633','#B08A4F','#5F7D8C','#2F5D7C','#5B7F6A']
MARKERS=['o','s','^','D','v']
LINES=['solid',(0,(4,2)),(0,(4,1.5,1,1.5)),(0,(1.2,1.7)),(0,(6,2))]


def finish_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True);spine.set_color('#333333');spine.set_linewidth(.7)
    ax.tick_params(direction='in',length=2.8,width=.6,pad=2.6)
    ax.yaxis.grid(True,color='#DEDEDE',linewidth=.5,linestyle=(0,(2,2)))
    ax.set_axisbelow(True)


def save_figure(fig,output,name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();canvas=fig.bbox
    outside=[]
    from matplotlib.text import Text
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():continue
        box=text.get_window_extent(renderer)
        if box.x0<canvas.x0-.5 or box.y0<canvas.y0-.5 or box.x1>canvas.x1+.5 or box.y1>canvas.y1+.5:
            outside.append(text.get_text())
    if outside:raise ValueError('Figure text outside canvas: '+repr(outside))
    for suffix in ('pdf','svg','png'):
        fig.savefig(Path(output)/(name+'.'+suffix),dpi=260)
    (Path(output)/(name+'.layout.json')).write_text(json.dumps({'outside_text':outside,
        'width_inches':fig.get_figwidth(),'height_inches':fig.get_figheight(),
        'abbreviations':{'BTPWR-NP':'BTPWR with promotion disabled only in the forecast',
        'C-seeded':'Trusted-seeded control','Mapped':'Mapped recruitment reference',
        'ETBP-P':'Resource-matched ETBP recruitment priority'},
        'intervals':'Two-sided 95% Student intervals; paired seed means in the MAE panel'},indent=2))


def plot(folder,manifest,output,partial=False):
    table,pairs,intervals,receipt=check(folder,manifest,output,partial)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure_style()
    methods=manifest['methods'];labels=figure_labels()
    data=table[table.stage=='main'].copy()
    if data.empty:raise ValueError('Main evidence is not yet available')
    fig,ax=plt.subplots(1,3,figsize=(7.16,2.22))
    fig.subplots_adjust(left=.060,right=.994,bottom=.260,top=.775,wspace=.380)
    groups=[('tdrive',1.),('tdrive',.75),('foursquare_tky',1.),('foursquare_tky',.75),('brightkite',1.),('brightkite',.75)]
    x=np.arange(len(groups));width=.146
    for i,(metric,title,ylabel) in enumerate([('TCR',r'$\mathbf{(a)}$ Task success','Successful tasks (%)'),
            ('payment',r'$\mathbf{(b)}$ Payment commitment','Payment ($10^3$ units)'),
            ('successes_per_1000_payment',r'$\mathbf{(c)}$ Service efficiency','Successes per\n$10^3$ payment units')]):
        for k,method in enumerate(methods):
            means=[];errors=[]
            for ds,b in groups:
                v=data[(data.dataset==ds)&(data.budget_factor==b)&(data.method==method)][metric].to_numpy()
                if metric=='payment':v=v/1000
                means.append(v.mean() if len(v) else np.nan)
                errors.append(student.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)) if len(v)>1 else 0.)
            ax[i].bar(x+(k-2)*width,means,width,color=COLORS[k],edgecolor='#2A2A2A',linewidth=.3,
                yerr=errors,capsize=1.0,error_kw={'elinewidth':.6,'capthick':.6},label=labels[k],zorder=2)
        ax[i].set_title(title,pad=5);ax[i].set_ylabel(ylabel,labelpad=3)
        ax[i].set_xticks(x,['TD\n100%','TD\n75%','FS\n100%','FS\n75%','BK\n100%','BK\n75%'],fontsize=7.2)
        ax[i].set_xlabel('Dataset / cap',labelpad=2.5)
        ax[i].set_xlim(-.55,5.55);ax[i].set_ylim(bottom=0)
        ax[i].margins(y=.12);finish_axes(ax[i])
    ax[0].set_ylim(0,88);ax[0].set_yticks([0,30,60,90])
    ax[1].set_ylim(0,27);ax[1].set_yticks([0,9,18,27])
    ax[2].set_ylim(0,140);ax[2].set_yticks([0,45,90,135])
    fig.legend(*ax[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.52,.995),
        ncol=5,fontsize=7.6,columnspacing=.7,handlelength=1.05,handletextpad=.4,borderaxespad=0)
    save_figure(fig,output,'recruitment_comparison')
    plt.close(fig)
    if not table[table.stage=='hourly'].empty:
        plot_hourly(table,intervals,manifest,output)
    return receipt


def plot_hourly(table,intervals,manifest,output):
    import matplotlib.pyplot as plt
    data=table[table.stage=='hourly'];ci=intervals[(intervals.stage=='hourly')&(intervals.metric=='MAE_change_percent')] if not intervals.empty else intervals
    if ci.empty:return
    figure_style();methods=manifest['methods'];labels=figure_labels()
    windows=['2014-03-01','2014-06-01','2014-09-01','2014-12-01'];x=np.arange(4)
    fig=plt.figure(figsize=(3.5,3.3))
    axes=[fig.add_axes([.175,.585,.785,.255]),fig.add_axes([.295,.125,.675,.275])]
    for i,method in enumerate(methods):
        values=[]
        for window in windows:
            v=data[(data.method==method)&(data.requested_start==window)].TCR.to_numpy()
            if len(v)!=3:raise ValueError('Three seeds per registered hourly window required')
            values.append(float(np.mean(v)))
        axes[0].plot(x,values,marker=MARKERS[i],markersize=4,linewidth=1.35 if i==0 else 1.05,
            linestyle=LINES[i],color=COLORS[i],markeredgewidth=.8,
            markerfacecolor=COLORS[i] if i==0 else 'white',label=labels[i])
    axes[0].set_xticks(x,['Mar.','Jun.','Sep.','Dec.']);axes[0].set_xlabel('Starting month (2014)',labelpad=3)
    axes[0].set_ylabel('Successful tasks (%)',labelpad=4)
    axes[0].set_title('(a) Hourly PM$_{2.5}$ service',pad=6)
    axes[0].set_xlim(-.15,3.15);axes[0].set_ylim(27,57);axes[0].set_yticks([30,40,50])
    for i,method in enumerate(methods[1:]):
        row=ci[ci.comparison==method]
        if len(row)!=1:raise ValueError('Missing paired hourly confidence interval')
        r=row.iloc[0]
        axes[1].errorbar(r['mean'],i,xerr=np.array([[r['mean']-r.lower],[r.upper-r['mean']]]),fmt=MARKERS[i+1],
            color=COLORS[i+1],markerfacecolor='white',markeredgewidth=1,markersize=4.5,
            capsize=2.5,linewidth=1.1,capthick=.8,zorder=3)
    axes[1].axvline(0,color='#777777',linestyle=(0,(3,2)),linewidth=.65,zorder=1)
    axes[1].set_yticks(np.arange(4),labels[1:],fontsize=8);axes[1].invert_yaxis()
    axes[1].set_xlabel('Relative MAE change (%)',labelpad=4)
    axes[1].set_title('(b) Pairwise MAE comparison',pad=6)
    axes[1].set_xlim(-44,44);axes[1].set_xticks([-40,-20,0,20,40])
    axes[1].set_ylim(3.45,-.45)
    for a in axes:finish_axes(a)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.52,.998),
        ncol=5,fontsize=6.8,columnspacing=.55,handlelength=1.0,handletextpad=.3,labelspacing=.3,borderaxespad=0)
    save_figure(fig,output,'hourly_recruitment')
    plt.close(fig)


"""Publication figures from validated Q36 measurements, never placeholders."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

DATASETS=['tdrive','foursquare_tky','brightkite']
TITLES=['T-Drive','Foursquare-TKY','Brightkite']
CORE=['BTPWR','BPP','VLR-BPP','ETBP-TD','REAPP']
COLORS={'BTPWR':'#B33940','BPP':'#456E95','VLR-BPP':'#8D719E','ETBP-TD':'#378373','REAPP':'#BA873D',
        'Random':'#777777','Greedy-C/R':'#576065','C-seeded':'#628E95'}
MARKERS=['s','o','^','D','v','P','X']
STYLES=['-','--','-.',':','--','-.',':']


def style():
    for f in ['C:/Windows/Fonts/times.ttf','C:/Windows/Fonts/timesbd.ttf','C:/Windows/Fonts/timesi.ttf']:
        if Path(f).exists():font_manager.fontManager.addfont(f)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                         'mathtext.fontset':'stix','font.size':9,'axes.labelsize':9,'axes.titlesize':9,
                         'axes.linewidth':.8,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})


def axes_format(ax):
    ax.spines[['top','right']].set_visible(True)
    ax.grid(axis='y',color='#D4D4D4',linewidth=.45,alpha=.65)
    ax.tick_params(direction='in',length=3,pad=3)
    ax.set_axisbelow(True)


def default_rows(df, varying=None):
    defaults={'n':80,'H':.7,'C':.1,'eps1':2.1,'eps2':2.7,'budget_override':450,'field':'legacy'}
    mask=np.ones(len(df),bool)
    for k,v in defaults.items():
        if k==varying:continue
        mask &= df[k].eq(v) if isinstance(v,str) else np.isclose(df[k],v)
    return df[mask]


def save(fig,name,output,manifest,description,source):
    handles,labels=[],[]
    for ax in fig.axes:
        h,l=ax.get_legend_handles_labels()
        for a,b in zip(h,l):
            if b not in labels:handles.append(a);labels.append(b)
    if handles:fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.025),ncol=min(5,len(labels)),frameon=False,handlelength=2,columnspacing=1.1,fontsize=8)
    fig.subplots_adjust(left=.075,right=.99,bottom=.11,top=.86,hspace=.48,wspace=.35)
    files=[]
    for ext in ['pdf','svg','png']:
        path=output/(name+'.'+ext);fig.savefig(path,dpi=220,bbox_inches='tight',pad_inches=.03)
        files.append(dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    plt.close(fig)
    manifest.append(dict(id=name,description=description,inputs=source,files=files))


def process(rounds,output,manifest):
    r=default_rows(rounds[rounds.group=='primary'])
    if r.empty:return
    if not all(len(r[(r.dataset==d)&(r.method=='BTPWR')].groupby('round'))==50 for d in DATASETS):return
    fig,axes=plt.subplots(2,3,figsize=(7.2,3.8))
    drawn=False
    for col,dataset in enumerate(DATASETS):
        dd=r[r.dataset==dataset]
        for row,metric in enumerate(['TCR','macro_F1']):
            ax=axes[row,col]
            for i,m in enumerate(CORE):
                v=dd[dd.method==m]
                if metric not in v or not v[metric].notna().any():continue
                g=v.groupby('round')[metric].mean().dropna()
                if g.empty:continue
                ax.plot(g.index,g.values,color=COLORS[m],marker=MARKERS[i],linestyle=STYLES[i],
                        markevery=[0,9,19,29,39,49],markersize=3.6,linewidth=1.15,
                        markerfacecolor=COLORS[m] if m=='BTPWR' else 'white',label=m)
                drawn=True
            ax.set_title(f'({chr(97+row*3+col)}) {TITLES[col]}');ax.set_xlabel('Round')
            ax.set_ylabel('Cumulative TCR (%)' if metric=='TCR' else 'Native macro F1 (%)')
            ax.set_xticks([1,10,20,30,40,50]);axes_format(ax)
    if drawn:save(fig,'E1_process',output,manifest,'50-round cumulative completion and native identification; inapplicable classifiers omitted',['round_metrics.csv'])
    else:plt.close(fig)


def sweep(summary,pairs,key,name,xlabel,output,manifest):
    d=default_rows(summary[summary.group=='sensitivity'],key)
    p=default_rows(pairs[pairs.group=='sensitivity'],key) if not pairs.empty else pairs
    if d.empty:return
    if not all(d[(d.dataset==ds)&(d.method=='BTPWR')][key].nunique()>=2 for ds in DATASETS):return
    if key!='H' and (p.empty or not all(p[(p.dataset==ds)&p.comparison.isin(['BPP','VLR-BPP'])][key].nunique()>=2 for ds in DATASETS)):return
    f1=(key=='H');fig,axes=plt.subplots(2,3,figsize=(7.2,3.8));drawn=False
    methods=CORE if f1 else ['BTPWR','BPP','VLR-BPP']
    for col,dataset in enumerate(DATASETS):
        local=d[d.dataset==dataset]
        for i,m in enumerate(methods):
            values=local[local.method==m]
            for row in [0,1]:
                ax=axes[row,col]
                if row==0 or f1:
                    metric='TCR' if row==0 else 'macro_F1'
                    if metric not in values:continue
                    g=values.groupby(key)[metric].mean().dropna()
                elif m=='BTPWR':continue
                else:
                    pp=p[(p.dataset==dataset)&(p.comparison==m)]
                    g=pp.groupby(key).RMSE_gap_baseline_minus_ours.mean().dropna()
                if len(g)<2:continue
                ax.plot(g.index,g.values,color=COLORS[m],marker=MARKERS[i],linestyle=STYLES[i],
                        markersize=4,linewidth=1.15,markerfacecolor=COLORS[m] if m=='BTPWR' else 'white',label=m)
                drawn=True
        for row in [0,1]:
            ax=axes[row,col];ax.set_title(f'({chr(97+row*3+col)}) {TITLES[col]}');ax.set_xlabel(xlabel)
            ax.set_ylabel('TCR (%)' if row==0 else 'Native macro F1 (%)' if f1 else 'RMSE gap: baseline − BTPWR')
            if row==1 and not f1:ax.axhline(0,color='#555555',lw=.8,zorder=1)
            axes_format(ax)
    if drawn:save(fig,name,output,manifest,'One-factor comparison; lower RMSE panels use each comparator’s pairwise common domain, positive gap favors BTPWR',['campaign_metrics.csv','pairwise_common_errors.csv'])
    else:plt.close(fig)


def frontiers(summary,output,manifest):
    d=default_rows(summary[summary.group=='sensitivity'],'budget_override')
    if d.empty:return
    if not all(d[(d.dataset==ds)&(d.method=='BTPWR')].budget_override.nunique()>=2 for ds in DATASETS):return
    methods=['BTPWR','BPP','VLR-BPP','Random','Greedy-C/R'];fig,axes=plt.subplots(2,3,figsize=(7.2,3.8));drawn=False
    for col,dataset in enumerate(DATASETS):
        local=d[d.dataset==dataset]
        for row,resource in enumerate(['settled','physical']):
            ax=axes[row,col]
            for i,m in enumerate(methods):
                g=local[local.method==m].groupby('budget_override')[[resource,'TCR']].mean().dropna()
                if len(g)<2:continue
                # Retain all registered operating points in budget order, even
                # when dominated. This is a tradeoff curve, not a fitted frontier.
                ax.plot(g[resource]/1000,g.TCR,color=COLORS[m],marker=MARKERS[i],linestyle=STYLES[i],markersize=3.6,linewidth=1.1,label=m,markerfacecolor=COLORS[m] if m=='BTPWR' else 'white')
                drawn=True
            ax.set_title(f'({chr(97+row*3+col)}) {TITLES[col]}');ax.set_ylabel('TCR (%)')
            ax.set_xlabel('Settled payment (thousands)' if resource=='settled' else 'Physical cost (thousands)');axes_format(ax)
    if drawn:save(fig,'E6_resource_tradeoffs',output,manifest,'Quality versus actual resources across budget points; all points retained, no interpolated matched-quality claim',['campaign_metrics.csv'])
    else:plt.close(fig)


def ablations(summary,output,manifest):
    d=default_rows(summary[summary.group=='primary'])
    groups=[('E7_recruitment_ablation',['BTPWR','Recruit-NoBalance','Recruit-NoOpportunity','Recruit-Neither'],['Full','No balance','No opportunity','Neither']),
            ('E8_feedback_ablation',['BTPWR','Feedback-NoCount','Feedback-NoSecondary','Feedback-Neither'],['Full','No count','No recovery','Neither'])]
    for name,methods,labels in groups:
        if not all(m in set(d[d.dataset==ds].method) for ds in DATASETS for m in methods):continue
        fig,axes=plt.subplots(2,3,figsize=(7.2,3.8));drawn=False
        for col,dataset in enumerate(DATASETS):
            local=d[d.dataset==dataset]
            for row,metric in enumerate(['TCR','macro_F1']):
                ax=axes[row,col];means=local.groupby('method')[metric].mean().reindex(methods)
                if means.notna().any():
                    bars=ax.bar(np.arange(4),means,color=['#B33940','#698BA4','#6D9D89','#AF956E'],width=.68,edgecolor='white',linewidth=.7)
                    for bar,hatch in zip(bars,['','//','\\\\','xx']):bar.set_hatch(hatch)
                    drawn=True
                ax.set_xticks(np.arange(4));ax.set_xticklabels(labels,rotation=24,ha='right',fontsize=7.2)
                ax.set_ylabel('TCR (%)' if metric=='TCR' else 'Native macro F1 (%)');ax.set_title(f'({chr(97+row*3+col)}) {TITLES[col]}');axes_format(ax)
        if drawn:save(fig,name,output,manifest,'Four-configuration core factorial; full numerical trajectories share paired potential worlds',['campaign_metrics.csv'])
        else:plt.close(fig)


def recognition_resources(rounds,output,manifest):
    """Whole measured trajectories, avoiding a post-hoc chosen F1 threshold."""
    r=default_rows(rounds[rounds.group=='primary'])
    methods=['BTPWR','ETBP-TD','REAPP']
    if not all(len(r[(r.dataset==ds)&(r.method==m)].groupby('round'))==50 for ds in DATASETS for m in methods):return
    fig,axes=plt.subplots(2,3,figsize=(7.2,3.8))
    for col,ds in enumerate(DATASETS):
        for row,resource in enumerate(['physical','selected_worker_rounds']):
            ax=axes[row,col]
            for i,m in enumerate(methods):
                g=r[(r.dataset==ds)&(r.method==m)].groupby('round')[[resource,'macro_F1']].mean()
                ax.plot(g[resource]/1000,g.macro_F1,color=COLORS[m],marker=MARKERS[i],linestyle=STYLES[i],
                        markevery=[0,9,19,29,39,49],markersize=3.6,linewidth=1.15,
                        markerfacecolor=COLORS[m] if m=='BTPWR' else 'white',label=m)
            ax.set_title(f'({chr(97+row*3+col)}) {TITLES[col]}');ax.set_ylabel('Native macro F1 (%)')
            ax.set_xlabel(r'Physical cost ($10^3$)' if row==0 else r'Worker-rounds ($10^3$)');axes_format(ax)
    save(fig,'S_identification_resources',output,manifest,
         'Native identification versus acquisition resources over all50measuredrounds; same worlds, native quota/payment scopes retained; no post-hoc threshold-cost inference',
         ['round_metrics.csv'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    source=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);style()
    summary=pd.read_csv(source/'campaign_metrics.csv');rounds=pd.read_csv(source/'round_metrics.csv')
    try:pairs=pd.read_csv(source/'pairwise_common_errors.csv')
    except pd.errors.EmptyDataError:pairs=pd.DataFrame()
    manifest=[]
    process(rounds,out,manifest)
    for key,name,label in [('H','E2_honest_share','True honest share'),('budget_override','E3_budget','Per-round budget'),('n','E4_workforce','Workers'),('eps1','E5_location_privacy',r'Location privacy $\epsilon_1$')]:
        sweep(summary,pairs,key,name,label,out,manifest)
    frontiers(summary,out,manifest);ablations(summary,out,manifest);recognition_resources(rounds,out,manifest)
    result=dict(figures=manifest,plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.csv')},
                error_bars=False,uncertainty='paired seed intervals in CSV/tables',
                accuracy_domain='Pairwise RMSE gaps avoid pretending BTPWR has one common domain across every comparator')
    (out/'figure_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(figures=[f['id'] for f in manifest])))


if __name__=='__main__':main()

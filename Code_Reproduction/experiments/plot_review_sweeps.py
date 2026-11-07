"""Publication figures from completed, verified sweep tables only."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METHODS=['BTPWR','BTPWR-NP','C-seeded','Mapped','ETBP-P','BPQA numeric','Random','Greedy-C/R']
COLORS=['#9b2636','#b78439','#497080','#344f8c','#538264','#73548a','#8c6e57','#428c91']
MARKERS=['o','s','^','D','v','P','X','<']
STYLES=['-','--','-.',':',(0,(5,2,1,2)),(0,(2,1)),(0,(4,1,1,1)),(0,(3,1,1,1,1,1))]
AXES={'workers':('Workers',False),'honest_fraction':('Honest workers (%)',True),
      'certified_fraction':('Initially certified workers (%)',True),
      'budget':('Per-round payment cap',False),'location_epsilon':(r'Location privacy $\epsilon_1$',False)}


def setup():
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':7.3,'axes.labelsize':7.1,'axes.titlesize':7.5,
        'xtick.labelsize':6.6,'ytick.labelsize':6.6,'axes.linewidth':.65,'pdf.fonttype':42,
        'ps.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white'})


def curve(ax, df, axis, metric):
    data=df[df['axis']==axis]
    if axis=='location_epsilon': data=data[data['method']!='BPQA numeric']
    for k,method in enumerate(METHODS):
        part=data[data['method']==method]
        if part.empty: continue
        per_seed=part.groupby(['x','seed'],as_index=False)[metric].mean()
        groups=per_seed.groupby('x')[metric]
        x=np.array(sorted(groups.groups),float)
        values=[groups.get_group(v).dropna().to_numpy(float) for v in x]
        mean=np.array([a.mean() if len(a) else np.nan for a in values])
        se=np.array([t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else np.nan for a in values])
        plotted=100*x if AXES[axis][1] else x
        ax.plot(plotted,mean,color=COLORS[k],marker=MARKERS[k],linestyle=STYLES[k],
                linewidth=1.15 if k==0 else .9,markersize=3.5,markerfacecolor=COLORS[k] if k==0 else 'white',
                markeredgewidth=.7,zorder=5 if k==0 else 3)
        if k==0: ax.fill_between(plotted,mean-se,mean+se,color=COLORS[k],alpha=.12,linewidth=0,zorder=1)
    ax.set_xlabel(AXES[axis][0]); ax.grid(axis='y',linestyle=(0,(2,2)),color='#d8d8d8',linewidth=.45)
    ax.tick_params(direction='in',length=2.4,width=.6,pad=2)
    for spine in ax.spines.values(): spine.set_linewidth(.65)


def legend(fig):
    handles=[Line2D([0],[0],color=COLORS[k],marker=MARKERS[k],linestyle=STYLES[k],
                    markersize=4,linewidth=1,markerfacecolor=COLORS[k] if k==0 else 'white')
             for k in range(len(METHODS))]
    fig.legend(handles,METHODS,loc='upper center',ncol=4,frameon=False,
               bbox_to_anchor=(.5,.995),columnspacing=1.1,handlelength=2.,fontsize=7.1)


def grid_figure(df, columns, metrics, labels, output, stem):
    fig,axes=plt.subplots(len(metrics),len(columns),figsize=(7.15,2.12*len(metrics)+.48),squeeze=False)
    fig.subplots_adjust(left=.075,right=.985,bottom=.075,top=.88,hspace=.47,wspace=.33)
    panel=0
    for row,(metric,label) in enumerate(zip(metrics,labels)):
        for col,axis in enumerate(columns):
            ax=axes[row,col];curve(ax,df,axis,metric)
            ax.set_ylabel(label);ax.set_title('('+chr(97+panel)+')',loc='left',pad=2);panel+=1
            if metric.endswith('_delta_vs_btpwr'): ax.axhline(0,color='#555555',linewidth=.55,zorder=0)
    legend(fig)
    for ext in ('pdf','svg','png'):
        fig.savefig(output/(stem+'.'+ext),dpi=260,facecolor='white')
    plt.close(fig)


def round_figure(df, output):
    fig,axes=plt.subplots(2,2,figsize=(7.15,4.9))
    fig.subplots_adjust(left=.09,right=.98,bottom=.09,top=.86,hspace=.42,wspace=.3)
    for ax,(metric,label) in zip(axes.flat,[('completion','Cumulative completion (%)'),('ce3','Cumulative CE(3)'),
                                            ('honest_reputation','Mean honest reputation'),
                                            ('malicious_reputation','Mean malicious reputation')]):
        for k,method in enumerate(METHODS):
            part=df[df['method']==method].groupby(['round','seed'],as_index=False)[metric].mean()
            if part.empty: continue
            x=part.groupby('round')[metric].mean()
            ax.plot(x.index,x.values,color=COLORS[k],linestyle=STYLES[k],marker=MARKERS[k],
                    markevery=[0,9,19,29,39,49],markersize=3,linewidth=1,
                    markerfacecolor=COLORS[k] if k==0 else 'white',markeredgewidth=.6)
        ax.set_xlabel('Round');ax.set_ylabel(label);ax.grid(axis='y',color='#d8d8d8',linewidth=.45,linestyle=':')
        ax.tick_params(direction='in',length=2.4,width=.6,pad=2)
    legend(fig)
    for ext in ('pdf','svg','png'): fig.savefig(output/('round_dynamics.'+ext),dpi=260,facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--tables',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(); tables=Path(a.tables);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    setup();df=pd.read_csv(tables/'sweep_cases.csv'); rd=pd.read_csv(tables/'rounds.csv')
    grid_figure(df,['budget','workers','honest_fraction'],
                ['completion','ce3','rmse_delta_vs_btpwr'],
                ['Completion (%)','CE(3)',r'$\Delta$ RMSE vs BTPWR'],out,'quality_sensitivity')
    grid_figure(df,['certified_fraction','honest_fraction','location_epsilon'],
                ['completion','honest_recognition','malicious_accepted'],
                ['Completion (%)','Honest identified','Malicious accepted'],out,'recognition_sensitivity')
    grid_figure(df,['budget','workers','honest_fraction'],['payment','physical'],
                ['Committed payment','Physical sampling cost'],out,'resource_sensitivity')
    round_figure(rd,out)


if __name__=='__main__': main()

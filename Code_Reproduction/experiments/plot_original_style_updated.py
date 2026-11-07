"""Reproduce the original six-panel sweep layout using current data."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[2]
out=ROOT.parents[1]/'outputs'
d=pd.read_csv(ROOT/'final_results'/'tables'/'sweep_cases.csv')
methods=['BTPWR','BTPWR-NP','C-seeded','Mapped','ETBP-P']
shared_rmse={}
for name in d['case'].unique():
    with np.load(ROOT/'final_results'/'review_sweeps'/(name+'.npz')) as z:
        pred=z['predictions'];truth=z['truth'];mask=z['task_mask'].astype(bool)
        shared=mask & np.isfinite(pred).all(axis=0)
        for i,method in enumerate(methods):
            shared_rmse[(name,method)]=float(np.sqrt(np.mean((pred[i][shared]-truth[shared])**2))) if shared.any() else np.nan
d['rmse_five']=[shared_rmse.get((case,method),np.nan) for case,method in zip(d['case'],d['method'])]
colors=['#a42b33','#b78439','#8199a6','#295d83','#538264']
markers=['o','s','^','D','v'];styles=['-','--',':','-.','--']
plt.rcParams.update({'font.family':'serif','font.size':8,'pdf.fonttype':42})
fig,axes=plt.subplots(2,3,figsize=(9.2,3.8),layout='constrained')
for row,axis in enumerate(('location_epsilon','budget')):
 for col,(metric,title) in enumerate((('completion','Completion'),('rmse_five','Common-task RMSE'),('payment','Payment'))):
  ax=axes[row,col]
  for k,m in enumerate(methods):
   q=d[(d.axis==axis)&(d.method==m)].groupby(['x','seed'])[metric].mean().reset_index()
   groups=q.groupby('x')[metric]; xs=np.array(sorted(groups.groups));vals=[groups.get_group(x).to_numpy() for x in xs]
   mean=np.array([v.mean() for v in vals]);err=np.array([t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)) for v in vals])
   if metric=='payment':mean/=1000;err/=1000
   ax.plot(xs,mean,color=colors[k],marker=markers[k],linestyle=styles[k],linewidth=1,
           markersize=3.3,markerfacecolor=colors[k])
  ax.set_title(f'({chr(97+row*3+col)}) {title}',pad=3)
  ax.set_xlabel(r'$\epsilon_1$' if axis=='location_epsilon' else 'Per-round budget')
  ax.set_ylabel(('Completion (%)','RMSE','Payment ($10^3$)')[col]);ax.grid(axis='y',alpha=.2,linestyle=':')
  ax.tick_params(direction='in',length=2.2)
handles=[Line2D([0],[0],color=colors[k],marker=markers[k],linestyle=styles[k],lw=1,markersize=4,label=m) for k,m in enumerate(methods)]
fig.legend(handles=handles,loc='outside upper center',ncol=5,frameon=False)
for ext in ('pdf','png'):fig.savefig(out/('BTPWR_原稿风格_新数据图.'+ext),dpi=250)
print('original-style updated figure complete')

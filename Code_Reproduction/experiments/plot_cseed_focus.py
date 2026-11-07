"""Paired comparison to C-seeded, with a zero-centered scale."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[2]
out=ROOT.parents[1]/'outputs'
d=pd.read_csv(ROOT/'final_results'/'tables'/'primary_cases.csv')
a=d[d.method=='BTPWR'].set_index(['dataset','seed'])
b=d[d.method=='C-seeded'].set_index(['dataset','seed'])
rows=[]
for col,label,relative in [('completion','Completion (percentage points)',False),
                           ('payment','Committed payment (%)',True),
                           ('physical','Physical sampling cost (%)',True)]:
    v=a[col]-b[col]
    if relative:v=100*v/b[col]
    v=v.groupby(level='seed').mean().to_numpy();e=t.ppf(.975,7)*v.std(ddof=1)/np.sqrt(8)
    rows.append((label,v.mean(),v.mean()-e,v.mean()+e))
s=pd.read_csv(ROOT/'final_results'/'tables'/'sweep_cases.csv')
plt.rcParams.update({'font.family':'serif','font.size':9,'pdf.fonttype':42})
fig,ax=plt.subplots(2,2,figsize=(10,6.8),layout='constrained')
x=ax[0,0]
for i,(label,m,lo,hi) in enumerate(rows):
    x.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color='#9b2636',capsize=3,linewidth=1.3)
    x.annotate(f'{m:+.2f} [{lo:+.2f}, {hi:+.2f}]',(m,i),xytext=(6,-12),textcoords='offset points',fontsize=8)
x.set_yticks(range(3),[r[0] for r in rows]);x.invert_yaxis();x.axvline(0,color='#555',lw=.8)
x.set_xlim(-16,7)
x.set_title('(a) Primary paired difference');x.grid(axis='x',alpha=.2)
axes=[('budget','Per-round budget'),('workers','Number of workers'),('honest_fraction','Honest workers (%)')]
for plot,(axis,label) in zip([ax[0,1],ax[1,0],ax[1,1]],axes):
    z=s[(s.axis==axis)&(s.method.isin(['BTPWR','C-seeded']))]
    p=z.pivot_table(index=['dataset','seed','x'],columns='method',values='completion')
    p['delta']=p['BTPWR']-p['C-seeded'];mat=p.reset_index().pivot_table(index='dataset',columns='x',values='delta')
    mat=mat.reindex(['tdrive','foursquare_tky','brightkite']);scale=max(3.,float(np.nanmax(np.abs(mat.to_numpy()))))
    image=plot.imshow(mat,aspect='auto',cmap='RdBu_r',vmin=-scale,vmax=scale)
    plot.set_xticks(range(len(mat.columns)),[f'{v*100:.0f}' if axis=='honest_fraction' else f'{v:.0f}' for v in mat.columns])
    plot.set_yticks(range(3),['T-Drive','Foursquare','Brightkite']);plot.set_xlabel(label)
    plot.set_title('('+chr(ord('b')+axes.index((axis,label)))+') Completion difference (pp)')
    for i in range(3):
        for j in range(len(mat.columns)):
            val=mat.iloc[i,j];plot.text(j,i,f'{val:+.1f}',ha='center',va='center',fontsize=8,color='white' if abs(val)>scale*.58 else '#222')
    fig.colorbar(image,ax=plot,shrink=.85,label='BTPWR − C-seeded (pp)')
for ext in ('pdf','png'):
    fig.savefig(out/('BTPWR与C-seeded_配对比较.'+ext),dpi=220)
print('created focused figures')

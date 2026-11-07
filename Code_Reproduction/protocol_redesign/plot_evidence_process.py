"""Original manuscript line/marker style, using only existing measured rows."""
from pathlib import Path
import hashlib,json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path(__file__).resolve().parents[1];workspace=base.parents[1]
source=workspace/'outputs'/'多区域证据力度_辨识过程.csv'
data=pd.read_csv(source)
data['dataset']=data['case'].str.replace(r'^main_','',regex=True).str.replace(r'_980[1-8]_b450$','',regex=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'mathtext.fontset':'stix','font.size':9,
                     'axes.linewidth':.75,'xtick.direction':'in','ytick.direction':'in','pdf.fonttype':42})
fig,axes=plt.subplots(2,3,figsize=(7.2,3.35),sharex=True,sharey=True)
styles=[('admission-mean-direct','Round mean / direct','#303030','--','o',False),
        ('admission-count-recovery','Bounded count / secondary', '#A73D30','-','s',True)]
for col,(ds,label) in enumerate([('tdrive','T-Drive'),('foursquare_tky','Foursquare-TKY'),('brightkite','Brightkite')]):
 for row,metric in enumerate(['honest','malicious']):
  ax=axes[row,col]
  for method,lab,color,ls,mark,filled in styles:
   d=data[(data.dataset==ds)&(data.method==method)].groupby('round')[metric].mean()
   if d.empty:raise ValueError('No measured rows for '+ds+' '+method)
   ax.plot(d.index,d.values,label=lab,color=color,ls=ls,marker=mark,markersize=3.2,lw=1.15,
           markerfacecolor=color if filled else 'white',markeredgewidth=.8)
  ax.set(ylim=(0,100),xlim=(3,52),xticks=[10,20,30,40,50],yticks=[0,25,50,75,100])
  ax.grid(axis='y',color='#dddddd',lw=.4)
  if row==0:ax.set_title(label,fontsize=10,pad=3)
  else:ax.set_xlabel('Round')
  if col==0:ax.set_ylabel(('Honest recall' if row==0 else 'Malicious recall')+' (%)')
  ax.text(.02,.94,'('+chr(97+row*3+col)+')',transform=ax.transAxes,va='top',fontsize=9)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.53,1.02),fontsize=9)
fig.subplots_adjust(left=.085,right=.99,bottom=.13,top=.87,hspace=.17,wspace=.08)
dest=base/'Paper_Source'/'figures'
fig.savefig(dest/'evidence_recognition.pdf');fig.savefig(dest/'evidence_recognition.png',dpi=240)
(dest/'evidence_recognition_manifest.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    study='Existing 24-world numerical dynamic-balance feedback comparison; not signed-grade protocol campaign',
    aggregation='mean of 8 seeds for each of 3 fixed trace conditions',error_bars=False,methods=[s[0] for s in styles]),indent=2),encoding='utf-8')

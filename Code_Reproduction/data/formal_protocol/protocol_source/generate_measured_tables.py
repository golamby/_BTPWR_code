from pathlib import Path
import pandas as pd
import numpy as np

base=Path(__file__).resolve().parents[1];out=base.parents[1]/'outputs'
d=pd.read_csv(out/'多区域证据力度_8种子验证.csv')
methods=[('admission-mean-direct','Mean/direct'),('admission-count-direct','Count/direct'),
         ('admission-count-recovery',r'Count/secondary $\beta=.5$'),
         ('admission-count-recovery-full',r'Count/secondary $\beta=1$')]
parts=[r'''\section{Dataset-Level Feedback and Reference Audits}\label{app:measured-details}
The following summaries expand the completed controlled feedback study, using the same numerical dynamic-balance selector as the main-paper table. Each dataset row averages eight seeds. The support model in these runs has different information from the signed-grade broker; this table therefore tests the feedback choices under its recorded numerical conditions.
\begin{table*}[t]\centering\small
\caption{Dataset-level feedback results. H/M recall uses hidden behavior labels. Payment and physical cost are 50-round totals in thousands.}
\begin{tabular}{llrrrrr}\toprule
Mobility & Feedback & TCR (\%) & H recall (\%) & M recall (\%) & Payment & Physical\\\midrule
''']
for ds,label in [('tdrive','T-Drive'),('foursquare_tky','Foursquare-TKY'),('brightkite','Brightkite')]:
 for m,lab in methods:
  v=d[(d.dataset==ds)&(d.method==m)]
  if len(v)!=8:raise ValueError('Unexpected paired cohort')
  a=v.mean(numeric_only=True)
  parts.append(f"{label} & {lab} & {a.completion:.2f} & {a.honest_recall:.2f} & {a.malicious_recall:.2f} & {a.payment/1000:.3f} & {a.physical/1000:.3f} "+r'\\'+'\n')
 parts.append(r'\midrule'+'\n')
parts.append(r'''\bottomrule\end{tabular}\end{table*}
The dataset breakdown distinguishes a pooled improvement from behavior tied to one trace mask. It also keeps acquisition cost beside recognition: a policy can identify more workers by obtaining more evaluable reports, so that gain should be read with its total payment and sampling. No row changes the initial honest share, certification convention or round cap relative to the main controlled study.

An earlier credible-only secondary-reference audit measures predicted reference errors against evaluator-held truth after the algorithm has run. The truth is not used online to accept or reject a reference. The next table pools the logged absolute/squared error sums over actual eligible worker-region checks, rather than averaging per-round means with different denominators. Those checks share anchors, regions and rounds and are not independent statistical replications.
\begin{table*}[t]\centering\small
\caption{Secondary-reference audit in nine development worlds of the earlier mean-update numerical variant. Errors are in checking units. The last column is the observed fraction within three checking units.}
\begin{tabular}{llrrrr}\toprule
Mobility & Secondary variant & Checks & MAE & RMSE & Within tolerance (\%)\\\midrule
''')
a=pd.read_csv(out/'次级参考_逐轮误差审计.csv')
a['dataset']=a['case'].str.extract(r'^sweep_(.*?)_970[1-3]_')[0]
for (ds,m),v in a.groupby(['dataset','method'],sort=True):
 n=int(v['count'].sum());label={'tdrive':'T-Drive','foursquare_tky':'Foursquare-TKY','brightkite':'Brightkite'}[ds]
 # This earlier mean-update audit used beta=1 and beta=.25, not the later
 # count-sensitive study's beta=.5. Keep the run-mode identities explicit.
 method={'admission-recovered':r'$\beta=1$', 'admission-recovered-025':r'$\beta=.25$'}[m]
 parts.append(f"{label} & {method} & {n} & {v.abs_error_sum.sum()/n:.3f} & {np.sqrt(v.sq_error_sum.sum()/n):.3f} & {100*v.within_3.sum()/n:.2f} "+r'\\'+'\n')
parts.append(r'''\bottomrule\end{tabular}\end{table*}
The tolerance column describes the reference prediction, not the final worker discrepancy: the latter additionally contains measurement noise or malicious behavior. Nor can it certify a reference during deployment, because the current regional truth is unavailable. It measures whether the observed predictions support using recovery as a discounted reference in the tested field. The new signed-grade campaign repeats the same audit under its own recruited anchors rather than importing this older audit as its accuracy estimate.

Reference exposure is reported both as worker-region checks and as distinct predicted region-rounds. The former measures how many actual reports can be evaluated, while the latter avoids counting one shared regional prediction as several independent recovered truths. Conditional reference errors, source counts and recovered coverage are analyzed together. A reference model that supplies many low-quality checks is not preferred merely because it yields a larger effective evidence count.
''')
(base/'Paper_Source'/'sections'/'measured_evidence_tables.tex').write_text(''.join(parts).replace(r'\midrule'+'\n'+r'\bottomrule',r'\bottomrule'),encoding='utf-8')
print('Generated measured tables from',len(d),'campaign rows and',len(a),'audit rows')

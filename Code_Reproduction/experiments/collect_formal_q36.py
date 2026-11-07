"""Validate frozen Q36 campaign records and derive manuscript data."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def quality(truth, prediction, mask):
    finite = mask & np.isfinite(prediction)
    error = np.abs(prediction-truth)
    success = finite & np.where(truth==0, error==0, error<.1*np.abs(truth))
    total = int(mask.sum()); count = int(finite.sum())
    return dict(TCR=100*success.sum()/total, AR=100*count/total,
                own_MAE=float(error[finite].mean()) if count else np.nan,
                own_RMSE=float(np.sqrt(np.mean(error[finite]**2))) if count else np.nan,
                answered_tasks=count, requested_tasks=total,
                CE3=float((error[success].sum()+3*(total-success.sum()))/total))


def classify(r, thresholds):
    if thresholds is None:
        return None
    credible = r>thresholds['high'] if thresholds.get('credible_strict',False) else r>=thresholds['high']
    return np.where(credible,2,np.where(r<thresholds['low'],0,1))


def identification(r, honest, certified, thresholds):
    label=classify(r,thresholds)
    if label is None:return {}
    honest=np.asarray(honest,bool);malicious=~honest
    f1=[]
    for truth,decision in [(honest,label==2),(malicious,label==0)]:
        tp=int((truth&decision).sum());fp=int((~truth&decision).sum());fn=int((truth&~decision).sum())
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan)
    unseeded=honest&~np.asarray(certified,bool)
    return dict(macro_F1=100*np.mean(f1), honest_recall=100*np.mean(label[honest]==2),
                malicious_recall=100*np.mean(label[malicious]==0),false_acceptance=100*np.mean(label[malicious]==2),
                false_exclusion=100*np.mean(label[honest]==0),unknown=100*np.mean(label==1),
                unseeded_honest_recall=100*np.mean(label[unseeded]==2) if unseeded.any() else np.nan,
                honest_mean=float(np.mean(r[honest])),honest_median=float(np.median(r[honest])),
                malicious_mean=float(np.mean(r[malicious])),honest_q10=float(np.quantile(r[honest],.1)),
                honest_q90=float(np.quantile(r[honest],.9)))


def resource_summary(resources):
    # Native quota systems have undefined payments throughout a campaign.
    return {name:float(resources[:,i].sum()) if np.isfinite(resources[:,i]).all() else np.nan
            for i,name in enumerate(['commitment','physical','selected_worker_rounds','actual_route_bid_sum','settled'])}


def validate(path):
    meta=json.loads(path.read_text(encoding='utf-8'));npz=path.with_name('result.npz')
    if meta['status']!='completed' or meta['rounds']!=50:raise ValueError('Incomplete '+str(path))
    if sha(npz)!=meta['output_sha256']:raise ValueError('Output hash mismatch '+str(path))
    with np.load(npz,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    c=meta['case'];n=c['n'];pred=arrays['predictions'];selected=arrays['selected'];res=arrays['resources']
    if pred.shape!=(50,100) or selected.shape!=(50,n) or res.shape!=(50,5):raise ValueError('Shape mismatch '+str(path))
    if np.isinf(pred).any() or not np.isfinite(arrays['truth']).all():raise ValueError('Invalid prediction/truth')
    if arrays['truth'].shape!=(50,100) or arrays['task_mask'].shape!=(50,100):raise ValueError('Evaluation domain mismatch')
    if not np.isfinite(res[:,1:4]).all() or np.any(res[:,1:4]<0):raise ValueError('Invalid physical resources')
    if not np.array_equal(selected.sum(1),res[:,2]):raise ValueError('Selected count differs from resource record')
    for col in [0,4]:
        values=res[:,col]
        if np.isnan(values).all():continue
        if not np.isfinite(values).all() or np.any(values<0) or np.any(values>c['budget_override']+1e-6):raise ValueError('Invalid monetary record')
    if not np.array_equal(np.isnan(res[:,0]),np.isnan(res[:,4])):raise ValueError('Mixed payment scope')
    if np.any(res[:,4]>res[:,0]+1e-6):raise ValueError('Settlement exceeds commitment')
    if 'reputations' in arrays:
        r=arrays['reputations']
        if r.shape!=(51,n) or not np.isfinite(r).all() or np.any((r<0)|(r>1)):raise ValueError('Invalid reputation state')
    elif meta.get('class_thresholds') is not None:raise ValueError('Classifier without state')
    return meta,arrays


def context(c,method):
    result={k:c[k] for k in ['dataset','seed','group','n','H','C','eps1','eps2','budget_override','field','factor','factor_value']}
    result.update(case=c['name'],method=method,stress=c.get('stress',''))
    return result


def confidence(values):
    v=np.asarray(values,float);v=v[np.isfinite(v)];n=len(v)
    if n<8:return dict(estimate=float(v.mean()) if n else np.nan,ci_low=np.nan,ci_high=np.nan,seeds=n,inference_complete=False)
    radius=float(student.ppf(.975,n-1)*v.std(ddof=1)/np.sqrt(n))
    return dict(estimate=float(v.mean()),ci_low=float(v.mean()-radius),ci_high=float(v.mean()+radius),seeds=n,inference_complete=True)


def escape(value):
    return str(value).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')


def tex_tables(summary,pairs,output):
    primary=summary[summary.group=='primary']
    if primary.empty:return
    means=primary.groupby('method').mean(numeric_only=True)
    methods=['BTPWR','BPP','VLR-BPP','ETBP-TD','REAPP','Random','Greedy-C/R','BPQA','MMCB']
    def fmt(x):return '--' if not np.isfinite(x) else f'{x:.2f}'
    lines=[r'\begin{table*}[t]\centering\small',r'\caption{Default 80-worker numerical campaigns. Undefined native payments and unsupported classifiers are shown as dashes. Dataset-level results and paired accuracy domains accompany this summary.}',r'\label{tab:q36-default}',r'\begin{tabular}{lrrrrrrr}\toprule',r'Method&TCR (\%)&AR (\%)&F1 (\%)&Commitment&Settlement&Physical&CE(3)\\\midrule']
    for m in methods:
        if m in means.index:
            row=means.loc[m];lines.append(escape(m)+'&'+'&'.join(fmt(row.get(k,np.nan)) for k in ['TCR','AR','macro_F1','commitment','settled','physical','CE3'])+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table*}']
    (output/'q36_default_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    pp=pairs[pairs.group=='primary'] if not pairs.empty else pairs
    if pp.empty:return
    lines=[r'\begin{table*}[t]\centering\small',r'\caption{Pairwise common requested-task accuracy at the default. Each comparator has its own common domain; both errors in a row use exactly that domain. Counts are campaign means.}',r'\label{tab:q36-common}',r'\begin{tabular}{llrrrrr}\toprule',r'Dataset&Comparator&Common $N$&BTPWR MAE&Baseline MAE&BTPWR RMSE&Baseline RMSE\\\midrule']
    for (d,m),row in pp.groupby(['dataset','comparison']).mean(numeric_only=True).iterrows():
        if m not in methods:continue
        lines.append(escape(d)+'&'+escape(m)+'&'+'&'.join(fmt(row[k]) for k in ['common_tasks','ours_MAE','baseline_MAE','ours_RMSE','baseline_RMSE'])+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table*}']
    (output/'q36_common_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def collect(root,output,partial=False):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    expected={};job_sources={}
    for f in root.glob('jobs_*.json'):
        j=json.loads(f.read_text(encoding='utf-8'));job_sources[f.name]=j.get('source_sha256')
        for row in j['jobs']:expected[(row['case'],row['method'])]=f.name
    if not expected and not partial:
        raise ValueError('Registered jobs manifests missing; collect jobs_*.json with results')
    records={};sources={};inputs=[];rows=[];rounds=[]
    for path in sorted((root/'cases').glob('*/*/result.json')):
        meta,a=validate(path);c=meta['case'];m=meta['method'];key=(c['name'],m)
        if key in records:raise ValueError('Duplicate case/method '+str(key))
        job=expected.get(key)
        if job and meta['source_sha256']!=job_sources[job]:raise ValueError('Job/source mismatch '+str(path))
        if sources and meta['source_sha256']!=next(iter(sources.values())):raise ValueError('Mixed frozen source versions')
        sources[str(path)]=meta['source_sha256'];records[key]=(meta,a)
        inputs.append(dict(path=str(path),meta_sha256=sha(path),npz_sha256=meta['output_sha256'],world_sha256=meta['world_sha256']))
        row=context(c,m);row.update(quality(a['truth'],a['predictions'],a['task_mask']));row.update(resource_summary(a['resources']))
        if 'reputations' in a:row.update(identification(a['reputations'][-1],a['honest'],a['certified'],meta.get('class_thresholds')))
        rows.append(row)
        for t in range(1,51):
            r=context(c,m);r['round']=t;r.update(quality(a['truth'][:t],a['predictions'][:t],a['task_mask'][:t]));r.update(resource_summary(a['resources'][:t]))
            if 'reputations' in a:r.update(identification(a['reputations'][t],a['honest'],a['certified'],meta.get('class_thresholds')))
            rounds.append(r)
    if not records:raise ValueError('No completed campaign records')
    missing=sorted(set(expected)-set(records))
    if missing and not partial:raise ValueError(f'{len(missing)} registered jobs incomplete; use --partial only for inspection')
    pairs=[]
    for (name,m),(meta,a) in records.items():
        if m=='BTPWR' or (name,'BTPWR') not in records:continue
        ours,o=records[(name,'BTPWR')]
        if meta['world_sha256']!=ours['world_sha256']:raise ValueError('Pair not same world '+name)
        for k in ['truth','task_mask','honest','certified']:
            np.testing.assert_array_equal(a[k],o[k],err_msg='Pair domain mismatch '+name)
        common=a['task_mask']&np.isfinite(a['predictions'])&np.isfinite(o['predictions']);n=int(common.sum())
        ea=(o['predictions']-o['truth'])[common];eb=(a['predictions']-a['truth'])[common]
        row=context(meta['case'],m);row['comparison']=m;row['common_tasks']=n
        for prefix,e in [('ours',ea),('baseline',eb)]:
            row[prefix+'_MAE']=float(np.mean(np.abs(e))) if n else np.nan;row[prefix+'_RMSE']=float(np.sqrt(np.mean(e*e))) if n else np.nan
        row['RMSE_gap_baseline_minus_ours']=row['baseline_RMSE']-row['ours_RMSE'];pairs.append(row)
    summary=pd.DataFrame(rows);trajectory=pd.DataFrame(rounds);paired=pd.DataFrame(pairs);intervals=[]
    for m in summary.method.unique():
        if m=='BTPWR':continue
        own=summary[(summary.group=='primary')&(summary.method=='BTPWR')]
        other=summary[(summary.group=='primary')&(summary.method==m)]
        join=own.merge(other,on=['dataset','seed'],suffixes=('_ours','_baseline'))
        for metric in ['TCR','AR','macro_F1','commitment','settled','physical']:
            if metric+'_ours' not in join or metric+'_baseline' not in join:continue
            join['difference']=join[metric+'_ours']-join[metric+'_baseline']
            valid=join[np.isfinite(join.difference)]
            complete=valid.groupby('seed').filter(lambda x:len(x)==3 and x.dataset.nunique()==3)
            seed_means=complete.groupby('seed').difference.mean()
            intervals.append(dict(comparison=m,metric=metric,direction='BTPWR minus comparator',**confidence(seed_means)))
        if not paired.empty:
            pp=paired[(paired.group=='primary')&(paired.comparison==m)].copy()
            pp['difference']=pp.ours_RMSE-pp.baseline_RMSE
            complete=pp[np.isfinite(pp.difference)].groupby('seed').filter(lambda x:len(x)==3 and x.dataset.nunique()==3)
            intervals.append(dict(comparison=m,metric='pairwise_common_RMSE',direction='BTPWR minus comparator',**confidence(complete.groupby('seed').difference.mean())))
    for name,df in [('campaign_metrics',summary),('round_metrics',trajectory),('pairwise_common_errors',paired),('primary_paired_intervals',pd.DataFrame(intervals))]:
        df.to_csv(output/(name+'.csv'),index=False,encoding='utf-8-sig')
    summary[summary.group=='primary'].groupby(['dataset','method']).mean(numeric_only=True).to_csv(output/'primary_dataset_summary.csv',encoding='utf-8-sig')
    summary[summary.group=='primary'].groupby('method').mean(numeric_only=True).to_csv(output/'primary_summary.csv',encoding='utf-8-sig')
    tex_tables(summary,paired,output)
    manifest=dict(status='partial_inspection' if missing or not expected else 'all_registered_jobs_collected',campaigns=len(records),missing_jobs=missing,
                  input_files=inputs,source_sha256=next(iter(sources.values())),collector_sha256=sha(__file__),
                  identification='native threshold decisions; unknown is a false negative for the true H/M class',
                  inference='seed paired; average three fixed datasets before Student interval; minimum8completepairedseeds',
                  physical_validation='resource rows and selected counts validated; raw route-cost recomputation needs the source world')
    (output/'collection_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['status','campaigns','missing_jobs']},ensure_ascii=False))
    return summary,trajectory,paired


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--partial',action='store_true');a=p.parse_args()
    collect(a.root,a.output,a.partial)

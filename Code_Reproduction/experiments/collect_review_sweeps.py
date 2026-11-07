"""Build privacy-safe aggregated tables from completed paired sweep campaigns."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.sweep_cases import cases
from src import data

ROOT=Path(__file__).resolve().parents[1]
LABELS={'admission':'BTPWR','admission-no-promotion':'BTPWR-NP',
        'template-first':'C-seeded','old':'Mapped','etbp':'ETBP-P','bpqa':'BPQA numeric',
        'random':'Random','greedy-cover':'Greedy-C/R'}


def quality(truth, pred, mask, common):
    valid=mask & np.isfinite(pred)
    err=np.abs(pred-truth)
    hit=valid & np.where(truth==0,err==0,err < .1*np.abs(truth))
    shared=common & valid
    return {'completion':float(100*hit.sum()/mask.sum()),
            'answered':float(100*valid.sum()/mask.sum()),
            'ce3':float((err[hit].sum()+3*(mask.sum()-hit.sum()))/mask.sum()),
            'mae_common':float(err[shared].mean()) if shared.any() else np.nan,
            'rmse_common':float(np.sqrt(np.mean(err[shared]**2))) if shared.any() else np.nan,
            'common_tasks':int(shared.sum())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--default-budget',required=True)
    parser.add_argument('--results',default=str(ROOT/'results'))
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=Path(args.results); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    all_rows=[]; rounds=[]; missing=[]
    for case in cases(default_budget=args.default_budget):
        name=case['name']; path=result/'review_sweeps'/(name+'.npz')
        metapath=path.with_suffix('.json')
        if not path.exists() or not metapath.exists():
            missing.append(str(path)); continue
        meta=json.loads(metapath.read_text(encoding='utf-8'))
        if meta['case']!=case: raise ValueError('Case mismatch: '+name)
        bp=result/'bpqa_native_sweeps'/(name+'.npz')
        if case['eps1']==2.1 and not bp.exists(): missing.append(str(bp)); continue
        with np.load(path,allow_pickle=False) as z:
            truth=z['truth']; mask=z['task_mask'].astype(bool)
            methods=z['methods'].tolist(); preds=z['predictions']; ours=preds[0]
            if methods[0]!='admission': raise ValueError('Unexpected primary ordering')
            b=None
            if bp.exists():
                bj=json.loads(bp.with_suffix('.json').read_text(encoding='utf-8'))
                if bj['world_sha256']!=meta['world_sha256'] or bj['error_aggregation']!='sum':
                    raise ValueError('BPQA world or error mode mismatch: '+name)
                b=np.load(bp,allow_pickle=False)
                np.testing.assert_array_equal(b['truth'],truth)
                np.testing.assert_array_equal(b['task_mask'],mask)
            q,honest,cert,_=data.population(case['n'],case['H'],case['C'],case['seed'])
            entries=[]
            for i,method in enumerate(methods):
                pred=preds[i]; common=mask & np.isfinite(ours) & np.isfinite(pred)
                stats=z[method+'__stats']; counts=z['worker_counts'][i]
                honest_total=int(honest.sum()); bad_total=int((~honest).sum())
                metrics=quality(truth,pred,mask,common)
                ours_metrics=quality(truth,ours,mask,common)
                metrics['mae_delta_vs_btpwr']=metrics['mae_common']-ours_metrics['mae_common']
                metrics['rmse_delta_vs_btpwr']=metrics['rmse_common']-ours_metrics['rmse_common']
                metrics.update(payment=float(stats[:,0].sum()),physical=float(stats[:,1].sum()),
                    honest_recognition=float(counts[-1,0]/honest_total) if honest_total else np.nan,
                    malicious_recognition=float(counts[-1,1]/bad_total) if bad_total else np.nan,
                    malicious_accepted=float(counts[-1,2]/bad_total) if bad_total else np.nan)
                entries.append((method,metrics,pred,z['reputations'][i]))
            trad=result/'traditional_sweeps'/(name+'.npz')
            if not trad.exists(): missing.append(str(trad)); continue
            tm=json.loads(trad.with_suffix('.json').read_text(encoding='utf-8'))
            if tm['world_sha256']!=meta['world_sha256']: raise ValueError('Traditional world mismatch: '+name)
            with np.load(trad,allow_pickle=False) as tr:
                np.testing.assert_array_equal(tr['truth'],truth)
                np.testing.assert_array_equal(tr['task_mask'],mask)
                for i,method in enumerate(tr['methods'].tolist()):
                    pred=tr['predictions'][i].copy(); rep=tr['reputations'][i].copy()
                    common=mask & np.isfinite(ours) & np.isfinite(pred)
                    metrics=quality(truth,pred,mask,common); ref=quality(truth,ours,mask,common)
                    stats=tr[method+'__stats'];counts=tr['worker_counts'][i]
                    metrics['mae_delta_vs_btpwr']=metrics['mae_common']-ref['mae_common']
                    metrics['rmse_delta_vs_btpwr']=metrics['rmse_common']-ref['rmse_common']
                    metrics.update(payment=float(stats[:,0].sum()),physical=float(stats[:,1].sum()),
                        honest_recognition=float(counts[-1,0]/honest_total),
                        malicious_recognition=float(counts[-1,1]/bad_total),
                        malicious_accepted=float(counts[-1,2]/bad_total))
                    entries.append((method,metrics,pred,rep))
            if b is not None:
                pred=b['predictions']; common=mask & np.isfinite(ours) & np.isfinite(pred)
                metrics=quality(truth,pred,mask,common); r=b['reputations']
                ours_metrics=quality(truth,ours,mask,common)
                metrics['mae_delta_vs_btpwr']=metrics['mae_common']-ours_metrics['mae_common']
                metrics['rmse_delta_vs_btpwr']=metrics['rmse_common']-ours_metrics['rmse_common']
                metrics.update(payment=float(b['resources'][:,0].sum()),physical=float(b['resources'][:,1].sum()),
                    settled_payment=float(b['resources'][:,4].sum()),
                    honest_recognition=float((r[-1,honest]>=.8).mean()),
                    malicious_recognition=float((r[-1,~honest]<.3).mean()),
                    malicious_accepted=float((r[-1,~honest]>=.8).mean()))
                entries.append(('bpqa',metrics,pred,r))
            for axis in case['sweep_axes']:
                for method,metrics,pred,reputation in entries:
                    all_rows.append({'case':name,'dataset':case['dataset'],'seed':case['seed'],
                                     'axis':axis['axis'],'x':axis['value'],'method':LABELS[method],**metrics})
                    if axis['axis']=='budget' and axis['value']==450:
                        for t in range(len(truth)):
                            mask_t=mask[:t+1]; truth_t=truth[:t+1]; pred_t=pred[:t+1]
                            valid=mask_t & np.isfinite(pred_t); err=np.abs(pred_t-truth_t)
                            hit=valid & np.where(truth_t==0,err==0,err<.1*np.abs(truth_t))
                            rounds.append({'dataset':case['dataset'],'seed':case['seed'],
                                'method':LABELS[method],'round':t+1,
                                'completion':float(100*hit.sum()/mask_t.sum()),
                                'ce3':float((err[hit].sum()+3*(mask_t.sum()-hit.sum()))/mask_t.sum()),
                                'honest_reputation':float(reputation[t+1,honest].mean()),
                                'malicious_reputation':float(reputation[t+1,~honest].mean())})
            if b is not None: b.close()
    if missing:
        raise FileNotFoundError(f'{len(missing)} unfinished cases; first: {missing[0]}')
    frame=pd.DataFrame(all_rows); frame.to_csv(out/'sweep_cases.csv',index=False)
    pd.DataFrame(rounds).to_csv(out/'rounds.csv',index=False)
    print(json.dumps({'complete':True,'worlds':len(cases(default_budget=args.default_budget)),
                      'rows':len(frame),'round_rows':len(rounds)}))


if __name__=='__main__':
    main()

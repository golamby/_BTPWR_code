"""Native complete-system context, separate from the matched primary study."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from . import metrics
METHODS=('ETBP-TD','MMCB','VLR-BPP','Random','No-Privacy')


def run_native(w, case, rounds=50):
    from src import privacy, baselines, metrics
    (T, n, D) = w['presence'].shape
    T = min(T, rounds)
    cfg = privacy.Config(n=n, m=D, T=T, eps1=case['eps1'], eps2=case['eps2'])
    state = baselines.etbp.initialize(n, w['initial_reputation'] >= 0.8, baselines.ETBP_CONFIG)
    predictions = {m: np.full((T, D), np.nan) for m in METHODS}
    selected = {m: np.zeros((T, n), bool) for m in METHODS}
    rows = []
    native_history = [state.trust.copy()]
    offset = float(w['native_offset'])
    scale = float(w['native_scale'])
    for t in range(T):
        loc = w['presence'][t]
        cost = w['actual_cost'][t]
        offers = w['offer_mask'][t]
        active = np.flatnonzero(offers)
        virtual = w['virtual_presence'][t]
        avg = np.divide(cost.sum(1), loc.sum(1), out=np.full(n, 3.25), where=loc.sum(1) > 0)
        packet = privacy.release(virtual, np.where(virtual, avg[:, None], 0), w['RR_coins'][t], cfg.eps1, cfg.eps2, cfg)
        for key in ('bits', 'hi', 'nonzero'):
            packet[key][~offers] = False
        tick = time.perf_counter()
        charges = baselines.bpp.published_pcp_costs(packet, cfg.eps2)
        order = np.random.default_rng(np.random.SeedSequence([case['seed'], 43501, t])).permutation(n)
        ids = []
        payment = 0.0
        for i in order[offers[order]]:
            if payment + charges[i] > w['budget'] + 1e-09:
                break
            ids.append(int(i))
            payment += float(charges[i])
        picks = {'Random': np.asarray(ids, int)}
        pays = {'Random': payment}
        elapsed = {'Random': time.perf_counter() - tick}
        tick = time.perf_counter()
        bids = cost.sum(1)
        (local, _) = baselines.ws.exact_ws(loc[active][:, w['requested_task_ids'][t]], bids[active], w['budget'])
        picks['No-Privacy'] = active[local]
        pays['No-Privacy'] = float(bids[active[local]].sum())
        elapsed['No-Privacy'] = time.perf_counter() - tick
        for (method, ids) in picks.items():
            raw = np.full((n, D), np.nan)
            raw[ids] = w['reports_check'][t, ids]
            predictions[method][t] = np.clip(offset + scale * ((baselines.mean_reports(raw) - 10) / 30), 0, 100)
            selected[method][t, ids] = True
            rows.append(dict(method=method, round=t + 1, commitment=pays[method], physical_cost=float(cost[ids].sum()), selected_workers=len(ids), budget=float(w['budget']), seconds=elapsed[method], constraint='native_monetary_budget'))
        tick = time.perf_counter()
        (values, picks, state, records) = baselines.native_round(w, t, state, case['seed'], case['dataset'])
        native_history.append(state.trust.copy())
        seconds = time.perf_counter() - tick
        for (method, ids) in picks.items():
            predictions[method][t, w['requested_task_ids'][t]] = np.clip(offset + scale * (values[method] - 10) / 30, 0, 100)
            selected[method][t, ids] = True
            rows.append(dict(method=method, round=t + 1, physical_cost=float(cost[ids].sum()), selected_workers=len(ids), budget=float(w['budget']), combined_native_seconds=seconds, **records[method]))
    tick = time.perf_counter()
    (pred, pick, records) = baselines.vlr_campaign(w, case, T)
    predictions['VLR-BPP'] = pred
    selected['VLR-BPP'] = pick
    for row in records:
        rows.append({**row, 'method': 'VLR-BPP'})
    elapsed_vlr = time.perf_counter() - tick
    result = []
    for method in METHODS:
        m = metrics.requested_metrics(w['truth_native_private'][:, :T].T, predictions[method], w['task_mask'][:T])
        local = [r for r in rows if r['method'] == method]
        monetary = all((r.get('commitment') is not None for r in local))
        if monetary and any((r['commitment'] > w['budget'] + 1e-07 for r in local)):
            raise AssertionError('Native payment budget exceeded')
        if np.any(selected[method] & ~w['offer_mask'][:T]):
            raise AssertionError('Nonparticipant recruited')
        result.append(dict(method=method, TCR=m['completion_percent'], own_MAE=m['all_answered_MAE'], own_RMSE=m['all_answered_RMSE'], payment=sum((r['commitment'] for r in local)) if monetary else None, physical=sum((r['physical_cost'] for r in local)), selected_worker_rounds=sum((r['selected_workers'] for r in local))))
    return (dict(predictions=np.array([predictions[m] for m in METHODS]), selected=np.array([selected[m] for m in METHODS]), native_reputations=np.asarray(native_history), truth=w['truth_native_private'][:, :T].T, task_mask=w['task_mask'][:T], methods=np.array(METHODS)), dict(results=result, resources=rows, vlr_seconds=elapsed_vlr))

def cases(manifest):
    return [c for c in manifest['cases'] if c['stage']=='main' and c['budget_factor']==1. and c['seed']<=9703]


def export(payload,result,case,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    resources=[]
    for method in METHODS:
        rows=sorted((r for r in result['resources'] if r['method']==method),key=lambda r:r['round'])
        if len(rows)!=50 or [r['round'] for r in rows]!=list(range(1,51)):raise ValueError('Incomplete native resources')
        resources.append([[np.nan if r['commitment'] is None else r['commitment'],r['physical_cost'],r['selected_workers']] for r in rows])
    np.savez_compressed(path,methods=np.asarray(METHODS),predictions=payload['predictions'],truth=payload['truth'],
        task_mask=payload['task_mask'],resources=np.asarray(resources),case_json=np.array(json.dumps(case)))


def check(folder,primary,manifest,output,partial=False):
    folder=Path(folder);primary=Path(primary);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];pairs=[];missing=[]
    for case in cases(manifest):
        path=folder/(case['name']+'.npz');main=primary/path.name
        if not path.exists() or not main.exists():missing.append(case['name']);continue
        with np.load(path,allow_pickle=False) as z,np.load(main,allow_pickle=False) as p:
            if p['methods'].tolist()!=manifest['methods'] or p['methods'][0]!='admission':raise ValueError('Primary method identity differs')
            if z['methods'].tolist()!=list(METHODS) or json.loads(str(z['case_json']))!=case:raise ValueError('Native case identity differs')
            if set(z.files)!={'methods','predictions','truth','task_mask','resources','case_json'}:raise ValueError('Unexpected evidence field')
            truth=z['truth'];mask=z['task_mask'];pred=z['predictions'];res=z['resources']
            np.testing.assert_array_equal(truth,p['truth']);np.testing.assert_array_equal(mask,p['task_mask'])
            if pred.shape!=(5,50,100) or res.shape!=(5,50,3):raise ValueError('Native shape differs')
            if not np.isnan(res[0,:,0]).all() or not np.isfinite(res[1:]).all():raise ValueError('Native payment semantics differ')
            if np.any(res[1:,:,0]<0) or np.any(res[1:,:,0]>case['budget_override']+1e-7):raise ValueError('Native cap violated')
            if not np.isfinite(res[:,:,1:]).all() or np.any(res[:,:,1:]<0):raise ValueError('Invalid native resources')
            ours=p['predictions'][0];pr=p['resources'][0]
            metric=metrics.requested_metrics(truth,ours,mask)
            rows.append(dict(case=case['name'],dataset=case['dataset'],seed=case['seed'],method='Admission-aware BTPWR',
                TCR=metric['completion_percent'],own_MAE=metric['all_answered_MAE'],own_RMSE=metric['all_answered_RMSE'],
                payment=float(pr[:,0].sum()),physical=float(pr[:,1].sum())))
            for i,method in enumerate(METHODS):
                metric=metrics.requested_metrics(truth,pred[i],mask)
                rows.append(dict(case=case['name'],dataset=case['dataset'],seed=case['seed'],method=method,TCR=metric['completion_percent'],
                    own_MAE=metric['all_answered_MAE'],own_RMSE=metric['all_answered_RMSE'],
                    payment=float(res[i,:,0].sum()),physical=float(res[i,:,1].sum())))
                common=mask&np.isfinite(ours)&np.isfinite(pred[i]);a=(ours-truth)[common];b=(pred[i]-truth)[common]
                pairs.append(dict(case=case['name'],dataset=case['dataset'],seed=case['seed'],comparison=method,common_tasks=int(common.sum()),
                    ours_MAE=float(np.abs(a).mean()),baseline_MAE=float(np.abs(b).mean()),ours_RMSE=float(np.sqrt((a*a).mean())),baseline_RMSE=float(np.sqrt((b*b).mean()))))
    if missing and not partial:raise ValueError('Missing secondary context cases: '+str(missing))
    if not rows:raise ValueError('No completed native context evidence')
    table=pd.DataFrame(rows);paired=pd.DataFrame(pairs)
    table.to_csv(output/'native_context_cases.csv',index=False);paired.to_csv(output/'native_context_common_errors.csv',index=False)
    receipt=dict(complete=not missing,checked_cases=len(rows)//6,missing=missing,
        role='Secondary native system context; not the registered matched privacy/payment comparison.',
        ETBP_TD='40-worker quota; payment is undefined. Physical costs are recorded. MMCB/No-Privacy have plaintext cost/location inputs.')
    (output/'native_context_check.json').write_text(json.dumps(receipt,indent=2))
    return table,paired,receipt


def plot(folder,primary,manifest,output,partial=False):
    table,paired,receipt=check(folder,primary,manifest,output,partial)
    from .recruitment_evidence import cited,figure_style,finish_axes
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':9,
        'axes.linewidth':.75,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
    figure_style()
    methods=['BTPWR']+list(METHODS);colors=['#A42B33','#4C7C69','#796287','#AF8144','#707070','#295D83']
    markers=['o','s','^','D','v','P'];styles=['-','-','-','-','--','--']
    datasets=['tdrive','foursquare_tky','brightkite'];labels=['T-Drive','Foursquare','Brightkite']
    fig,axes=plt.subplots(2,3,figsize=(7.15,3.65),layout='constrained');x=np.arange(1,51);curve_rows=[]
    for column,dataset in enumerate(datasets):
        tracks=[]
        for case in cases(manifest):
            if case['dataset']!=dataset:continue
            path=Path(folder)/(case['name']+'.npz');main=Path(primary)/path.name
            if not path.exists() or not main.exists():continue
            with np.load(path,allow_pickle=False) as z,np.load(main,allow_pickle=False) as p:
                pred=np.concatenate([p['predictions'][:1],z['predictions']],axis=0);truth=z['truth'];task=z['task_mask']
                valid=task[None]&np.isfinite(pred);error=np.abs(pred-truth[None])
                success=valid&np.where(truth[None]==0,error==0,error<.1*np.abs(truth[None]))
                completed=np.cumsum(success.sum(2),axis=1)/np.cumsum(task.sum(1))[None]*100
                counts=np.cumsum(valid.sum(2),axis=1);sse=np.cumsum(np.where(valid,error*error,0).sum(2),axis=1)
                rmse=np.sqrt(np.divide(sse,counts,out=np.full_like(sse,np.nan),where=counts>0))
                tracks.append(np.stack([completed,rmse]))
                for m,method in enumerate(methods):
                    for t in range(50):curve_rows.append(dict(case=case['name'],dataset=dataset,seed=case['seed'],method=method,round=t+1,
                        cumulative_TCR=completed[m,t],cumulative_answered_RMSE=rmse[m,t]))
        if not tracks:raise ValueError('No native context curves for '+dataset)
        values=np.mean(np.asarray(tracks),axis=0)
        for row in range(2):
            ax=axes[row,column]
            for i,method in enumerate(methods):
                label=cited(method) if method in ('ETBP-TD','MMCB','VLR-BPP') else method
                ax.plot(x,values[row,i],color=colors[i],marker=markers[i],linestyle=styles[i],linewidth=1.1,
                    markersize=3.3,markevery=[0,9,19,29,39,49],markerfacecolor='white' if i else colors[i],label=label)
            ax.set_title('('+chr(ord('a')+row*3+column)+') '+labels[column],pad=3)
            ax.set_xlabel('Service round');ax.set_ylabel('Task completion (%)' if row==0 else 'Own-answer RMSE (RH pp)')
            ax.set_xticks([1,10,20,30,40,50]);finish_axes(ax)
    pd.DataFrame(curve_rows).to_csv(Path(output)/'native_context_round_curves.csv',index=False)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=6,frameon=False,columnspacing=.8,handlelength=1.4,fontsize=8)
    for suffix in ('pdf','svg'):fig.savefig(Path(output)/('native_context_comparison.'+suffix),bbox_inches='tight',pad_inches=.02)
    plt.close(fig)
    return receipt

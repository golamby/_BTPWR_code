"""Execute complete numerical policies on source-bound Q36 paired worlds."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
from multiprocessing import Pool
from collections import OrderedDict
from dataclasses import asdict
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import data,privacy,trust,recovery,recruitment,baselines,bpqa_native
from src import signed_grade_policy as policy
from src.simulate import world_digest,atomic_json
from experiments.formal_q36_cases import cases,methods_for

INPUTS=None;WORLDS=OrderedDict();HEADS={};BPQA_CAL={};OUTPUT=None
RESOURCE_COLUMNS=['commitment','physical','selected_workers','actual_route_bid_sum','settled']

def init(output):
    global INPUTS,OUTPUT
    INPUTS=data.load_inputs();OUTPUT=Path(output)

def perturb(world,c):
    """Simulator-only adversary; these hidden labels never enter a policy."""
    w=dict(world);raw=world['reports_check'].copy();T,n,D=raw.shape
    attack=c.get('stress');h=w['honest_private'];rng=np.random.default_rng(np.random.SeedSequence([c['seed'],771204]))
    if attack in ('sole_bad_certificate','two_bad_certificates'):
        count=1 if attack=='sole_bad_certificate' else 2
        old=np.flatnonzero(w['certified']);bad=np.flatnonzero(~h)[:count]
        cert=w['certified'].copy();r=w['initial_reputation'].copy()
        cert[old[:count]]=False;r[old[:count]]=.5;cert[bad]=True
        r[bad]=np.rint(rng.uniform(.95,1,len(bad))*10000)/10000
        w['certified']=cert;w['initial_reputation']=r
    elif attack in ('small_bias','large_bias','on_off'):
        truth_check=10+30*w['truth_normalized_private'].T
        for t in range(T):
            values=truth_check[t][None,:]+(1. if attack=='small_bias' else 4.)
            if attack=='on_off':
                if t>=20:continue
                values=truth_check[t][None,:]+rng.normal(0,.05,(n,D))
            raw[t,~h]=np.where(w['presence'][t,~h],np.clip(np.broadcast_to(values,(n,D))[~h],10,40),np.nan)
    elif attack=='reference_noise':
        cert=w['certified'];noise=rng.normal(0,3.,raw[:,cert].shape)
        raw[:,cert]=np.where(np.isfinite(raw[:,cert]),np.clip(raw[:,cert]+noise,10,40),np.nan)
    elif attack is not None:raise ValueError(attack)
    # Same codec precision as actual reports. Authorized clipping is explicit,
    # not an outcome-dependent removal of difficult observations.
    w['reports_check']=np.clip(raw,10,40).astype(np.float32).astype(float)
    return w

def get_world(c):
    key=c['name']
    if key not in WORLDS:
        WORLDS[key]=perturb(data.build_world(c,INPUTS),c)
        while len(WORLDS)>2:WORLDS.popitem(last=False)
    return WORLDS[key]

def head_for(w):
    k=hashlib.sha256(w['public_reference'].tobytes()+np.asarray([w['native_offset'],w['native_scale']],dtype='<f8').tobytes()).hexdigest()
    if k not in HEADS:HEADS[k]=recovery.initialize(w['public_reference'],w['native_offset'],w['native_scale'])
    # Fitted-cache entries are pure public fits indexed by actual anchor set.
    return HEADS[k]

def packet_for(w,c,t,virtual=True):
    loc=w['presence'][t];cost=w['actual_cost'][t]
    cfg=privacy.Config(n=len(loc),m=loc.shape[1],T=50,eps1=c['eps1'],eps2=c['eps2'])
    if virtual:
        bits=w['virtual_presence'][t]
        avg=np.divide(cost.sum(1),loc.sum(1),out=np.full(len(loc),3.25),where=loc.sum(1)>0)
        supplied=np.where(bits,avg[:,None],0.)
    else:bits=loc;supplied=cost
    p=privacy.release(bits,supplied,w['RR_coins'][t],cfg.eps1,cfg.eps2,cfg)
    for k in ('bits','hi','nonzero'):p[k][~w['offer_mask'][t]]=False
    return p,cfg

def grade(r):
    z=np.rint(np.asarray(r)*10000).astype(int)
    return (np.minimum(z//1000,9)*1000+500)/10000.

def traditional(method,p,active,r,tasks,cfg,B,seed,t):
    eligible=active&(trust.classify(r)!=0);profile=recruitment.PriceProfile.from_config(cfg)
    limit=recruitment.budget_limit(profile,B);chosen=np.zeros(len(r),bool);bill=0
    def quote(mask):return int(recruitment.quote_numerators(mask[None],p,profile)[0])
    if method=='Random':
        for i in np.random.default_rng(np.random.SeedSequence([seed,43501,t])).permutation(len(r)):
            if not eligible[i]:continue
            trial=chosen.copy();trial[i]=True;q=quote(trial)
            if q<=limit:chosen=trial;bill=q
    elif method=='ETBP-P':
        state,ecfg=baselines.priority_state(r)
        order=baselines.etbp.recruit_offers(p['bits'][:,tasks],min(40,int(eligible.sum())),state,
             np.random.default_rng(np.random.SeedSequence([seed,78102,t])),ecfg,eligible)['selected']
        ids,num,den=recruitment.prefix_budget(order,p,cfg,B)
        return np.asarray(ids,int),float(num/den)
    else:
        p1=1/(1+np.exp(-cfg.eps1));prior=.05
        pos=prior*p1/(prior*p1+(1-prior)*(1-p1));neg=prior*(1-p1)/(prior*(1-p1)+(1-prior)*p1)
        support=np.where(p['bits'],pos,neg)*r[:,None]
        for _ in range(len(r)):
            before=1-np.prod(1-support[chosen],axis=0) if chosen.any() else np.zeros(p['bits'].shape[1])
            best=None
            for i in np.flatnonzero(eligible&~chosen):
                trial=chosen.copy();trial[i]=True;q=quote(trial)
                if q>limit:continue
                gain=float(((1-before[tasks])*support[i,tasks]).mean())
                key=((gain+1e-9)/max(q-bill,1),-int(i))
                if best is None or key>best[0]:best=(key,i,q)
            if best is None:break
            chosen[best[1]]=True;bill=best[2]
    return np.flatnonzero(chosen),bill/profile.cost_denominator

def run_common(w,c,method):
    T,n,D=w['presence'].shape;r=w['initial_reputation'].copy();head=head_for(w)
    pred=np.full((T,D),np.nan);selected=np.zeros((T,n),bool);hist=[r.copy()];resources=[];records=[]
    balance=method not in ('Recruit-NoBalance','Recruit-Neither','C-seeded')
    opportunity=method not in ('Recruit-NoOpportunity','Recruit-Neither','C-seeded')
    count=method not in ('Feedback-NoCount','Feedback-Neither')
    secondary=method not in ('Feedback-NoSecondary','Feedback-Neither')
    beta=1. if method=='Secondary-FullWeight' else .5
    cap=1. if method=='Evidence-Cap1' else 9. if method=='Evidence-Cap9' else 4.
    for t in range(T):
        p,cfg=packet_for(w,c,t);rg=grade(r);start=time.perf_counter();collisions=0
        if method in ('Random','Greedy-C/R','ETBP-P'):
            ids,charge=traditional(method,p,w['offer_mask'][t],rg,w['virtual_task_cells'][t],cfg,w['budget'],c['seed'],t)
        else:
            a=policy.select_round(p,w['offer_mask'][t],rg,w['virtual_task_cells'][t],w['budget'],T-t,
                                  c['eps1'],c['eps2'],balance=balance,opportunity=opportunity)
            ids=np.asarray(a['members'],int);charge=round(a['cost_numerator']/a['cost_denominator']*1e6)/1e6
            collisions=a['active_packet_tie_collisions']
        selection_seconds=time.perf_counter()-start;selected[t,ids]=True
        raw=w['reports_check'][t,ids].copy();start=time.perf_counter()
        fb=policy.requester_round(raw,r[ids],head,count=count,secondary=secondary,secondary_weight=beta,evidence_cap=cap,gamma=c['gamma'])
        pred[t]=fb['prediction'];r[ids]=fb['next_reputation'];hist.append(r.copy())
        physical=float(w['actual_cost'][t,ids].sum());resources.append([charge,physical,len(ids),physical,charge])
        direct=fb.get('direct',{})
        records.append(dict(round=t+1,selector_seconds=selection_seconds,requester_seconds=time.perf_counter()-start,
            direct_regions=int(np.asarray(direct.get('has_credible',[]),bool).sum()),
            recovered_regions=int(np.isfinite(fb['secondary_reference']).sum()),
            scored_workers=int((fb['scored_count']>0).sum()),scored_worker_regions=int(fb['scored_count'].sum()),
            effective_evidence=float(fb['effective_evidence'].sum()),active_packet_tie_collisions=collisions))
        if charge>w['budget']+1e-6:raise AssertionError('Budget excess')
    head['cache'].clear();head['blocked_fill_cache'].clear()
    return dict(predictions=pred,selected=selected,reputations=np.asarray(hist),resources=np.asarray(resources),records=records,
                class_thresholds=dict(low=.3,high=.8,credible_strict=False),constraint='common monetary budget and released broker inputs')

def run_bpp(w,c):
    T,n,D=w['presence'].shape;pred=np.full((T,D),np.nan);chosen=np.zeros((T,n),bool);res=[];records=[]
    for t in range(T):
        p,cfg=packet_for(w,c,t,False);active=np.flatnonzero(w['offer_mask'][t]);tasks=w['requested_task_ids'][t]
        costs=baselines.bpp.published_pcp_costs(p,cfg.eps2)
        local,diag=baselines.bpp.exact_ws(p['bits'][active],costs[active],w['budget'],tasks,cfg.eps1)
        ids=active[local];chosen[t,ids]=True
        raw=w['reports_check'][t,ids];mean=baselines.mean_reports(raw)
        pred[t]=np.clip(w['native_offset']+w['native_scale']*(mean-10)/30,0,100)
        charge=float(costs[ids].sum());physical=float(w['actual_cost'][t,ids].sum())
        res.append([charge,physical,len(ids),physical,charge]);records.append(dict(round=t+1,**diag))
    return dict(predictions=pred,selected=chosen,resources=np.asarray(res),records=records,class_thresholds=None,
                constraint='native private worker selection plus documented actual-report mean head')

def run_native_other(w,c,method):
    T,n,D=w['presence'].shape
    if method=='REAPP':
        from src.reapp_native import campaign
        return campaign(w,c,T)
    if method=='VLR-BPP':
        pred,selected,records=baselines.vlr_campaign(w,c,T)
        res=np.asarray([[x['commitment'],x['physical_cost'],x['selected_workers'],x['physical_cost'],x['commitment']] for x in records])
        return dict(predictions=pred,selected=selected,resources=res,records=records,class_thresholds=None,
                    constraint='author-manuscript-based virtual replacement and BGMC; fixed numeric conventions')
    if method=='BPQA':
        from experiments.run_bpqa_native import calibrate
        key=(c['dataset'],c['n'],c['H'],c['C'],c['budget_override'],c['field'])
        if key not in BPQA_CAL:BPQA_CAL[key]=calibrate(INPUTS,c['dataset'],c['budget_factor'],template=c)
        cal=BPQA_CAL[key];r=w['initial_reputation'].copy();hist=[r.copy()];pred=np.full((T,D),np.nan);sel=np.zeros((T,n),bool);res=[]
        cfg=bpqa_native.Config()
        for t in range(T):
            tasks=w['requested_task_ids'][t];bids=w['actual_cost'][t].sum(1)
            a=bpqa_native.allocate(w['presence'][t][:,tasks],bids,r,w['offer_mask'][t],w['budget'],cal['multiplier'],cfg)
            ids=a['selected'];sel[t,ids]=True;raw=np.full((n,D),np.nan);raw[ids]=w['reports_check'][t,ids];raw[:,~w['task_mask'][t]]=np.nan
            valid=np.isfinite(raw);passed=valid.any(1)&np.all(~valid|((raw>=10)&(raw<=40)),1);raw[~passed]=np.nan
            z,weights=bpqa_native.truth_and_weights(raw,cfg)
            pred[t]=np.clip(w['native_offset']+w['native_scale']*(z-10)/30,0,100)
            r=bpqa_native.update_reputation(r,weights,np.isfinite(raw).any(1),cfg);hist.append(r.copy())
            physical=float(w['actual_cost'][t,ids].sum());res.append([a['total_payment'],physical,len(ids),float(bids[ids].sum()),float(a['payments'][passed].sum())])
        return dict(predictions=pred,selected=sel,reputations=np.asarray(hist),resources=np.asarray(res),records=[],class_thresholds=None,
                    calibration=cal,config=asdict(cfg),constraint='native monetary critical allocation with sparse numeric conventions')
    state=baselines.etbp_initial_state(w);pred=np.full((T,D),np.nan);sel=np.zeros((T,n),bool);hist=[state.trust.copy()];res=[];records=[]
    for t in range(T):
        values,picks,state,record=baselines.native_round(w,t,state,c['seed'],c['dataset'])
        ids=picks[method];sel[t,ids]=True
        pred[t,w['requested_task_ids'][t]]=np.clip(w['native_offset']+w['native_scale']*(values[method]-10)/30,0,100)
        charge=record[method]['commitment'];charge=np.nan if charge is None else charge
        physical=float(w['actual_cost'][t,ids].sum());res.append([charge,physical,len(ids),physical,charge]);records.append(dict(round=t+1,**record[method]))
        hist.append(state.trust.copy())
    result=dict(predictions=pred,selected=sel,resources=np.asarray(res),records=records,
                class_thresholds=dict(low=.4,high=.9,credible_strict=True) if method=='ETBP-TD' else None,
                constraint='native quota' if method=='ETBP-TD' else 'native monetary auction plus documented mean service head')
    if method=='ETBP-TD':result['reputations']=np.asarray(hist)
    return result

def hashes():
    files=list((ROOT/'src').glob('*.py'))+[Path(__file__),Path(__file__).with_name('formal_q36_cases.py')]
    return {str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}

def one(job):
    c,method=job;safe=method.replace('/','_');folder=OUTPUT/'cases'/c['name']/safe;folder.mkdir(parents=True,exist_ok=True)
    source=hashes();wid=None;started=time.perf_counter()
    try:
        w=get_world(c);wid=world_digest(w)
        if (folder/'result.json').exists():
            old=json.loads((folder/'result.json').read_text(encoding='utf-8'))
            output_file=folder/'result.npz'
            intact=output_file.is_file() and hashlib.sha256(output_file.read_bytes()).hexdigest()==old.get('output_sha256')
            if old['source_sha256']==source and old['world_sha256']==wid and intact:return dict(case=c['name'],method=method,status='verified-resume')
            raise ValueError('Refuse source/input-mismatched resume')
        atomic_json(folder/'progress.json',dict(status='running',case=c,method=method,pid=os.getpid()))
        result=run_bpp(w,c) if method=='BPP' else run_native_other(w,c,method) if method in ('ETBP-TD','MMCB','VLR-BPP','REAPP','BPQA') else run_common(w,c,method)
        T,n,D=w['presence'].shape
        if result['predictions'].shape!=(50,100) or result['selected'].shape!=(50,n) or result['resources'].shape!=(50,5):raise AssertionError('Incomplete campaign')
        if np.any(result['selected']&~w['offer_mask']):raise AssertionError('Recruited unavailable worker')
        finite=np.isfinite(result['resources'][:,0])
        if np.any(result['resources'][finite,0]>w['budget']+1e-6):raise AssertionError('Monetary budget exceeded')
        arrays={k:result[k] for k in ('predictions','selected','resources','reputations') if k in result}
        arrays.update(truth=w['truth_native_private'].T,task_mask=w['task_mask'],honest=w['honest_private'],certified=w['certified'])
        np.savez_compressed(folder/'result.npz',**arrays)
        meta={k:v for k,v in result.items() if k not in arrays}
        meta.update(case=c,method=method,status='completed',rounds=50,seconds=time.perf_counter()-started,
                    resource_columns=RESOURCE_COLUMNS,source_sha256=source,world_sha256=wid,
                    output_sha256=hashlib.sha256((folder/'result.npz').read_bytes()).hexdigest(),report_codec='clipped checking [10,40] float32 -> float64')
        atomic_json(folder/'result.json',meta);atomic_json(folder/'progress.json',dict(status='completed',rounds=50))
        return dict(case=c['name'],method=method,status='completed',seconds=meta['seconds'])
    except Exception:
        atomic_json(folder/'progress.json',dict(status='failed',traceback=traceback.format_exc(),world_sha256=wid))
        raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--stage',choices=['primary','sensitivity','traceable','stress'],default='primary')
    ap.add_argument('--workers',type=int,default=12);ap.add_argument('--methods',nargs='+');ap.add_argument('--ablations',action='store_true');a=ap.parse_args()
    worlds=cases(a.stage);jobs=[(c,m) for c in worlds for m in methods_for(c,a.ablations) if a.methods is None or m in a.methods]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/('jobs_'+a.stage+'.json'),dict(worlds=worlds,jobs=[dict(case=c['name'],method=m) for c,m in jobs],source_sha256=hashes()))
    print(json.dumps(dict(stage=a.stage,worlds=len(worlds),jobs=len(jobs),workers=a.workers)),flush=True)
    with Pool(a.workers,initializer=init,initargs=(a.output,)) as pool:
        for i,r in enumerate(pool.imap_unordered(one,jobs)):
            print(json.dumps(dict(done=i+1,total=len(jobs),**r)),flush=True)
    atomic_json(out/('done_'+a.stage+'.json'),dict(status='completed',jobs=len(jobs),worlds=len(worlds)))

if __name__=='__main__':main()

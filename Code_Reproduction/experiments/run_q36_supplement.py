"""Q36 supplemental controls; frozen primary sources and arrays stay unchanged."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
from multiprocessing import Pool
from types import SimpleNamespace
import numpy as np

SOURCE=Path(os.environ.get('BTPWR_Q36_SOURCE',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(SOURCE))
from experiments import run_formal_q36 as formal
from experiments.formal_q36_cases import cases,base,finish
from src import trust,recovery,recruitment,footprints
from src.fast_qualification import DynamicBalanced
from src.simulate import atomic_json,world_digest

OUTPUT=None;PRIMARY=None
FIXED=['Fixed-Acquisition-ActualOnly','Fixed-Acquisition-PublicAffine',
       'Fixed-Acquisition-AllObservedKRR','Fixed-Acquisition-CredibleKRR']
INFO=['ExactScore-Privileged','PaidFootprint-Privileged','Nov3-Mapped-CurrentInputs']


def initialize(output,primary):
    global OUTPUT,PRIMARY
    OUTPUT=Path(output);PRIMARY=Path(primary);formal.init(output)


def source_hashes():
    return dict(formal.hashes(),supplement_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def canonical_order(packet,r):
    keys=[hashlib.sha256(np.asarray([r[i]],dtype='<f8').tobytes()+b''.join(
        np.asarray(packet[k][i],np.uint8).tobytes() for k in ('bits','hi','nonzero'))).hexdigest() for i in range(len(r))]
    return np.array(sorted(range(len(r)),key=lambda i:(keys[i],i)),int)


def select_control(method,p,active,r,support,tasks,cfg,budget,remaining):
    """Canonical packet/grade hash tie rule; no latent behavior input."""
    order=canonical_order(p,r);packet={k:np.asarray(v)[order] for k,v in p.items()}
    if method=='Nov3-Mapped-CurrentInputs':
        ans=recruitment.greedy(packet,np.asarray(active)[order],r[order],tasks,cfg,budget)
    else:
        state=SimpleNamespace(privacy_config=cfg,trust_config=trust.Config(),reputation=r[order],
            support=np.asarray(support)[order],offers=np.asarray(active)[order],packet=packet,
            budget=float(budget),requested_tasks=np.asarray(tasks,int))
        ans=DynamicBalanced(state,remaining).select()
    return order[np.asarray(ans['members'],int)],round(ans['cost_numerator']/ans['cost_denominator']*1e6)/1e6


def current_support(packet,epsilon):
    p1=1/(1+np.exp(-epsilon));prior=.05
    pos=prior*p1/(prior*p1+(1-prior)*(1-p1));neg=prior*(1-p1)/(prior*(1-p1)+(1-prior)*p1)
    return np.where(packet['bits'],pos,neg)


def run_information(w,c,method):
    T,n,D=w['presence'].shape;r=w['initial_reputation'].copy();head=formal.head_for(w)
    pred=np.full((T,D),np.nan);chosen=np.zeros((T,n),bool);history=[r.copy()];resources=[];records=[]
    template=footprints.TemplateSupport(n,w['public_reference'],c['seed']) if method=='PaidFootprint-Privileged' else None
    for t in range(T):
        p,cfg=formal.packet_for(w,c,t);rg=r.copy() if method=='ExactScore-Privileged' else formal.grade(r)
        if template is None:support=current_support(p,c['eps1']);receipt=None;tasks=w['virtual_task_cells'][t]
        else:
            support,receipt=template.predict(p['bits'],c['eps1'],w['offer_mask'][t],trust.classify(rg),t)
            tasks=w['requested_task_ids'][t]
        tick=time.perf_counter()
        if method=='ExactScore-Privileged':
            a=formal.policy.select_round(p,w['offer_mask'][t],rg,tasks,w['budget'],T-t,c['eps1'],c['eps2'])
            ids=np.asarray(a['members'],int);charge=round(a['cost_numerator']/a['cost_denominator']*1e6)/1e6
        else:ids,charge=select_control(method,p,w['offer_mask'][t],rg,support,tasks,cfg,w['budget'],T-t)
        chosen[t,ids]=True
        if template is not None:template.observe(ids,w['presence'][t,ids],t)
        fb=formal.policy.requester_round(w['reports_check'][t,ids].copy(),r[ids],head,gamma=c['gamma'])
        pred[t]=fb['prediction'];r[ids]=fb['next_reputation'];history.append(r.copy())
        physical=float(w['actual_cost'][t,ids].sum());resources.append([charge,physical,len(ids),physical,charge])
        records.append(dict(round=t+1,seconds=time.perf_counter()-tick,support_coordinates='physical' if template else 'virtual',
            prior_history=receipt,selected_workers=len(ids),scored_workers=int((fb['scored_count']>0).sum())))
        if charge>w['budget']+1e-6:raise AssertionError('Budget excess')
    head['cache'].clear();head['blocked_fill_cache'].clear()
    return dict(predictions=pred,selected=chosen,reputations=np.asarray(history),resources=np.asarray(resources),records=records,
                class_thresholds=dict(low=.3,high=.8,credible_strict=False),
                constraint='common monetary cap; privileged numerical input control' if method!='Nov3-Mapped-CurrentInputs' else 'common current released input and signed grade; November3 mapped recruitment objective',
                interpretation='own complete50round numerical trajectory; not a cryptographic/privacy performance comparison')


def reference_audit(raw,r,direct,secondary,w,t):
    """Truth enters only this post-computation evaluation, not source eligibility."""
    classes=trust.classify(r)
    eligible=np.isfinite(raw)&(classes[:,None]==1)&~np.asarray(direct['has_credible'],bool)[None,:]&np.isfinite(secondary)[None,:]
    predicted_check=10+30*(secondary-w['native_offset'])/w['native_scale']
    truth_check=10+30*w['truth_normalized_private'][:,t]
    errors=np.broadcast_to(predicted_check-truth_check,raw.shape)[eligible]
    return dict(secondary_worker_region_checks=int(eligible.sum()),
        secondary_distinct_regions=int(np.any(eligible,axis=0).sum()),
        secondary_sum_absolute_error=float(np.abs(errors).sum()),secondary_sum_squared_error=float((errors*errors).sum()),
        secondary_within3=int((np.abs(errors)<=3.).sum()),
        direct_regions=int(np.asarray(direct['has_credible'],bool).sum()),
        direct_worker_region_checks=int(np.isfinite(direct['reference_by_worker']).sum()))


def fixed_replay(w,c,stored):
    T,n,D=w['presence'].shape;head=formal.head_for(w)
    predictions={k:np.full((T,D),np.nan) for k in FIXED};audit=[]
    for t in range(T):
        ids=np.flatnonzero(stored['selected'][t]);r=stored['reputations'][t,ids]
        raw=w['reports_check'][t,ids].copy()
        if not len(ids):
            audit.append(dict(round=t+1,secondary_worker_region_checks=0,secondary_distinct_regions=0,
                secondary_sum_absolute_error=0.,secondary_sum_squared_error=0.,secondary_within3=0,direct_regions=0,direct_worker_region_checks=0));continue
        if np.any(trust.classify(r)==0):raise AssertionError('Primary selected excluded worker')
        direct=trust.feedback(raw,r,trust.Config(gamma=c['gamma']))
        actual=recovery.decode_actual(direct['consensus'],head)
        credible=direct['credible_consensus'];anchor=recovery.decode_actual(credible,head)
        predictions[FIXED[0]][t]=actual
        predictions[FIXED[1]][t]=recovery.linear_complete(credible,actual,head['linear'])[0]
        predictions[FIXED[2]][t]=recovery.complete(direct['consensus'],actual,head)[0]
        predictions[FIXED[3]][t]=recovery.complete(credible,actual,head)[0]
        expected=stored['predictions'][t];replayed=predictions[FIXED[3]][t]
        if not np.array_equal(np.isfinite(expected),np.isfinite(replayed)):raise AssertionError('Primary domain mismatch')
        if not np.allclose(expected,replayed,atol=1e-10,rtol=0,equal_nan=True):raise AssertionError('Primary recovery replay mismatch')
        secondary=recovery.complete(credible,anchor,head)[0]
        secondary[np.asarray(direct['has_credible'],bool)]=np.nan
        record=reference_audit(raw,r,direct,secondary,w,t);record['round']=t+1;audit.append(record)
    head['cache'].clear();head['blocked_fill_cache'].clear()
    return predictions,audit


def save(c,method,w,result,extra):
    folder=OUTPUT/'cases'/c['name']/method;folder.mkdir(parents=True,exist_ok=True)
    arrays={k:result[k] for k in ('predictions','selected','reputations','resources')}
    arrays.update(truth=w['truth_native_private'].T,task_mask=w['task_mask'],honest=w['honest_private'],certified=w['certified'])
    if arrays['predictions'].shape!=(50,100) or arrays['selected'].shape!=(50,c['n']):raise AssertionError('Incomplete campaign')
    np.savez_compressed(folder/'result.npz',**arrays)
    meta={k:v for k,v in result.items() if k not in arrays}
    meta.update(case=c,method=method,status='completed',rounds=50,resource_columns=formal.RESOURCE_COLUMNS,
        source_sha256=source_hashes(),world_sha256=world_digest(w),output_sha256=hashlib.sha256((folder/'result.npz').read_bytes()).hexdigest(),**extra)
    atomic_json(folder/'result.json',meta)


def intact(c,methods,w):
    for method in methods:
        folder=OUTPUT/'cases'/c['name']/method
        if not (folder/'result.json').exists():return False
        old=json.loads((folder/'result.json').read_text());f=folder/'result.npz'
        if old['source_sha256']!=source_hashes() or old['world_sha256']!=world_digest(w) or not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=old['output_sha256']:
            raise ValueError('Refuse source/input/output mismatched resume')
    return True


def one(job):
    c,method=job;family='fixed-acquisition' if method=='fixed' else 'information'
    log=OUTPUT/'progress'/(c['name']+'_'+method+'.json');log.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    try:
        w=formal.get_world(c);names=FIXED if method=='fixed' else [method]
        if intact(c,names,w):return dict(case=c['name'],method=method,status='verified-resume')
        atomic_json(log,dict(status='running',pid=os.getpid(),source_sha256=source_hashes()))
        if method=='fixed':
            source=PRIMARY/'cases'/c['name']/'BTPWR';meta=json.loads((source/'result.json').read_text())
            f=source/'result.npz'
            if hashlib.sha256(f.read_bytes()).hexdigest()!=meta['output_sha256'] or world_digest(w)!=meta['world_sha256']:raise AssertionError('Primary source mismatch')
            if meta['source_sha256']!=formal.hashes():raise AssertionError('Frozen primary code mismatch')
            with np.load(f) as z:stored={k:z[k].copy() for k in ('predictions','selected','reputations','resources')}
            pred,audit=fixed_replay(w,c,stored)
            for name in names:
                result=dict(predictions=pred[name],selected=stored['selected'],reputations=stored['reputations'],resources=stored['resources'],
                    records=audit,class_thresholds=dict(low=.3,high=.8,credible_strict=False),
                    interpretation='fixed acquisition and BTPWR reputation path; alternate service predictions only, not a full alternate feedback campaign')
                save(c,name,w,result,dict(family=family,primary_result_sha256=meta['output_sha256'],feedback_recruitment_fixed=True))
        else:save(c,method,w,run_information(w,c,method),dict(family=family,feedback_recruitment_fixed=False))
        atomic_json(log,dict(status='completed',seconds=time.perf_counter()-start))
        return dict(case=c['name'],method=method,status='completed',seconds=time.perf_counter()-start)
    except Exception:
        atomic_json(log,dict(status='failed',traceback=traceback.format_exc()));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--primary',required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--family',choices=['fixed','information','all'],default='all');a=p.parse_args()
    fixed=[(c,'fixed') for c in cases('primary')]
    information=[(finish(base(d,s)),m) for d in formal.data.DATASETS for s in range(9701,9704) for m in INFO]
    jobs=(fixed if a.family=='fixed' else information if a.family=='information' else fixed+information)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'jobs_supplement.json',dict(status='registered',jobs=[dict(case=c,method=m) for c,m in jobs],source_sha256=source_hashes(),primary_root=a.primary))
    print(json.dumps(dict(jobs=len(jobs),fixed_worlds=len(fixed),information_method_worlds=len(information),workers=a.workers)),flush=True)
    with Pool(a.workers,initializer=initialize,initargs=(a.output,a.primary)) as pool:
        for i,answer in enumerate(pool.imap_unordered(one,jobs)):print(json.dumps(dict(done=i+1,total=len(jobs),**answer)),flush=True)
    atomic_json(out/'done_supplement.json',dict(status='completed',jobs=len(jobs),fixed_worlds=24,information_method_worlds=27))


if __name__=='__main__':
    import signal
    out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
    if hasattr(signal,'SIGALRM'):
        def timed_out(*args):raise TimeoutError('Supplement campaign hard timeout2700seconds')
        signal.signal(signal.SIGALRM,timed_out);signal.alarm(2700)
    try:
        main();atomic_json(out/'exit.json',dict(status='completed',exit_code=0))
    except BaseException:
        atomic_json(out/'exit.json',dict(status='failed',exit_code=1,traceback=traceback.format_exc()));raise

"""Five complete policies on one shared 50-round world."""
from pathlib import Path
import copy,json,time,hashlib
import numpy as np
from . import data,privacy,trust,recruitment,recovery,baselines,metrics
from .simulate import calibration_for,json_default,world_digest,scientific_code_digest
from .footprints import TemplateSupport
from .admission import AdmissionPlanner,LegalState,ReleasedFrame,ScoredRound,select_additive
from . import _bpqa

METHODS=('admission','admission-no-promotion','template-first','old','etbp')
LABELS=('Admission-aware policy','No predicted admission','C-first','Frozen BTPWR','Matched ETBP priority')
WORKER_COLUMNS=('honest_C','malicious_M','malicious_C','honest_M','honest_U','malicious_U',
                'paid_pre_C','paid_pre_U','paid_new_C')


def worker_audit(reputations,selected,honest):
    """Evaluation labels are simulated behavior, never selector inputs."""
    end=np.asarray(reputations)[1:];pre=np.asarray(reputations)[:-1];honest=np.asarray(honest,bool)[None,:]
    C=end>=.8;M=end<.3;U=~(C|M)
    new=(np.asarray(reputations)[0]<.8)[None,:]
    result=np.stack([(C&honest).sum(1),(M&~honest).sum(1),(C&~honest).sum(1),(M&honest).sum(1),
        (U&honest).sum(1),(U&~honest).sum(1),(selected&(pre>=.8)).sum(1),
        (selected&(pre>=.3)&(pre<.8)).sum(1),(selected&(pre>=.8)&new).sum(1)],axis=1)
    if not np.all(result[:,:6].sum(1)==end.shape[1]):raise AssertionError('Worker audit partition mismatch')
    return result


def scored_events(reports, feedback, reputation, round_index, cfg, factors=None):
    """Represent exactly the feedback transitions applied in a round."""
    events=[]
    for i in np.flatnonzero(feedback['scored_count']):
        valid=np.isfinite(feedback['reference_by_worker'][i])&np.isfinite(reports[i])
        if valid.sum()!=feedback['scored_count'][i]:
            raise AssertionError('Feedback history and applied score counts differ')
        e=np.abs(reports[i,valid]-feedback['reference_by_worker'][i,valid])/cfg.score_width
        good=e<=cfg.score_tolerance
        f=1. if factors is None else factors[i,valid]
        a=float(np.mean(np.where(good,f*(1-e/cfg.score_tolerance),0)))
        b=float(np.mean(np.where(good,0,f*(1-cfg.score_tolerance/np.maximum(e,1e-300)))))
        events.append(ScoredRound(int(i),round_index,float(reputation[i]),tuple(map(float,e)),a,b,float(feedback['next_reputation'][i])))
    return events


def run_case(case,world,output,calibration=None,methods=METHODS,planner_types=None,
             reference_limits=None,round_observer=None,reference_iteration_cap=None,
             reputation_modes=None,selection_modes=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    w=world;T,n,D=w['presence'].shape
    if T!=50:raise ValueError('The registered experiment uses a 50-round public contract')
    H=w['public_reference'];cfg=privacy.Config(n=n,m=D,T=T,eps1=case['eps1'],eps2=case['eps2'])
    tc=trust.Config(gamma=case['gamma'])
    cal=calibration_for(w,output) if calibration is None else calibration
    arrays={};predictions=[];histories=[];rows=[];worker_counts=[]
    planner_types = {} if planner_types is None else planner_types
    reference_limits = {} if reference_limits is None else reference_limits
    reputation_modes = {} if reputation_modes is None else reputation_modes
    selection_modes = {} if selection_modes is None else selection_modes
    for method in methods:
        started=time.perf_counter()
        head=recovery.initialize(H,w['native_offset'],w['native_scale'],cal)
        r=w['initial_reputation'].copy();hist=[r.copy()];pred=[];pre=[];selected=[];stats=[];timing=[]
        bpqa_arm = method in selection_modes
        planning = (method.startswith('admission') or method in planner_types
                    or method in ('bpqa-coupled', 'deng-coupled'))
        c_seeded = method in ("template-first", "template-first-panel")
        model=TemplateSupport(n,H,case['seed']) if (planning or c_seeded) and not bpqa_arm else None
        priority,pcfg=baselines.priority_state(r)
        events=[];frames=[];offer_counts=np.zeros(n,int)
        for t in range(T):
            loc=w['presence'][t];cost=w['actual_cost'][t];v=w['virtual_presence'][t];offers=w['offer_mask'][t]
            avg=np.divide(cost.sum(1),loc.sum(1),out=np.full(n,3.25),where=loc.sum(1)>0)
            packet=privacy.release(v,np.where(v,avg[:,None],0),w['RR_coins'][t],cfg.eps1,cfg.eps2,cfg)
            for key in ('bits','hi','nonzero'):packet[key][~offers]=False
            if bpqa_arm:
                order, num, den = _bpqa.select(packet, offers, r, w['virtual_task_cells'][t], cfg, w['budget'])
                ids = np.asarray(order, int)
            elif method == 'random':
                key=[case['seed'],79101,baselines.DATASETS.index(case['dataset']),t]
                order=np.random.default_rng(np.random.SeedSequence(key)).permutation(np.flatnonzero(offers))
                ids,num,den=recruitment.prefix_budget(order,packet,cfg,w['budget'])
            elif method == 'greedy-cover':
                ids,num,den=_bpqa.select(packet,offers,r,w['virtual_task_cells'][t],cfg,w['budget'])
            elif planning:
                start=time.perf_counter();prior=copy.deepcopy(model)
                support,receipt=model.predict(packet['bits'],cfg.eps1,offers,trust.classify(r,tc),t)
                state=LegalState(t,case['seed'],packet,offers,r.copy(),w['requested_task_ids'][t],w['virtual_task_cells'][t],cfg,tc,
                    float(w['budget']),H,support,receipt,tuple(events),offer_counts.copy(),t,tuple(frames),prior)
                tick=time.perf_counter();planner=planner_types.get(method,AdmissionPlanner)(state,T-t,allow_admission=method!='admission-no-promotion')
                prepare=time.perf_counter()-tick;tick=time.perf_counter();answer=planner.select();search=time.perf_counter()-tick
                timing.append([time.perf_counter()-start,receipt['seconds'],prepare,search,answer['value_evaluations'],
                    ['template-additive','template-first','rollout-greedy','fidelity-proposal','coverage-portfolio'].index(answer['source']),answer['value']])
                ids=answer['members'];num=answer['cost_numerator'];den=answer['cost_denominator']
            elif c_seeded:
                support,_=model.predict(packet['bits'],cfg.eps1,offers,trust.classify(r,tc),t)
                answer=select_additive(packet,offers,r,w['requested_task_ids'][t],cfg,w['budget'],tc,support,True)
                ids=answer['members'];num=answer['cost_numerator'];den=answer['cost_denominator']
            elif method=='old':
                answer=recruitment.greedy(packet,offers,r,w['virtual_task_cells'][t],cfg,w['budget'],tc)
                ids=answer['members'];num=answer['cost_numerator'];den=answer['cost_denominator']
            else:
                key=[case['seed'],78102,t] if case.get('field')=='beijing_hourly' else [case['seed'],78102,baselines.DATASETS.index(case['dataset']),t]
                order=baselines.etbp.recruit_offers(packet['bits'][:,w['virtual_task_cells'][t]],min(40,int(offers.sum())),priority,
                    np.random.default_rng(np.random.SeedSequence(key)),pcfg,offers)['selected']
                ids,num,den=recruitment.prefix_budget(order,packet,cfg,w['budget'])
            if num/den>w['budget']+1e-7 or not np.all(offers[ids]):raise AssertionError('Invalid payment or participation')
            pre.append(r.copy());raw=np.full((n,D),np.nan);raw[ids]=w['reports_check'][t,ids]
            if model is not None:model.observe(np.asarray(ids,int),np.isfinite(raw[ids]),t)
            raw[trust.classify(r,tc)==0]=np.nan
            fb=trust.feedback(raw,r,tc,reference_limit=reference_limits.get(method),
                              reference_iteration_cap=reference_iteration_cap)
            recovery_start=time.perf_counter() if round_observer is not None else None
            actual=recovery.decode_actual(fb['consensus'],head)
            pred_cur=recovery.complete(fb['credible_consensus'],actual,head)[0]
            pred.append(pred_cur)
            fb_used=fb
            feedback_reports=raw
            feedback_factors=None
            mode=reputation_modes.get(method)
            if mode in ('bpqa','deng'):
                # Recovery returns native physical units; feedback expects the
                # registered check scale [10, 40].  Convert before filling.
                pred_check=10.0+30.0*(pred_cur-float(head['offset']))/float(head['scale'])
                filled=raw.copy()
                rows_to_fill=np.arange(n) if mode=='bpqa' else np.asarray(ids,int)
                for i in rows_to_fill:
                    missing=~np.isfinite(filled[i]) & np.isfinite(pred_check)
                    filled[i,missing]=pred_check[missing]
                fb_used=trust.feedback(filled,r,tc,reference_limit=reference_limits.get(method),
                                       reference_iteration_cap=reference_iteration_cap)
                feedback_reports=filled
            elif mode in ('recovery-ref','recovery-ref-025'):
                from .reference_experiments import reference_feedback
                beta=1. if mode=='recovery-ref' else .25
                # Construct the secondary reference from actual credible anchors
                # only. In particular, never feed this worker's own report into
                # the estimate used to score it.
                credible_native=recovery.decode_actual(fb['credible_consensus'],head)
                secondary,_=recovery.complete(fb['credible_consensus'],credible_native,head)
                secondary[np.asarray(fb['has_credible'],bool)]=np.nan
                nxt,_,refs,feedback_factors,counts=reference_feedback(raw,r,secondary,None,tc,beta,
                    float(head['offset']),float(head['scale']))
                fb_used=dict(fb,next_reputation=nxt,reference_by_worker=refs,scored_count=counts,
                    recovery_reference_mask=np.isfinite(refs)&~np.isfinite(fb['reference_by_worker']))
            elif mode in ('count-direct','count-recovery','count-recovery-full'):
                from .count_feedback import feedback as count_feedback
                secondary=None
                if mode!='count-direct':
                    credible_native=recovery.decode_actual(fb['credible_consensus'],head)
                    secondary,_=recovery.complete(fb['credible_consensus'],credible_native,head)
                    secondary[np.asarray(fb['has_credible'],bool)]=np.nan
                fb_used=count_feedback(raw,r,secondary,tc,float(head['offset']),float(head['scale']),
                                       .5 if mode=='count-recovery' else 1.)
            elif mode in ('region-direct','region-recovery'):
                from .region_feedback import feedback as region_feedback
                secondary=None
                if mode=='region-recovery':
                    credible_native=recovery.decode_actual(fb['credible_consensus'],head)
                    secondary,_=recovery.complete(fb['credible_consensus'],credible_native,head)
                    secondary[np.asarray(fb['has_credible'],bool)]=np.nan
                fb_used=region_feedback(raw,r,secondary,tc,float(head['offset']),float(head['scale']))
            if planning:
                # The history supplied to the next planner must describe the
                # same transitions actually applied by this experimental arm.
                # The all-worker synthetic control can also score nonselected
                # workers, so record every worker with a scored transition.
                if mode not in ('region-direct','region-recovery','count-direct','count-recovery','count-recovery-full'):
                    events.extend(scored_events(feedback_reports,fb_used,r,t,tc,feedback_factors))
                offer_counts+=offers
                frames.append(ReleasedFrame(t,copy.deepcopy(packet),offers.copy(),w['requested_task_ids'][t].copy(),w['virtual_task_cells'][t].copy()))
            if round_observer is not None:
                round_observer(
                    method=method, index=t, selected=ids, numerator=num, denominator=den,
                    packet=packet, offers=offers, pre=r, reports=raw, feedback=fb_used,
                    predictions=pred_cur, events=events, frames=frames,
                    selection_stats=timing[-1] if planning else None,
                    recovery_seconds=time.perf_counter()-recovery_start,
                )
            pick=np.zeros(n,bool);pick[ids]=True;selected.append(pick)
            stats.append([num/den,cost[ids].sum(),fb_used['has_credible'].sum(),fb_used['scored_count'].sum(),np.any(trust.classify(r,tc)[ids]==2),len(ids)])
            r=fb_used['next_reputation'];hist.append(r.copy())
            if method=='etbp':
                priority.trust=r.copy();priority.evaluation_count+=fb['scored_count'];priority.recruited_count[ids]+=1
                priority.history.append(r.copy());priority.period+=1
        pred=np.asarray(pred);stats=np.asarray(stats);predictions.append(pred);histories.append(hist)
        arrays[method+'__pre']=np.asarray(pre);arrays[method+'__selected']=np.asarray(selected);arrays[method+'__stats']=stats
        worker_counts.append(worker_audit(hist,np.asarray(selected),w['honest_private']))
        if timing:arrays[method+'__selector_stats']=np.asarray(timing)
        m=metrics.requested_metrics(w['truth_native_private'].T,pred,w['task_mask'])
        row={k:m[k] for k in ('completion_percent','answered_rate_percent','all_answered_MAE','all_answered_RMSE')}
        row.update(method=method,seconds=time.perf_counter()-started,payment=float(stats[:,0].sum()),physical=float(stats[:,1].sum()),anchors=float(stats[:,2].sum()))
        rows.append(row);print(json.dumps(dict(case=case['name'],**row)),flush=True)
    np.savez_compressed(output/(case['name']+'.npz'),methods=np.array(methods),predictions=np.asarray(predictions),
        reputations=np.asarray(histories),truth=w['truth_native_private'].T,task_mask=w['task_mask'],
        worker_counts=np.asarray(worker_counts),worker_columns=np.array(WORKER_COLUMNS),**arrays)
    (output/(case['name']+'.json')).write_text(json.dumps(dict(case=case,results=rows,
        world_sha256=world_digest(w),scientific_code_sha256=scientific_code_digest()),indent=2,default=json_default))
    return rows


def export_evidence(source,destination,case):
    """Publish predictions and aggregate costs; exclude worker-level state/masks."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    with np.load(source,allow_pickle=False) as z:
        methods=z['methods'].tolist();unit=10 if case.get('field')=='beijing_hourly' else 1
        stats=np.asarray([z[m+'__stats'][:,:6] for m in methods])
        result=dict(methods=np.array(methods),predictions=z['predictions']*unit,truth=z['truth']*unit,task_mask=z['task_mask'],resources=stats,
                    case_json=np.array(json.dumps(case)),error_units=np.array('microgram/m3' if unit==10 else 'controlled field units'))
        if 'worker_counts' in z:
            result.update(worker_counts=z['worker_counts'],worker_columns=z['worker_columns'])
        else:
            _,honest,_,_=data.population(int(case['n']),float(case['H']),float(case['C']),int(case['seed']))
            counts=[worker_audit(z['reputations'][i],z[m+'__selected'],honest) for i,m in enumerate(methods)]
            result.update(worker_counts=np.asarray(counts),worker_columns=np.array(WORKER_COLUMNS))
        np.savez_compressed(destination/(case['name']+'.npz'),**result)
    return destination/(case['name']+'.npz')


def audit_worker_feedback(world,source,case,reference_limits=None):
    """Replay every real worker transition independently of the recruitment planner."""
    w=world;tc=trust.Config(gamma=case['gamma']);rounds=0;scored=0
    reference_limits = {} if reference_limits is None else reference_limits
    with np.load(source,allow_pickle=False) as z:
        for k,method in enumerate(z['methods'].tolist()):
            history=z['reputations'][k];pre=z[method+'__pre'];selected=z[method+'__selected'];stats=z[method+'__stats']
            np.testing.assert_array_equal(history[0],w['initial_reputation'])
            np.testing.assert_array_equal(pre,history[:-1])
            if np.any(selected&~w['offer_mask']):raise ValueError('A nonparticipant was recruited')
            for t,pick in enumerate(selected):
                raw=np.full(w['reports_check'][t].shape,np.nan);raw[pick]=w['reports_check'][t,pick]
                raw[trust.classify(pre[t],tc)==0]=np.nan
                feedback=trust.feedback(raw,pre[t],tc,reference_limit=reference_limits.get(method))
                np.testing.assert_array_equal(feedback['next_reputation'],history[t+1])
                np.testing.assert_allclose(stats[t,1],w['actual_cost'][t,pick].sum(),rtol=0,atol=1e-8)
                if stats[t,2]!=feedback['has_credible'].sum() or stats[t,3]!=feedback['scored_count'].sum():raise ValueError('Feedback aggregates differ')
                if not 0<=stats[t,0]<=w['budget']+1e-7:raise ValueError('Recorded payment exceeds budget')
                rounds+=1;scored+=int(feedback['scored_count'].sum())
    return dict(method_rounds=rounds,worker_state_transitions=rounds*w['presence'].shape[1],scored_reports=scored,
        checks='Initial-state continuity, pre/post continuity, actual participation, complete real feedback replay, physical costs, payment cap and evidence counts.')

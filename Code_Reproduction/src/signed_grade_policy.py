"""Shared permitted-input policy for plaintext replay and role services."""
import hashlib
from types import SimpleNamespace
import numpy as np
from . import privacy, trust, recovery, recruitment
from .fast_qualification import DynamicBalanced

def select_round(packet, active, reputation_grade, virtual_task_ids, budget,
                 remaining=50, eps1=2.1, eps2=2.7, balance=True, opportunity=True):
    r=np.asarray(reputation_grade,float);n=len(r);D=np.asarray(packet['bits']).shape[1]
    keys=[hashlib.sha256(np.asarray([r[i]],dtype='<f8').tobytes()+b''.join(np.asarray(packet[k][i],np.uint8).tobytes() for k in ('bits','hi','nonzero'))).hexdigest() for i in range(n)]
    order=np.array(sorted(range(n),key=lambda i:(keys[i],i)),int)
    p={k:np.asarray(v)[order] for k,v in packet.items()};rr=r[order]
    p1=1/(1+np.exp(-eps1));prior=.05
    positive=prior*p1/(prior*p1+(1-prior)*(1-p1));negative=prior*(1-p1)/(prior*(1-p1)+(1-prior)*p1)
    state=SimpleNamespace(privacy_config=privacy.Config(n=n,m=D,T=50,eps1=eps1,eps2=eps2),trust_config=trust.Config(),reputation=rr,
        support=np.where(p['bits'],positive,negative),offers=np.asarray(active,bool)[order],packet=p,budget=float(budget),requested_tasks=np.asarray(virtual_task_ids,int))
    class Policy(DynamicBalanced):
        def value(self,ids):
            if opportunity:return super().value(ids)
            ids=np.asarray(ids,int);c=ids[self.classes[ids]==2];ref=1-np.prod(1-self.p[c],axis=0) if len(c) else np.zeros(D)
            self.evals+=1
            return float(ref[np.asarray(self.s.requested_tasks,int)].mean())
        def select(self):
            if balance:return super().select()
            selected=np.zeros(n,bool);current=0;eligible=self.s.offers&(self.classes!=0)
            for _ in range(n):
                base=self.value(np.flatnonzero(selected));best=None
                for i in np.flatnonzero(eligible&~selected):
                    temp=selected.copy();temp[i]=True;c=int(recruitment.quote_numerators(temp[None],self.s.packet,self.price)[0])
                    if c>self.limit:continue
                    gain=(max(self.value(np.flatnonzero(temp))-base,0.)+1e-9)*(.5+.5*float(self.s.reputation[i]))
                    key=(gain/max(c-current,1),-int(i))
                    if best is None or key>best[0]:best=(key,int(i),c)
                if best is None:break
                selected[best[1]]=True;current=best[2]
            return dict(members=np.flatnonzero(selected),cost_numerator=current,cost_denominator=self.price.cost_denominator,value=self.value(np.flatnonzero(selected)),value_evaluations=self.evals)
    ans=Policy(state,remaining).select();ans['members']=order[ans['members']]
    eligible_keys=[keys[i] for i in np.flatnonzero(active)]
    ans['active_packet_tie_collisions']=len(eligible_keys)-len(set(eligible_keys))
    return ans

def requester_round(selected_raw_float32,selected_exact_r,head,count=True,secondary=True,secondary_weight=.5,evidence_cap=4.,gamma=.6):
    raw=np.asarray(selected_raw_float32,dtype=np.float32).astype(float);r=np.asarray(selected_exact_r,float)
    tc=trust.Config(gamma=gamma)
    if not len(r):return dict(prediction=np.full(raw.shape[1],np.nan),next_reputation=r.copy(),effective_evidence=np.zeros(0),scored_count=np.zeros(0,int),secondary_reference=np.full(raw.shape[1],np.nan))
    raw[trust.classify(r,tc)==0]=np.nan
    direct=trust.feedback(raw,r,tc)
    actual=recovery.decode_actual(direct['consensus'],head)
    service,_=recovery.complete(direct['credible_consensus'],actual,head)
    anchor=recovery.decode_actual(direct['credible_consensus'],head)
    ref,_=recovery.complete(direct['credible_consensus'],anchor,head)
    ref[np.asarray(direct['has_credible'],bool)]=np.nan
    refs=direct['reference_by_worker'].copy();classes=trust.classify(r,tc)
    if secondary:
        check=10+30*(ref-head['offset'])/head['scale']
        add=np.isfinite(raw)&~np.isfinite(refs)&np.isfinite(check)[None,:]&(classes[:,None]==1)
        refs=np.where(add,check[None,:],refs)
    else:add=np.zeros_like(raw,bool)
    valid=np.isfinite(raw)&np.isfinite(refs);weights=np.where(add,secondary_weight,1.)*valid
    errors=np.abs(raw-refs)/tc.score_width
    A=(np.exp(-r/tc.score_shape)-np.exp(-1/tc.score_shape))/(1-np.exp(-1/tc.score_shape))
    scores=np.where(errors<=tc.score_tolerance,tc.gamma*A[:,None]*(1-errors/tc.score_tolerance),-tc.gamma*(1-A[:,None])*(1-tc.score_tolerance/np.maximum(errors,1e-300)))
    effective=weights.sum(1)
    mean=np.divide(np.where(valid,weights*scores,0.).sum(1),effective,out=np.zeros(len(r)),where=effective>0)
    multiplier=np.sqrt(np.minimum(effective,evidence_cap)) if count else np.ones(len(r))
    nxt=np.rint(np.clip(r+mean*multiplier,0,1)*tc.reputation_scale)/tc.reputation_scale
    nxt=np.where((effective>0)&(classes!=0),nxt,r)
    return dict(prediction=service,next_reputation=nxt,effective_evidence=effective,scored_count=valid.sum(1),secondary_reference=ref,direct=direct,reference_by_worker=refs)

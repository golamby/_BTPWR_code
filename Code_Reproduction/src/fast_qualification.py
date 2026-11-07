"""Exploratory one-step reference-opportunity selector; no latent labels."""
import numpy as np
from . import recruitment,trust

class FastQualification:
    def __init__(self,state,remaining_contract_rounds,allow_admission=True):
        self.s=state;self.remaining=remaining_contract_rounds
        self.price=recruitment.PriceProfile.from_config(state.privacy_config)
        self.limit=recruitment.budget_limit(self.price,state.budget)
        self.classes=trust.classify(state.reputation,state.trust_config)
        self.p=np.clip(np.asarray(state.support,float),0,1)
        self.evals=0
    def value(self,ids):
        self.evals+=1
        ids=np.asarray(ids,int);n=len(self.classes)
        if not len(ids):return 0.
        credible=ids[self.classes[ids]==2];unknown=ids[self.classes[ids]==1]
        ref=1-np.prod(1-self.p[credible],axis=0) if len(credible) else np.zeros(self.p.shape[1])
        coverage=float(ref[np.asarray(self.s.requested_tasks,int)].mean())
        # Estimated probability of at least one other-credible opportunity.
        opportunities=1-np.prod(1-self.p[unknown]*ref[None,:],axis=1) if len(unknown) else np.zeros(0)
        progress=np.clip((self.s.reputation[unknown]-.3)/.5,0,1)
        return coverage+3*float(np.sum(opportunities*(.5+.5*progress)))/n
    def select(self):
        n=len(self.classes);selected=np.zeros(n,bool);current=0;den=self.price.cost_denominator
        candidates=self.s.offers & (self.classes!=0)
        def charge(mask):return int(recruitment.quote_numerators(mask[None],self.s.packet,self.price)[0])
        # Ensure a reference can be obtained before optimizing paired opportunity.
        possible=[]
        for i in np.flatnonzero(candidates & (self.classes==2)):
            temp=selected.copy();temp[i]=True;c=charge(temp)
            if c<=self.limit:possible.append((self.value([i])/max(c,1),int(i),c))
        if possible:
            _,i,current=max(possible,key=lambda a:(a[0],-a[1]));selected[i]=True
        else:
            from .admission import select_additive
            answer=select_additive(self.s.packet,self.s.offers,self.s.reputation,self.s.requested_tasks,
                self.s.privacy_config,self.s.budget,self.s.trust_config,self.s.support)
            return dict(members=answer['members'],source='template-additive',value=0.,
                cost_numerator=answer['cost_numerator'],cost_denominator=answer['cost_denominator'],value_evaluations=self.evals)
        for _ in range(n):
            base=self.value(np.flatnonzero(selected));best=None
            for i in np.flatnonzero(candidates & ~selected):
                temp=selected.copy();temp[i]=True;c=charge(temp)
                if c>self.limit:continue
                gain=self.value(np.flatnonzero(temp))-base
                if gain<=1e-12:continue
                key=(gain/max(c-current,1),-int(i))
                if best is None or key>best[0]:best=(key,int(i),c)
            if best is None:break
            selected[best[1]]=True;current=best[2]
        return dict(members=np.flatnonzero(selected),source='template-additive',
                    value=self.value(np.flatnonzero(selected)),cost_numerator=current,
                    cost_denominator=den,value_evaluations=self.evals)

class WeightedOpportunity(FastQualification):
    """Original count-coupled class weights plus current reference opportunity."""
    def value(self,ids):
        self.evals+=1
        ids=np.asarray(ids,int);n=len(self.classes)
        if not len(ids):return 0.
        weights,_,_=recruitment.mapweights(self.s.reputation,self.s.trust_config)
        credible=ids[self.classes[ids]==2];unknown=ids[self.classes[ids]==1]
        ref=1-np.prod(1-self.p[credible],axis=0) if len(credible) else np.zeros(self.p.shape[1])
        coverage=float(ref[np.asarray(self.s.requested_tasks,int)].mean())
        opportunities=1-np.prod(1-self.p[unknown]*ref[None,:],axis=1) if len(unknown) else np.zeros(0)
        cweight=float(weights[credible[0]]) if len(credible) else 0.
        return cweight*coverage+3*float(np.sum(opportunities*weights[unknown]))/n

class DynamicBalanced(FastQualification):
    """Adaptive credible/unknown recruitment using only estimated state."""
    def select(self):
        n=len(self.classes);selected=np.zeros(n,bool);current=0
        c_fraction=float(np.mean(self.classes==2))
        unknown_target=float(np.clip(.5*(1-max(0.,c_fraction-.1)/.7),.1,.5))
        eligible=self.s.offers & (self.classes!=0)
        def charge(mask):return int(recruitment.quote_numerators(mask[None],self.s.packet,self.price)[0])
        for _ in range(n):
            c_count=int(np.sum(selected & (self.classes==2)))
            u_count=int(np.sum(selected & (self.classes==1)))
            preferred=2 if c_count==0 else (1 if u_count<unknown_target*(c_count+u_count+1) else 2)
            base=self.value(np.flatnonzero(selected));best=None
            # If the preferred class is absent or unaffordable, use the other class.
            for wanted in (preferred,3-preferred):
                for i in np.flatnonzero(eligible & ~selected & (self.classes==wanted)):
                    temp=selected.copy();temp[i]=True;c=charge(temp)
                    if c>self.limit:continue
                    gain=self.value(np.flatnonzero(temp))-base
                    # Within each class, higher estimated reputation receives higher weight.
                    factor=.5+.5*float(self.s.reputation[i])
                    gain=(max(gain,0.)+1e-9)*factor
                    key=(gain/max(c-current,1),-int(i))
                    if best is None or key>best[0]:best=(key,int(i),c)
                if best is not None:break
            if best is None:break
            selected[best[1]]=True;current=best[2]
        return dict(members=np.flatnonzero(selected),source='template-additive',
            value=self.value(np.flatnonzero(selected)),cost_numerator=current,
            cost_denominator=self.price.cost_denominator,value_evaluations=self.evals)

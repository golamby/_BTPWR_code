"""Numerical REAPP: source Eq.(10)--(23), not its cryptographic protocol.

See reapp_audit.md for conventions where the paper leaves recruitment/sparse
trajectory details open. Native costs are worker counts, not invented auctions.
"""
from dataclasses import dataclass, asdict
import time
import numpy as np

try:
    from numba import njit
    JIT = True
except ImportError:
    JIT = False
    def njit(*args, **kwargs):
        return lambda f: f


@dataclass(frozen=True)
class Config:
    high: float = .9
    low: float = .3
    score_threshold: float = .8
    mu: float = .7
    phi: int = 3
    unknown_per_region: int = 3
    mature_region_fraction: float = .5
    recovery_start_cycle: int = 5
    rank: int = 10
    learning_rate: float = .0005
    regularization: float = .01
    max_epochs: int = 1000
    relative_loss_tolerance: float = 1e-5


@njit(cache=True)
def _sgd(L, P, br, bc, rows, cols, values, mean, alpha, beta, epochs, tol):
    """Source Eq.(23), simultaneous old-factor values per observed entry."""
    previous = np.inf
    used = 0
    loss = np.inf
    for epoch in range(epochs):
        for z in range(len(values)):
            i, j = rows[z], cols[z]
            error = values[z] - mean - br[i] - bc[j]
            for k in range(L.shape[1]):
                error -= L[i,k]*P[j,k]
            for k in range(L.shape[1]):
                old_l, old_p = L[i,k], P[j,k]
                L[i,k] += alpha*(2*error*old_p-beta*old_l)
                P[j,k] += alpha*(2*error*old_l-beta*old_p)
            br[i] += alpha*(2*error-beta*br[i])
            bc[j] += alpha*(2*error-beta*bc[j])
        loss = 0.
        for z in range(len(values)):
            i,j=rows[z],cols[z]
            error=values[z]-mean-br[i]-bc[j]
            for k in range(L.shape[1]): error-=L[i,k]*P[j,k]
            loss += error*error
        # Source Eq.(20); the source SGD uses per-observation shrinkage.
        loss += .5*beta*((L*L).sum()+(P*P).sum()+(br*br).sum()+(bc*bc).sum())
        used=epoch+1
        if abs(previous-loss) <= tol*max(1.,abs(previous)) and epoch>0:
            break
        previous=loss
    return L,P,br,bc,used,loss


class MFDC:
    """Masked biased factorization. Only finite observed entries enter SGD."""
    def __init__(self, config=Config(), seed=0):
        self.config=config
        self.rng=np.random.default_rng(seed)
        self.state=None

    def complete(self, observed):
        x=np.asarray(observed,float)
        rows,cols=np.nonzero(np.isfinite(x))
        if not len(rows):
            return np.full_like(x,np.nan),dict(epochs=0,loss=None,observations=0)
        cfg=self.config;rank=min(cfg.rank,min(x.shape)-1)
        if rank<1: raise ValueError('MFDC requires at least two regions and columns')
        mean=float(x[rows,cols].mean())
        if self.state is None or self.state[0].shape!=(x.shape[0],rank):
            L=self.rng.normal(0,.1,(x.shape[0],rank))
            P=self.rng.normal(0,.1,(x.shape[1],rank))
            br=np.zeros(x.shape[0]);bc=np.zeros(x.shape[1])
        else:
            L,P,br,bc=self.state
            L=L.copy();br=br.copy()
            extra=x.shape[1]-len(P)
            if extra<0:raise ValueError('History cannot shrink')
            P=np.vstack((P,self.rng.normal(0,.1,(extra,rank))))
            bc=np.r_[bc,np.zeros(extra)]
        L,P,br,bc,epochs,loss=_sgd(L,P,br,bc,rows,cols,x[rows,cols],mean,
            cfg.learning_rate,cfg.regularization,cfg.max_epochs,cfg.relative_loss_tolerance)
        if not np.isfinite(loss):raise FloatingPointError('Nonfinite MFDC objective')
        self.state=(L,P,br,bc)
        result=mean+br[:,None]+bc[None,:]+L@P.T
        result[rows,cols]=x[rows,cols]
        return result,dict(epochs=int(epochs),loss=float(loss),observations=len(rows),numba=JIT)


def classes(reputation, config=Config()):
    r=np.asarray(reputation)
    return np.where(r>config.high,2,np.where(r<config.low,0,1)).astype(np.int8)


def select(presence, offers, reputation, rng, config=Config(), costs=None, cap=None):
    """Native top-phi regional credible + bounded unknown identification.

    Whole-worker route union avoids charging/recruiting a worker twice. cap=None
    is native. Optional cap adapter uses actual route bid, explicitly labeled.
    """
    loc=np.asarray(presence,bool);active=np.asarray(offers,bool)
    c=classes(reputation,config);regions=np.arange(loc.shape[1])
    all_covered=np.all(np.any(loc & (active&(c==2))[:,None],axis=0))
    if all_covered:
        regions=np.sort(rng.choice(regions,max(1,int(np.ceil(len(regions)*config.mature_region_fraction))),replace=False))
    trusted=[];unknown=[]
    for j in regions:
        candidates=np.flatnonzero(loc[:,j]&active&(c==2))
        ordered=sorted(candidates,key=lambda i:(-reputation[i],int(i)))
        trusted.extend(ordered[:config.phi])
        if not all_covered:
            ids=np.flatnonzero(loc[:,j]&active&(c==1))
            unknown.extend(rng.permutation(ids)[:config.unknown_per_region].tolist())
    order=list(dict.fromkeys(trusted+unknown));selected=[];bill=0.
    bids=None if costs is None else np.asarray(costs).sum(1)
    for i in order:
        charge=0. if cap is None else float(bids[i])
        if cap is None or bill+charge<=cap+1e-9:
            selected.append(i);bill+=charge
    return np.array(selected,int),dict(mature=bool(all_covered),regions_considered=len(regions),
                                       candidate_workers=len(order),adapter_charge=bill)


def credible_reference(raw_physical, reputation, config=Config()):
    raw=np.asarray(raw_physical,float);r=np.asarray(reputation)
    ref=np.full(raw.shape[1],np.nan);c=classes(r,config)
    for j in range(raw.shape[1]):
        ids=np.flatnonzero(np.isfinite(raw[:,j])&(c==2))
        ids=np.array(sorted(ids,key=lambda i:(-r[i],int(i)))[:config.phi],int)
        if len(ids):ref[j]=np.dot(r[ids],raw[ids,j])/r[ids].sum()
    return ref


def feedback(raw_physical, reputation, reference, config=Config()):
    raw=np.asarray(raw_physical,float);ref=np.asarray(reference,float);r=np.asarray(reputation,float)
    valid=np.isfinite(raw)&np.isfinite(ref)[None,:]
    error=np.abs(raw-ref[None,:])
    # Source denominator is reference, assumed positive environmental quantity.
    # At zero use the continuous exact-agreement convention, not a hidden truth.
    denom=ref[None,:]
    relative=np.divide(error,denom,out=np.full_like(raw,np.inf),where=denom!=0)
    relative=np.where((denom==0)&(error==0),0.,relative)
    rho=1-relative
    count=valid.sum(1)
    score=np.divide(np.where(valid,rho,0).sum(1),count,out=np.zeros(len(r)),where=count>0)
    good=(1-config.mu)*score+config.mu*r
    bad=-(1-config.mu)*(1-score)+config.mu*r
    updated=np.clip(np.where(score>=config.score_threshold,good,bad),0,1)
    return np.where((count>0)&(classes(r,config)!=0),updated,r),count


def campaign(world, case, rounds=50, config=Config(), cap_adapter=False):
    T,n,D=world['presence'].shape;T=min(T,int(rounds));seed=int(case['seed'])
    r=np.asarray(world['initial_reputation'],float).copy()
    history=float(world['native_offset'])+float(world['native_scale'])*np.asarray(world['public_reference'],float)
    head=MFDC(config,seed+41901)
    predictions=np.full((T,D),np.nan);chosen=np.zeros((T,n),bool)
    reputations=np.full((T+1,n),np.nan);reputations[0]=r
    resources=np.full((T,5),np.nan);records=[]
    for t in range(T):
        rng=np.random.default_rng(np.random.SeedSequence([seed,41902,t]))
        ids,diag=select(world['presence'][t],world['offer_mask'][t],r,rng,config,
                        world['actual_cost'][t],float(world['budget']) if cap_adapter else None)
        chosen[t,ids]=True
        raw=np.full((n,D),np.nan)
        raw[ids]=float(world['native_offset'])+float(world['native_scale'])*(world['reports_check'][t,ids]-10)/30
        raw[classes(r,config)==0]=np.nan
        reference=credible_reference(raw,r,config);direct=np.isfinite(reference)
        matrix=np.column_stack([history,reference]);start=time.perf_counter()
        completion=dict(epochs=0,loss=None,observations=int(np.isfinite(matrix).sum()))
        if t+1>=config.recovery_start_cycle:
            completed,completion=head.complete(matrix)
            reference=np.where(direct,reference,completed[:,-1])
        # Before completion begins, unknown-only reports can still serve a task
        # but cannot become credible references (source distinguishes the two).
        available=np.isfinite(raw).sum(0)
        actual=np.divide(np.nansum(raw,axis=0),available,out=np.full(D,np.nan),where=available>0)
        predictions[t]=np.where(np.isfinite(reference),reference,actual)
        r,counts=feedback(raw,r,reference,config)
        reputations[t+1]=r
        history=matrix  # observed credible entries only; imputed values stay missing
        physical=float(world['actual_cost'][t,ids].sum())
        money=diag['adapter_charge'] if cap_adapter else np.nan
        resources[t]=[money,physical,len(ids),physical,money]
        records.append(dict(round=t+1,method='REAPP-Cap' if cap_adapter else 'REAPP',
            constraint='actual-route-bid cap adaptation' if cap_adapter else 'native equal worker cost; no monetary cap',
            native_worker_cost=len(ids),selected_workers=len(ids),physical_cost=physical,
            direct_regions=int(direct.sum()),recovered_regions=int((~direct&np.isfinite(reference)).sum()),
            scored_workers=int((counts>0).sum()),compute_seconds=time.perf_counter()-start,
            MFDC=completion,recruitment=diag))
    return dict(predictions=predictions,selected=chosen,reputations=reputations,resources=resources,
                records=records,class_thresholds=dict(low=config.low,high=config.high,credible_strict=True),
                config=asdict(config),implementation='paper numerical reconstruction; see reapp_audit.md',
                cost_columns=['commitment','physical','selected_workers','actual_route_bid_sum','settled'])

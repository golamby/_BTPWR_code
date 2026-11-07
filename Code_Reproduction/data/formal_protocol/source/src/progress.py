"""Planning-only progress states and authenticated-transition count interface."""
from dataclasses import dataclass
import numpy as np

LABELS = ('M', 'U3', 'U2', 'U1', 'C')
M, U3, U2, U1, C = range(5)


@dataclass(frozen=True)
class Profile:
    scale: int
    grid_states: np.ndarray
    allowed: np.ndarray
    boundaries: tuple
    max_steps: int
    parameters: tuple


def _A(r, shape):
    return (np.exp(-np.asarray(r)/shape)-np.exp(-1/shape))/(1-np.exp(-1/shape))


def _round_state(r, cfg):
    return np.rint(np.clip(r,0,1)*cfg.reputation_scale)/cfg.reputation_scale


def innovation_transition(r, a, b, cfg):
    r, a, b = np.broadcast_arrays(np.asarray(r,float), np.asarray(a,float), np.asarray(b,float))
    if (not np.isfinite(r).all() or np.any((r<0)|(r>1)) or
            not np.isfinite(a).all() or not np.isfinite(b).all() or
            np.any(a<0) or np.any(b<0) or np.any(a+b>1+1e-12)):
        raise ValueError('Finite reputation and feasible worker-round innovation required')
    candidate = _round_state(r+cfg.gamma*((a+b)*_A(r,cfg.score_shape)-b),cfg)
    return np.where(r<cfg.malicious_threshold,r,candidate)


def make_profile(cfg):
    scale=int(cfg.reputation_scale)
    if (not np.isfinite(cfg.gamma) or not np.isfinite(cfg.score_shape) or
            cfg.gamma<=0 or cfg.score_shape<=0 or scale<=0):
        raise ValueError('Positive finite registered score parameters required')
    if cfg.gamma>cfg.score_shape*(1-np.exp(-1/cfg.score_shape))+1e-15:
        raise ValueError('Registered progress proof requires the monotone update domain')
    grid=np.arange(scale+1,dtype=float)/scale
    states=np.full(scale+1,-1,dtype=np.int8)
    states[grid<cfg.malicious_threshold]=M
    states[grid>=cfg.credible_threshold]=C
    unknown=np.flatnonzero(states<0)
    r=grid[unknown].copy();steps=np.zeros(len(r),int)
    for k in range(1,101):
        pending=steps==0
        if not pending.any():break
        r[pending]=innovation_transition(r[pending],1.,0.,cfg)
        steps[(steps==0)&(r>=cfg.credible_threshold)]=k
    if np.any(steps==0) or steps.max()!=3:
        raise ValueError('Five-state profile requires exactly three possible U progress levels')
    states[unknown]=4-steps
    allowed=np.zeros((5,5),bool);allowed[M,M]=True
    boundaries=[]
    for s in range(5):
        ids=np.flatnonzero(states==s)
        if not len(ids):raise ValueError('Every registered progress state must be nonempty')
        boundaries.append((int(ids.min()),int(ids.max())))
        if s==M:continue
        low=innovation_transition(grid[ids[0]],0.,1.,cfg)
        high=innovation_transition(grid[ids[-1]],1.,0.,cfg)
        reachable=states[int(np.rint(low*scale)):int(np.rint(high*scale))+1]
        allowed[s,np.unique(reachable)]=True
    assert not allowed[U3,U1] and not allowed[U3,C] and not allowed[U2,C]
    states.setflags(write=False);allowed.setflags(write=False)
    parameters=(cfg.gamma,cfg.score_shape,cfg.malicious_threshold,cfg.credible_threshold,scale)
    return Profile(scale,states,allowed,tuple(boundaries),3,parameters)


def progress_state(reputation, profile):
    r=np.asarray(reputation,float)
    if not np.isfinite(r).all() or np.any((r<0)|(r>1)):
        raise ValueError('Finite registered reputation required')
    integers=np.rint(r*profile.scale).astype(np.int64)
    if np.any(np.abs(r-integers/profile.scale)>1e-10):
        raise ValueError('Progress interface requires the registered quantized reputation')
    return profile.grid_states[integers]


def _get(event,key):
    return event[key] if isinstance(event,dict) else getattr(event,key)


def counts_from_history(scored_rounds, round_index, profile):
    """Input provenance must be enforced by the feedback-history integration."""
    counts=np.zeros((5,5),np.int64);seen=set()
    for event in scored_rounds:
        t=int(_get(event,'round'));worker=int(_get(event,'worker'))
        if t<0 or t>=round_index:raise ValueError('Only strictly previous scored rounds are allowed')
        if (t,worker) in seen:raise ValueError('One transition per scored worker-round is required')
        seen.add((t,worker))
        r0=float(_get(event,'pre_r'));r1=float(_get(event,'next_r'))
        a=int(progress_state(r0,profile))
        b=int(progress_state(r1,profile))
        if a==M:raise ValueError('Ordinary M feedback is not an admissible scored event')
        if not profile.allowed[a,b]:raise ValueError('Transition violates exact feedback reachability')
        gamma,shape,_,_,scale=profile.parameters
        amplitude=float(_A(r0,shape))
        low=np.rint(np.clip(r0-gamma*(1-amplitude),0,1)*scale)/scale
        high=np.rint(np.clip(r0+gamma*amplitude,0,1)*scale)/scale
        if r1<low-1e-12 or r1>high+1e-12:
            raise ValueError('Exact pre/post transition exceeds the admissible score range')
        counts[a,b]+=1
    return counts


def transition_kernel(counts, profile):
    counts=np.asarray(counts)
    if (counts.shape!=(5,5) or not np.issubdtype(counts.dtype,np.integer) or
            np.any(counts<0) or np.any(counts[~profile.allowed]) or np.any(counts[M])):
        raise ValueError('Valid five-state scored transition counts required')
    probabilities=np.zeros((5,5),float);probabilities[M,M]=1.
    for s in (U3,U2,U1):
        numerator=counts[s].astype(float)+.5*profile.allowed[s]
        probabilities[s]=numerator/numerator.sum()
    numerator=counts[C].astype(float);numerator[C]+=1.
    probabilities[C]=numerator/numerator.sum()
    assert np.allclose(probabilities.sum(1),1) and not np.any(probabilities[~profile.allowed])
    probabilities.setflags(write=False)
    return probabilities


def step_progress(states, referenced, coins, kernel):
    """All class tests/reference masks use pre-step states; no within-round cascade."""
    states=np.asarray(states,np.int8);referenced=np.asarray(referenced,bool);coins=np.asarray(coins,float)
    if states.ndim!=1 or states.shape!=referenced.shape or states.shape!=coins.shape:
        raise ValueError('One reference flag and fixed coin per slot required')
    if (np.any((states<0)|(states>4)) or not np.isfinite(coins).all() or
            np.any((coins<0)|(coins>=1))):
        raise ValueError('Invalid state or transition coin')
    kernel=np.asarray(kernel,float)
    if (kernel.shape!=(5,5) or not np.isfinite(kernel).all() or
            np.any(kernel<0) or not np.allclose(kernel.sum(1),1,rtol=0,atol=1e-12)):
        raise ValueError('A normalized finite transition kernel is required')
    result=states.copy()
    for i in np.flatnonzero(referenced&(states!=M)):
        result[i]=min(4,int(np.searchsorted(np.cumsum(kernel[states[i]]),coins[i],side='right')))
    return result


def planning_depth(remaining_contract_rounds, profile):
    if int(remaining_contract_rounds)!=remaining_contract_rounds or remaining_contract_rounds<0:
        raise ValueError('Nonnegative published remaining contract rounds required')
    return min(int(remaining_contract_rounds),profile.max_steps+1)

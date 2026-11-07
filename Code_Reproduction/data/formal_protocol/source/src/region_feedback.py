"""One update per distinct reported region; all references frozen pre-update."""
import numpy as np
from . import trust
from .reference_experiments import reference_feedback

def feedback(raw,r,secondary,tc,offset,scale):
    base=trust.feedback(raw,r,tc)
    if secondary is None:
        refs=base['reference_by_worker'].copy()
    else:
        _,_,refs,_,_=reference_feedback(raw,r,secondary,None,tc,1.,offset,scale)
    initial=trust.classify(r,tc);nxt=np.asarray(r,float).copy()
    counts=(np.isfinite(raw)&np.isfinite(refs)).sum(1)
    for i in np.flatnonzero((initial!=0)&(counts>0)):
        for j in np.flatnonzero(np.isfinite(raw[i])&np.isfinite(refs[i])):
            error=abs(raw[i,j]-refs[i,j])/tc.score_width
            A=(np.exp(-nxt[i]/tc.score_shape)-np.exp(-1/tc.score_shape))/(1-np.exp(-1/tc.score_shape))
            delta=tc.gamma*A*(1-error/tc.score_tolerance) if error<=tc.score_tolerance else -tc.gamma*(1-A)*(1-tc.score_tolerance/error)
            nxt[i]=np.rint(np.clip(nxt[i]+delta,0,1)*tc.reputation_scale)/tc.reputation_scale
    return dict(base,next_reputation=nxt,reference_by_worker=refs,scored_count=counts)

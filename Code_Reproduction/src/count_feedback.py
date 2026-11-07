"""Count-sensitive bounded feedback with frozen per-round references."""
import numpy as np
from . import trust
from .reference_experiments import reference_feedback

def feedback(raw,r,secondary,tc,offset,scale,secondary_weight=.5):
    base=trust.feedback(raw,r,tc)
    refs=base['reference_by_worker'].copy()
    if secondary is not None:
        _,_,refs,_,_=reference_feedback(raw,r,secondary,None,tc,1.,offset,scale)
    valid=np.isfinite(raw)&np.isfinite(refs)
    added=valid & ~np.isfinite(base['reference_by_worker'])
    weights=np.where(added,secondary_weight,1.)*valid
    errors=np.abs(raw-refs)/tc.score_width
    A=(np.exp(-np.asarray(r)/tc.score_shape)-np.exp(-1/tc.score_shape))/(1-np.exp(-1/tc.score_shape))
    scores=np.where(errors<=tc.score_tolerance,
        tc.gamma*A[:,None]*(1-errors/tc.score_tolerance),
        -tc.gamma*(1-A[:,None])*(1-tc.score_tolerance/np.maximum(errors,1e-300)))
    effective=weights.sum(1)
    mean=np.divide(np.where(valid,weights*scores,0.).sum(1),effective,out=np.zeros(len(r)),where=effective>0)
    multiplier=np.sqrt(np.minimum(effective,4.))
    nxt=np.rint(np.clip(r+mean*multiplier,0,1)*tc.reputation_scale)/tc.reputation_scale
    nxt=np.where((effective>0)&(trust.classify(r,tc)!=0),nxt,r)
    return dict(base,next_reputation=nxt,reference_by_worker=refs,scored_count=valid.sum(1),
                effective_evidence=effective,evidence_multiplier=multiplier)

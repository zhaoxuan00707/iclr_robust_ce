"""Refine a first sampled crossing without assuming a monotone probability path."""
import torch

@torch.no_grad()
def first_sampled_crossing(probability, count, tau, *, device='cpu', intervals=4096, bisections=24, chunk=8, tolerance=1e-4):
    zero=torch.zeros(count,device=device);one=torch.ones_like(zero)
    start=probability(zero);end=probability(one)
    eligible=(start<tau)&(end>=tau)&torch.isfinite(start)&torch.isfinite(end)
    found=torch.zeros(count,dtype=torch.bool,device=device);lo=zero.clone();hi=one.clone()
    previous=start.clone();previous_alpha=0.
    for first in range(1,intervals+1,chunk):
        if bool((found|~eligible).all()):break
        alphas=torch.arange(first,min(first+chunk,intervals+1),device=device,dtype=torch.float32)/intervals
        values=probability(alphas[:,None].expand(-1,count))
        for alpha,values_at_alpha in zip(alphas,values):
            crossing=eligible&~found&(previous<tau)&(values_at_alpha>=tau)
            lo[crossing]=previous_alpha;hi[crossing]=alpha;found|=crossing
            previous=values_at_alpha;previous_alpha=alpha
    initial_lo=lo.clone();initial_hi=hi.clone()
    for _ in range(bisections):
        mid=(lo+hi)/2;p=probability(mid);above=p>=tau
        hi=torch.where(found&above,mid,hi);lo=torch.where(found&~above,mid,lo)
    lp=probability(lo);hp=probability(hi)
    success=found&(lp<tau)&(hp>=tau)&(hp<=tau+tolerance)&torch.isfinite(hp)
    alpha=torch.where(success,hi,one)
    reasons=[]
    for i in range(count):
        reasons.append('refined' if success[i] else 'start_already_positive' if start[i]>=tau else 'endpoint_below_target' if end[i]<tau else 'numerical_or_missing_bracket')
    return {'alpha':alpha,'success':success,'start_probability':start,'end_probability':end,'initial_lower':initial_lo,'initial_upper':initial_hi,'lower':lo,'upper':hi,'lower_probability':lp,'upper_probability':hp,'reason':reasons}

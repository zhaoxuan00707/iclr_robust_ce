"""Exact finite-sentence population objective; no token/decoder gradients."""
import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def update(x,y,w,b,steps=3,lr=.5):
    # Analytic SGD gradients retain the complete update graph through x.
    for _ in range(steps):
        error=(x@w+b).sigmoid()-y
        w,b=w-lr*(x.T@error)/len(x),b-lr*error.mean()
    return w,b

def losses(q,w,b):
    return .05*torch.nn.functional.softplus((.5-(q@w+b).sigmoid())/.05)

def cache(x,y,q,w,b,steps=3,lr=.5):
    x=x.detach().clone().requires_grad_(True)
    nw,nb=update(x,y,w,b,steps,lr);f=losses(q,nw,nb)
    adj=torch.stack([torch.autograd.grad(v,x,retain_graph=True)[0].flatten() for v in f])
    return f.detach(),adj.detach()

def objective(ids,f,adj,n,r):
    nominal=f[ids].mean();sens=math.sqrt(n)*adj[ids].mean(0).norm()
    return nominal+r*sens,nominal,sens

def w2(a,b):
    assert len(a)==len(b)
    cost=cdist(np.asarray(a),np.asarray(b),'sqeuclidean');i,j=linear_sum_assignment(cost)
    return float(np.sqrt(max(0,cost[i,j].mean()))),j

def geometry(base,q,ref):
    return w2(base,q)[0]+w2(q,ref)[0]

def solve(codes,base_ids,ref,f,adj,n,r,beta,sweeps=10,tol=1e-8):
    # Exhaustive one-support replacements, on the exact same scalar objective.
    codes=np.asarray(codes);f=np.asarray(f);adj=np.asarray(adj);ids=np.array(base_ids).copy();base=codes[ids];D=w2(base,ref)[0];budget=beta*D;trace=[]
    total=adj[ids].sum(0);fsum=f[ids].sum();m=len(ids)
    def val(v,s):return float(v/m+r*math.sqrt(n)*np.linalg.norm(s)/m)
    score=val(fsum,total);stopped=False
    for sweep in range(sweeps):
        changed=False
        for k in range(m):
            proposed_s=total[None]-adj[ids[k]][None]+adj
            scores=(fsum-f[ids[k]]+f)/m+r*math.sqrt(n)*np.linalg.norm(proposed_s,axis=1)/m
            order=np.argsort(scores,kind='stable');chosen=None
            for candidate in order:
                if scores[candidate]>=score-1e-10:break
                proposal=ids.copy();proposal[k]=candidate
                if geometry(base,codes[proposal],ref)<=budget+tol:chosen=int(candidate);break
            if chosen is not None:
                total=proposed_s[chosen];fsum=fsum-f[ids[k]]+f[chosen];ids[k]=chosen;score=val(fsum,total);changed=True
        trace.append({'sweep':sweep+1,'surrogate':score,'geometry':geometry(base,codes[ids],ref),'changed':changed})
        if not changed:stopped=True;break
    _,assignment=w2(base,codes[ids]);deployed=ids[assignment]
    return ids,deployed,trace,stopped

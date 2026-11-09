"""Unregularized Euclidean W2 plans and constrained particle geometry.

The envelope gradient differentiates transport costs with an optimal plan held
fixed. This is a numerical OT subgradient, not differentiation through a solver.
"""
from dataclasses import dataclass
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import ot


@dataclass
class ExactPlan:
    source_index: torch.Tensor
    target_index: torch.Tensor
    weights: torch.Tensor
    distance: float
    marginal_residual: float
    solver: str

    def differentiable_distance(self, source, target):
        i=self.source_index.to(source.device);j=self.target_index.to(target.device)
        weights=self.weights.to(source)
        return torch.linalg.vector_norm((source[i]-target[j])*weights.sqrt()[:,None])

    def state_dict(self):return dict(vars(self))


def exact_w2(source, target, a=None, b=None):
    """Balanced quadratic OT, with an exact equal-uniform assignment shortcut."""
    x=source.detach().cpu().double().numpy();y=target.detach().cpu().double().numpy()
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1] or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('finite compatible support matrices required')
    a=np.full(len(x),1/len(x)) if a is None else np.asarray(a,dtype=np.float64)
    b=np.full(len(y),1/len(y)) if b is None else np.asarray(b,dtype=np.float64)
    if (a<=0).any() or (b<=0).any() or abs(a.sum()-1)>1e-8 or abs(b.sum()-1)>1e-8:
        raise ValueError('positive probability masses required')
    cost=cdist(x,y,'sqeuclidean')
    if len(x)==len(y) and np.allclose(a,1/len(x),atol=1e-14,rtol=0) and np.allclose(b,1/len(y),atol=1e-14,rtol=0):
        i,j=linear_sum_assignment(cost);mass=a.copy();solver='SciPy linear_sum_assignment; exact equal-mass discrete quadratic OT'
    else:
        plan,log=ot.emd(a,b,cost,numItermax=1000000,log=True,numThreads=1)
        if log['warning'] is not None:raise RuntimeError(log['warning'])
        i,j=np.nonzero(plan>0);mass=plan[i,j];solver='POT network simplex; unregularized quadratic OT'
    residual=max(np.abs(np.bincount(i,weights=mass,minlength=len(x))-a).max(),np.abs(np.bincount(j,weights=mass,minlength=len(y))-b).max())
    if residual>1e-8:raise RuntimeError('OT marginals failed')
    distance=float(np.sqrt(np.sum(mass*cost[i,j])))
    return ExactPlan(torch.from_numpy(i.copy()),torch.from_numpy(j.copy()),torch.from_numpy(mass.copy()),distance,float(residual),solver)


def excess(source,control,target,D=None):
    p0=exact_w2(source,control);p1=exact_w2(control,target)
    D=exact_w2(source,target).distance if D is None else D
    return p0,p1,p0.distance+p1.distance-D


def project_basic(points, reference, current, lower, upper):
    """Box projection and per-particle backtracking to a current-valid reference.

    This enforces the declared particle constraint, not a near-target radius.
    No claim is made about continuous validity along the backtracking segment.
    """
    with torch.no_grad():
        result=points.clamp(lower,upper)
        if (current(reference)<0).any():raise ValueError('reference must satisfy f0>=.5')
        for _ in range(24):
            invalid=current(result)<0
            if not invalid.any():return result
            result[invalid]=(result[invalid]+reference[invalid])/2
        invalid=current(result)<0;result[invalid]=reference[invalid]
        return result


def glued_arclength_path(source,control,target,p0,p1,lambdas):
    """Glue two exact plans through control masses; retain factual identities."""
    # Equal target/control masses give a permutation for the second segment.
    if len(p1.weights)!=len(control) or len(torch.unique(p1.source_index))!=len(control):
        raise ValueError('this gluing adapter requires equal-mass second-leg assignment')
    permutation=torch.empty(len(control),dtype=torch.long)
    permutation[p1.source_index]=p1.target_index
    i,j=p0.source_index,p0.target_index;k=permutation[j]
    L=p0.distance+p1.distance
    pivot=p0.distance/L if L>0 else 0.
    points=[]
    for t in lambdas:
        if not 0<=t<=1:raise ValueError('lambda outside [0,1]')
        if L==0:q=source[i]
        elif t<=pivot and p0.distance>1e-12:
            s=t/pivot;q=(1-s)*source[i]+s*control[j]
        else:
            s=(t-pivot)/(1-pivot) if p1.distance>1e-12 else 1.
            q=(1-s)*control[j]+s*target[k]
        points.append(q)
    return {'q':torch.stack(points),'weights':p0.weights,'source_index':i,'target_index':k,'control_index':j,'lambda_W':pivot,'d0':p0.distance,'d1':p1.distance}

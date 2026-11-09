"""Empirical update-aware Wasserstein surrogate, PDF Section 3.4."""
from collections import OrderedDict
import math
import torch
from torch.func import functional_call
from .geodesic_excess import exact_w2


def unroll(model, x, y, steps=3, lr=.01, differentiable=True):
    params=OrderedDict((k,v.detach().clone().requires_grad_(True)) for k,v in model.named_parameters())
    buffers=OrderedDict((k,v.detach().clone()) for k,v in model.named_buffers())
    for _ in range(steps):
        logits=functional_call(model,(params,buffers),(x,))
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
        grads=torch.autograd.grad(loss,tuple(params.values()),create_graph=differentiable)
        params=OrderedDict((k,p-lr*g) for (k,p),g in zip(params.items(),grads))
        if not differentiable:params=OrderedDict((k,p.detach().requires_grad_(True)) for k,p in params.items())
    return params,buffers


def surrogate(model,x,y,q,r,steps=3,lr=.01,decode=None,hinge_temperature=0.):
    h=torch.zeros_like(x,requires_grad=True)
    params,buffers=unroll(model,x+h,y,steps,lr,True)
    margin=.5-functional_call(model,(params,buffers),(q if decode is None else decode(q),)).sigmoid()
    if hinge_temperature<0:raise ValueError('hinge temperature must be nonnegative')
    losses=torch.nn.functional.softplus(margin/hinge_temperature)*hinge_temperature if hinge_temperature>0 else torch.relu(margin)
    nominal=losses.mean()
    g=torch.autograd.grad(nominal,h,create_graph=True)[0]
    sensitivity=math.sqrt(len(x))*torch.linalg.vector_norm(g)
    return nominal+r*sensitivity,nominal,sensitivity


def differentiable_w2(a,b):
    plan=exact_w2(a.detach().flatten(1).cpu(),b.detach().flatten(1).cpu())
    i=plan.source_index.to(a.device);j=plan.target_index.to(b.device);w=plan.weights.to(a.device,dtype=a.dtype)
    squared=((a[i]-b[j]).flatten(1).square().sum(1)*w).sum()
    # At identical supports use the zero subgradient of the norm.
    distance=squared.sqrt() if float(squared.detach())>1e-20 else squared*0
    return distance,plan

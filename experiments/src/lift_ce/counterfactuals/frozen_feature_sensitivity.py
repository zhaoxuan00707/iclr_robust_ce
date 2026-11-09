"""Exact pixel pullback norm for a fixed feature extractor, with bounded memory.

Only the adjoint is differentiated by this operator. The input distribution and
feature extractor are fixed during Q optimization. No derivative wrt their
parameters or their pixels is claimed by this custom first-order operator.
"""
import math
import torch
import torch.nn.functional as F


def head_update(features, labels, weight, bias, steps, lr, differentiable=True):
    w=weight.detach().clone().requires_grad_(True)
    b=bias.detach().clone().requires_grad_(True)
    for _ in range(steps):
        loss=F.cross_entropy(F.linear(features,w,b),labels)
        gw,gb=torch.autograd.grad(loss,(w,b),create_graph=differentiable)
        w,b=w-lr*gw,b-lr*gb
        if not differentiable:w,b=w.detach().requires_grad_(),b.detach().requires_grad_()
    return w,b


class _PixelNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, adjoint, images, extractor, batch):
        pieces=[];squared=adjoint.new_zeros(())
        with torch.enable_grad():
            for start in range(0,len(images),batch):
                x=images[start:start+batch].detach().requires_grad_()
                u=adjoint[start:start+batch].detach().requires_grad_()
                feature=extractor(x)
                pixel=torch.autograd.grad(feature,x,u,create_graph=True)[0]
                norm2=pixel.square().sum()
                action=torch.autograd.grad(norm2,u)[0]/2
                squared+=norm2.detach();pieces.append(action.detach())
        norm=squared.sqrt();scale=math.sqrt(len(images))
        derivative=torch.cat(pieces)*(scale/norm) if float(norm)>0 else torch.zeros_like(adjoint)
        ctx.save_for_backward(derivative)
        return scale*norm
    @staticmethod
    def backward(ctx, grad):
        derivative,=ctx.saved_tensors
        return grad*derivative,None,None,None


def pixel_sensitivity(adjoint,images,extractor,batch=2):
    return _PixelNorm.apply(adjoint,images,extractor,batch)


def smooth_target(logits,target,temperature=.05):
    return temperature*F.softplus((.5-logits.softmax(-1)[:,target])/temperature).mean()

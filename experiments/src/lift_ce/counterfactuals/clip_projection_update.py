"""Complete differentiable SGD update of the original CLIP projection."""
import torch
import torch.nn.functional as F

def projection_update(features,labels,initial,text,scale,steps,lr,differentiable=True):
    p=initial.detach().clone().requires_grad_(True)
    for _ in range(steps):
        logits=scale*F.normalize(features@p,dim=-1)@text.T
        loss=F.cross_entropy(logits,labels)
        grad=torch.autograd.grad(loss,p,create_graph=differentiable)[0]
        p=p-lr*grad
        if not differentiable:p=p.detach().requires_grad_(True)
    return p

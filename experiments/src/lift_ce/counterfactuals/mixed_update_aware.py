"""Appendix A.5 relaxation in the existing standardized feature coordinates."""
import math
import torch
from torch.func import functional_call
from .update_aware import unroll


class MixedDomain:
    def __init__(self, train, groups):
        self.groups = {k: list(v) for k, v in groups.items() if v}
        self.low = torch.zeros(train.shape[1], dtype=train.dtype, device=train.device)
        self.scale = torch.ones_like(self.low)
        for ids in self.groups.values():
            self.low[ids] = train[:, ids].amin(0)
            self.scale[ids] = train[:, ids].amax(0) - self.low[ids]
            if (self.scale[ids] <= 0).any():
                raise ValueError('Degenerate categorical coordinate')
        self.numeric = [j for j in range(train.shape[1]) if all(j not in v for v in self.groups.values())]

    def perturb(self, x, u, temperature=1., smoothing=1e-4):
        z = x + u
        low, scale = self.low.to(x), self.scale.to(x)
        for ids in self.groups.values():
            p = ((x[:, ids]-low[ids])/scale[ids]).clamp(0, 1)
            p = (1-smoothing)*p + smoothing/len(ids)
            p = p/p.sum(1, keepdim=True)
            alpha = temperature*p.log()
            z = z.clone()
            z[:, ids] = torch.softmax((alpha+u[:, ids])/temperature, 1)*scale[ids]+low[ids]
        return z

    def project(self, x, hard=False):
        z = x.clone(); low, scale = self.low.to(x), self.scale.to(x)
        for ids in self.groups.values():
            p = (x[:, ids]-low[ids])/scale[ids]
            if hard:
                p = torch.nn.functional.one_hot(p.argmax(1), len(ids)).to(x)
            else:
                # Euclidean projection in standardized coordinates: weighted simplex.
                weights = scale[ids].square()
                lo = ((p-1)*weights).amin(1, keepdim=True)
                hi = (p*weights).amax(1, keepdim=True)
                for _ in range(45):
                    mid = (lo+hi)/2
                    large = (p-mid/weights).clamp_min(0).sum(1, keepdim=True)>1
                    lo = torch.where(large, mid, lo); hi = torch.where(large, hi, mid)
                p = (p-(lo+hi)/2/weights).clamp_min(0)
            z[:, ids] = p*scale[ids]+low[ids]
        return z


def mixed_surrogate(model, x, y, q, r, domain, steps=10, lr=.01,
                    temperature=.05, cat_temperature=1., smoothing=1e-4,
                    adversary_mode='population', sensitivity_chunk_size=1):
    if adversary_mode not in ('population', 'individual'):
        raise ValueError('adversary_mode must be population or individual')
    u = torch.zeros_like(x, requires_grad=True)
    perturbed = domain.perturb(x, u, cat_temperature, smoothing)
    params, buffers = unroll(model, perturbed, y, steps, lr, True)
    probability = functional_call(model, (params, buffers), (q,)).sigmoid()
    losses = temperature*torch.nn.functional.softplus((.5-probability)/temperature)
    nominal = losses.mean()
    if adversary_mode == 'individual':
        norms = []
        for derivative in sensitivity_chunks(losses, u, sensitivity_chunk_size, True):
            norms.append(torch.linalg.vector_norm(derivative.flatten(1), dim=1))
        sensitivity = math.sqrt(len(x))*torch.cat(norms).mean()
        return nominal+r*sensitivity, nominal, sensitivity
    derivative = torch.autograd.grad(nominal, u, create_graph=True)[0]
    sensitivity = math.sqrt(len(x))*torch.linalg.vector_norm(derivative)
    return nominal+r*sensitivity, nominal, sensitivity


def sensitivity_chunks(losses, perturbation, chunk_size=1, create_graph=True):
    """Per-CF loss gradients in training-perturbation coordinates, without 1/n.

    All VJPs share the same differentiable update graph. Chunking bounds the
    batched VJP dimension, not the complete higher-order optimization graph.
    """
    if chunk_size < 1 or losses.ndim != 1 or not len(losses):
        raise ValueError('Positive chunk size and nonempty vector losses required')
    for start in range(0, len(losses), chunk_size):
        stop = min(start+chunk_size, len(losses))
        basis = losses.new_zeros((stop-start, len(losses)))
        basis[torch.arange(stop-start, device=losses.device),
              torch.arange(start, stop, device=losses.device)] = 1
        yield torch.autograd.grad(losses, perturbation, grad_outputs=basis,
                                  is_grads_batched=True, create_graph=create_graph,
                                  retain_graph=True)[0]


def sensitivity_aggregation_diagnostics(model, x, y, q, domain, steps=10, lr=.01,
                                      temperature=.05, cat_temperature=1.,
                                      smoothing=1e-4, chunk_size=1):
    """Compute both norms from identical individual vectors; no optimizer use."""
    u = torch.zeros_like(x, requires_grad=True)
    params, buffers = unroll(model, domain.perturb(x,u,cat_temperature,smoothing),
                            y, steps, lr, True)
    p = functional_call(model,(params,buffers),(q,)).sigmoid()
    losses = temperature*torch.nn.functional.softplus((.5-p)/temperature)
    vector_sum = torch.zeros_like(x)
    norm_sum = x.new_zeros(())
    for vectors in sensitivity_chunks(losses,u,chunk_size,False):
        vector_sum += vectors.sum(0)
        norm_sum += vectors.flatten(1).norm(dim=1).sum()
    mean_vector = vector_sum/len(q)
    direct = torch.autograd.grad(losses.mean(),u)[0]
    pop = float(math.sqrt(len(x))*mean_vector.norm())
    ind = float(math.sqrt(len(x))*norm_sum/len(q))
    return dict(population_sensitivity=pop, individual_sensitivity=ind,
                kappa=pop/ind if ind>0 else None, zero_denominator=ind==0,
                mean_vector_max_error=float((mean_vector-direct).abs().max()))

"""Mass-preserving factual association of a fixed robust population."""
import torch
from .geodesic_excess import exact_w2


def direct_realization(base, control, require_one_per_factual=False):
    if require_one_per_factual and len(base) != len(control):
        raise ValueError('One-to-one realization requires equal support counts; resampling Q is not permitted.')
    plan = exact_w2(base, control)
    source_mass = torch.zeros(len(base), dtype=torch.float64).scatter_add_(0, plan.source_index, plan.weights)
    target_mass = torch.zeros(len(control), dtype=torch.float64).scatter_add_(0, plan.target_index, plan.weights)
    torch.testing.assert_close(source_mass, torch.full_like(source_mass, 1/len(base)), atol=1e-8, rtol=0)
    torch.testing.assert_close(target_mass, torch.full_like(target_mass, 1/len(control)), atol=1e-8, rtol=0)
    permutation = len(base) == len(control) and len(plan.weights) == len(base) and len(plan.target_index.unique()) == len(control)
    if require_one_per_factual and not permutation:
        raise ValueError('Solver did not return a one-to-one assignment.')
    return {'points': control[plan.target_index].clone(), 'weights': plan.weights,
            'source_index': plan.source_index, 'control_index': plan.target_index,
            'one_per_factual': permutation, 'plan': plan.state_dict()}

"""Fixed-budget path optimization with exact OT and explicit rejection.

The primary risk is mean_lambda(max_model(weighted_mean_CF(true_hinge))).
Gradients use fixed exact plans and smooth losses. All retained paths are checked
again using unregularized OT and reference-CPU current-model scores.
"""
import copy
import time

import torch
import torch.nn.functional as F

from .geodesic_excess import exact_w2
from .path_preflight import check_path_preconditions


def path_state(source, controls, target, lambdas):
    """Glue uniform target-sized controls, preserving source identities and mass.

    Equal uniform intermediate/target masses give permutation plans after the
    first leg. Unsupported nonuniform control masses require a general adapter.
    """
    if controls.ndim != 3 or controls.shape[1:] != target.shape:
        raise ValueError('controls must have shape [K, target_count, dimension]')
    if not lambdas or any(not 0 <= t <= 1 for t in lambdas):
        raise ValueError('nonempty lambda grid in [0,1] required')
    supports = [source, *controls.unbind(0), target]
    plans = [exact_w2(a, b) for a, b in zip(supports, supports[1:])]
    distances = [p.differentiable_distance(a.double(), b.double())
                 for p, a, b in zip(plans, supports, supports[1:])]
    length = torch.stack(distances).sum()
    indices = [plans[0].source_index.to(source.device), plans[0].target_index.to(source.device)]
    for plan in plans[1:]:
        if len(plan.weights) != len(target) or len(plan.source_index.unique()) != len(target):
            raise ValueError('uniform control-to-control permutation required')
        permutation = torch.empty(len(target), dtype=torch.long, device=source.device)
        permutation[plan.source_index.to(source.device)] = plan.target_index.to(source.device)
        indices.append(permutation[indices[-1]])
    chain = [support[index] for support, index in zip(supports, indices)]
    points = []
    for lam in lambdas:
        if lam == 0 or float(length.detach()) == 0:
            q = chain[0]
        elif lam == 1:
            q = chain[-1]
        else:
            position = lam*length
            offset = length*0
            q = chain[-1]
            for k, distance in enumerate(distances):
                if float(distance.detach()) > 0 and float(position.detach()) <= float((offset+distance).detach()):
                    fraction = ((position-offset)/distance).to(source.dtype)
                    q = chain[k] + fraction*(chain[k+1]-chain[k])
                    break
                offset = offset+distance
        points.append(q)
    weights = plans[0].weights.to(source.device)
    knots = [0.]
    total = sum(p.distance for p in plans)
    for p in plans:
        knots.append(knots[-1]+p.distance/total if total else 0.)
    return {'q': torch.stack(points), 'weights': weights, 'source_index': indices[0],
            'target_index': indices[-1], 'control_indices': indices[1:-1],
            'length_tensor': length, 'total_length': total,
            'leg_distances': [p.distance for p in plans], 'knots': knots,
            'plans': plans, 'marginal_residual': max(p.marginal_residual for p in plans)}


def weighted_path_risk(bank, q, weights, beta_h=50., beta_m=100., chunk=64, gradient=True):
    """Stream model chunks; never take model maxima separately per CF.

    Return per-lambda true/smooth risks and the derivative w.r.t. q. The gradient
    already includes the mean across lambda. Reattach it to the path graph with
    a vector-Jacobian product outside this function.
    """
    points = q.detach().requires_grad_(gradient)
    count, n, dim = points.shape
    mass = weights.to(device=q.device, dtype=torch.float64)
    if mass.shape != (n,) or bool((mass <= 0).any()) or abs(float(mass.sum())-1) > 1e-8:
        raise ValueError('positive normalized population weights required')
    smooths, trues = [], []
    with torch.no_grad():
        for start in range(0, bank.count, chunk):
            score = torch.sigmoid(bank.logits(points.reshape(-1, dim), start, start+chunk))
            score = score.reshape(-1, count, n)
            smooths.append(((F.softplus(beta_h*(.5-score))/beta_h).double()*mass).sum(-1))
            trues.append((torch.relu(.5-score).double()*mass).sum(-1))
        smooth = torch.cat(smooths)
        true = torch.cat(trues).amax(0)
        soft_weights = torch.softmax(beta_m*smooth, dim=0)
        values = torch.logsumexp(beta_m*smooth, dim=0)/beta_m
    grad = torch.zeros_like(points)
    if gradient:
        for start in range(0, bank.count, chunk):
            score = torch.sigmoid(bank.logits(points.reshape(-1, dim), start, start+chunk)).reshape(-1, count, n)
            per_model = ((F.softplus(beta_h*(.5-score))/beta_h).double()*mass).sum(-1)
            loss = (soft_weights[start:start+chunk]*per_model).sum()/count
            grad.add_(torch.autograd.grad(loss, points)[0])
    return values, true, grad


@torch.no_grad()
def audit_path(state, controls, current_cpu, lower, upper, budget, tolerance):
    """Reference-CPU validity at every grid point AND every control support."""
    q = state['q'].detach().cpu()
    w = controls.detach().cpu()
    scores = torch.sigmoid(current_cpu(q.reshape(-1, q.shape[-1]))).reshape(q.shape[:2])
    control_scores = torch.sigmoid(current_cpu(w.reshape(-1, w.shape[-1])))
    mass = state['weights'].cpu().double()
    invalid = ((scores < .5).double()*mass).sum(1)
    lo, hi = lower.cpu(), upper.cpu()
    box = ((w < lo-1e-7) | (w > hi+1e-7)).any(-1)
    path_box = (((q < lo-1e-5) | (q > hi+1e-5)).any(-1).double()*mass).sum(1)
    length = state['total_length']
    return {'total_length': length, 'B_geo': budget, 'slack': budget-length,
            'maximum_grid_invalid_mass': float(invalid.max()),
            'minimum_f0_score': float(scores.min()),
            'control_invalid_count': int((control_scores < .5).sum()),
            'control_box_violation_count': int(box.sum()),
            'grid_current_valid_mass': (1-invalid).tolist(),
            'grid_minimum_score': scores.amin(1).tolist(),
            'grid_range_violation_mass': path_box.tolist(),
            'marginal_residual': state['marginal_residual'],
            'feasible': bool(length <= budget+tolerance and not (scores < .5).any()
                             and not (control_scores < .5).any() and not box.any())}


def frozen_state(state):
    """Serialization without tensors attached to an autograd graph."""
    return {'q': state['q'].detach().cpu(), 'weights': state['weights'].cpu(),
            'source_index': state['source_index'].cpu(), 'target_index': state['target_index'].cpu(),
            'control_indices': [i.cpu() for i in state['control_indices']],
            'total_length': state['total_length'], 'leg_distances': state['leg_distances'],
            'knots': state['knots'], 'plans': [p.state_dict() for p in state['plans']]}


def straight_derived_initial(source, target, count=2):
    """Target-conditional means are starts only, never claimed exact geodesics."""
    plan = exact_w2(source, target)
    i, j = plan.source_index.to(source.device), plan.target_index.to(source.device)
    mass = plan.weights.to(source)
    averages = torch.zeros_like(target)
    averages.index_add_(0, j, source[i]*mass[:, None])
    averages *= len(target)
    return torch.stack([(1-t)*averages+t*target for t in [k/(count+1) for k in range(1, count+1)]])


def solve_path(bank, current, source, target, initial, budget, D_train,
               lower, upper, lambdas, settings, on_step=None, incumbent=None):
    """Constrained train or held-out solve; same algorithm and absolute budget.

    Current-validity constraint is the nonnegative average weighted hinge over
    grid populations and controls. Its zero set requires every positive-mass
    point to be valid. Final acceptance uses strict pointwise checks, not its
    small average. No future outcomes or fresh-bank models enter this API.
    """
    reference = copy.deepcopy(current).cpu().eval().requires_grad_(False)
    tolerance = settings.get('feasibility_relative_tolerance', 1e-6)*max(1., D_train)
    with torch.no_grad():
        src_invalid = float((reference(source.cpu()) < 0).double().mean())
        tgt_invalid = float((reference(target.cpu()) < 0).double().mean())
    preflight = check_path_preconditions(exact_w2(source, target).distance, budget,
                                         src_invalid, tgt_invalid, geometry_tolerance=tolerance)
    if not preflight['necessary_conditions_pass']:
        return {'status': 'infeasible_endpoints', 'preflight': preflight, 'candidate': None, 'trace': []}
    w = initial.detach().clamp(lower, upper).requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=settings['learning_rate'])
    rho_geo = rho_valid = settings['rho_initial']
    nu_geo = nu_valid = 0.
    stalled = [float('inf'), float('inf')]
    trace, best = [], None
    # A carried incumbent is independently re-evaluated under this batch/budget.
    if incumbent is not None:
        carried = incumbent.to(source.device)
        state = path_state(source, carried, target, lambdas)
        audit = audit_path(state, carried, reference, lower, upper, budget, tolerance)
        if audit['feasible']:
            _, risks, _ = weighted_path_risk(bank, state['q'], state['weights'], gradient=False,
                                           chunk=settings.get('model_chunk', 64))
            best = {'controls': carried.cpu(), 'path': frozen_state(state), 'audit': audit,
                    'true_path_risk': float(risks.mean()), 'step': -1, 'risk_curve': risks.cpu()}
    for step in range(settings['steps']+1):
        started = time.perf_counter()
        state = path_state(source, w, target, lambdas)
        audit = audit_path(state, w, reference, lower, upper, budget, tolerance)
        smooth, risk, gradient = weighted_path_risk(bank, state['q'], state['weights'],
            settings['beta_h'], settings['beta_m'], settings.get('model_chunk', 64), step < settings['steps'])
        q = state['q']
        score = torch.sigmoid(current(q.reshape(-1, q.shape[-1]))).reshape(q.shape[:2])
        control_score = torch.sigmoid(current(w.reshape(-1, w.shape[-1])))
        grid_violation = (torch.relu(.5-score).double()*state['weights']).sum(1).mean()
        control_violation = torch.relu(.5-control_score).double().mean()
        g_valid = (grid_violation+control_violation)/2
        g_geo = state['length_tensor']-budget
        if audit['feasible'] and (best is None or float(risk.mean()) < best['true_path_risk']):
            best = {'controls': w.detach().cpu().clone(), 'path': frozen_state(state), 'audit': audit,
                    'true_path_risk': float(risk.mean()), 'step': step, 'risk_curve': risk.cpu()}
        row = {k: v for k, v in audit.items() if not isinstance(v, list)}
        row.update(step=step, true_path_risk=float(risk.mean()), smooth_path_risk=float(smooth.mean()),
                   g_geo=float(g_geo.detach()), g_valid=float(g_valid.detach()), nu_geo=nu_geo,
                   nu_valid=nu_valid, rho_geo=rho_geo, rho_valid=rho_valid)
        if step < settings['steps']:
            penalty = (nu_geo*g_geo+rho_geo/2*torch.relu(g_geo).square()
                       +nu_valid*g_valid+rho_valid/2*g_valid.square())
            objective = (q*gradient).sum()+penalty
            optimizer.zero_grad(); objective.backward()
            if not torch.isfinite(w.grad).all():
                raise ValueError('nonfinite path gradient')
            optimizer.step()
            with torch.no_grad(): w.clamp_(lower, upper)
            if (step+1) % settings['dual_interval'] == 0:
                nu_geo = max(0., nu_geo+rho_geo*row['g_geo'])
                nu_valid = max(0., nu_valid+rho_valid*row['g_valid'])
            if (step+1) % settings['stall_interval'] == 0:
                violations = [max(0., row['g_geo']), row['g_valid']]
                if violations[0] > tolerance and violations[0] >= .9*stalled[0]: rho_geo *= 2
                if violations[1] > 0 and violations[1] >= .9*stalled[1]: rho_valid *= 2
                stalled = violations
        row['seconds'] = time.perf_counter()-started
        trace.append(row)
        if on_step is not None: on_step(row)
    return {'status': 'feasible' if best is not None else 'no_feasible_iterate',
            'preflight': preflight, 'candidate': best, 'trace': trace,
            'final_controls_diagnostic_only': w.detach().cpu()}

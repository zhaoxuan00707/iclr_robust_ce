"""Graph-supported WRO in the current favorable region, with no validity penalty.

Graph distances approximate the intrinsic metric. Discrete support search is a
local numerical solver; graph OT is unregularized and checked independently.
"""
from dataclasses import dataclass
import numpy as np
import ot
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from scipy.sparse.csgraph import dijkstra

from .risk_geometry import reconstruct_route, traverse_polyline


@dataclass
class GraphPlan:
    source_index: np.ndarray
    target_index: np.ndarray
    weights: np.ndarray
    distance: float
    marginal_residual: float


def graph_ot(distances, source_ids, target_ids, source_mass=None, target_mass=None):
    """Exact unregularized quadratic graph OT; never replace infinity by a cost."""
    source_ids, target_ids = np.asarray(source_ids), np.asarray(target_ids)
    n, m = len(source_ids), len(target_ids)
    a = np.full(n, 1/n) if source_mass is None else np.asarray(source_mass, dtype=float)
    b = np.full(m, 1/m) if target_mass is None else np.asarray(target_mass, dtype=float)
    if (a <= 0).any() or (b <= 0).any() or not np.isclose(a.sum(), 1) or not np.isclose(b.sum(), 1):
        raise ValueError('positive normalized masses required')
    cost = np.asarray(distances[np.ix_(source_ids, target_ids)], dtype=float)**2
    if not np.isfinite(cost).all():
        # Solve finite component blocks, preserving their mass rather than
        # using an arbitrary large substitute for disconnected transport.
        remaining = set(range(n)); pieces = []
        while remaining:
            i = min(remaining)
            js = np.flatnonzero(np.isfinite(cost[i]))
            if not len(js): raise ValueError('source mass has no finite target route')
            ii = np.flatnonzero(np.isfinite(cost[:, js[0]]))
            if not np.isclose(a[ii].sum(), b[js].sum(), atol=1e-10, rtol=0):
                raise ValueError('incompatible component masses')
            mass = a[ii].sum()
            plan = ot.emd(a[ii]/mass, b[js]/mass, cost[np.ix_(ii, js)], numItermax=1000000)
            x, y = np.nonzero(plan > 0)
            pieces.append((ii[x], js[y], plan[x, y]*mass)); remaining.difference_update(ii.tolist())
        i, j, weights = [np.concatenate([p[k] for p in pieces]) for k in range(3)]
    elif n == m and np.allclose(a, 1/n, atol=1e-14, rtol=0) and np.allclose(b, 1/m, atol=1e-14, rtol=0):
        i, j = linear_sum_assignment(cost); weights = a[i]
    else:
        plan, log = ot.emd(a, b, cost, numItermax=1000000, log=True, numThreads=1)
        if log['warning'] is not None: raise RuntimeError(log['warning'])
        i, j = np.nonzero(plan > 0); weights = plan[i, j]
    residual = max(np.abs(np.bincount(i, weights=weights, minlength=n)-a).max(),
                   np.abs(np.bincount(j, weights=weights, minlength=m)-b).max())
    if residual > 1e-8: raise ValueError('OT marginals failed')
    return GraphPlan(i, j, weights, float(np.sqrt(np.sum(weights*cost[i,j]))), float(residual))


def bank_risk(node_hinge, node_ids):
    """One shared model acts on the complete equal-mass particle distribution."""
    counts = np.bincount(node_ids, minlength=node_hinge.shape[1])/len(node_ids)
    risks = node_hinge @ counts
    return float(risks.max()), risks


def geometry(distances, source_ids, control_ids, target_ids):
    first = graph_ot(distances, source_ids, control_ids)
    second = graph_ot(distances, control_ids, target_ids)
    return first, second, first.distance+second.distance


def solve_nodes(distances, adjacency, node_hinge, source_ids, target_ids, budget,
                initial=None, incumbent=None, steps=32, proposal_models_temperature=100.,
                backtracks=12, tolerance=1e-6, on_step=None):
    """Monotone feasible discrete descent using neighbor proposals and exact OT.

    All particles have the original target mass. Moves stay on favorable graph
    vertices. Rank proposals by a smooth outer-model directional derivative;
    accept using TRUE bank risk and exact total graph W2. No geometry or
    current-validity term is added to the risk objective.
    """
    target_ids = np.asarray(target_ids, dtype=np.int64)
    start = target_ids.copy() if initial is None else np.asarray(initial, dtype=np.int64).copy()
    if start.shape != target_ids.shape: raise ValueError('one particle per target anchor required')
    D = graph_ot(distances, source_ids, target_ids).distance
    if D > budget+tolerance: return {'status':'infeasible_endpoint_budget','candidate':None,'trace':[]}
    candidates = [target_ids, start]
    if incumbent is not None: candidates.append(np.asarray(incumbent, dtype=np.int64))
    admissible = []
    for candidate in candidates:
        if candidate.min() < 0 or candidate.max() >= adjacency.shape[0]: raise ValueError('support must be favorable graph nodes')
        p0, p1, length = geometry(distances, source_ids, candidate, target_ids)
        if length <= budget+tolerance:
            risk,_ = bank_risk(node_hinge,candidate)
            admissible.append((risk,candidate.copy(),p0,p1,length))
    risk,control,p0,p1,length = min(admissible,key=lambda r:r[0])
    trace=[]
    for step in range(steps+1):
        record={'step':step,'true_bank_hinge':risk,'d0':p0.distance,'d1':p1.distance,
                'total_length':length,'D':D,'B_geo':budget,'slack':budget-length,
                'lambda_Q':p0.distance/length if length else 0.,'marginal_residual':max(p0.marginal_residual,p1.marginal_residual)}
        trace.append(record)
        if on_step: on_step(record)
        if step == steps: break
        _,model_risks = bank_risk(node_hinge,control)
        model_weights = softmax(proposal_models_temperature*model_risks)
        potential = model_weights @ node_hinge
        destinations=control.copy(); gains=np.zeros(len(control))
        for node in np.unique(control):
            neighbors=adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node+1]]
            choices=np.r_[node,neighbors]
            best=int(choices[np.argmin(potential[choices])])
            where=np.flatnonzero(control==node)
            destinations[where]=best; gains[where]=potential[node]-potential[best]
        order=np.argsort(-gains,kind='stable');order=order[gains[order]>1e-14]
        if not len(order): record['termination']='no_improving_neighbor_proposal'; break
        accepted=False; attempts=[]
        sizes=sorted(set(max(1,len(order)//(2**k)) for k in range(backtracks)),reverse=True)
        for size in sizes:
            proposal=control.copy();proposal[order[:size]]=destinations[order[:size]]
            trial,_=bank_risk(node_hinge,proposal)
            if trial >= risk-1e-12: continue
            a,b,total=geometry(distances,source_ids,proposal,target_ids)
            attempts.append({'particles':size,'risk':trial,'length':total})
            if total <= budget+tolerance:
                risk,control,p0,p1,length=trial,proposal,a,b,total;accepted=True;break
        record['proposal_evaluations']=attempts
        if not accepted: record['termination']='no_feasible_improving_batch_in_declared_proposals';break
    return {'status':'feasible_local_search','candidate':{'node_ids':control,'true_bank_hinge':risk,
            'd0':p0.distance,'d1':p1.distance,'total_length':length,'slack':budget-length},'trace':trace}


def intrinsic_path(nodes, distances, predecessor, source_ids, control_ids, target_ids, lambdas):
    """Fixed-plan graph-geodesic interpolation; glue through uniform particles."""
    p0,p1,length=geometry(distances,source_ids,control_ids,target_ids)
    if len(p1.weights)!=len(control_ids) or len(np.unique(p1.source_index))!=len(control_ids):
        raise ValueError('second-leg uniform assignment required')
    permutation=np.empty(len(control_ids),dtype=np.int64);permutation[p1.source_index]=p1.target_index
    i,j=p0.source_index,p0.target_index;k=permutation[j]
    first_routes=[nodes[reconstruct_route(predecessor[int(source_ids[a])],int(source_ids[a]),int(control_ids[b]))] for a,b in zip(i,j)]
    second_routes=[nodes[reconstruct_route(predecessor[int(control_ids[b])],int(control_ids[b]),int(target_ids[c]))] for b,c in zip(j,k)]
    pivot=p0.distance/length if length else 0.
    outputs=[]
    for lam in lambdas:
        if not 0 <= lam <= 1:raise ValueError('lambda outside [0,1]')
        if lam==0:values=nodes[np.asarray(source_ids)[i]]
        elif lam==1:values=nodes[np.asarray(target_ids)[k]]
        elif length==0:values=nodes[np.asarray(source_ids)[i]]
        elif lam<=pivot and p0.distance>0:
            values=np.stack([traverse_polyline(route,np.ones(len(route)),lam/pivot) for route in first_routes])
        else:
            fraction=(lam-pivot)/(1-pivot) if p1.distance>0 else 1.
            values=np.stack([traverse_polyline(route,np.ones(len(route)),fraction) for route in second_routes])
        outputs.append(values)
    return {'q':np.stack(outputs),'weights':p0.weights,'source_index':i,'target_index':k,
            'control_index':j,'lambda_Q':pivot,'d0':p0.distance,'d1':p1.distance,'total_length':length,
            'first_routes':first_routes,'second_routes':second_routes,
            'marginal_residual':max(p0.marginal_residual,p1.marginal_residual)}

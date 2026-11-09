"""Fixed undirected graph approximation to a bounded spatial CF metric.

This module changes CF geometry only. It has no data-adversary budget or update
logic. Feasibility is checked at finitely many edge samples, not continuously.
"""
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from sklearn.neighbors import NearestNeighbors


def knn_edges(nodes, neighbors=20):
    """Union-symmetrized kNN topology, with unique nonzero undirected edges."""
    x = np.asarray(nodes, dtype=np.float64)
    if len(np.unique(x, axis=0)) != len(x):
        raise ValueError("deduplicate nodes and preserve endpoint mappings first")
    ids = NearestNeighbors(n_neighbors=min(neighbors+1, len(x)), algorithm="brute").fit(x).kneighbors(x, return_distance=False)
    pairs = {(min(i, int(j)), max(i, int(j))) for i, row in enumerate(ids) for j in row if i != j}
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def edge_samples(nodes, edges, quadrature=9):
    t = np.linspace(0., 1., quadrature)
    x = np.asarray(nodes)
    return (1-t[None, :, None])*x[edges[:, 0], None]+t[None, :, None]*x[edges[:, 1], None]


def edge_lengths(nodes, edges, values):
    """Trapezoidal integral of the stored piecewise-linear positive weight."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(edges) or values.shape[1] < 2:
        raise ValueError("one quadrature weight vector is required per edge")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("metric weights must be finite and positive")
    distance = np.linalg.norm(np.asarray(nodes)[edges[:, 1]]-np.asarray(nodes)[edges[:, 0]], axis=1)
    if (distance <= 0).any():
        raise ValueError("zero-length edges require node deduplication")
    return distance*np.mean((values[:, :-1]+values[:, 1:])/2, axis=1)


def sparse_graph(node_count, edges, lengths):
    lengths = np.asarray(lengths)
    if not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError("edge costs must be finite and positive")
    return coo_matrix((np.r_[lengths, lengths],
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(node_count, node_count)).tocsr()


def distances(graph, sources, targets):
    d, predecessor = dijkstra(graph, directed=False, indices=np.asarray(sources), return_predecessors=True)
    return d[:, np.asarray(targets)], predecessor


def component_mass_audit(graph, sources, targets, source_mass=None, target_mass=None):
    count, component = connected_components(graph, directed=False)
    a = np.full(len(sources), 1/len(sources)) if source_mass is None else np.asarray(source_mass)
    b = np.full(len(targets), 1/len(targets)) if target_mass is None else np.asarray(target_mass)
    source = np.bincount(component[sources], weights=a, minlength=count)
    target = np.bincount(component[targets], weights=b, minlength=count)
    return {"components": count, "source_mass": source.tolist(), "target_mass": target.tolist(),
            "maximum_mass_mismatch": float(np.max(np.abs(source-target))),
            "compatible": bool(np.allclose(source, target, atol=1e-9, rtol=0))}


def balanced_plan(distance, source_mass=None, target_mass=None):
    """Unregularized finite LP; unreachable pairs are absent variables."""
    distance = np.asarray(distance, dtype=np.float64)
    n, m = distance.shape
    a = np.full(n, 1/n) if source_mass is None else np.asarray(source_mass, dtype=float)
    b = np.full(m, 1/m) if target_mass is None else np.asarray(target_mass, dtype=float)
    if (not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any()
            or not np.isclose(a.sum(), 1.) or not np.isclose(b.sum(), 1.)):
        raise ValueError("endpoint masses must be probability distributions")
    ii, jj = np.where(np.isfinite(distance))
    if not len(ii) or (distance[ii, jj] < 0).any():
        raise ValueError("no finite nonnegative transport distances")
    cols = np.arange(len(ii))
    constraints = coo_matrix((np.ones(2*len(ii)),
        (np.r_[ii, n+jj], np.r_[cols, cols])), shape=(n+m, len(ii))).tocsr()
    solution = linprog(distance[ii, jj]**2, A_eq=constraints, b_eq=np.r_[a, b],
                       bounds=(0, None), method="highs")
    if not solution.success:
        raise ValueError(f"finite-distance mass transport infeasible: {solution.message}")
    active = solution.x > 0
    i, j, mass = ii[active], jj[active], solution.x[active]
    residual = max(np.max(np.abs(np.bincount(i, weights=mass, minlength=n)-a)),
                   np.max(np.abs(np.bincount(j, weights=mass, minlength=m)-b)))
    if residual > 1e-8:
        raise RuntimeError("OT marginal residual exceeds tolerance")
    return i, j, mass, {"solver": "scipy HiGHS linear programming", "regularization": 0.,
                       "marginal_residual": float(residual), "objective": float(solution.fun)}


def reconstruct_route(predecessor, source, target):
    route = [int(target)]
    while route[-1] != source:
        next_node = int(predecessor[route[-1]])
        if next_node < 0 or next_node in route:
            raise ValueError("unreachable or cyclic predecessor chain")
        route.append(next_node)
    return np.asarray(route[::-1], dtype=np.int64)


def traverse_polyline(points, point_weights, lam):
    """Invert the SAME piecewise-linear weight integral used in edge costs."""
    if not 0 <= lam <= 1:
        raise ValueError("lambda must be in [0,1]")
    points, w = np.asarray(points, dtype=np.float64), np.asarray(point_weights, dtype=np.float64)
    if len(points) == 1 or lam == 0:
        return points[0].copy()
    if lam == 1:
        return points[-1].copy()
    length = np.linalg.norm(np.diff(points, axis=0), axis=1)
    integral = length*(w[:-1]+w[1:])/2
    cumulative = np.r_[0., np.cumsum(integral)]
    if cumulative[-1] == 0:
        return points[0].copy()
    goal = lam*cumulative[-1]
    idx = min(int(np.searchsorted(cumulative, goal, side="right")-1), len(length)-1)
    unit_integral = (goal-cumulative[idx])/length[idx]
    # Solve w0*t + (w1-w0)*t^2/2 = unit_integral without cancellation.
    disc = max(0., w[idx]**2+2*(w[idx+1]-w[idx])*unit_integral)
    t = 2*unit_integral/(w[idx]+np.sqrt(disc))
    return (1-t)*points[idx]+t*points[idx+1]


@dataclass
class RoutePopulation:
    """Weighted split atoms; factual identity is stored separately from routes."""
    source_index: torch.Tensor
    target_index: torch.Tensor
    weights: torch.Tensor
    route_points: list
    route_weights: list
    route_lengths: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor

    def at(self, lam):
        result = np.stack([traverse_polyline(p, w, lam) for p, w in zip(self.route_points, self.route_weights)])
        return torch.as_tensor(result, dtype=self.source.dtype)

    @property
    def speed_bound(self):
        return float(torch.sqrt((self.weights.double()*self.route_lengths.double().square()).sum()))

    def state_dict(self):
        return dict(vars(self))


def route_population(nodes, edges, values, sources, targets, source, target):
    """Freeze a balanced coupling and its discrete shortest routes."""
    x = np.asarray(nodes)
    lengths = edge_lengths(x, edges, values)
    graph = sparse_graph(len(x), edges, lengths)
    components = component_mass_audit(graph, sources, targets)
    if not components["compatible"]:
        raise ValueError(f"component mass mismatch: {components}")
    distance, predecessor = distances(graph, sources, targets)
    i, j, mass, audit = balanced_plan(distance)
    lookup = {tuple(pair): e for e, pair in enumerate(edges)}
    route_points, route_weights, route_nodes, route_lengths = [], [], [], []
    samples = edge_samples(x, edges, values.shape[1])
    for a, b in zip(i, j):
        route = reconstruct_route(predecessor[a], sources[a], targets[b])
        points, weights = [], []
        for u, v in zip(route[:-1], route[1:]):
            e = lookup[(min(u, v), max(u, v))]
            pp, ww = samples[e], values[e]
            if u > v:
                pp, ww = pp[::-1], ww[::-1]
            points.extend(pp[:-1]); weights.extend(ww[:-1])
        points.append(x[route[-1]])
        weights.append(float(ww[-1]) if len(route) > 1 else 1.)
        pp, ww = np.asarray(points), np.asarray(weights)
        length = float((np.linalg.norm(np.diff(pp, axis=0), axis=1)*(ww[:-1]+ww[1:])/2).sum())
        if not np.isclose(length, distance[a, b], atol=1e-7):
            raise RuntimeError("route traversal and graph edge costs disagree")
        route_points.append(torch.from_numpy(pp.copy()))
        route_weights.append(torch.from_numpy(ww.copy()))
        route_nodes.append(route.tolist()); route_lengths.append(length)
    population = RoutePopulation(torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(mass),
        route_points, route_weights, torch.tensor(route_lengths, dtype=torch.float64), source, target)
    return population, {**audit, "components": components, "speed_bound": population.speed_bound,
                        "route_nodes": route_nodes}, distance

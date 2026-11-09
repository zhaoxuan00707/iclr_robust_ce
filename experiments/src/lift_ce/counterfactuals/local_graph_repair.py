"""Minimal component-local edges, with no relaxation of graph feasibility."""
import numpy as np
import torch
from scipy.sparse.csgraph import connected_components

from .risk_geometry import sparse_graph, component_mass_audit


def relu_segment_minimum(model, start, end):
    """Enumerate affine pieces of a Linear/ReLU MLP on one segment.

    Float64 breakpoint arithmetic provides a numerical check of all pieces,
    not an interval-arithmetic certificate or a physical actionability claim.
    """
    layers = list(model.features)+[model.classifier]
    pieces = [(0., 1., np.asarray(end, dtype=float)-np.asarray(start, dtype=float), np.asarray(start, dtype=float))]
    for layer in layers:
        if isinstance(layer, torch.nn.Linear):
            w = layer.weight.detach().cpu().double().numpy()
            b = layer.bias.detach().cpu().double().numpy()
            pieces = [(lo, hi, w@a, w@c+b) for lo, hi, a, c in pieces]
        elif isinstance(layer, torch.nn.ReLU):
            updated = []
            for lo, hi, a, c in pieces:
                active = np.abs(a) > 1e-15
                roots = -c[active]/a[active]
                cuts = np.unique(np.r_[lo, roots[(roots > lo+1e-12)&(roots < hi-1e-12)], hi])
                for left, right in zip(cuts[:-1], cuts[1:]):
                    positive = a*((left+right)/2)+c > 0
                    updated.append((left, right, a*positive, c*positive))
            pieces = updated
        else:
            raise ValueError('segment verification requires a Linear/ReLU classifier')
    minimum = min(float(np.minimum(a*lo+c, a*hi+c).min()) for lo, hi, a, c in pieces)
    return {'minimum_logit': minimum, 'affine_pieces': len(pieces), 'precision': 'float64 numerical breakpoint enumeration'}


def minimal_component_bridges(nodes, edges, sources, targets, allowed):
    """Add the nearest admissible cross-component edge, only where mass differs.

    ``allowed(u,v)`` performs the existing domain check and returns diagnostics
    or None. Stop after component source/target masses balance. No new nodes,
    no global neighborhood expansion, and no artificial finite costs.
    """
    nodes = np.asarray(nodes)
    current = [tuple(map(int, edge)) for edge in edges]
    added, attempts = [], []
    while True:
        e = np.asarray(current, dtype=np.int64)
        g = sparse_graph(len(nodes), e, np.linalg.norm(nodes[e[:, 0]]-nodes[e[:, 1]], axis=1))
        audit = component_mass_audit(g, np.asarray(sources), np.asarray(targets))
        if audit['compatible']:
            return np.asarray(added, dtype=np.int64).reshape(-1, 2), attempts, audit
        _, labels = connected_components(g, directed=False)
        excess = np.asarray(audit['source_mass'])-np.asarray(audit['target_mass'])
        source_components = np.flatnonzero(excess > 1e-9)
        target_components = np.flatnonzero(excess < -1e-9)
        options = []
        for u in np.flatnonzero(np.isin(labels, source_components)):
            order = np.argsort(np.linalg.norm(nodes-nodes[u], axis=1), kind='stable')
            order = order[order != u]
            for rank, v in enumerate(order, 1):
                if labels[v] in target_components:
                    options.append((float(np.linalg.norm(nodes[u]-nodes[v])), int(u), int(v), rank))
        found = False
        for distance, u, v, rank in sorted(options):
            evidence = allowed(u, v)
            attempts.append({'u': u, 'v': v, 'neighbor_rank': rank, 'euclidean_length': distance,
                             'accepted': evidence is not None, 'evidence': evidence})
            if evidence is not None:
                pair = (min(u, v), max(u, v))
                current.append(pair); added.append(pair); found = True
                break
        if not found:
            raise ValueError('No admissible local edge between mass-incompatible components; no relaxation applied')

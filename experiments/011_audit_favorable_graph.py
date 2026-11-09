"""Read-only historical input audit; write only a new task-003 diagnostic run."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from run import bootstrap
bootstrap()
from lift_ce.experiments import fixed_risk_geometry_v1 as e
from lift_ce.counterfactuals.local_graph_repair import relu_segment_minimum, minimal_component_bridges
from lift_ce.counterfactuals.risk_geometry import sparse_graph, component_mass_audit
from scipy.sparse.csgraph import dijkstra, connected_components

parent = Path('/home/xu.zhao/robust_ce_results/fixed_risk_weighted_ot_geometry_v1/20260914_heloc_local_repair_v1/heloc')
instance = Path('/home/xu.zhao/robust_ce_results/single_budget_geodesic_wro_v1/20260915_single_budget_v1/heloc')
out = Path('/home/xu.zhao/robust_ce_results/iclr_runs/favorable_region_geodesic_wro_v1/20260916_graph_audit')
out.mkdir(parents=True, exist_ok=False)
model = e.legacy.load_model(instance/'current.pt', 22)
graph = torch.load(parent/'graph.pt', weights_only=True)
nodes = np.asarray(graph['nodes'], dtype=np.float64)
edges = np.asarray(graph['edges'], dtype=np.int64)
source, target = np.asarray(graph['source_nodes']), np.asarray(graph['target_nodes'])
pop = torch.load(instance/'train_population.pt', weights_only=True)
train = torch.load(instance/'training.pt', weights_only=True)
np.testing.assert_array_equal(nodes[source], pop['path']['base'].numpy())
np.testing.assert_array_equal(nodes[target], train['target'].numpy())
if e.p.file_hash(parent/'current.pt') != e.p.file_hash(instance/'current.pt'):
    raise ValueError('f0 mismatch')
with torch.no_grad():
    scores = torch.sigmoid(model(torch.from_numpy(nodes).float()))
    if (scores < .5).any(): raise ValueError('unfavorable node')
rows = []; start = time.perf_counter()
for i, (u, v) in enumerate(edges):
    check = relu_segment_minimum(model, nodes[u], nodes[v])
    rows.append({'u': int(u), 'v': int(v), **check, 'accepted': check['minimum_logit'] >= 0})
    if i%5000 == 0: print('checked', i, 'seconds', time.perf_counter()-start, flush=True)
pd.DataFrame(rows).to_csv(out/'edge_checks.csv', index=False)
retained = edges[[r['accepted'] for r in rows]]
def allowed(u, v):
    check = relu_segment_minimum(model, nodes[u], nodes[v])
    return check if check['minimum_logit'] >= 0 else None
added, attempts, audit = minimal_component_bridges(nodes, retained, source, target, allowed)
retained = np.concatenate([retained, added])
lengths = np.linalg.norm(nodes[retained[:, 0]]-nodes[retained[:, 1]], axis=1)
adjacency = sparse_graph(len(nodes), retained, lengths)
distances, predecessors = dijkstra(adjacency, directed=False, return_predecessors=True)
np.save(out/'distances.npy', distances); np.save(out/'predecessors.npy', predecessors)
e.save({'nodes': torch.from_numpy(nodes), 'edges': torch.from_numpy(retained),
        'source_nodes': torch.from_numpy(source), 'target_nodes': torch.from_numpy(target),
        'lengths': torch.from_numpy(lengths)}, out/'graph.pt')
report = {'historical_graph': str(parent/'graph.pt'), 'historical_graph_sha256': e.p.file_hash(parent/'graph.pt'),
          'instance': str(instance), 'current_sha256': e.p.file_hash(instance/'current.pt'),
          'nodes': len(nodes), 'original_edges': len(edges), 'rejected_edges': len(edges)-sum(r['accepted'] for r in rows),
          'added_edges': added.tolist(), 'local_repair_attempts': attempts, 'components': audit,
          'minimum_node_score': float(scores.min()), 'seconds': time.perf_counter()-start,
          'metric': 'unit Euclidean edge length; all ReLU affine pieces checked in float64; numerical, not interval certified',
          'files': {f: e.p.file_hash(out/f) for f in ['graph.pt', 'distances.npy', 'predecessors.npy', 'edge_checks.csv']}}
e.p.write_json(out/'audit.json', report)
print(json.dumps(report, indent=2), flush=True)

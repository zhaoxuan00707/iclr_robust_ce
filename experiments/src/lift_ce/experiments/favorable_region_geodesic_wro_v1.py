"""Task 003: direct finite-particle WRO on a frozen favorable-region graph.

Graph-node feasible descent is a local solver, not unrestricted distributional
optimization. Historical controls remain unchanged and explicitly Euclidean.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from scipy.sparse.csgraph import connected_components

from . import fixed_risk_geometry_v1 as e
from . import transport_spline_path_v1 as merger
from ..counterfactuals.favorable_region import bank_risk, graph_ot, solve_nodes, intrinsic_path
from ..counterfactuals.local_graph_repair import relu_segment_minimum
from ..counterfactuals.risk_geometry import sparse_graph, component_mass_audit
from ..counterfactuals.robust_waypoints import ModelBank

PARENT = Path('/home/xu.zhao/robust_ce_results/single_budget_geodesic_wro_v1/20260915_single_budget_v1/heloc')
AUDIT = Path('/home/xu.zhao/robust_ce_results/iclr_runs/favorable_region_geodesic_wro_v1/20260916_graph_audit')
BETAS = [1., 1.05, 1.10, 1.20]
INTRINSIC = [f'F_beta{round(b*100):03d}' for b in BETAS]
CONTROLS = ['S_straight', 'S_old_waypoint', 'S_beta100', 'S_beta105', 'S_beta110', 'S_beta120']
LAMBDAS = [round(i*.05, 2) for i in range(21)]
OPT = {'steps': 32, 'proposal_models_temperature': 100., 'backtracks': 12,
       'tolerance': 1e-6}


def read(file):
    return json.loads(Path(file).read_text())


def verify_files(out, name):
    manifest = read(out/name)
    for file, digest in manifest['files'].items():
        if e.p.file_hash(out/file) != digest:
            raise ValueError(f'frozen input changed: {file}')
    return manifest


def verify_bank(records, meta, train):
    """Verify each saved perturbation once; no held-out/oracle construction."""
    grouped = defaultdict(list)
    checked = []
    for a in records:
        if a['split'] != 'train' or a['full_update_seed'] != meta['full_update_seed']:
            raise ValueError('incompatible candidate provenance')
        if 'oracle' in str(a.get('origin', '')).lower():
            raise ValueError('oracle model forbidden')
        if e.p.file_hash(Path(a['model_path'])) != a['model_sha256']:
            raise ValueError('model changed')
        grouped[a['delta_file']].append(a)
    for name, members in grouped.items():
        digest = e.p.file_hash(Path(name))
        obj = torch.load(name, weights_only=False)
        if obj.get('discovery_split') != 'train' or obj.get('full_update_seed') != meta['full_update_seed']:
            raise ValueError('saved perturbation provenance mismatch')
        for a in members:
            if a.get('delta_file_sha256', digest) != digest:
                raise ValueError('saved perturbation changed')
            value = obj['delta'] if a.get('delta_key') == 'delta' else obj['candidates'][a['delta_index']]
            delta = value['delta'] if isinstance(value, dict) else value
            norm = float(e.rms(delta))
            if delta.shape != train['x'].shape or not torch.isfinite(delta).all():
                raise ValueError('invalid displacement')
            if norm > min(3., a['nominal_radius'])*meta['b_ref']+1e-5:
                raise ValueError('candidate outside fixed data budget')
            checked.append({**a, 'measured_budget': norm, 'delta_file_sha256': digest})
    if len(set(a['candidate_id'] for a in checked)) != len(checked):
        raise ValueError('duplicate candidate IDs')
    return checked


def prepare(args):
    out = args.output/'heloc'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'construction_manifest.json').exists():
        verify_files(out, 'construction_manifest.json'); return
    audit = read(AUDIT/'audit.json')
    for name, digest in audit['files'].items():
        if e.p.file_hash(AUDIT/name) != digest: raise ValueError('graph audit changed')
    for name in ['instance_manifest.json', 'training.pt', 'current.pt', 'train_population.pt',
                 'data_audit.pt', 'frozen_waypoint.pt', 'waypoint_manifest.json']:
        shutil.copyfile(PARENT/name, out/name)
    _, meta, train, current = e.context(args)
    for name in ['graph.pt', 'distances.npy', 'predecessors.npy', 'edge_checks.csv', 'audit.json']:
        shutil.copyfile(AUDIT/name, out/name)
    records = verify_bank(read(PARENT/'evaluation_bank.json')['candidates'], meta, train)
    records = [{**a, 'parent_phase': a['phase'], 'phase': 'old'} for a in records]
    if len(records) != 5387: raise ValueError('incomplete historical bank')
    e.p.write_json(out/'old_bank.json', {'candidates': records, 'parent': str(PARENT/'evaluation_bank.json'),
                                       'parent_sha256': e.p.file_hash(PARENT/'evaluation_bank.json')})
    pd.DataFrame(records).to_csv(out/'attack_bank_manifest.csv', index=False)
    g = torch.load(out/'graph.pt', weights_only=True)
    nodes, s, t = g['nodes'].numpy(), g['source_nodes'].numpy(), g['target_nodes'].numpy()
    pop = torch.load(out/'train_population.pt', weights_only=True)
    np.testing.assert_array_equal(nodes[s], pop['path']['base'])
    np.testing.assert_array_equal(nodes[t], train['target'])
    distance = np.load(out/'distances.npy', mmap_mode='r')
    D = graph_ot(distance, s, t).distance
    config = {'task': 'tasks/003_baseline.md', 'dataset': 'heloc', 'tau': .5,
              'fixed_radius_over_b_ref': 3., 'absolute_data_budget': 3*meta['b_ref'],
              'beta_geo': BETAS, 'D': D, 'lambdas': LAMBDAS, 'optimizer': OPT,
              'decision': '3336 equal-mass particles on favorable construction graph nodes',
              'objective': 'max_model population_mean_true_hinge at Q itself',
              'constraint': 'graph_W2(P_CF,Q) + graph_W2(Q,P1) <= beta_geo*D',
              'metric': 'unweighted shortest paths on graph with float64 ReLU affine-piece edge checks',
              'regularization': 0., 'solver': 'local discrete feasible descent, exact OT at each accepted move',
              'initialization': 'target plus retained smaller-budget feasible incumbent',
              'domain': 'historical all-mutable continuous representation; F0 at tau=.5; no new feature box',
              'query_rule': 'single nearest admissible frozen-node leaf attachment per query; deterministic distance/index order; same Q, separate exact OT; no training graph shortcut or mutation',
              'heldout_budget': 'report fixed training B_geo feasibility per query batch; never assume transfer',
              'selection': 'fixed r=3; exact calibration-bank argmin, all exact ties, smallest lambda; beta not selected',
              'search': {('softplus_sharpness' if k == 'kappa' else k): v for k,v in e.SETTINGS['search'].items()},
              'source_lambdas': [.25,.5,.75], 'replications': 3,
              'fresh_methods': INTRINSIC, 'controls': CONTROLS,
              'historical_controls_fresh_search': 'already contained in frozen bank; new equal-budget searches on all four new paths',
              'confirmation': 'previously inspected retrospective benchmark, not untouched confirmation',
              'bank_count': len(records), 'graph_audit': str(AUDIT/'audit.json')}
    e.p.write_json(out/'resolved_config.json', config)
    pd.DataFrame([{'beta_geo': b, 'D': D, 'B_geo': b*D} for b in BETAS]).to_csv(out/'beta_geo_grid.csv', index=False)
    graph = sparse_graph(len(nodes), g['edges'].numpy(), g['lengths'].numpy())
    count, labels = connected_components(graph, directed=False)
    mass = component_mass_audit(graph, s, t)
    pd.DataFrame([{'component': i, 'source_mass': mass['source_mass'][i], 'target_mass': mass['target_mass'][i]} for i in range(count)]).to_csv(out/'component_diagnostics.csv', index=False)
    pd.DataFrame([{'nodes': len(nodes), 'edges': len(g['edges']), 'components': count,
                   'rejected_historical_edges': audit['rejected_edges'], 'added_edges': len(audit['added_edges']),
                   'mass_compatible': mass['compatible']}]).to_csv(out/'graph_connectivity.csv', index=False)
    files = ['instance_manifest.json','training.pt','current.pt','train_population.pt','graph.pt',
             'distances.npy','predecessors.npy','old_bank.json','resolved_config.json']
    e.p.write_json(out/'construction_manifest.json', {'files': {n: e.p.file_hash(out/n) for n in files},
                                                    'heldout_outcomes_used': False})
    print('prepared intrinsic D', D, 'bank', len(records), flush=True)


def smoke(args):
    out, meta, train, current = e.context(args)
    verify_files(out, 'construction_manifest.json')
    g = torch.load(out/'graph.pt', weights_only=True)
    nodes = g['nodes'].numpy(); sources = g['source_nodes'].numpy(); targets = g['target_nodes'].numpy()
    distance = np.load(out/'distances.npy'); predecessor = np.load(out/'predecessors.npy')
    adjacency = sparse_graph(len(nodes), g['edges'].numpy(), g['lengths'].numpy())
    model = e.legacy.load_model(Path(read(out/'old_bank.json')['candidates'][0]['model_path']), 22)
    with torch.no_grad():
        hinge = torch.relu(.5-torch.sigmoid(model(torch.from_numpy(nodes).float()))).numpy()[None]
    start = time.perf_counter()
    result = solve_nodes(distance, adjacency, hinge, sources[:8], targets[:32],
                         1.2*graph_ot(distance, sources[:8], targets[:32]).distance, steps=2)
    candidate = result['candidate']['node_ids']
    path = intrinsic_path(nodes, distance, predecessor, sources[:8], candidate, targets[:32], [0.,.5,1.])
    with torch.no_grad():
        probability = torch.sigmoid(current(torch.from_numpy(path['q'].reshape(-1,22)).float()))
    if (probability < .5).any(): raise ValueError('smoke path invalid')
    small_seconds = time.perf_counter()-start
    start=time.perf_counter()
    full=graph_ot(distance,sources,targets)
    same=graph_ot(distance,targets,targets)
    seconds=time.perf_counter()-start
    e.p.write_json(out/'smoke.json', {'small_sources':8,'small_targets':32,'small_steps':2,
                   'small_seconds':small_seconds,'full_geometry_seconds':seconds,'D':full.distance,
                   'target_self_distance':same.distance,'minimum_path_probability':float(probability.min()),
                   'excluded_from_candidate_pool':True,
                   'max_proposal_trials':4*OPT['steps']*OPT['backtracks'],
                   'rough_geometry_seconds_upper_proxy':4*OPT['steps']*OPT['backtracks']*seconds,
                   'node_cache_bytes_float32':5387*len(nodes)*4,
                   'discovery_jobs':36,'retained_candidates_per_job':15,
                   'fresh_runtime_estimate':'measure first equal-strength job; do not reduce resources for one beta'})
    print('smoke',small_seconds,'full geometry',seconds,flush=True)


def cache(args):
    out, meta, train, current = e.context(args)
    verify_files(out, 'construction_manifest.json')
    records = read(out/'old_bank.json')['candidates']
    nodes = torch.load(out/'graph.pt', weights_only=True)['nodes'].float().to(args.device)
    chosen = records[args.worker::args.workers]
    start = time.perf_counter()
    models = [e.legacy.load_model(Path(a['model_path']), nodes.shape[1]) for a in chosen]
    bank = ModelBank(models, args.device)
    values = []
    with torch.no_grad():
        for i in range(0, len(models), 32):
            values.append(torch.relu(.5-torch.sigmoid(bank.logits(nodes, i, i+32))).cpu().numpy())
    np.savez_compressed(out/f'node_hinge_{args.worker}.npz', values=np.concatenate(values),
                        indices=np.arange(args.worker, len(records), args.workers))
    e.p.write_json(out/f'cache_timing_{args.worker}.json', {'seconds': time.perf_counter()-start,
                      'models': len(models), 'nodes': len(nodes), 'device': args.device})
    print('cached', args.worker, len(models), flush=True)


def optimize(args):
    out, meta, train, current = e.context(args)
    verify_files(out, 'construction_manifest.json')
    config = read(out/'resolved_config.json')
    g = torch.load(out/'graph.pt', weights_only=True)
    nodes = g['nodes'].numpy(); s = g['source_nodes'].numpy(); t = g['target_nodes'].numpy()
    adjacency = sparse_graph(len(nodes), g['edges'].numpy(), g['lengths'].numpy())
    distances = np.load(out/'distances.npy', mmap_mode='r')
    hinge = np.zeros((config['bank_count'], len(nodes)), dtype=np.float32)
    seen = []
    for worker in range(args.workers):
        part = np.load(out/f'node_hinge_{worker}.npz')
        hinge[part['indices']] = part['values']; seen.extend(part['indices'].tolist())
    if sorted(seen) != list(range(len(hinge))): raise ValueError('incomplete node bank cache')
    np.save(out/'node_hinge.npy', hinge)
    controls = []; traces = []; incumbent = None
    started = time.perf_counter()
    for beta, method in zip(BETAS, INTRINSIC):
        file = out/f'learned_distribution_{method}.pt'
        if file.exists():
            chosen = torch.load(file, weights_only=True)
            incumbent = chosen['node_ids'].numpy()
            controls.append(read(file.with_suffix('.json')))
            traces.extend(read(file.with_suffix('.trace.json'))); continue
        def progress(row):
            print(method, 'step', row['step'], 'risk', row['true_bank_hinge'],
                  'length', row['total_length'], 'slack', row['slack'], flush=True)
        result = solve_nodes(distances, adjacency, hinge, s, t, beta*config['D'],
                             incumbent=incumbent, on_step=progress, **OPT)
        candidate = result['candidate']
        if candidate is None: raise ValueError('no feasible control')
        incumbent = candidate.pop('node_ids')
        points = torch.from_numpy(nodes[incumbent]).float()
        with torch.no_grad(): prob = torch.sigmoid(current(points))
        if (prob < .5).any(): raise ValueError('control left F0')
        e.save({'node_ids': torch.from_numpy(incumbent), 'points': points,
                'weights': torch.full((len(t),), 1/len(t), dtype=torch.float64)}, file)
        row = {**candidate, 'beta_geo': beta, 'method': method, 'D': config['D'], 'B_geo': beta*config['D'],
               'lambda_Q': candidate['d0']/candidate['total_length'], 'current_validity': 1.,
               'minimum_current_score': float(prob.min()), 'control_file': file.name,
               'representation': 'fixed target-count equal particles on graph nodes',
               'solver_status': result['status']}
        trace = [{**r, 'beta_geo': beta, 'method': method} for r in result['trace']]
        e.p.write_json(file.with_suffix('.json'), row); e.p.write_json(file.with_suffix('.trace.json'), trace)
        traces.extend(trace); controls.append(row)
    risks = [r['true_bank_hinge'] for r in controls]
    if np.any(np.diff(risks) > 1e-12): raise ValueError('nested incumbent worsened')
    e.p.write_json(out/'controls_manifest.json', controls)
    pd.DataFrame(controls).to_csv(out/'geometry_feasibility.csv', index=False)
    pd.DataFrame(traces).to_csv(out/'optimization_trace.csv', index=False)
    e.p.write_json(out/'optimization_runtime.json', {'seconds': time.perf_counter()-started})


def overlay(out, pop, current):
    """Query leaves preserve every construction distance and route exactly."""
    g = torch.load(out/'graph.pt', weights_only=True)
    base = g['nodes'].numpy(); old_d = np.load(out/'distances.npy'); old_p = np.load(out/'predecessors.npy')
    query = pop['path']['base'].numpy().astype(np.float64)
    nodes = list(base); ids = []; attachments = []; diagnostics = []
    for index, point in enumerate(query):
        exact = np.flatnonzero(np.all(base == point, axis=1))
        if len(exact):
            ids.append(int(exact[0])); continue
        lengths = np.linalg.norm(base-point, axis=1)
        found = None
        for rank, j in enumerate(np.argsort(lengths, kind='stable'), 1):
            check = relu_segment_minimum(current, point, base[j])
            if check['minimum_logit'] >= 0:
                found = (int(j), float(lengths[j])); break
        if found is None: raise ValueError('query CF has no admissible frozen-node attachment; no relaxation')
        node = len(nodes); nodes.append(point); ids.append(node); attachments.append((node, *found))
        diagnostics.append({'source_index': index, 'query_node': node, 'base_node': found[0],
                            'length': found[1], 'neighbor_rank': rank, **check})
    size = len(nodes); n = len(base)
    distance = np.zeros((size, size)); predecessor = np.full((size, size), -9999, dtype=np.int32)
    distance[:n, :n] = old_d; predecessor[:n, :n] = old_p
    for node, j, length in attachments:
        distance[node, :n] = distance[:n, node] = old_d[j]+length
        predecessor[node, :n] = old_p[j]; predecessor[node, j] = node
        predecessor[:n, node] = j
    for node, j, length in attachments:
        for other, k, leg in attachments:
            if node != other:
                distance[node, other] = length+old_d[j,k]+leg
                predecessor[node, other] = k
    return np.asarray(nodes), distance, predecessor, np.asarray(ids), g['target_nodes'].numpy(), diagnostics


def paths(args):
    out, meta, train, current = e.context(args)
    verify_files(out, 'construction_manifest.json')
    split = args.split
    if split != 'train':
        verify_files(out, 'frozen_paths.json')
        if split == 'test' and not (out/'selection.json').exists():
            raise ValueError('test path extension requires frozen selection')
        src = PARENT/f'{split}_population.pt'
        reference = Path(meta['v3_instance'])/src.name
        if e.p.file_hash(src) != reference.with_suffix('.sha256').read_text().strip():
            raise ValueError('query population changed')
        shutil.copyfile(src, out/src.name)
    pop = torch.load(out/f'{split}_population.pt', weights_only=True)
    g = torch.load(out/'graph.pt', weights_only=True)
    if split == 'train':
        nodes = g['nodes'].numpy(); dist = np.load(out/'distances.npy'); pred = np.load(out/'predecessors.npy')
        sources = g['source_nodes'].numpy(); targets = g['target_nodes'].numpy(); diagnostics = []
    else:
        nodes, dist, pred, sources, targets, diagnostics = overlay(out, pop, current)
    pd.DataFrame(diagnostics).to_csv(out/f'query_attachments_{split}.csv', index=False)
    previous = torch.load(PARENT/f'paths_{split}.pt', weights_only=True)
    torch.testing.assert_close(previous['source'], pop['source'], rtol=0, atol=0)
    torch.testing.assert_close(previous['source_ids'], pop['source_ids'], rtol=0, atol=0)
    collection = {'source': pop['source'], 'source_ids': pop['source_ids'], 'methods': {m: previous['methods'][m] for m in CONTROLS}}
    checks = []; geometry = []; config = read(out/'resolved_config.json')
    for item in read(out/'controls_manifest.json'):
        method = item['method']
        ids = torch.load(out/item['control_file'], weights_only=True)['node_ids'].numpy()
        path = intrinsic_path(nodes, dist, pred, sources, ids, targets, LAMBDAS)
        knot = intrinsic_path(nodes, dist, pred, sources, ids, targets, [path['lambda_Q']])
        np.testing.assert_allclose(knot['q'][0], nodes[ids[knot['control_index']]], atol=1e-7, rtol=0)
        for key, count in [('source_index',len(sources)), ('target_index',len(targets)), ('control_index',len(ids))]:
            np.testing.assert_allclose(np.bincount(path[key], weights=path['weights'], minlength=count), 1/count, atol=1e-8, rtol=0)
        e.save({'first_routes': [torch.from_numpy(x) for x in path.pop('first_routes')],
                'second_routes': [torch.from_numpy(x) for x in path.pop('second_routes')]}, out/f'routes_{split}_{method}.pt')
        collection['methods'][method] = {k: torch.from_numpy(v).float() if k=='q' else torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in path.items()}
        geometry.append({'split': split, 'method': method, 'D': graph_ot(dist,sources,targets).distance,
                         'B_geo': item['B_geo'], 'total_length': path['total_length'], 'd0': path['d0'], 'd1': path['d1'],
                         'lambda_Q': path['lambda_Q'], 'slack': item['B_geo']-path['total_length'],
                         'budget_feasible': path['total_length'] <= item['B_geo']+OPT['tolerance'],
                         'marginal_residual': path['marginal_residual']})
    with torch.no_grad():
        for method, path in collection['methods'].items():
            for lam, q in zip(LAMBDAS, path['q']):
                probability = torch.sigmoid(current(q))
                mass = float(((probability < .5).double()*path['weights']).sum())
                checks.append({'split': split, 'method': method, 'lambda': lam,
                               'current_invalid_mass': mass, 'minimum_score': float(probability.min()), 'projected': False})
                if method in INTRINSIC and mass > 0:
                    pd.DataFrame(checks).to_csv(out/f'FAILED_validity_{split}.csv', index=False)
                    raise ValueError('intrinsic path realization left F0; no penalty or projection applied')
    pd.DataFrame(checks).to_csv(out/f'current_validity_{split}.csv', index=False)
    pd.DataFrame(geometry).to_csv(out/f'geometry_{split}.csv', index=False)
    e.save(collection, out/f'paths_{split}.pt')
    if split == 'train':
        files = ['controls_manifest.json','paths_train.pt']+[a['control_file'] for a in read(out/'controls_manifest.json')]
        e.p.write_json(out/'frozen_paths.json', {'files': {n:e.p.file_hash(out/n) for n in files},
                  'construction_sha256': e.p.file_hash(out/'construction_manifest.json'), 'frozen_before_fresh_attacks': True})
    else:
        e.p.write_json(out/f'frozen_paths_{split}.json', {'files':{f'paths_{split}.pt':e.p.file_hash(out/f'paths_{split}.pt')},
                  'training_paths_sha256':e.p.file_hash(out/'frozen_paths.json'), 'training_graph_changed':False})
    print('paths frozen', split, flush=True)


def discover(args):
    out, meta, train, current = e.context(args)
    frozen = verify_files(out, 'frozen_paths.json')
    collection = torch.load(out/'paths_train.pt', weights_only=True)
    jobs = [(method, lam, rep) for method in INTRINSIC for lam in [.25,.5,.75] for rep in range(3)]
    for job,(method,lam,rep) in enumerate(jobs):
        if job%args.workers != args.worker: continue
        file = out/'fresh_attacks'/f'{method}_r3_l{lam:.2f}_rep{rep}.pt'
        if file.with_suffix('.models.json').exists():
            if torch.load(file,weights_only=True)['frozen_geometry'] != frozen: raise ValueError('fresh input changed')
            continue
        start = time.perf_counter()
        seed = 163000000+INTRINSIC.index(method)*100000+round(lam*100)*10+rep
        path = collection['methods'][method]
        candidates,tags,trace,full,runtime,peak = e.run_search(current,train,path['q'][LAMBDAS.index(lam)],path['weights'],3*meta['b_ref'],seed,meta['full_update_seed'],args.device)
        obj = {'instance_id':meta['instance_id'],'phase':'fresh','method':method,'lambda':lam,'radius':3.,
               'replication':rep,'search_seed':seed,'full_update_seed':meta['full_update_seed'],
               'candidates':candidates,'shifted_training_distributions':[train['x']+d.cpu() for d in candidates],
               'source_beta_geo':BETAS[INTRINSIC.index(method)],'tags':tags,'discovery_split':'train','frozen_geometry':frozen}
        e.save(obj,file); trace.to_csv(file.with_suffix('.truncated.csv'),index=False); full.to_csv(file.with_suffix('.full.csv'),index=False)
        e.replay_candidates(out,meta,train,current,file,obj,args.device)
        replay = read(file.with_suffix('.models.json'))
        for i,a in enumerate(replay['candidates']):
            a.update(candidate_id=f'{args.output.name}::{a["candidate_id"]}', source_beta_geo=obj['source_beta_geo'],
                     shifted_training_file=str(file), shifted_training_index=i, labels_and_row_ids_reference=str(out/'training.pt'))
        e.p.write_json(file.with_suffix('.models.json'),replay)
        e.p.write_json(file.with_suffix('.runtime.json'),{'seconds':time.perf_counter()-start,'truncated_seconds':runtime,'peak_cuda_bytes':peak})
        verify_files(out,'frozen_paths.json'); print('fresh completed',method,lam,rep,flush=True)


def freeze_bank(args):
    out,meta,train,current = e.context(args)
    verify_files(out,'frozen_paths.json')
    fresh = []
    for method in INTRINSIC:
        for lam in [.25,.5,.75]:
            for rep in range(3):
                file=out/'fresh_attacks'/f'{method}_r3_l{lam:.2f}_rep{rep}.models.json'
                fresh.extend(read(file)['candidates'])
    fresh = verify_bank(fresh,meta,train)
    old = verify_bank(read(out/'old_bank.json')['candidates'],meta,train)
    records = old+fresh
    if len(set(a['candidate_id'] for a in records)) != len(records): raise ValueError('bank identity collision')
    e.p.write_json(out/'evaluation_bank.json',{'candidates':records,'frozen_paths_sha256':e.p.file_hash(out/'frozen_paths.json')})
    pd.DataFrame(fresh).to_csv(out/'fresh_attack_manifest.csv',index=False)
    pd.DataFrame(records).to_csv(out/'full_attack_bank_manifest.csv',index=False)
    print('expanded bank',len(old),'+',len(fresh),'=',len(records),flush=True)


def score(args):
    out,meta,train,current = e.context(args)
    verify_files(out,'frozen_paths.json')
    records = read(out/'evaluation_bank.json')['candidates']
    e.curve_evaluation(out,meta,train,current,records[args.worker::args.workers],args.split,args.device,f'shard{args.worker}')
    if args.split == 'train':
        chosen=records[args.worker::args.workers]
        bank=ModelBank([e.legacy.load_model(Path(a['model_path']),22) for a in chosen],args.device)
        hinges=[]
        for method in INTRINSIC:
            points=torch.load(out/f'learned_distribution_{method}.pt',weights_only=True)['points'].to(args.device)
            weights=torch.full((len(points),),1/len(points),dtype=torch.float64,device=args.device)
            hinges.append(bank.risks(points,weights=weights).cpu().numpy())
        np.savez_compressed(out/f'direct_Q_shard{args.worker}.npz',hinge=np.stack(hinges,axis=1),
                            candidate_ids=np.array([a['candidate_id'] for a in chosen]),methods=INTRINSIC)



def merge(args):
    merger.METHODS = CONTROLS+INTRINSIC
    # Merger verifies manifests through this experiment's frozen files.
    e.verify_eval_bank = lambda out,meta,train: read(out/'evaluation_bank.json')['candidates']
    merger.merge(args)
    data=np.load(args.output/'heloc'/f'bank_{args.split}_replay.npz')
    for index in range(1,len(data['methods'])):
        np.testing.assert_allclose(data['hinge'][:,0,[0,-1]],data['hinge'][:,index,[0,-1]],atol=1e-10,rtol=0)
    if args.split == 'train':
        out=args.output/'heloc';parts=[np.load(out/f'direct_Q_shard{i}.npz') for i in range(args.workers)]
        h=np.concatenate([p['hinge'] for p in parts]);ids=np.concatenate([p['candidate_ids'] for p in parts])
        old=set(a['candidate_id'] for a in read(out/'old_bank.json')['candidates'])
        mask=np.array([i in old for i in ids]);rows=[]
        for i,method in enumerate(INTRINSIC):
            rows.append({'method':method,'training_bank_hinge':float(h[mask,i].max()),'fresh_bank_hinge':float(h[:,i].max())})
        pd.DataFrame(rows).to_csv(out/'direct_Q_risk.csv',index=False)
        np.savez_compressed(out/'direct_Q_replay.npz',hinge=h,candidate_ids=ids,methods=INTRINSIC)



def select(args):
    out = args.output/'heloc'
    verify_files(out,'frozen_paths.json');verify_files(out,'frozen_paths_cal.json')
    frame = pd.read_csv(out/'bank_cal_curves.csv')
    selected = []
    for method, group in frame.groupby('method',sort=False):
        tied = group[group.bank_hinge == group.bank_hinge.min()].sort_values('lambda')
        row = tied.iloc[0].to_dict(); row['tied_lambdas'] = tied['lambda'].tolist()
        selected.append(row)
    e.p.write_json(out/'selection.json', {'selected':selected,'selection_split':'cal','deployment_calibrated':True,
                 'evaluation_bank_sha256':e.p.file_hash(out/'evaluation_bank.json'),
                 'calibration_paths_sha256':e.p.file_hash(out/'paths_cal.pt'),
                 'rule':'exact bank argmin, smallest lambda on exact ties; fixed radius3; beta not selected'})
    pd.DataFrame(selected).to_csv(out/'selected_configs.csv',index=False)


def future(args):
    out,meta,train,current = e.context(args)
    selection = read(out/'selection.json')
    if selection['evaluation_bank_sha256'] != e.p.file_hash(out/'evaluation_bank.json'): raise ValueError('selected bank changed')
    if selection['calibration_paths_sha256'] != e.p.file_hash(out/'paths_cal.pt'): raise ValueError('selection paths changed')
    collection = torch.load(out/'paths_test.pt',weights_only=True)
    historical = read(PARENT/'evaluation_manifest.json')
    manifest = pd.read_csv(Path(meta['original_instance'])/'future_final_manifest.csv')
    if e.p.file_hash(Path(meta['original_instance'])/'future_final_manifest.csv') != historical['future_manifest_sha256']:
        raise ValueError('future scenarios changed')
    rows=[]
    for protocol in ['aligned_full_update','scratch_stress']:
        files = [PARENT/'aligned_future_models'/f'model{i}.pt' for i in manifest.index] if protocol=='aligned_full_update' else [Path(p) for p in manifest.path]
        for file in files:
            if e.p.file_hash(file) != historical['model_hashes'][str(file)]: raise ValueError('historical future model changed')
        bank = ModelBank([e.legacy.load_model(file,22) for file in files],args.device)
        with torch.no_grad():
            for method,path in collection['methods'].items():
                w=path['weights'].to(args.device)
                for lam,q in zip(LAMBDAS,path['q']):
                    pred=torch.sigmoid(bank.logits(q.to(args.device)))
                    hinges=(torch.relu(.5-pred).double()*w).sum(1).cpu().numpy()
                    valid=((pred>=.5).double()*w).sum(1).cpu().numpy()
                    for i,(h,v) in enumerate(zip(hinges,valid)):
                        rows.append({'protocol':protocol,'method':method,'lambda':lam,'model_index':i,'hinge':h,'validity':v})
                print('future',protocol,method,flush=True)
    raw=pd.DataFrame(rows);raw.to_csv(out/'future_per_model.csv',index=False)
    summary=raw.groupby(['protocol','method','lambda'],sort=False).agg(mean_future_hinge=('hinge','mean'),worst_future_hinge=('hinge','max'),
                    mean_future_validity=('validity','mean'),worst_future_validity=('validity','min'),validity_sd=('validity','std')).reset_index()
    summary.to_csv(out/'future_curves.csv',index=False)
    selected=pd.DataFrame(selection['selected'])[['method','lambda','bank_hinge','before_fresh_hinge','tied_lambdas']].rename(columns={'bank_hinge':'calibration_bank_hinge','before_fresh_hinge':'calibration_before_fresh_hinge'})
    final=selected.merge(summary,on=['method','lambda']).merge(pd.read_csv(out/'bank_test_curves.csv')[['method','lambda','bank_hinge','before_fresh_hinge','recourse_cost','current_validity']],on=['method','lambda'])
    final.to_csv(out/'final_metrics.csv',index=False)
    e.p.write_json(out/'future_evaluation_manifest.json',{'historical_evaluation_manifest':str(PARENT/'evaluation_manifest.json'),
                  'sha256':e.p.file_hash(PARENT/'evaluation_manifest.json'),'selection_sha256':e.p.file_hash(out/'selection.json'),
                  'confirmatory':False,'mean_uncertainty':'SD across declared stresses; not an IID confidence interval'})


def report(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=args.output/'heloc';(out/'plots').mkdir(exist_ok=True)
    metrics=pd.read_csv(out/'final_metrics.csv'); geom=pd.read_csv(out/'geometry_feasibility.csv')
    curves=pd.read_csv(out/'bank_cal_curves.csv'); future_curves=pd.read_csv(out/'future_curves.csv')
    for name,frame,value in [('bank_hinge',curves,'bank_hinge'),('recourse_cost',pd.read_csv(out/'bank_test_curves.csv'),'recourse_cost'),
                             ('future_validity',future_curves[future_curves.protocol=='aligned_full_update'],'mean_future_validity')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for method,group in frame.groupby('method',sort=False):
            ax.plot(group['lambda'],group[value],label=method,linewidth=1)
        ax.set(xlabel='lambda',ylabel=value);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'plots'/f'{name}.png',dpi=150);plt.close(fig)
    validity=pd.concat([pd.read_csv(out/f'current_validity_{s}.csv') for s in ['train','cal','test']])
    validity.to_csv(out/'current_validity_checks.csv',index=False)
    pd.concat([pd.read_csv(out/f'bank_{s}_curves.csv') for s in ['train','cal','test']]).to_csv(out/'lambda_curves.csv',index=False)
    pd.concat([pd.read_csv(out/f'geometry_{s}.csv') for s in ['train','cal','test']]).to_csv(out/'geometry_by_split.csv',index=False)
    # Long-form replay is explicit, in addition to compressed matrices.
    chunks=[]
    for split in ['train','cal','test']:
        data=np.load(out/f'bank_{split}_replay.npz')
        for i,method in enumerate(data['methods']):
            for j,lam in enumerate(data['lambdas']):
                chunks.append(pd.DataFrame({'split':split,'candidate_id':data['candidate_ids'],'method':method,'lambda':lam,
                                            'hinge':data['hinge'][:,i,j],'invalidity':data['invalidity'][:,i,j]}))
    pd.concat(chunks).to_csv(out/'cross_eval_matrix.csv',index=False)
    text = '# Task 003: favorable-region direct WRO\n\n'
    text += 'Local graph-node optimization of Q itself; no current-validity or geometry penalty. All three OT terms use the same unregularized quadratic graph OT.\n\n'
    text += 'Graph validity uses float64 ReLU affine-piece enumeration, not interval certification. Query CFs use frozen single admissible leaf attachments; no query point changes training routes or Q. Budgets are checked separately per batch.\n\n'
    text += '## Learned distributions\n\n'+geom.to_markdown(index=False)+'\n\n'
    text += '## Matched selected outcomes\n\n'+metrics.to_markdown(index=False)+'\n\n'
    text += '## Limits\n\nFinite construction and fresh banks, discrete support and local optimization, numerical graph OT. Beta=1 equality checks the represented shortest-geodesic restriction; target-mass graph-node particles do not reproduce every continuous intermediate distribution. No unrestricted global solution or continuous safety certificate. All benchmark future scenarios were previously inspected; no untouched confirmation. Scratch retraining is separate stress, not the modeled full-update guarantee. Historical Euclidean controls can violate current validity. Mean validity differences require paired stress-level interpretation; scenario SD is not an IID confidence interval.\n'
    (out/'report.md').write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','smoke','cache','optimize','paths','discover','freeze_bank','score','merge','select','future','report'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--dataset',default='heloc')
    parser.add_argument('--device',default='cpu');parser.add_argument('--split',default='train')
    parser.add_argument('--worker',type=int,default=0);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    e.SETTINGS={**e.SETTINGS,'evaluation_radii':[3.],'lambdas':LAMBDAS}
    e.verify_geometry=lambda out:verify_files(out,'frozen_paths.json')
    globals()[args.phase](args)

if __name__=='__main__':main()

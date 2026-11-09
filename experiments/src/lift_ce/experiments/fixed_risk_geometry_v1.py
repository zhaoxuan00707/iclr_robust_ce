"""Isolated, fixed training-derived risk geometry experiment (no default change)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from ..counterfactuals import risk_geometry as geo
from ..counterfactuals.population_path import PopulationPath, empirical_ot_path, population_mean
from ..counterfactuals.robust_waypoints import ModelBank
from ..data.tabular import _load_raw_frame, _binarize_labels
from . import heloc_diabetes_bank_protocol_v3 as p
from . import heloc_diabetes_official_lift_v1 as legacy
from . import finite_waypoint_paths_v2 as finite
from .full_update_adversary_v1 import differentiable_full_update
from .heloc_conditional_adversary_strengthened_v1 import optimize, rms, search_loss
from .heloc_gradient_sensitive_adversary_v1 import fine_tune
from .diabetes_original_lift_comparison import shift

RESULTS = Path('/home/xu.zhao/robust_ce_results')
OUTPUT = RESULTS/'fixed_risk_weighted_ot_geometry_v1/20260914_fixed_v1'
ORIGINAL = RESULTS/'heloc_diabetes_wro_lambda_v2'
SEEDS = {'heloc': 1, 'diabetes': 0}
SETTINGS = {'neighbors': 20, 'quadrature': 9, 'geometry_radii': [.5, 1., 3.],
    'evaluation_radii': [.5, 1., 3.], 'lambdas': [round(i*.05, 2) for i in range(21)],
    'source_lambdas': [.25, .5, .75], 'replications': 3,
    'search': {'truncated_steps': 60, 'unroll_K': 5, 'full_refinement_steps': 10,
               'lr': .075, 'surrogate': 'softplus', 'kappa': 10., 'radial_fractions': [.25, .5, .75, 1.]},
    'weight': '2 - min_model target_probability', 'weight_coefficient_tuned': False,
    'domain': 'all features mutable continuous encoded space; f0>=.5 at 9 edge samples; no added box constraint',
    'query_rule': 'batch overlay: attach each query CF to 20 nearest frozen construction nodes, union undirected edges; no query-query edges; do not mutate base graph',
    'lambda_tie_rule': 'exact numerical minimum; report all exact ties, choose smallest lambda; no cost or tolerance',
    'confirmation': 'retrospective exploratory; existing final benchmark already inspected',
    'diabetes_scope': 'no frozen calibration CFs: fixed-population diagnostic only; auxiliary training CFs for attack construction, original CFs preserved as retrospective query batch'}


def save(obj, file):
    file.parent.mkdir(parents=True, exist_ok=True)
    legacy.atomic_torch_save(obj, file)


def context(args):
    out = args.output/args.dataset
    meta = json.loads((out/'instance_manifest.json').read_text())
    for name in ['training.pt', 'train_population.pt', 'current.pt']:
        if p.file_hash(out/name) != meta['files'][name]:
            raise ValueError(f'frozen instance changed: {name}')
    train = torch.load(out/'training.pt', weights_only=True)
    current = legacy.load_model(out/'current.pt', train['x'].shape[1])
    if meta['update_code_sha256'] != hashlib.sha256(inspect.getsource(fine_tune).encode()).hexdigest():
        raise ValueError('official Update changed')
    return out, meta, train, current


def audit_prepare(args):
    out = args.output/args.dataset
    out.mkdir(parents=True, exist_ok=True)
    if (out/'instance_manifest.json').exists():
        context(args)
        return
    ds, seed = args.dataset, SEEDS[args.dataset]
    sd, d, common, current = legacy.load_common(ORIGINAL, ds, seed, '/home/xu.zhao/robust_ce_data/openml')
    raw, labels = _load_raw_frame(legacy.DATASETS[ds]['config'], '/home/xu.zhao/robust_ce_data/openml')
    if len(raw.select_dtypes(include=['object', 'category', 'bool']).columns):
        raise ValueError('categorical segment constraints require an explicit supported adapter')
    numeric = raw.mask(raw.isin(legacy.DATASETS[ds]['config'].get('missing_values', [])), np.nan)
    imputer = SimpleImputer(strategy='median').fit(numeric.iloc[d['train_ids']])
    imputed = imputer.transform(numeric)
    scaler = StandardScaler().fit(imputed[d['train_ids']])
    scaled = torch.tensor(scaler.transform(imputed), dtype=torch.float32)
    for split in ['train', 'cal', 'test']:
        if not torch.equal(scaled[d[f'{split}_ids']], d[f'{split}_x']):
            raise ValueError('reconstructed frozen preprocessing differs')
    old_cf = torch.load(sd/'lift_original.pt', weights_only=True)['counterfactuals']
    old_path = torch.load(sd/'lift_lift_path.pt', weights_only=True)
    if not torch.equal(common['source'], d['test_x'][common['test_indices']]):
        raise ValueError('original factual identities changed')
    with torch.no_grad():
        target_mask = torch.sigmoid(current(d['train_x'])) >= .5
    target = d['train_x'][target_mask]
    train = {'x': d['train_x'], 'y': d['train_y'], 'ids': torch.as_tensor(d['train_ids']),
             'target': target, 'target_ids': torch.as_tensor(d['train_ids'])[target_mask]}
    save(train, out/'training.pt'); save(current.state_dict(), out/'current.pt')
    snapshot = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v for k, v in d.items()}
    snapshot.update({'original_source': common['source'], 'original_source_ids': common['source_ids'],
        'original_base': old_cf, 'original_barycentric_destination': old_path['destination'],
        'all_raw_ids': torch.arange(len(raw)), 'training_inclusion_mask': torch.tensor(np.isin(np.arange(len(raw)), d['train_ids'])),
        'calibration_inclusion_mask': torch.tensor(np.isin(np.arange(len(raw)), d['cal_ids'])),
        'test_inclusion_mask': torch.tensor(np.isin(np.arange(len(raw)), d['test_ids'])),
        'imputation_medians': torch.tensor(imputer.statistics_), 'scaler_mean': torch.tensor(scaler.mean_),
        'scaler_scale': torch.tensor(scaler.scale_), 'raw_missing_mask': torch.tensor(numeric.isna().to_numpy())})
    save(snapshot, out/'data_audit.pt')
    v3 = RESULTS/'heloc_diabetes_bank_protocol_v3'/ds/f'seed{seed}'
    if ds == 'heloc':
        frozen = torch.load(v3/'train.pt', weights_only=True)
        if not (torch.equal(frozen['x'], train['x']) and torch.equal(frozen['y'], train['y'])
                and torch.equal(frozen['target'], target)):
            raise ValueError('v3 training instance incompatible')
        for split in ['train', 'cal', 'test']:
            pop = torch.load(v3/f'{split}_population.pt', weights_only=True)
            if split != 'train' and p.file_hash(v3/f'{split}_population.pt') != (v3/f'{split}_population.sha256').read_text().strip():
                raise ValueError('heldout CF artifact hash mismatch')
            save(pop, out/f'{split}_population.pt')
        train_population_origin = str(v3/'train_population.pt')
    else:
        # Exactly one deterministic training-only extension using the already
        # documented frozen G. Original 50 CFs/factuals remain unchanged.
        population = p.make_population(current, train['x'], train['ids'], target, ds, seed+2, 128)
        save(population, out/'train_population.pt')
        save({'source': common['source'], 'source_ids': common['source_ids'],
              'path': empirical_ot_path(old_cf, target).state_dict()}, out/'diagnostic_population.pt')
        train_population_origin = 'one deterministic auxiliary training-only application of existing G; not regenerated original CFs'
    excluded = pd.read_csv(RESULTS/'fixed_instance_attack_bank_audit_v1/candidate_manifest.csv')
    excluded = excluded[excluded.dataset == ds].copy()
    excluded['exclusion_reason'] = 'discovery targeted original test CF path; per-attack update seeds also differ from fixed update seed'
    excluded.to_csv(out/'excluded_legacy_candidates.csv', index=False)
    baseline_audit = []
    for method in ['trex', 'roar', 'proplace', 'ellice']:
        f = sd/f'{method}_original.pt'
        if f.exists():
            artifact = torch.load(f, weights_only=True)
            baseline_audit.append({'method': method, 'artifact': str(f), 'sha256': p.file_hash(f),
                'source_count': len(artifact['counterfactuals']), 'paired_source_count': len(common['source']),
                'reuse': 'original outputs only; old aggregate metrics differ in CF representation and scratch protocol; no retuning'})
    p.write_json(out/'baseline_audit.json', baseline_audit)
    metadata = {'dataset': ds, 'seed': seed, 'original_instance': str(sd), 'v3_instance': str(v3) if v3.exists() else None,
        'b_ref': float(common['b_ref']), 'absolute_budgets': {str(r): float(r*common['b_ref']) for r in SETTINGS['evaluation_radii']},
        'threshold': .5, 'full_update_seed': 700000+seed, 'update_code_sha256': hashlib.sha256(inspect.getsource(fine_tune).encode()).hexdigest(),
        'update': {'optimizer': 'AdamW', 'initialization': 'deepcopy frozen f0; fresh zero moments each update',
                   'lr': .0002, 'weight_decay': .0001, 'epochs': 8, 'batch_size': 256,
                   'batch_order': 'randperm with seed full_update_seed+epoch', 'labels': 'unchanged'},
        'feature_names': d['feature_names'], 'categorical_columns': [],
        'labels': legacy.DATASETS[ds]['favorable'], 'raw_label_counts': labels.astype(str).value_counts().to_dict(),
        'generator': 'Wachter; HELOC 300 steps lr .03 cost .05 penalty20; Diabetes solve_wachter 5 restarts 500 steps, no immutable constraints',
        'train_population_origin': train_population_origin, 'calibration_CF_available': ds == 'heloc',
        'validation_population': 'calibration split used historically for base f0 early stopping; no independent untouched validation split',
        'original_representation': 'barycentric projection of transport; Q(1) is barycentric destination, not empirical P1',
        'new_and_matched_straight_representation': 'weighted sparse balanced coupling; one atom per positive transport mass',
        'uncertainty': 'unchanged fixed-label sample displacements with RMS <= r*b_ref; not changed by CF metric',
        'actionability': 'historical all-features-mutable continuous relaxation retained; Diabetes age/preg immutability not enforced historically; report violations, no exact actionability claim',
        'original_files': {str(f): p.file_hash(f) for f in [sd/'common.pt', sd/'models/current.pt', sd/'lift_original.pt', sd/'lift_lift_path.pt', sd/'metadata.json']},
        'settings': SETTINGS, 'files': {f.name: p.file_hash(f) for f in out.glob('*.pt')}}
    metadata['instance_id'] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    p.write_json(out/'resolved_config.json', SETTINGS)
    p.write_json(out/'instance_manifest.json', metadata)
    print(ds, 'frozen', len(train['x']), len(target), flush=True)


def verified_records(records, meta, train):
    """Verify actual deltas and declared full-update provenance, not names."""
    checked = []
    cache = {}
    for a in records:
        if a['split'] != 'train' or a['full_update_seed'] != meta['full_update_seed']:
            raise ValueError('non-training or incompatible update candidate')
        if p.file_hash(Path(a['model_path'])) != a['model_sha256']:
            raise ValueError('full model changed')
        file = Path(a['delta_file'])
        if str(file) not in cache:
            cache[str(file)] = torch.load(file, weights_only=False)
        saved = cache[str(file)]
        actual_hash = p.file_hash(file)
        if a.get('delta_file_sha256', actual_hash) != actual_hash:
            raise ValueError('frozen candidate perturbation changed')
        if saved.get('discovery_split') != 'train' or saved.get('full_update_seed') != meta['full_update_seed']:
            raise ValueError('perturbation provenance contradicts candidate manifest')
        if a.get('delta_key') == 'delta':
            delta = saved['delta']
        else:
            candidate = saved['candidates'][a['delta_index']]
            delta = candidate['delta'] if isinstance(candidate, dict) else candidate
        norm = float(rms(delta))
        if (delta.shape != train['x'].shape or not torch.isfinite(delta).all()
                or norm > a['nominal_radius']*meta['b_ref']+1e-5):
            raise ValueError('infeasible candidate displacement')
        checked.append({**a, 'measured_budget': norm, 'delta_file_sha256': p.file_hash(file)})
    return checked


def import_heloc_bank(args):
    out, meta, train, _ = context(args)
    v3 = Path(meta['v3_instance'])
    m3 = json.loads((v3/'instance.json').read_text())
    finite_out = v3/'finite_waypoint_paths_v2'
    bank = finite.verified_bank(finite_out)
    old = []
    for a in bank['attacks']:
        name, index = a['attack_id'].rsplit(':', 1)
        if name.startswith('finite_v2/'):
            file = finite_out/'attacks'/name.split('/', 1)[1]
        elif a.get('origin') == 'waypoint_training_discovery':
            file = v3/'robust_waypoint_path_v1/attacks'/name
        else:
            file = v3/'attacks'/name
        saved = torch.load(file, weights_only=False)
        if (saved.get('instance_id', saved.get('parent_instance_id')) != m3['instance_id']
                or saved.get('discovery_split') != 'train' or saved['full_update_seed'] != meta['full_update_seed']):
            raise ValueError('legacy candidate provenance mismatch')
        old.append({**a, 'candidate_id': a['attack_id'], 'nominal_radius': a['radius_ratio'],
            'delta_file': str(file), 'delta_index': int(index), 'split': 'train',
            'full_update_seed': meta['full_update_seed'], 'phase': 'old'})
    old = verified_records(old, meta, train)
    _, geom = p.verify_bank(v3, m3, 'bank_full_update_v1')
    ids = {a['attack_id'] for a in geom['attacks']}
    geom_records = [a for a in old if a['candidate_id'] in ids]
    assert len(geom_records) == len(ids)
    p.write_json(out/'old_bank.json', {'instance_id': meta['instance_id'], 'candidates': old})
    p.write_json(out/'geometry_bank.json', {'instance_id': meta['instance_id'],
        'policy': 'all 851 bank_full_update_v1 full-update candidates at construction radii .5,1,3',
        'candidates': geom_records, 'frozen_before_geometry': True})


def run_search(current, train, q, weights, radius, search_seed, update_seed, device):
    model = copy.deepcopy(current).to(device).train()
    x, y = train['x'].to(device), train['y'].to(device)
    q, weights = q.to(device), weights.to(device)
    settings = SETTINGS['search']
    delta, trace, runtime, peak = optimize(model, x, y, q, radius, search_seed,
        settings['surrogate'], settings['kappa'], settings['unroll_K'], settings['lr'],
        settings['truncated_steps'], weights)
    delta = delta.to(device).requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=settings['lr'])
    candidates, full_trace = [], []
    for step in range(settings['full_refinement_steps']+1):
        pred = differentiable_full_update(model, x, y, delta, q, update_seed)
        gradient = torch.autograd.grad(search_loss(pred, 'softplus', 10., weights), delta)[0]
        if not torch.isfinite(gradient).all() or not torch.isfinite(pred).all():
            raise ValueError('nonfinite full-update search')
        candidates.append(delta.detach().cpu().clone())
        full_trace.append({'iteration': step, 'true_hinge': float(population_mean(torch.relu(.5-pred), weights)),
                           'gradient_norm': float(gradient.norm()), 'finite': True})
        if step == settings['full_refinement_steps']:
            break
        opt.zero_grad(); delta.grad = -gradient; opt.step()
        with torch.no_grad():
            norm = rms(delta)
            if norm > radius:
                delta.mul_(radius/norm)
    unit = candidates[-1]/rms(candidates[-1]).clamp_min(1e-12)
    tags = [f'full_iter{step}' for step in range(len(candidates))]
    for fraction in settings['radial_fractions']:
        candidates.append(unit*(radius*fraction)); tags.append(f'radial_{fraction:g}')
    return candidates, tags, trace, pd.DataFrame(full_trace), runtime, peak


def replay_candidates(out, meta, train, current, file, obj, device):
    records = []
    for i, delta in enumerate(obj['candidates']):
        if float(rms(delta)) > obj['radius']*meta['b_ref']+1e-5:
            raise ValueError('projected candidate outside declared data budget')
        model = fine_tune(current, train['x']+delta, train['y'], meta['full_update_seed'], device)
        target = out/'models'/f'{file.stem}_candidate{i}.pt'
        save(model.state_dict(), target)
        records.append({'candidate_id': f'{file.stem}:{i}', 'model_path': str(target),
            'model_sha256': p.file_hash(target), 'nominal_radius': obj['radius'], 'delta_file': str(file),
            'delta_index': i, 'split': 'train', 'full_update_seed': meta['full_update_seed'],
            'search_seed': obj['search_seed'], 'replication': obj['replication'], 'phase': obj['phase'],
            'source_method': obj.get('method', 'geometry_bootstrap'), 'source_lambda': obj.get('lambda'),
            'candidate_type': obj['tags'][i], 'measured_budget': float(rms(delta))})
    p.write_json(file.with_suffix('.models.json'), {'discovery_sha256': p.file_hash(file), 'candidates': records})


def bootstrap(args):
    out, meta, train, current = context(args)
    if args.dataset == 'heloc':
        import_heloc_bank(args)
        return
    pop = torch.load(out/'train_population.pt', weights_only=True)
    # Only training-derived CFs and the unchanged training target anchor enter
    # bootstrap. No old test-targeted attack is admitted.
    q = torch.cat([pop['path']['base'], train['target']])
    weights = torch.cat([torch.full((len(pop['source']),), .5/len(pop['source'])),
                         torch.full((len(train['target']),), .5/len(train['target']))])
    jobs = [(r, rep) for r in SETTINGS['geometry_radii'] for rep in range(3)]
    for job, (r, rep) in enumerate(jobs):
        if job % args.workers != args.worker:
            continue
        file = out/'bootstrap'/f'geom_r{r:g}_rep{rep}.pt'
        if file.with_suffix('.models.json').exists():
            continue
        seed = 91000000+int(r*1000)+rep
        candidates, tags, trace, full, runtime, peak = run_search(current, train, q, weights, r*meta['b_ref'], seed, meta['full_update_seed'], args.device)
        obj = {'instance_id': meta['instance_id'], 'phase': 'geometry', 'radius': r, 'replication': rep,
               'search_seed': seed, 'candidates': candidates, 'tags': tags, 'discovery_split': 'train',
               'full_update_seed': meta['full_update_seed']}
        save(obj, file); trace.to_csv(file.with_suffix('.truncated.csv'), index=False); full.to_csv(file.with_suffix('.full.csv'), index=False)
        replay_candidates(out, meta, train, current, file, obj, args.device)
        print('bootstrap complete', r, rep, flush=True)


def freeze_geometry_bank(args):
    out, meta, train, _ = context(args)
    if (out/'geometry_bank.json').exists():
        return
    records = []
    for r in SETTINGS['geometry_radii']:
        for rep in range(3):
            file = out/'bootstrap'/f'geom_r{r:g}_rep{rep}.pt'
            side = json.loads(file.with_suffix('.models.json').read_text())
            if p.file_hash(file) != side['discovery_sha256']:
                raise ValueError('bootstrap perturbations changed')
            records.extend(side['candidates'])
    records = verified_records(records, meta, train)
    p.write_json(out/'old_bank.json', {'instance_id': meta['instance_id'], 'candidates': records})
    p.write_json(out/'geometry_bank.json', {'instance_id': meta['instance_id'], 'candidates': records,
        'policy': 'three training-only replications at each construction radius .5,1,3; full updates including all radial candidates',
        'frozen_before_geometry': True})


def score_geometry(out, train, points, cache_name, device):
    """Evaluate the entire frozen bank; cache all scores in a disk-backed array."""
    bank_file = out/'geometry_bank.json'
    bank = json.loads(bank_file.read_text())
    records = bank['candidates']
    models = [legacy.load_model(Path(a['model_path']), train['x'].shape[1]) for a in records]
    ensemble = ModelBank(models, device)
    shape = (len(points), len(records))
    filename = out/'scores'/f'{cache_name}.npy'
    filename.parent.mkdir(exist_ok=True)
    score_cache = np.lib.format.open_memmap(filename, mode='w+', dtype=np.float32, shape=shape)
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(points), 1024):
            q = torch.as_tensor(points[start:start+1024], dtype=torch.float32, device=device)
            block = np.empty((len(q), len(records)), dtype=np.float32)
            for first in range(0, len(records), 32):
                pred = torch.sigmoid(ensemble.logits(q, first, first+32)).cpu().numpy().T
                block[:, first:first+pred.shape[1]] = pred
            score_cache[start:start+len(q)] = block
    score_cache.flush()
    minimum = np.empty(len(points), dtype=np.float32)
    winner = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), 8192):
        block = np.asarray(score_cache[start:start+8192])
        minimum[start:start+len(block)] = block.min(1)
        winner[start:start+len(block)] = block.argmin(1)
    weights = 2-minimum.astype(np.float64)
    if not np.isfinite(weights).all() or (weights < 1-1e-7).any() or (weights > 2+1e-7).any():
        raise ValueError('risk weights outside [1,2]')
    count = np.bincount(winner, minlength=len(records))
    stats = {'points': len(points), 'models': len(records), 'score_cache_bytes': int(np.prod(shape)*4),
        'elapsed_seconds': time.perf_counter()-started, 'min': float(weights.min()), 'max': float(weights.max()),
        'mean': float(weights.mean()), 'std': float(weights.std()),
        'quantiles': np.quantile(weights, [0, .01, .1, .5, .9, .99, 1]).tolist(),
        'dominant_model_fraction': float(count.max()/len(points)),
        'dominant_model_id': records[int(count.argmax())]['candidate_id'],
        'geometry_bank_sha256': p.file_hash(bank_file), 'pointwise_minimum_role': 'spatial cost only; never population WRO reduction'}
    p.write_json(filename.with_suffix('.json'), stats)
    pd.DataFrame({'candidate_id': [a['candidate_id'] for a in records], 'minimum_count': count,
                  'minimum_fraction': count/len(points)}).to_csv(filename.with_suffix('.dominance.csv'), index=False)
    return weights, winner, stats


def check_edges(current, nodes, edges):
    samples = geo.edge_samples(nodes, edges, SETTINGS['quadrature'])
    flat = samples.reshape(-1, nodes.shape[1])
    values = []
    with torch.no_grad():
        for start in range(0, len(flat), 4096):
            values.append(torch.sigmoid(current(torch.tensor(flat[start:start+4096], dtype=torch.float32))).numpy())
    scores = np.concatenate(values).reshape(len(edges), -1)
    allowed = (scores >= .5).all(1)
    return edges[allowed], samples[allowed], {'candidate_edges': len(edges), 'accepted_edges': int(allowed.sum()),
        'rejected_edges': int((~allowed).sum()), 'interior_rejections_despite_valid_endpoints': int(((scores[:, [0, -1]] >= .5).all(1) & ~allowed).sum()),
        'minimum_accepted_sample_f0': float(scores[allowed].min()) if allowed.any() else None,
        'numerical_only': True, 'continuous_actionability_certificate': False}


def graph_construct(args):
    out, meta, train, current = context(args)
    if (out/'frozen_geometry.json').exists():
        verify_geometry(out)
        return
    bank = json.loads((out/'geometry_bank.json').read_text())
    verified_records(bank['candidates'], meta, train)
    pop = torch.load(out/'train_population.pt', weights_only=True)
    base, target = pop['path']['base'], train['target']
    nodes, inverse = np.unique(torch.cat([base, target]).numpy(), axis=0, return_inverse=True)
    edges = geo.knn_edges(nodes, SETTINGS['neighbors'])
    retained, samples, admissibility = check_edges(current, nodes, edges)
    estimate = {'nodes': len(nodes), 'candidate_edges': len(edges), 'accepted_edges': len(retained),
        'quadrature_points': len(retained)*SETTINGS['quadrature'], 'models': len(bank['candidates']),
        'full_scores_disk_bytes': len(retained)*SETTINGS['quadrature']*len(bank['candidates'])*4,
        'point_batch': 1024, 'model_batch': 32, 'same_resolution_for_controls': True}
    p.write_json(out/'compute_estimate.json', estimate)
    print(meta['dataset'], 'graph score estimate', estimate, flush=True)
    values, winners, stats = score_geometry(out, train, samples.reshape(-1, nodes.shape[1]), 'train_edges', args.device)
    values = values.reshape(len(retained), SETTINGS['quadrature'])
    graph = {'nodes': torch.from_numpy(nodes), 'edges': torch.from_numpy(retained), 'values': torch.from_numpy(values),
        'source_nodes': torch.from_numpy(inverse[:len(base)]), 'target_nodes': torch.from_numpy(inverse[len(base):]),
        'unweighted_costs': torch.from_numpy(geo.edge_lengths(nodes, retained, np.ones_like(values))),
        'weighted_costs': torch.from_numpy(geo.edge_lengths(nodes, retained, values)),
        'minimum_model_indices': torch.from_numpy(winners.reshape(values.shape))}
    save(graph, out/'graph.pt')
    p.write_json(out/'graph_audit.json', {**admissibility, 'risk_weight': stats,
        'topology': 'union-symmetrized 20-NN; duplicate nodes coalesced with endpoint maps',
        'constraints': meta['actionability']})
    freeze_split_paths(out, meta, train, pop, graph, 'train')
    p.write_json(out/'frozen_geometry.json', {'instance_id': meta['instance_id'],
        'geometry_bank_sha256': p.file_hash(out/'geometry_bank.json'), 'graph_sha256': p.file_hash(out/'graph.pt'),
        'train_paths_sha256': p.file_hash(out/'paths_train.pt'), 'settings_sha256': p.file_hash(out/'resolved_config.json')})


def freeze_split_paths(out, meta, train, pop, graph, split):
    nodes, edges, values = (graph[k].numpy() for k in ['nodes', 'edges', 'values'])
    sources, targets = graph['source_nodes'].numpy(), graph['target_nodes'].numpy()
    paths, audits, distance_matrices = {}, {}, {}
    straight = PopulationPath(**pop['path'])
    paths['A_straight'] = {'q': torch.stack([straight.at(l) for l in SETTINGS['lambdas']]),
        'weights': straight.weights.double(), 'source_index': straight.source_index, 'target_index': straight.target_index}
    for name, weights in [('B_unweighted', np.ones_like(values)), ('C_risk_weighted', values)]:
        try:
            route, audit, distance = geo.route_population(nodes, edges, weights, sources, targets,
                                                          pop['path']['base'], train['target'])
        except ValueError as error:
            p.write_json(out/f'geometry_infeasibility_{split}.json', {'method': name, 'reason': str(error), 'mass_not_dropped': True})
            raise
        q = torch.stack([route.at(l) for l in SETTINGS['lambdas']])
        paths[name] = {'q': q, 'weights': route.weights, 'source_index': route.source_index,
            'target_index': route.target_index, 'routes': route.state_dict(), 'speed_bound': route.speed_bound}
        audits[name] = audit; distance_matrices[name] = distance
        if not torch.equal(q[0], pop['path']['base'][route.source_index]) or not torch.equal(q[-1], train['target'][route.target_index]):
            raise ValueError('represented endpoint mismatch')
    if meta['dataset'] == 'heloc':
        wpfile = Path(meta['v3_instance'])/'robust_waypoint_path_v1/frozen_path.pt'
        fit = torch.load(wpfile, weights_only=True)
        if fit['split'] != 'train' or fit['parent_instance_id'] != json.loads((Path(meta['v3_instance'])/'instance.json').read_text())['instance_id']:
            raise ValueError('waypoint control provenance mismatch')
        from ..counterfactuals.population_path import WaypointPopulationPath
        wp = WaypointPopulationPath(straight, fit['waypoints'])
        paths['D_waypoint'] = {'q': torch.stack([wp.at(l) for l in SETTINGS['lambdas']]), 'weights': straight.weights.double(),
                               'source_index': straight.source_index, 'target_index': straight.target_index,
                               'control_sha256': p.file_hash(wpfile)}
    d0, dw = distance_matrices['B_unweighted'], distance_matrices['C_risk_weighted']
    finite_pairs = np.isfinite(d0)
    if not np.array_equal(finite_pairs, np.isfinite(dw)) or not (dw[finite_pairs] >= d0[finite_pairs]-1e-7).all() or not (dw[finite_pairs] <= 2*d0[finite_pairs]+1e-7).all():
        raise ValueError('bounded graph metric comparison failed')
    b, c = paths['B_unweighted'], paths['C_risk_weighted']
    def plan_map(v):
        return {(int(i), int(j)): float(w) for i, j, w in zip(v['source_index'], v['target_index'], v['weights'])}
    mb, mc = plan_map(b), plan_map(c)
    common = set(mb)&set(mc)
    rb = {key: route for key, route in zip(mb, audits['B_unweighted']['route_nodes'])}
    rc = {key: route for key, route in zip(mc, audits['C_risk_weighted']['route_nodes'])}
    overlap_mass = sum(min(mb[k], mc[k]) for k in common)
    diag = {'coupling_L1_change': sum(abs(mb.get(k, 0)-mc.get(k, 0)) for k in set(mb)|set(mc)),
        'shared_pair_mass': overlap_mass, 'changed_route_shared_pairs': sum(rb[k] != rc[k] for k in common),
        'shared_pair_count': len(common), 'weighted_over_unweighted_distance_quantiles': np.quantile(dw[finite_pairs]/np.maximum(d0[finite_pairs], 1e-20), [0, .5, .9, 1]).tolist(),
        'changes_both_assignment_and_routes': bool(any(abs(mb.get(k, 0)-mc.get(k, 0)) > 1e-9 for k in set(mb)|set(mc)) and any(rb[k] != rc[k] for k in common)), 'split_population': split,
        'endpoint_population_equality': True, 'endpoint_cost_may_differ': 'factual-target assignment changes despite identical target distribution',
        'B': audits['B_unweighted'], 'C': audits['C_risk_weighted']}
    # Both Euclidean and weighted lengths of the actually selected routes.
    for name in ['B_unweighted', 'C_risk_weighted']:
        raw = paths[name]['routes']
        euclidean = [float(torch.linalg.vector_norm(torch.diff(pp, dim=0), dim=1).sum()) for pp in raw['route_points']]
        paths[name]['route_euclidean_lengths'] = torch.tensor(euclidean)
    save({'methods': paths, 'source': pop['source'], 'source_ids': pop['source_ids'],
          'lambdas': SETTINGS['lambdas'], 'split': split}, out/f'paths_{split}.pt')
    p.write_json(out/f'path_audit_{split}.json', diag)
    np.savez_compressed(out/f'distances_{split}.npz', unweighted=d0, risk_weighted=dw)
    print(meta['dataset'], split, 'frozen paths', {k: len(v['weights']) for k, v in paths.items()}, flush=True)


def verify_geometry(out):
    manifest = json.loads((out/'frozen_geometry.json').read_text())
    for file, key in [('geometry_bank.json', 'geometry_bank_sha256'), ('graph.pt', 'graph_sha256'),
                      ('paths_train.pt', 'train_paths_sha256'), ('resolved_config.json', 'settings_sha256')]:
        if p.file_hash(out/file) != manifest[key]:
            raise ValueError('frozen geometry changed after construction')
    return manifest


def query_paths(args, split):
    out, meta, train, current = context(args)
    frozen = verify_geometry(out)
    if (out/f'paths_{split}.pt').exists():
        return
    pop = torch.load(out/f'{split}_population.pt', weights_only=True)
    graph = torch.load(out/'graph.pt', weights_only=True)
    base_nodes = graph['nodes'].numpy()
    nodes = list(base_nodes)
    lookup = {tuple(x): i for i, x in enumerate(base_nodes)}
    source_nodes = []
    for q in pop['path']['base'].numpy():
        if tuple(q) not in lookup:
            lookup[tuple(q)] = len(nodes); nodes.append(q)
        source_nodes.append(lookup[tuple(q)])
    nodes = np.asarray(nodes)
    new_ids = sorted(set(source_nodes)-set(range(len(base_nodes))))
    edges = set()
    if new_ids:
        neighbor = NearestNeighbors(n_neighbors=min(SETTINGS['neighbors'], len(base_nodes)), algorithm='brute').fit(base_nodes)
        attachments = neighbor.kneighbors(nodes[new_ids], return_distance=False)
        edges = {(min(i, int(j)), max(i, int(j))) for i, row in zip(new_ids, attachments) for j in row}
    edges = np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)
    retained, samples, audit = check_edges(current, nodes, edges)
    values, winners, stats = score_geometry(out, train, samples.reshape(-1, nodes.shape[1]), f'{split}_query_edges', args.device)
    overlay = {'nodes': torch.from_numpy(nodes), 'edges': torch.cat([graph['edges'], torch.from_numpy(retained)]),
        'values': torch.cat([graph['values'], torch.from_numpy(values.reshape(len(retained), SETTINGS['quadrature']))]),
        'source_nodes': torch.tensor(source_nodes), 'target_nodes': graph['target_nodes']}
    save(overlay, out/f'overlay_{split}.pt')
    p.write_json(out/f'query_audit_{split}.json', {**audit, 'risk_weight': stats, 'base_geometry': frozen,
                                                'query_rule': SETTINGS['query_rule'], 'training_graph_mutated': False})
    freeze_split_paths(out, meta, train, pop, overlay, split)
    verify_geometry(out)


def discover(args):
    out, meta, train, current = context(args)
    frozen = verify_geometry(out)
    collection = torch.load(out/'paths_train.pt', weights_only=True)
    jobs = [(method, r, lam, rep) for method in collection['methods']
            for r in SETTINGS['evaluation_radii'] for lam in SETTINGS['source_lambdas'] for rep in range(3)]
    for job, (method, r, lam, rep) in enumerate(jobs):
        if job % args.workers != args.worker:
            continue
        file = out/'fresh_attacks'/f'{method}_r{r:g}_l{lam:.2f}_rep{rep}.pt'
        if file.with_suffix('.models.json').exists():
            saved = torch.load(file, weights_only=False)
            if saved['frozen_geometry'] != frozen:
                raise ValueError('new attacks cannot update geometry')
            continue
        path = collection['methods'][method]
        q = path['q'][SETTINGS['lambdas'].index(lam)]
        seed = 93000000+SEEDS[args.dataset]*1000000+list(collection['methods']).index(method)*100000+int(r*10000)+int(lam*100)*10+rep
        started = time.perf_counter()
        candidates, tags, trace, full, runtime, peak = run_search(current, train, q, path['weights'], r*meta['b_ref'], seed, meta['full_update_seed'], args.device)
        obj = {'instance_id': meta['instance_id'], 'phase': 'fresh', 'method': method, 'lambda': lam,
            'radius': r, 'replication': rep, 'search_seed': seed, 'full_update_seed': meta['full_update_seed'],
            'candidates': candidates, 'tags': tags, 'discovery_split': 'train', 'frozen_geometry': frozen}
        save(obj, file); trace.to_csv(file.with_suffix('.truncated.csv'), index=False); full.to_csv(file.with_suffix('.full.csv'), index=False)
        replay_candidates(out, meta, train, current, file, obj, args.device)
        p.write_json(file.with_suffix('.runtime.json'), {'total_seconds': time.perf_counter()-started,
            'truncated_seconds': runtime, 'peak_cuda_bytes_truncated': peak, 'retained_models': len(candidates)})
        verify_geometry(out)
        print(args.dataset, 'fresh completed', method, r, lam, rep, flush=True)


def freeze_eval_bank(args):
    out, meta, train, _ = context(args)
    frozen = verify_geometry(out)
    records = json.loads((out/'old_bank.json').read_text())['candidates']
    methods = torch.load(out/'paths_train.pt', weights_only=True)['methods']
    for method in methods:
        for r in SETTINGS['evaluation_radii']:
            for lam in SETTINGS['source_lambdas']:
                for rep in range(3):
                    file = out/'fresh_attacks'/f'{method}_r{r:g}_l{lam:.2f}_rep{rep}.pt'
                    obj = torch.load(file, weights_only=False)
                    if (obj['instance_id'] != meta['instance_id'] or obj['discovery_split'] != 'train'
                            or obj['frozen_geometry'] != frozen or obj['full_update_seed'] != meta['full_update_seed']):
                        raise ValueError('fresh candidate provenance mismatch')
                    side = json.loads(file.with_suffix('.models.json').read_text())
                    if p.file_hash(file) != side['discovery_sha256']:
                        raise ValueError('radial or search candidate artifact changed')
                    records.extend(side['candidates'])
    records = verified_records(records, meta, train)
    if len({a['candidate_id'] for a in records}) != len(records):
        raise ValueError('duplicate evaluation candidate identifiers')
    p.write_json(out/'evaluation_bank.json', {'instance_id': meta['instance_id'], 'frozen_geometry': frozen,
        'candidates': records, 'compatibility_rule': 'measured RMS displacement <= actual budget + numerical feasibility tolerance',
        'selection_outcomes_used': False})
    print(args.dataset, 'evaluation bank frozen', len(records), flush=True)


def verify_eval_bank(out, meta, train):
    bank = json.loads((out/'evaluation_bank.json').read_text())
    if bank['instance_id'] != meta['instance_id'] or bank['frozen_geometry'] != verify_geometry(out):
        raise ValueError('evaluation bank/geometry mismatch')
    return verified_records(bank['candidates'], meta, train)


def curve_evaluation(out, meta, train, current, records, split, device, prefix):
    collection = torch.load(out/f'paths_{split}.pt', weights_only=True)
    methods = list(collection['methods'])
    models = [legacy.load_model(Path(a['model_path']), train['x'].shape[1]) for a in records]
    bank = ModelBank(models, device)
    shape = (len(records), len(methods), len(SETTINGS['lambdas']))
    hinges, invalidities = np.zeros(shape), np.zeros(shape)
    summary = []
    pop_file = out/f'{split}_population.pt'
    population = torch.load(pop_file, weights_only=True)
    endpoint_cache = {}
    with torch.no_grad():
        for mi, method in enumerate(methods):
            path = collection['methods'][method]
            for li, lam in enumerate(SETTINGS['lambdas']):
                output = path['q'][li]
                weights = path['weights'].to(device)
                if lam in (0., 1.):
                    q = population['path']['base'] if lam == 0 else train['target']
                    score_weights = torch.full((len(q),), 1/len(q), dtype=torch.float64, device=device)
                else:
                    q, score_weights = output, weights
                q = q.to(device)
                if lam in endpoint_cache:
                    h, inv = endpoint_cache[lam]
                else:
                    h, inv = [], []
                    for start in range(0, len(records), 32):
                        pred = torch.sigmoid(bank.logits(q, start, start+32))
                        h.extend((torch.relu(.5-pred).double()*score_weights).sum(1).cpu().tolist())
                        inv.extend(((pred < .5).double()*score_weights).sum(1).cpu().tolist())
                    h, inv = np.asarray(h), np.asarray(inv)
                    if lam in (0., 1.):
                        endpoint_cache[lam] = (h, inv)
                hinges[:, mi, li], invalidities[:, mi, li] = h, inv
                source = collection['source'][path['source_index']]
                cost = float((torch.linalg.vector_norm(output-source, dim=1).double()*path['weights']).sum())
                cur = float(population_mean((torch.sigmoid(current(output)) >= .5).double(), path['weights']))
                for r in SETTINGS['evaluation_radii']:
                    eligible = np.array([a['measured_budget'] <= r*meta['b_ref']+max(1e-7, r*meta['b_ref']*1e-6) for a in records])
                    old = eligible & np.array([a['phase'] != 'fresh' for a in records])
                    if not eligible.any() or not old.any():
                        raise ValueError('empty compatible bank at an evaluation radius')
                    indices = np.flatnonzero(eligible)
                    win = indices[np.argmax(h[eligible])]
                    row = {'dataset': meta['dataset'], 'split': split, 'method': method, 'r_over_b_ref': r,
                        'lambda': lam, 'bank_hinge': float(h[eligible].max()), 'before_fresh_hinge': float(h[old].max()),
                        'bank_invalidity': float(inv[eligible].max()), 'hinge_witness_validity': float(1-inv[win]),
                        'winning_attack_id': records[win]['candidate_id'], 'bank_size': int(eligible.sum()),
                        'current_validity': cur, 'recourse_cost': cost}
                    for rep in range(3):
                        replica = old | (eligible & np.array([a['phase'] == 'fresh' and a['replication'] == rep for a in records]))
                        row[f'rep{rep}_hinge'] = float(h[replica].max())
                    summary.append(row)
            print(meta['dataset'], split, 'scored', method, flush=True)
    np.savez_compressed(out/f'{prefix}_{split}_replay.npz', hinge=hinges, invalidity=invalidities,
                        candidate_ids=np.asarray([a['candidate_id'] for a in records]), methods=np.asarray(methods), lambdas=SETTINGS['lambdas'])
    frame = pd.DataFrame(summary)
    frame['fresh_risk_increase'] = frame.bank_hinge-frame.before_fresh_hinge
    if (frame.fresh_risk_increase < -1e-10).any():
        raise ValueError('bank enlargement decreased a column maximum')
    frame.to_csv(out/f'{prefix}_{split}_curves.csv', index=False)
    return frame


def exact_lambda_choices(frame):
    choices = []
    for (method, r), rows in frame.groupby(['method', 'r_over_b_ref']):
        minimum = rows.bank_hinge.min()
        ties = rows[rows.bank_hinge == minimum].sort_values('lambda')
        choices.append({**ties.iloc[0].to_dict(), 'exact_ties': ties['lambda'].tolist(),
                        'tie_rule': 'smallest lambda among exact numerical minima'})
    return choices


def select(args, evaluated_frames=None):
    out, meta, train, current = context(args)
    records = verify_eval_bank(out, meta, train)
    # No calibration artifact is invented for Diabetes. Training minima below
    # are fixed-population diagnostics, not deployment-calibrated decisions.
    split = 'cal' if meta['calibration_CF_available'] else 'train'
    if split == 'cal':
        query_paths(args, split)
    if evaluated_frames is None:
        train_frame = curve_evaluation(out, meta, train, current, records, 'train', args.device, 'bank')
        frame = curve_evaluation(out, meta, train, current, records, split, args.device, 'bank') if split != 'train' else train_frame
    else:
        frame = evaluated_frames[split]
    choices = exact_lambda_choices(frame)
    selected = []
    if split == 'cal':
        files = sorted((Path(meta['v3_instance'])/'update_alignment_feasibility_v1/matched_models').glob('model*.pt'), key=lambda q: int(q.stem[5:]))
        if len(files) != 45:
            raise ValueError('expected frozen aligned calibration models are missing')
        models = [legacy.load_model(file, train['x'].shape[1]) for file in files]
        paths = torch.load(out/'paths_cal.pt', weights_only=True)['methods']
        with torch.no_grad():
            for row in choices:
                path = paths[row['method']]
                q = path['q'][SETTINGS['lambdas'].index(row['lambda'])]
                values = [float(population_mean((torch.sigmoid(model(q)) >= .5).double(), path['weights'])) for model in models]
                row.update(cal_worst=min(values), cal_mean=float(np.mean(values)))
        for method in paths:
            group = pd.DataFrame([row for row in choices if row['method'] == method])
            selected.append(group.sort_values(['cal_worst', 'cal_mean', 'recourse_cost', 'r_over_b_ref'],
                             ascending=[False, False, True, True]).iloc[0].to_dict())
        radius_rule = 'existing aligned-calibration worst validity, mean validity, cost, radius; lambda uses only exact bank hinge'
        cal_hashes = {str(file): p.file_hash(file) for file in files}
    else:
        selected = choices
        radius_rule = 'fixed-r diagnostic only; no r_star and no deployment calibration'
        cal_hashes = {}
    p.write_json(out/'selection.json', {'instance_id': meta['instance_id'], 'frozen_geometry': verify_geometry(out),
        'evaluation_bank_sha256': p.file_hash(out/'evaluation_bank.json'), 'selection_split': split,
        'selection_paths_sha256': p.file_hash(out/f'paths_{split}.pt'), 'calibration_model_hashes': cal_hashes,
        'deployment_calibrated': split == 'cal', 'selected': selected, 'fixed_radius_choices': choices,
        'radius_rule': radius_rule})
    pd.DataFrame(choices).to_csv(out/'fixed_radius_choices.csv', index=False)
    print(meta['dataset'], 'selection', [(s['method'], s['r_over_b_ref'], s['lambda']) for s in selected], flush=True)


def verify_selection(out):
    selection = json.loads((out/'selection.json').read_text())
    if (selection['frozen_geometry'] != verify_geometry(out)
            or selection['evaluation_bank_sha256'] != p.file_hash(out/'evaluation_bank.json')
            or selection['selection_paths_sha256'] != p.file_hash(out/f"paths_{selection['selection_split']}.pt")):
        raise ValueError('selection inputs changed')
    for file, digest in selection['calibration_model_hashes'].items():
        if p.file_hash(Path(file)) != digest:
            raise ValueError('calibration model changed')
    return selection


def future_models(args):
    out, meta, train, current = context(args)
    verify_selection(out)
    original = Path(meta['original_instance'])
    manifest = pd.read_csv(original/'future_final_manifest.csv')
    features = legacy.predictive_features(train['x'], train['y'])
    for index, row in manifest.iterrows():
        if index % args.workers != args.worker:
            continue
        target = out/'aligned_future_models'/f'model{index}.pt'
        if target.exists():
            continue
        x, y, _ = shift(row.environment, train['x'], train['y'], row.severity_control,
                         int(row.realization), int(row.seed), meta['b_ref'], features)
        model = fine_tune(current, x, y, meta['full_update_seed'], args.device)
        save(model.state_dict(), target)
    print(meta['dataset'], 'future replay worker', args.worker, 'done', flush=True)


def evaluate(args, score_bank=True):
    out, meta, train, current = context(args)
    selection = verify_selection(out)
    records = verify_eval_bank(out, meta, train)
    # Retrospective fixed batch only: there is no untouched confirmation set.
    split = 'test' if meta['calibration_CF_available'] else 'diagnostic'
    query_paths(args, split)
    if score_bank:
        curve_evaluation(out, meta, train, current, records, split, args.device, 'bank')
    future = pd.read_csv(Path(meta['original_instance'])/'future_final_manifest.csv')
    rows, individual, curves = [], [], []
    splits = [split] if meta['calibration_CF_available'] else ['train', 'diagnostic']
    for population_split in splits:
        collection = torch.load(out/f'paths_{population_split}.pt', weights_only=True)
        for operator in ['aligned_full_update', 'scratch_stress']:
            files = ([out/'aligned_future_models'/f'model{i}.pt' for i in future.index]
                     if operator == 'aligned_full_update' else list(map(Path, future.path)))
            models = [legacy.load_model(file, train['x'].shape[1]) for file in files]
            with torch.no_grad():
                for method, path in collection['methods'].items():
                    for li, lam in enumerate(SETTINGS['lambdas']):
                        q, w = path['q'][li], path['weights']
                        h, validity = [], []
                        for index, model in enumerate(models):
                            pred = torch.sigmoid(model(q))
                            h.append(float(population_mean(torch.relu(.5-pred).double(), w)))
                            validity.append(float(population_mean((pred >= .5).double(), w)))
                            individual.append({'split': population_split, 'operator': operator, 'method': method,
                                'lambda': lam, 'model_index': index, 'hinge': h[-1], 'validity': validity[-1]})
                        source = collection['source'][path['source_index']]
                        cost = float(population_mean(torch.linalg.vector_norm(q-source, dim=1).double(), w))
                        row = {'dataset': meta['dataset'], 'split': population_split, 'operator': operator, 'method': method,
                            'lambda': lam, 'mean_future_hinge': float(np.mean(h)), 'worst_future_hinge': max(h),
                            'mean_future_validity': float(np.mean(validity)), 'worst_future_validity': min(validity),
                            'recourse_cost': cost, 'current_validity': float(population_mean((torch.sigmoid(current(q)) >= .5).double(), w)),
                            'deployment_calibrated': selection['deployment_calibrated'], 'confirmatory': False}
                        curves.append(row)
                        for chosen in selection['selected']:
                            if chosen['method'] == method and chosen['lambda'] == lam:
                                rows.append({**row, 'r_over_b_ref': chosen['r_over_b_ref']})
    pd.DataFrame(rows).to_csv(out/'selected_future_results.csv', index=False)
    pd.DataFrame(curves).to_csv(out/'future_curves.csv', index=False)
    pd.DataFrame(individual).to_csv(out/'future_per_model.csv', index=False)
    p.write_json(out/'evaluation_manifest.json', {'selection_sha256': p.file_hash(out/'selection.json'),
        'population_paths_sha256': {s: p.file_hash(out/f'paths_{s}.pt') for s in splits},
        'future_manifest_sha256': p.file_hash(Path(meta['original_instance'])/'future_final_manifest.csv'),
        'model_hashes': {str(f): p.file_hash(f) for f in list((out/'aligned_future_models').glob('model*.pt'))+list(map(Path, future.path))},
        'interpretation': 'retrospective benchmark; no untouched confirmation claim'})
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['audit_prepare', 'bootstrap', 'freeze_geometry_bank', 'graph_construct',
                                        'discover', 'freeze_eval_bank', 'select', 'future_models', 'evaluate'])
    parser.add_argument('--dataset', choices=['heloc', 'diabetes'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--worker', type=int, default=0)
    args = parser.parse_args()
    globals()[args.phase](args)


if __name__ == '__main__':
    main()

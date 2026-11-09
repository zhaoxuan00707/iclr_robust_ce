#!/usr/bin/env python3
"""Independent lightweight completion checks for task 003; no raw mutation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run import bootstrap
bootstrap()
from lift_ce.experiments import favorable_region_geodesic_wro_v1 as experiment
from lift_ce.experiments import fixed_risk_geometry_v1 as e
from lift_ce.counterfactuals.favorable_region import geometry

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/home/xu.zhao/robust_ce_results/iclr_runs/favorable_region_geodesic_wro_v1/20260916_intrinsic_v1/heloc')
checks={}
for name in ['construction_manifest.json','frozen_paths.json','frozen_paths_cal.json','frozen_paths_test.json']:
    experiment.verify_files(OUT,name)
checks['frozen_construction_and_paths_unchanged']=True
records=experiment.read(OUT/'evaluation_bank.json')['candidates']
assert len(records)==5927
assert len({a['candidate_id'] for a in records})==len(records)
assert sum(a['phase']=='fresh' for a in records)==540
checks['bank_counts']={'historical':5387,'fresh':540,'total':5927}
checks['fresh_coverage']=[]
for method in experiment.INTRINSIC:
    for lam in [.25,.5,.75]:
        for rep in range(3):
            selected=[a for a in records if a['phase']=='fresh' and a['source_method']==method and a['source_lambda']==lam and a['replication']==rep]
            assert len(selected)==15
            assert sum(a['candidate_type'].startswith('radial') for a in selected)==4
            checks['fresh_coverage'].append({'method':method,'lambda':lam,'replication':rep,'models':15,'radial_models':4})
current=e.legacy.load_model(OUT/'current.pt',22)
g=torch.load(OUT/'graph.pt',weights_only=True);d=np.load(OUT/'distances.npy')
config=experiment.read(OUT/'resolved_config.json')
geom=pd.read_csv(OUT/'geometry_feasibility.csv')
checks['independent_geometry']=[]
for row in geom.to_dict('records'):
    learned=torch.load(OUT/row['control_file'],weights_only=True)
    p0,p1,length=geometry(d,g['source_nodes'].numpy(),learned['node_ids'].numpy(),g['target_nodes'].numpy())
    assert length<=row['B_geo']+1e-6
    assert abs(length-row['total_length'])<1e-8
    assert p0.marginal_residual<1e-8 and p1.marginal_residual<1e-8
    with torch.no_grad():assert (torch.sigmoid(current(learned['points']))>=.5).all()
    checks['independent_geometry'].append({'method':row['method'],'total':length,'budget':row['B_geo'],'slack':row['B_geo']-length})
assert abs(geom.iloc[0].total_length-config['D'])<1e-6
assert np.max(np.diff(geom.true_bank_hinge))<=1e-12
checks['beta1_triangle_equality']=True
checks['nested_best_achieved_risk_nonincreasing']=True
selection=experiment.read(OUT/'selection.json')
for split in ['train','cal','test']:
    collection=torch.load(OUT/f'paths_{split}.pt',weights_only=True)
    original=torch.load(experiment.PARENT/f'paths_{split}.pt',weights_only=True)
    pop=torch.load(OUT/f'{split}_population.pt',weights_only=True)
    for method,path in collection['methods'].items():
        torch.testing.assert_close(path['q'][0],pop['path']['base'][path['source_index']],rtol=0,atol=1e-6)
        torch.testing.assert_close(path['q'][-1],pop['path']['target'][path['target_index']],rtol=0,atol=1e-6)
        np.testing.assert_allclose(torch.bincount(path['source_index'],weights=path['weights']).numpy(),1/128,atol=1e-8)
        np.testing.assert_allclose(torch.bincount(path['target_index'],weights=path['weights']).numpy(),1/3336,atol=1e-8)
        if method in experiment.CONTROLS:
            for key in ['q','weights','source_index','target_index']:
                torch.testing.assert_close(path[key],original['methods'][method][key],rtol=0,atol=0)
        else:
            with torch.no_grad():assert (torch.sigmoid(current(path['q'].flatten(0,1)))>=.5).all()
    curves=pd.read_csv(OUT/f'bank_{split}_curves.csv')
    assert (curves.bank_hinge>=curves.before_fresh_hinge-1e-12).all()
    replay=np.load(OUT/f'bank_{split}_replay.npz')
    assert replay['hinge'].shape==(5927,10,21)
    for mi,method in enumerate(replay['methods']):
        np.testing.assert_allclose(replay['hinge'][:,mi,[0,-1]],replay['hinge'][:,0,[0,-1]],atol=1e-10,rtol=0)
        calculated=replay['hinge'][:,mi].max(0)
        reported=curves[curves.method==method].sort_values('lambda').bank_hinge.to_numpy()
        np.testing.assert_allclose(calculated,reported,atol=1e-12,rtol=0)
    if split=='cal':
        for selected in selection['selected']:
            sub=curves[curves.method==selected['method']]
            tied=sub[sub.bank_hinge==sub.bank_hinge.min()]['lambda'].sort_values().tolist()
            assert selected['lambda']==tied[0] and selected['tied_lambdas']==tied
checks['all_population_endpoints_and_marginals']=True
checks['all_historical_controls_bitwise_unchanged']=True
checks['all_intrinsic_grid_points_current_valid']=True
checks['all_bank_maxima_and_selection_recomputed']=True
query=pd.read_csv(OUT/'geometry_by_split.csv')
checks['query_budget_failures']=query[~query.budget_feasible].to_dict('records')
checks['interpretation']='Exact budget checks refer to unregularized OT on the numerical graph, not exact continuous intrinsic distances. Any query budget failures are explicitly unsupported transfer, not discarded mass or repaired by retuning.'
(ROOT/'results/006_favorable_region_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print('Task-003 review checks passed',flush=True)

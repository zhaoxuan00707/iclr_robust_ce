#!/usr/bin/env python3
"""Independent checks of completed task004 frozen artifacts and selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run import bootstrap
bootstrap()
from lift_ce.experiments import minimal_validity_repair_wro_v1 as run
from lift_ce.experiments import fixed_risk_geometry_v1 as e
from lift_ce.counterfactuals.minimal_validity_repair import geometry,validity,valid_intervals
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/home/xu.zhao/robust_ce_results/iclr_runs/minimal_validity_repair_wro_v1/20260916_minimal_v2/heloc')
config=run.read(OUT/'resolved_config.json');checks={}
for manifest in ['construction_manifest.json','frozen_raw.json','frozen_repair.json','frozen_paths.json']:
    run.verify_files(OUT,manifest)
checks['frozen_files_unchanged']=True
train=torch.load(OUT/'training.pt',weights_only=True);pop=torch.load(OUT/'train_population.pt',weights_only=True)
current=e.legacy.load_model(OUT/'current.pt',22);source=pop['path']['base'];target=train['target']
with torch.no_grad():assert (torch.sigmoid(current(target))>=.5).all()
checks['all_corresponding_target_anchors_valid']=True
alpha=pd.read_csv(OUT/'repair_alpha.csv');manifest=pd.read_csv(OUT/'repair_manifest.csv');geometry_rows=[]
for beta,u,p in zip(run.BETAS,run.RAW,run.REPAIRED):
    raw=torch.load(OUT/f'raw_supports_{u}.pt',weights_only=True)['points']
    _,_,g=geometry(source,raw,target,config['D'],beta*config['D'])
    assert g['slack']>=-config['geometry_tolerance']
    row=manifest[manifest.method==p].iloc[0]
    if row.status!='feasible':
        checks.setdefault('infeasible_betas',[]).append(beta);continue
    fixed=torch.load(OUT/f'repaired_supports_{p}.pt',weights_only=True)['points']
    a=alpha[alpha.method==p].sort_values('support_index')
    invalid=torch.tensor(a.raw_invalid.to_numpy())
    torch.testing.assert_close(raw[~invalid],fixed[~invalid],atol=0,rtol=0)
    assert validity(current,fixed)['invalid_support_count']==0
    assert ((fixed>=train['x'].min(0).values-1e-7)&(fixed<=train['x'].max(0).values+1e-7)).all()
    _,_,g2=geometry(source,fixed,target,config['D'],beta*config['D'])
    assert g2['slack']>=-config['geometry_tolerance']
    for item in a[a.raw_invalid].to_dict('records'):
        i=item['support_index'];intervals=valid_intervals(current,raw[i].numpy(),target[i].numpy())
        if not row.joint_budget_repair_used:
            # Root-finding tolerance plus explicitly recorded float32 evaluation allowance.
            assert abs(item['alpha']-intervals[0][0])<2e-5,(p,i,item['alpha'],intervals[0])
    geometry_rows.append({'beta_geo':beta,'raw_total_length':g['total_length'],'repaired_total_length':g2['total_length'],'B_geo':beta*config['D'],'raw_slack':g['slack'],'repair_slack':g2['slack']})
checks['independent_exact_geometry']=geometry_rows
checks['valid_supports_bitwise_unchanged']=True
checks['repaired_controls_current_valid_and_in_feature_box']=True
checks['first_crossings_checked']=True
records=run.read(OUT/'evaluation_bank.json')['candidates'];old_count=sum(a['phase']!='fresh' for a in records)
assert old_count==5927 and len({a['candidate_id'] for a in records})==len(records)
methods=torch.load(OUT/'paths_train.pt',weights_only=True)['methods']
active=[m for m in run.RAW+run.REPAIRED if m in methods]
assert sum(a['phase']=='fresh' for a in records)==len(active)*3*3*15
for m in active:
    for l in [.25,.5,.75]:
        for rep in range(3):
            r=[a for a in records if a['phase']=='fresh' and a['source_method']==m and a['source_lambda']==l and a['replication']==rep]
            assert len(r)==15 and sum(a['candidate_type'].startswith('radial') for a in r)==4
checks['bank_counts']={'historical':old_count,'fresh':len(records)-old_count,'total':len(records)}
checks['all_raw_repaired_discovery_cells_and_radial_models_present']=True
selection=run.read(OUT/'selection.json')
assert selection['selection_split']=='train'
assert selection['evaluation_bank_sha256']==e.p.file_hash(OUT/'evaluation_bank.json')
assert selection['selection_paths_sha256']==e.p.file_hash(OUT/'paths_train.pt')
for split in ['train','test']:
    curves=pd.read_csv(OUT/f'bank_{split}_curves.csv');replay=np.load(OUT/f'bank_{split}_replay.npz')
    collection=torch.load(OUT/f'paths_{split}.pt',weights_only=True)
    population=torch.load(OUT/f'{split}_population.pt',weights_only=True)
    original=torch.load(run.PARENT/f'paths_{split}.pt',weights_only=True)['methods']
    graph=torch.load(run.GRAPH/f'paths_{split}.pt',weights_only=True)['methods']
    for mi,m in enumerate(replay['methods']):
        path=collection['methods'][m]
        torch.testing.assert_close(path['q'][0],population['path']['base'][path['source_index']],atol=1e-6,rtol=0)
        torch.testing.assert_close(path['q'][-1],target[path['target_index']],atol=1e-6,rtol=0)
        np.testing.assert_allclose(np.bincount(path['source_index'],weights=path['weights']),1/128,atol=1e-8)
        np.testing.assert_allclose(np.bincount(path['target_index'],weights=path['weights']),1/3336,atol=1e-8)
        if m in run.CONTROLS:
            parent=graph if m.startswith('F_') else original
            for key in ['q','weights','source_index','target_index']:torch.testing.assert_close(path[key],parent[m][key],rtol=0,atol=0)
        np.testing.assert_allclose(replay['hinge'][:,mi].max(0),curves[curves.method==m].sort_values('lambda').bank_hinge,atol=1e-12,rtol=0)
        np.testing.assert_allclose(replay['hinge'][:,mi,[0,-1]],replay['hinge'][:,0,[0,-1]],atol=1e-10,rtol=0)
    assert (curves.bank_hinge>=curves.before_fresh_hinge-1e-12).all()
    if split=='train':
        v=pd.read_csv(OUT/'path_validity_train.csv')
        for item in selection['selected']:
            group=curves[curves.method==item['method']]
            if item['method'] in run.REPAIRED:
                valid=v[(v.method==item['method'])&(v.invalid_support_count==0)]['lambda']
                group=group[group['lambda'].isin(valid)]
            tied=group[group.bank_hinge==group.bank_hinge.min()]['lambda'].sort_values().tolist()
            assert item['lambda']==tied[0] and item['tied_lambdas']==tied
checks['all_endpoint_masses_and_historical_controls_preserved']=True
checks['shared_bank_reduction_and_enlargement_checked']=True
checks['repaired_selection_uses_only_valid_training_outputs']=True
metrics=pd.read_csv(OUT/'all_metrics.csv')
checks['heldout_repaired_output_failures']=metrics[(metrics.kind=='selected_path')&(metrics.split=='test')&(metrics.method.isin(run.REPAIRED))&((metrics.invalid_mass>0)|(metrics.final_geometry_feasible==False))][['method','invalid_mass','slack']].to_dict('records')
(ROOT/'results/007_minimal_repair_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print('Task004 independent review passed',flush=True)

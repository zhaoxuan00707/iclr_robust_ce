"""Matched spline experiment; reuses official discovery/evaluation operators.

Only this entry-point process binds the existing protocol's path-verification
and query hooks to fixed Euclidean spline paths. No graph or risk map is built.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import fixed_risk_geometry_v1 as e
from ..counterfactuals.population_path import PopulationPath, WaypointPopulationPath
from ..counterfactuals.transport_spline import TransportSplinePath, bend_diagnostics

OUTPUT = e.RESULTS/'transport_spline_path_v1/20260915_matched_v1'
PARENT = e.OUTPUT
METHODS = ['straight','waypoint','spline']


def verify_paths(out):
    manifest = json.loads((out/'frozen_paths.json').read_text())
    for name, digest in manifest['files'].items():
        if e.p.file_hash(out/name) != digest:
            raise ValueError(f'frozen matched path input changed: {name}')
    return manifest


def prepare(args):
    out = args.output/args.dataset
    out.mkdir(parents=True,exist_ok=True)
    if args.dataset == 'diabetes':
        original = json.loads((PARENT/'diabetes/instance_manifest.json').read_text())
        e.p.write_json(out/'instance_manifest.json',original)
        e.p.write_json(out/'blocked.json',{'reason':'No compatible pre-existing training-only waypoint found; no frozen calibration CF artifact. No waypoint training or CF regeneration allowed.',
            'searched_root':str(e.RESULTS),'missing_cells':'All three matched-path comparisons, all radii/lambdas, fresh discovery and held-out selection/evaluation.',
            'fixed_instance':str(PARENT/'diabetes'),'status':'awaiting existing waypoint artifact; no fabricated results'})
        return
    if (out/'frozen_paths.json').exists():
        verify_paths(out)
        return
    parent = PARENT/args.dataset
    for name in ['instance_manifest.json','training.pt','current.pt','train_population.pt','data_audit.pt','old_bank.json']:
        shutil.copyfile(parent/name,out/name)
    _,meta,train,current = e.context(args)
    v3 = Path(meta['v3_instance'])
    wp = v3/'robust_waypoint_path_v1'
    manifest = json.loads((wp/'path_manifest.json').read_text())
    instance = json.loads((v3/'instance.json').read_text())
    if manifest['parent_instance_id'] != instance['instance_id'] or manifest['split'] != 'train' or manifest['calibration_used_for_fit']:
        raise ValueError('waypoint provenance incompatible')
    if e.p.file_hash(wp/'frozen_path.pt') != manifest['path_sha256']:
        raise ValueError('waypoint changed')
    bank_dir, _ = e.p.verify_bank(v3,instance,'bank_full_update_v1')
    if e.p.file_hash(bank_dir/'manifest.json') != manifest['training_bank_sha256']:
        raise ValueError('waypoint training bank mismatch')
    fit = torch.load(wp/'frozen_path.pt',weights_only=True)
    if fit['split'] != 'train' or fit['parent_instance_id'] != instance['instance_id']:
        raise ValueError('waypoint tensor provenance mismatch')
    shutil.copyfile(wp/'frozen_path.pt',out/'frozen_waypoint.pt')
    e.p.write_json(out/'waypoint_manifest.json',{**manifest,'source':str(wp),'selection':'existing training-objective winner seed5, no refit or heldout reselection'})
    config = {k:e.SETTINGS[k] for k in ['evaluation_radii','lambdas','source_lambdas','replications','search','lambda_tie_rule','confirmation']}
    config.update(methods=METHODS, pairing='unchanged stored squared-Euclidean sparse OT, per split; common to A/B/C',
        waypoint='unchanged training-objective winner',spline='natural Euclidean minimum acceleration; no projection',
        constraint_policy=meta['actionability'],range_diagnostic='observed training min/max, diagnostic not newly imposed hard bounds',
        construction_bank='all 1742 compatible historical candidates; fresh all-method union',
        gpu_ids=[0,1,2,3,5,6,7],new_geometry=False)
    e.p.write_json(out/'resolved_config.json',config)
    build_paths(args,'train',verify=False)
    files=['training.pt','current.pt','train_population.pt','frozen_waypoint.pt','waypoint_manifest.json','resolved_config.json','paths_train.pt','fixed_ot_coupling_train.pt','spline_coefficients_train.pt','old_bank.json']
    e.p.write_json(out/'frozen_paths.json',{'instance_id':meta['instance_id'],'files':{name:e.p.file_hash(out/name) for name in files},'no_risk_geometry':True,'frozen_before_new_discovery':True})
    records=e.verified_records(json.loads((out/'old_bank.json').read_text())['candidates'],meta,train)
    if len(records)!=1742:raise ValueError('historical bank incomplete')
    print('Frozen HELOC paths: same pairing and existing waypoint; historical bank',len(records),flush=True)


def build_paths(args,split,verify=True):
    out,meta,train,current=e.context(args)
    if verify:verify_paths(out)
    path_file=out/f'paths_{split}.pt'
    if path_file.exists():return
    if split!='train':
        if split=='test':e.verify_selection(out)
        source=Path(meta['v3_instance'])/f'{split}_population.pt'
        if e.p.file_hash(source)!=source.with_suffix('.sha256').read_text().strip():
            raise ValueError('frozen query CF population changed')
        shutil.copyfile(source,out/source.name)
    pop=torch.load(out/f'{split}_population.pt',weights_only=True)
    p=PopulationPath(**pop['path'])
    if not torch.equal(p.target,train['target']):raise ValueError('target anchor changed')
    # Float64 aggregation retains stored masses; do not renormalize/recompute OT.
    source_mass=torch.bincount(p.source_index,weights=p.weights.double(),minlength=len(p.base))
    target_mass=torch.bincount(p.target_index,weights=p.weights.double(),minlength=len(p.target))
    if not (torch.allclose(source_mass,torch.full_like(source_mass,1/len(p.base)),atol=1e-8) and torch.allclose(target_mass,torch.full_like(target_mass,1/len(p.target)),atol=1e-8)):
        raise ValueError('stored OT marginals fail')
    wp=torch.load(out/'frozen_waypoint.pt',weights_only=True)['waypoints']
    classes=[p,WaypointPopulationPath(p,wp),TransportSplinePath(p,wp)]
    collection={'source':pop['source'],'source_ids':pop['source_ids'],'methods':{}}
    feasibility=[]
    with torch.no_grad():
        for method,path in zip(METHODS,classes):
            q=torch.stack([path.at(t) for t in e.SETTINGS['lambdas']])
            torch.testing.assert_close(q[0],p.base[p.source_index],atol=1e-6,rtol=0)
            torch.testing.assert_close(q[-1],p.target[p.target_index],atol=1e-6,rtol=0)
            collection['methods'][method]={'q':q,'weights':p.weights.double(),'source_index':p.source_index,'target_index':p.target_index}
            for t,points in zip(e.SETTINGS['lambdas'],q):
                invalid=torch.sigmoid(current(points))<.5
                outside=((points<train['x'].min(0).values-1e-6)|(points>train['x'].max(0).values+1e-6)).any(1)
                feasibility.append({'split':split,'method':method,'lambda':t,'current_invalid_mass':float((invalid*p.weights.double()).sum()),
                    'training_range_violation_mass':float((outside*p.weights.double()).sum()),'immutable_violation_mass':0.,'categorical_violation_mass':0.,
                    'constraint_scope':'no immutable/categorical constraints declared in inherited continuous representation; physical actionability not certified','projected':False})
    torch.testing.assert_close(collection['methods']['waypoint']['q'][10],collection['methods']['spline']['q'][10],atol=1e-6,rtol=0)
    e.save(collection,path_file)
    e.save(p.state_dict(),out/f'fixed_ot_coupling_{split}.pt')
    e.save({'z':p.base[p.source_index],'y':p.target[p.target_index],'w':wp[p.target_index],
        'u':wp[p.target_index]-(p.base[p.source_index]+p.target[p.target_index])/2,'weights':p.weights,
        'basis_left':[0.,3.,0.,-4.],'right':'b(1-t)','knots':[0.,.5,1.]},out/f'spline_coefficients_{split}.pt')
    diag=bend_diagnostics(p,wp,e.SETTINGS['lambdas'])
    pd.DataFrame([{'split':split,'lambda':t,**{k:v for k,v in diag.items() if k!='deviations'},'deviation':d}
        for t,d in zip(e.SETTINGS['lambdas'],diag['deviations'])]).to_csv(out/f'bend_{split}.csv',index=False)
    pd.DataFrame(feasibility).to_csv(out/f'feasibility_{split}.csv',index=False)
    e.p.write_json(out/f'path_audit_{split}.json',{'source_marginal_residual':float((source_mass-1/len(p.base)).abs().max()),
        'target_marginal_residual':float((target_mass-1/len(p.target)).abs().max()),'same_coupling_all_methods':True,'exact_knots_checked':True,
        'stored_original_coupling':True,'no_rematching':True,'source_population_sha256':e.p.file_hash(out/f'{split}_population.pt'),**diag})


def query_paths(args,split):build_paths(args,split)


def score(args):
    out,meta,train,current=e.context(args)
    records=e.verify_eval_bank(out,meta,train)
    e.curve_evaluation(out,meta,train,current,records[args.worker::args.workers],args.split,args.device,f'shard{args.worker}')


def merge(args):
    out,meta,train,_=e.context(args)
    records=e.verify_eval_bank(out,meta,train)
    chunks=[np.load(out/f'shard{i}_{args.split}_replay.npz') for i in range(args.workers)]
    ids=np.concatenate([c['candidate_ids'] for c in chunks]);lookup={v:i for i,v in enumerate(ids)}
    if len(lookup)!=len(records) or set(lookup)!={a['candidate_id'] for a in records}:raise ValueError('shard membership mismatch')
    order=[lookup[a['candidate_id']] for a in records]
    h=np.concatenate([c['hinge'] for c in chunks])[order];inv=np.concatenate([c['invalidity'] for c in chunks])[order]
    frame=pd.read_csv(out/f'shard0_{args.split}_curves.csv')
    for index,row in frame.iterrows():
        mi=METHODS.index(row['method']);li=e.SETTINGS['lambdas'].index(row['lambda']);r=row.r_over_b_ref
        eligible=np.array([a['measured_budget']<=r*meta['b_ref']+max(1e-7,r*meta['b_ref']*1e-6) for a in records])
        old=eligible & np.array([a['phase']!='fresh' for a in records])
        indices=np.flatnonzero(eligible);win=indices[np.argmax(h[eligible,mi,li])]
        frame.loc[index,['bank_hinge','before_fresh_hinge','bank_invalidity','hinge_witness_validity','winning_attack_id','bank_size']]=[
            h[win,mi,li],h[old,mi,li].max(),inv[eligible,mi,li].max(),1-inv[win,mi,li],records[win]['candidate_id'],int(eligible.sum())]
        for rep in range(3):
            chosen=old|(eligible & np.array([a['phase']=='fresh' and a['replication']==rep for a in records]))
            frame.loc[index,f'rep{rep}_hinge']=h[chosen,mi,li].max()
    frame['fresh_risk_increase']=frame.bank_hinge-frame.before_fresh_hinge
    if (frame.fresh_risk_increase < -1e-12).any():raise ValueError('bank enlargement decreased maximum')
    for mi in [1,2]:np.testing.assert_allclose(h[:,0,[0,-1]],h[:,mi,[0,-1]],atol=1e-10)
    frame.to_csv(out/f'bank_{args.split}_curves.csv',index=False)
    np.savez_compressed(out/f'bank_{args.split}_replay.npz',hinge=h,invalidity=inv,candidate_ids=np.array([a['candidate_id'] for a in records]),methods=METHODS,lambdas=e.SETTINGS['lambdas'])
    print('Merged',args.split,h.shape,flush=True)


def select(args):
    out=args.output/args.dataset
    e.select(args,evaluated_frames={s:pd.read_csv(out/f'bank_{s}_curves.csv') for s in ['train','cal']})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','paths','discover','freeze_eval_bank','score','merge','select','future_models','evaluate'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',choices=['heloc','diabetes'],default='heloc')
    parser.add_argument('--split',default='train');parser.add_argument('--device',default='cpu')
    parser.add_argument('--workers',type=int,default=7);parser.add_argument('--worker',type=int,default=0)
    args=parser.parse_args()
    # Reuse shared-bank, full-update and heldout-selection protocol unchanged;
    # replace graph-specific hooks only inside this isolated executable process.
    e.verify_geometry=verify_paths
    e.query_paths=query_paths
    if args.phase=='paths':build_paths(args,args.split)
    elif args.phase=='evaluate':e.evaluate(args,score_bank=False)
    elif args.phase in ['discover','freeze_eval_bank','future_models']:getattr(e,args.phase)(args)
    else:globals()[args.phase](args)

if __name__=='__main__':main()

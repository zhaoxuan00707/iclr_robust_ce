"""Direct geodesic-excess constrained WRO; isolated fixed HELOC experiment."""
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from . import fixed_risk_geometry_v1 as e
from . import transport_spline_path_v1 as old
from ..counterfactuals.population_path import PopulationPath,WaypointPopulationPath
from ..counterfactuals.geodesic_excess import exact_w2,excess,glued_arclength_path,project_basic
from ..counterfactuals.geodesic_excess_solver import solve_control,bank_value_gradient,geometry_record
from ..counterfactuals.robust_waypoints import ModelBank

OUTPUT=e.RESULTS/'geodesic_excess_wro_v1/20260915_direct_v2_cpu_verified'
PARENT=old.OUTPUT/'heloc'
KAPPAS=[0.,.05,.10,.20]
INITIALIZATIONS=['target','old_waypoint','random','barycentric_midpoint']
METHODS=['A_straight','B_old_waypoint']+[f'C_k{int(k*100):03d}' for k in KAPPAS]
OPT={'steps':120,'learning_rate':.03,'beta_h':50.,'beta_m':100.,'rho_initial':1.,
     'dual_interval':5,'stall_interval':20,'feasibility_relative_tolerance':1e-6,
     'restoration_steps':12,'random_seed':20260915,'random_sd':.02}


def verify_construction(out):
    manifest=json.loads((out/'construction_manifest.json').read_text())
    for file,digest in manifest['files'].items():
        if e.p.file_hash(out/file)!=digest:raise ValueError(f'construction changed: {file}')
    return manifest


def verify_paths(out):
    verify_construction(out)
    manifest=json.loads((out/'frozen_paths.json').read_text())
    for file,digest in manifest['files'].items():
        if e.p.file_hash(out/file)!=digest:raise ValueError(f'fixed path changed: {file}')
    return manifest


def prepare(args):
    out=args.output/'heloc';out.mkdir(parents=True,exist_ok=True)
    if (out/'construction_manifest.json').exists():verify_construction(out);return
    old.verify_paths(PARENT)
    for name in ['instance_manifest.json','training.pt','current.pt','train_population.pt','data_audit.pt','frozen_waypoint.pt','waypoint_manifest.json']:
        shutil.copyfile(PARENT/name,out/name)
    _,meta,train,current=e.context(args)
    bank=json.loads((PARENT/'evaluation_bank.json').read_text())
    records=e.verified_records(bank['candidates'],meta,train)
    # Previous fresh attacks now belong to the frozen construction bank.
    records=[{**a,'parent_phase':a['phase'],'phase':'old'} for a in records]
    if len(records)!=2957:raise ValueError('expected complete compatible historical bank')
    e.p.write_json(out/'old_bank.json',{'instance_id':meta['instance_id'],'source':str(PARENT/'evaluation_bank.json'),'source_sha256':e.p.file_hash(PARENT/'evaluation_bank.json'),'candidates':records})
    pd.DataFrame(records).to_csv(out/'attack_bank_manifest.csv',index=False)
    pop=torch.load(out/'train_population.pt',weights_only=True)
    z=pop['path']['base'];y=train['target'];D=exact_w2(z,y).distance
    w=torch.load(out/'frozen_waypoint.pt',weights_only=True)['waypoints']
    p0,p1,E=excess(z,w,y,D)
    config={'dataset':'heloc','fixed_radius_over_b_ref':3.,'absolute_data_radius':3*meta['b_ref'],
        'kappa_over_D':KAPPAS,'D':D,'lambdas':e.SETTINGS['lambdas'],'initializations':INITIALIZATIONS,
        'optimizer':OPT,'tau':.5,'current_validity_constraint':.5,'construction_margin_diagnostic_only':.7,
        'particle_masses':'same uniform b_j=1/3336 as P1; no particle dropping or reweighting',
        'geometry':'unregularized Euclidean W2 for all three terms, no Sinkhorn',
        'OT_gradient':'envelope subgradient of optimal discrete plans, recomputed every step',
        'lens_audit':'fixed-instance metadata has no instantiated delta/epsilon; unrelated YAML initial ratios are not adopted',
        'basic_constraints':'observed training feature box and f0>=.5; inherited all-mutable continuous representation; no physical integer/immutable guarantee',
        'query_rule':'same frozen learned Q on cal/test; separately solve both exact OT legs and use batch-specific arclength; report each batch excess, no Q refit or automatic feasibility transfer',
        'midpoint_initialization':'target-conditional barycentric mean of split OT midpoint, not an exact geodesic population',
        'nested_candidate_rule':'pool every retained training candidate across starts and kappa jobs; select minimum true risk among exact-feasible candidates for each kappa; never select kappa',
        'historical_control_excess':{'d0':p0.distance,'d1':p1.distance,'E_geo':E,'E_geo_over_D':E/D},
        'search':e.SETTINGS['search'],'source_lambdas':e.SETTINGS['source_lambdas'],'replications':3,
        'gpu_ids':[0,1,2,3,5,6,7],'confirmation':'previously inspected fixed benchmark; no fresh confirmatory claim'}
    e.p.write_json(out/'resolved_config.json',config)
    pd.DataFrame([{'kappa_over_D':k,'kappa':k*D,'D':D} for k in KAPPAS]).to_csv(out/'kappa_grid.csv',index=False)
    files=['instance_manifest.json','training.pt','current.pt','train_population.pt','frozen_waypoint.pt','old_bank.json','resolved_config.json']
    e.p.write_json(out/'construction_manifest.json',{'instance_id':meta['instance_id'],'files':{f:e.p.file_hash(out/f) for f in files},'heldout_outcomes_used':False})
    print('Frozen direct problem D=',D,'bank=',len(records),'old excess=',E,flush=True)


def load_solver(args):
    out,meta,train,current=e.context(args);verify_construction(out)
    records=e.verified_records(json.loads((out/'old_bank.json').read_text())['candidates'],meta,train)
    eligible=[a for a in records if a['measured_budget']<=3*meta['b_ref']+3*meta['b_ref']*1e-6]
    models=[e.legacy.load_model(Path(a['model_path']),train['x'].shape[1]) for a in eligible]
    bank=ModelBank(models,args.device)
    source=torch.load(out/'train_population.pt',weights_only=True)['path']['base'].to(args.device)
    target=train['target'].to(args.device);current=current.to(args.device).eval().requires_grad_(False)
    return out,meta,train,current,bank,source,target


def initial_points(out,train,kind,device):
    target=train['target']
    if kind=='target':return target.to(device).clone()
    if kind=='old_waypoint':return torch.load(out/'frozen_waypoint.pt',weights_only=True)['waypoints'].to(device)
    if kind=='random':return (target+OPT['random_sd']*torch.randn(target.shape,generator=torch.Generator().manual_seed(OPT['random_seed']))).to(device)
    pop=torch.load(out/'train_population.pt',weights_only=True)['path']
    numerator=torch.zeros_like(target,dtype=torch.float64)
    numerator.index_add_(0,pop['target_index'],pop['base'][pop['source_index']].double()*pop['weights'].double()[:,None])
    masses=torch.bincount(pop['target_index'],weights=pop['weights'].double(),minlength=len(target))
    return ((target.double()+numerator/masses[:,None])/2).float().to(device)


def optimize(args):
    out,meta,train,current,bank,source,target=load_solver(args)
    config=json.loads((out/'resolved_config.json').read_text());D=config['D']
    lower=train['x'].min(0).values.to(args.device);upper=train['x'].max(0).values.to(args.device)
    jobs=[(k,kind) for k in KAPPAS for kind in INITIALIZATIONS]
    for job,(k,kind) in enumerate(jobs):
        if job%args.workers!=args.worker:continue
        file=out/'optimization'/f'k{int(k*100):03d}_{kind}.pt'
        if file.exists():
            saved=torch.load(file,weights_only=True)
            if saved['construction']!=verify_construction(out):raise ValueError('solver inputs changed')
            continue
        initial=initial_points(out,train,kind,args.device)
        started=time.perf_counter()
        retained,trace=solve_control(bank,current,source,target,initial,D,k*D,lower,upper,OPT)
        e.save({'construction':verify_construction(out),'kappa_ratio':k,'initialization':kind,'candidates':retained,'seconds':time.perf_counter()-started},file)
        frame=pd.DataFrame(trace);frame['initialization']=kind;frame['kappa_ratio']=k
        frame.to_csv(file.with_suffix('.csv'),index=False)
        print('Completed',k,kind,'best risk',min(a['true_bank_hinge'] for a in retained),'seconds',time.perf_counter()-started,flush=True)


def smoke(args):
    out,meta,train,current,bank,source,target=load_solver(args)
    settings={**OPT,'steps':2};lower=train['x'].min(0).values.to(args.device);upper=train['x'].max(0).values.to(args.device)
    D=json.loads((out/'resolved_config.json').read_text())['D'];started=time.perf_counter()
    retained,trace=solve_control(bank,current,source,target,target,D,.1*D,lower,upper,settings)
    e.p.write_json(out/'smoke.json',{'seconds':time.perf_counter()-started,'full_support':True,'full_model_bank':True,'steps':2,'retained_feasible':len(retained),'solver_trace':trace,'excluded_from_primary_candidate_pool':True})
    print('full-size smoke passed',time.perf_counter()-started,flush=True)


def freeze_controls(args):
    out,meta,train,current=e.context(args);verify_construction(out)
    if (out/'frozen_paths.json').exists():verify_paths(out);return
    config=json.loads((out/'resolved_config.json').read_text());D=config['D'];tol=OPT['feasibility_relative_tolerance']*max(1,D)
    source=torch.load(out/'train_population.pt',weights_only=True)['path']['base'];target=train['target']
    pool=[];traces=[]
    for k in KAPPAS:
        for kind in INITIALIZATIONS:
            file=out/'optimization'/f'k{int(k*100):03d}_{kind}.pt'
            obj=torch.load(file,weights_only=True)
            if obj['construction']!=verify_construction(out):raise ValueError('candidate construction mismatch')
            pool.extend([{**a,'origin_kappa_ratio':k,'initialization':kind,'file':str(file),'candidate_index':i} for i,a in enumerate(obj['candidates'])])
            traces.append(pd.read_csv(file.with_suffix('.csv')))
    frame=pd.concat(traces);frame.to_csv(out/'optimization_trace.csv',index=False)
    frame[['kappa_ratio','initialization','step','nu','rho_aug','g']].to_csv(out/'dual_trace.csv',index=False)
    lower=train['x'].min(0).values;upper=train['x'].max(0).values
    # The final reporting predicate is CPU probability >= tau. GPU boundary
    # rounding can disagree, so recheck every candidate before any selection.
    with torch.no_grad():
        for candidate in pool:
            probability=torch.sigmoid(current(candidate['points']))
            candidate['gpu_current_invalid_mass']=candidate['current_invalid_mass']
            candidate['current_invalid_mass']=float((probability<.5).double().mean())
            candidate['reference_minimum_probability']=float(probability.min())
            candidate['box_violation_mass']=float(((candidate['points']<lower-1e-7)|(candidate['points']>upper+1e-7)).any(1).double().mean())
    controls=[]
    for k in KAPPAS:
        candidates=[a for a in pool if a['E_geo']<=k*D+tol and a['current_invalid_mass']==0 and a['box_violation_mass']==0]
        chosen=min(candidates,key=lambda a:(a['true_bank_hinge'],a['origin_kappa_ratio'],a['initialization'],a['step']))
        p0,p1,audit=geometry_record(source,chosen['points'],target,D,k*D,current,lower,upper)
        if audit['E_geo']>k*D+tol or audit['current_invalid_mass']!=0 or audit['box_violation_mass']!=0:
            raise ValueError('final exact or basic feasibility failed')
        method=f'C_k{int(k*100):03d}';file=out/f'learned_control_distribution_k{int(k*100):03d}.pt'
        e.save({'points':chosen['points'],'weights':torch.full((len(target),),1/len(target),dtype=torch.float64),'first_plan':p0.state_dict(),'second_plan':p1.state_dict()},file)
        controls.append({**{key:value for key,value in chosen.items() if key!='points'},**audit,'method':method,'kappa_ratio':k,'control_file':file.name,'control_sha256':e.p.file_hash(file),'exact_tolerance':tol,'candidate_pool_size':len(candidates)})
    risks=[a['true_bank_hinge'] for a in controls]
    if any(b>a+1e-12 for a,b in zip(risks,risks[1:])):raise ValueError('nested feasible set incumbent risk increased')
    e.p.write_json(out/'controls_manifest.json',controls)
    pd.DataFrame(controls).to_csv(out/'control_metrics.csv',index=False)
    ablation=[]
    for k in KAPPAS:
        for kind in INITIALIZATIONS:
            candidates=[a for a in pool if a['initialization']==kind and a['E_geo']<=k*D+tol and a['current_invalid_mass']==0 and a['box_violation_mass']==0]
            c=min(candidates,key=lambda a:a['true_bank_hinge'])
            ablation.append({'kappa_ratio':k,'initialization':kind,'best_feasible_true_hinge':c['true_bank_hinge'],'E_geo':c['E_geo'],'selected_tag':c['tag']})
    pd.DataFrame(ablation).to_csv(out/'initialization_ablation.csv',index=False)
    build_paths(args,'train',verify=False)
    files=['controls_manifest.json','paths_train.pt']+[a['control_file'] for a in controls]
    if (out/'selection_numerics.json').exists():files.append('selection_numerics.json')
    e.p.write_json(out/'frozen_paths.json',{'construction':verify_construction(out),'files':{f:e.p.file_hash(out/f) for f in files},'frozen_before_post_construction_discovery':True})
    print(pd.DataFrame(controls)[['method','true_bank_hinge','E_geo','kappa','slack','lambda_W']].to_string(index=False),flush=True)


def build_paths(args,split,verify=True):
    out,meta,train,current=e.context(args)
    if verify:verify_paths(out)
    if (out/f'paths_{split}.pt').exists():return
    if split!='train':
        if split=='test':e.verify_selection(out)
        src=Path(meta['v3_instance'])/f'{split}_population.pt'
        if e.p.file_hash(src)!=src.with_suffix('.sha256').read_text().strip():raise ValueError('heldout population changed')
        shutil.copyfile(src,out/src.name)
    pop=torch.load(out/f'{split}_population.pt',weights_only=True);p=PopulationPath(**pop['path'])
    wp=torch.load(out/'frozen_waypoint.pt',weights_only=True)['waypoints']
    collection={'source':pop['source'],'source_ids':pop['source_ids'],'methods':{}}
    for method,path in [('A_straight',p),('B_old_waypoint',WaypointPopulationPath(p,wp))]:
        collection['methods'][method]={'q':torch.stack([path.at(t) for t in e.SETTINGS['lambdas']]),'weights':p.weights.double(),'source_index':p.source_index,'target_index':p.target_index}
    D=exact_w2(p.base,p.target).distance;audits=[]
    for item in json.loads((out/'controls_manifest.json').read_text()):
        w=torch.load(out/item['control_file'],weights_only=True)['points']
        p0,p1,E=excess(p.base,w,p.target,D)
        path=glued_arclength_path(p.base,w,p.target,p0,p1,e.SETTINGS['lambdas'])
        collection['methods'][item['method']]=path
        # Verify the off-grid control knot explicitly, without adding it to lambda selection.
        knot=glued_arclength_path(p.base,w,p.target,p0,p1,[path['lambda_W']])['q'][0]
        torch.testing.assert_close(knot,w[path['control_index']],atol=1e-6,rtol=0)
        for key,count in [('source_index',len(p.base)),('control_index',len(w)),('target_index',len(p.target))]:
            np.testing.assert_allclose(torch.bincount(path[key],weights=path['weights']).numpy(),1/count,atol=1e-8)
        audits.append({'split':split,'method':item['method'],'D':D,'d0':p0.distance,'d1':p1.distance,'E_geo':E,'kappa':item['kappa'],'slack':item['kappa']-E,'lambda_W':path['lambda_W'],'marginal_residual':max(p0.marginal_residual,p1.marginal_residual),'feasible_for_this_batch':E<=item['kappa']+item['exact_tolerance']})
        e.save({'first_plan':p0.state_dict(),'second_plan':p1.state_dict()},out/f'ot_plans_{split}_{item["method"]}.pt')
    feasibility=[]
    with torch.no_grad():
        for method,path in collection['methods'].items():
            for t,q in zip(e.SETTINGS['lambdas'],path['q']):
                weights=path['weights'];score=torch.sigmoid(current(q))
                outside=((q<train['x'].min(0).values-1e-7)|(q>train['x'].max(0).values+1e-7)).any(1)
                feasibility.append({'split':split,'method':method,'lambda':t,'current_invalid_mass':float(((score<.5)*weights).sum()),'below_0.7_mass':float(((score<.7)*weights).sum()),'training_range_violation_mass':float((outside*weights).sum()),'projected':False})
    e.save(collection,out/f'paths_{split}.pt')
    pd.DataFrame(audits).to_csv(out/f'ot_distances_{split}.csv',index=False)
    pd.DataFrame(feasibility).to_csv(out/f'feasibility_{split}.csv',index=False)


def discover(args):
    out,meta,train,current=e.context(args);frozen=verify_paths(out)
    paths=torch.load(out/'paths_train.pt',weights_only=True)['methods']
    jobs=[(method,lam,rep) for method in METHODS for lam in e.SETTINGS['source_lambdas'] for rep in range(3)]
    for job,(method,lam,rep) in enumerate(jobs):
        if job%args.workers!=args.worker:continue
        file=out/'fresh_attacks'/f'{method}_r3_l{lam:.2f}_rep{rep}.pt'
        if file.with_suffix('.models.json').exists():
            if torch.load(file,weights_only=True)['frozen_geometry']!=frozen:raise ValueError('fresh path mismatch')
            continue
        seed=152000000+METHODS.index(method)*100000+int(lam*100)*10+rep
        start=time.perf_counter();path=paths[method];q=path['q'][e.SETTINGS['lambdas'].index(lam)]
        candidates,tags,trace,full,runtime,peak=e.run_search(current,train,q,path['weights'],3*meta['b_ref'],seed,meta['full_update_seed'],args.device)
        obj={'instance_id':meta['instance_id'],'phase':'fresh','method':method,'lambda':lam,'radius':3.,'replication':rep,'search_seed':seed,'full_update_seed':meta['full_update_seed'],'candidates':candidates,'tags':tags,'discovery_split':'train','frozen_geometry':frozen}
        e.save(obj,file);trace.to_csv(file.with_suffix('.truncated.csv'),index=False);full.to_csv(file.with_suffix('.full.csv'),index=False)
        e.replay_candidates(out,meta,train,current,file,obj,args.device)
        e.p.write_json(file.with_suffix('.runtime.json'),{'total_seconds':time.perf_counter()-start,'truncated_seconds':runtime,'peak_cuda_bytes_truncated':peak})
        verify_paths(out);print('fresh completed',method,lam,rep,flush=True)


def score(args):
    out,meta,train,current=e.context(args);records=e.verify_eval_bank(out,meta,train)
    e.curve_evaluation(out,meta,train,current,records[args.worker::args.workers],args.split,args.device,f'shard{args.worker}')


def merge(args):
    # Reuse the previously validated global shard merger with this fixed method list.
    old.METHODS=METHODS
    old.merge(args)


def select(args):
    out=args.output/'heloc'
    e.select(args,evaluated_frames={s:pd.read_csv(out/f'bank_{s}_curves.csv') for s in ['train','cal']})


def freeze_eval_bank(args):
    e.freeze_eval_bank(args)
    out,meta,train,_=e.context(args)
    extra=out/'preserved_discovery_bank.json'
    if extra.exists():
        manifest=json.loads((out/'evaluation_bank.json').read_text())
        records=manifest['candidates']+json.loads(extra.read_text())['candidates']
        if len({a['candidate_id'] for a in records})!=len(records):raise ValueError('duplicate preserved attack')
        manifest['candidates']=e.verified_records(records,meta,train)
        manifest['preserved_discovery_manifest_sha256']=e.p.file_hash(extra)
        e.p.write_json(out/'evaluation_bank.json',manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','smoke','optimize','freeze_controls','paths','discover','freeze_eval_bank','score','merge','select','future_models','evaluate'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--dataset',default='heloc',choices=['heloc'])
    parser.add_argument('--device',default='cpu');parser.add_argument('--split',default='train')
    parser.add_argument('--worker',type=int,default=0);parser.add_argument('--workers',type=int,default=7)
    args=parser.parse_args()
    e.SETTINGS={**e.SETTINGS,'evaluation_radii':[3.]}
    e.verify_geometry=verify_paths;e.query_paths=lambda a,split:build_paths(a,split)
    if args.phase=='paths':build_paths(args,args.split)
    elif args.phase=='evaluate':e.evaluate(args,score_bank=False)
    elif args.phase=='future_models':e.future_models(args)
    else:globals()[args.phase](args)

if __name__=='__main__':main()
